#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <chrono>

#define TOTAL_TASKS 1000000
#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32

using real = double; 

__device__ real heavy_compute(int workload) {
    real acc = 0.0;
    for (int i = 0; i < workload * 10000; ++i) {
        acc += sinf(i * 0.0001f) * cosf(i * 0.0002f);
    }
    return acc;
}

// --- Kernel 1: Static scheduling ---
__global__ void static_kernel(int* workloads, real* results, int total_tasks) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < total_tasks; i += stride) {
        int w = workloads[i];
        real r = heavy_compute(w);
        results[i] = r;
    }
}

// --- Kernel 2: Basic dynamic scheduling ---
__global__ void dynamic_kernel(int* workloads, int* task_counter, real* results, int total_tasks) {
    while (true) {
        int task_id = atomicAdd(task_counter, 1);
        if (task_id >= total_tasks) break;
        int w = workloads[task_id];
        real r = heavy_compute(w);
        results[task_id] = r;
        __threadfence();
    }
}

// --- Kernel 3: Hierarchical warp-level dynamic scheduling ---
__global__ void hierarchical_kernel(int* workloads, int* counter, real* results, int total_tasks) {
    const int lane_id = threadIdx.x % WARP_SIZE;

    while (true) {
        int task_id = -1;
        if (lane_id == 0) {
            task_id = atomicAdd(counter, WARP_SIZE);  // each warp take a task
        }
        task_id = __shfl_sync(0xffffffff, task_id, 0);
        if (task_id >= total_tasks) break;

        int my_task = task_id + lane_id;
        if (my_task < total_tasks) {
            int w = workloads[my_task];
            real r = heavy_compute(w);
            results[my_task] = r;
            __threadfence();
        }
    }
}

// --- Helper ---
void run_kernel(const char* name, int mode) {
    std::vector<int> workloads(TOTAL_TASKS);
    std::vector<real> results(TOTAL_TASKS, 0.0);

    // Generate heavy-tailed workload: 90% light, 10% heavy
    for (int i = 0; i < TOTAL_TASKS; ++i)
        workloads[i] = (rand() % 100 < 90) ? 1 : 80 + rand() % 40;

    int *d_workloads, *d_counter = nullptr;
    real *d_results;

    cudaMalloc(&d_workloads, TOTAL_TASKS * sizeof(int));
    cudaMemcpy(d_workloads, workloads.data(), TOTAL_TASKS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&d_results, TOTAL_TASKS * sizeof(real));
    cudaMemset(d_results, 0, TOTAL_TASKS * sizeof(real));

    if (mode != 0) {
        cudaMalloc(&d_counter, sizeof(int));
        cudaMemset(d_counter, 0, sizeof(int));
    }

    dim3 blocks(128), threads(THREADS_PER_BLOCK);
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    switch (mode) {
        case 0: static_kernel<<<blocks, threads>>>(d_workloads, d_results, TOTAL_TASKS); break;
        case 1: dynamic_kernel<<<blocks, threads>>>(d_workloads, d_counter, d_results, TOTAL_TASKS); break;
        case 2: hierarchical_kernel<<<blocks, threads>>>(d_workloads, d_counter, d_results, TOTAL_TASKS); break;
        default: std::cerr << "Invalid mode\n"; return;
    }

    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float elapsed = 0;
    cudaEventElapsedTime(&elapsed, start, stop);

    cudaMemcpy(results.data(), d_results, TOTAL_TASKS * sizeof(real), cudaMemcpyDeviceToHost);

    // checksum sampling
    real partial_sum = 0, full_sum = 0;
    for (int i = 0; i < TOTAL_TASKS; ++i) {
        full_sum += results[i];
        if (i % (TOTAL_TASKS / 100) == 0) partial_sum += results[i];
    }

    std::cout << name << " scheduling took " << elapsed << " ms. ";
    std::cout << "Checksum (partial): " << partial_sum << " | Total: " << full_sum << "\n";

    cudaFree(d_workloads);
    cudaFree(d_results);
    if (d_counter) cudaFree(d_counter);
    cudaEventDestroy(start); cudaEventDestroy(stop);
}

// --- Main ---
int main(int argc, char** argv) {
    std::cout << "CUDA Dynamic Load Balancing Demo\n";

    if (argc < 2) {
        std::cout << "Usage: ./balanced_cuda [static|dynamic|hierarchical]\n";
        return 1;
    }

    std::string mode = argv[1];
    if (mode == "static") run_kernel("Static", 0);
    else if (mode == "dynamic") run_kernel("Dynamic", 1);
    else if (mode == "hierarchical") run_kernel("Hierarchical", 2);
    else {
        std::cerr << "Unknown mode.\n";
        return 1;
    }

    return 0;
}
