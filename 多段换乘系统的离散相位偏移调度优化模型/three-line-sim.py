import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from scipy.ndimage import minimum_filter

# --- 1. 中文显示设置 ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# --- 2. 系统参数 (min) ---
T1 = 6.0
T2 = 6.5
T3 = 7.0
LCM_GRAND = 546.0
EPSILON = 1e-9

# --- 3. 扫描精度 ---
# 使用比较高的精度以获得更平滑的图像 (这里是约0.6s)
n_points_delta = 1000

# --- 4. 核心计算函数 ---
def calculate_wait_times(t_arrival, T, delta, epsilon):
    """一个健壮且支持向量化广播的等待时间计算函数"""
    time_since_delta_epoch = t_arrival - delta
    num_prev_intervals = np.floor(time_since_delta_epoch / T)
    t_prev_or_current_depart = delta + num_prev_intervals * T
    t_next_depart = t_prev_or_current_depart + T
    passenger_departs_at = np.where(
        (t_arrival - t_prev_or_current_depart) > epsilon,
        t_next_depart,
        t_prev_or_current_depart
    )
    wait_times = passenger_departs_at - t_arrival
    return wait_times, passenger_departs_at

# --- 5. 定义相位偏移扫描范围 ---
offsets_2 = np.linspace(0, T2, n_points_delta)
offsets_3 = np.linspace(0, T3, n_points_delta)

print("--- 您定义的最终模型：基于L1离散班次时刻的期望值分析 ---")

# --- 6. 模拟过程 ---
# 输入源是L1在一个周期内所有的、离散的发车时刻
t_depart_A_schedule = np.arange(T1, LCM_GRAND + T1, T1)
print(f"输入源: L1 的 {len(t_depart_A_schedule)} 个离散发车班次 (t = 6, 12, ..., 546)")
print(f"变量: {n_points_delta}x{n_points_delta} 个 (δ₂, δ₃) 偏移组合")
start_time = time.time()

# 准备存储网格
E_W1 = T1 / 2  # E[W₁] 固定为 3.0
E_W2_grid = np.zeros(n_points_delta)
E_W3_grid = np.zeros((n_points_delta, n_points_delta))

print("\n步骤1/2: 遍历所有偏移组合并计算平均换乘等待...")
# 遍历所有(δ₂, δ₃)组合，计算91次初始乘坐可能的平均情况
for i, delta_2 in enumerate(tqdm(offsets_2, desc="扫描 L2 偏移 (δ₂)")):
    W2, t_depart_B = calculate_wait_times(t_depart_A_schedule, T2, delta_2, EPSILON)
    E_W2_grid[i] = np.mean(W2)
    
    offsets_3_col = offsets_3[:, np.newaxis]
    W3_grid_col, _ = calculate_wait_times(t_depart_B, T3, offsets_3_col, EPSILON)
    E_W3_grid[i, :] = np.mean(W3_grid_col, axis=1)
    
    # 输出中间相关信息
    if (i + 1) % 100 == 0 or i == n_points_delta - 1:
        # 临时计算到目前为止的最小总等待时间
        temp_total_wait = E_W1 + E_W2_grid[:i+1, np.newaxis] + E_W3_grid[:i+1, :]
        current_min_wait = np.min(temp_total_wait)
        print(f"\n[中间状态] 已处理 {i+1}/{n_points_delta} 个δ₂。目前找到的最小总等待: {current_min_wait:.4f} 分钟")

print("\n步骤2/2: 整合数据并进行分析...")

total_wait_grid = E_W1 + E_W2_grid[:, np.newaxis] + E_W3_grid
end_time = time.time()
print(f"\n--- 计算完成，总耗时: {end_time - start_time:.2f} 秒 ---")

# --- 7. 结果分析 ---
footprint = np.ones((3, 3))
local_min_mask = (total_wait_grid == minimum_filter(total_wait_grid, footprint=footprint))
local_min_indices = np.where(local_min_mask)
local_min_waits = total_wait_grid[local_min_indices]
local_min_delta2 = offsets_2[local_min_indices[0]]
local_min_delta3 = offsets_3[local_min_indices[1]]
sorted_local_minima = sorted(zip(local_min_waits, local_min_delta2, local_min_delta3))

print("\n" + "="*50)
print("              所有局部极小值点分析")
print("="*50)
if not sorted_local_minima:
    print("未找到局部极小值点。")
else:
    print(f"共找到 {len(sorted_local_minima)} 个局部极小值点。")
    print("按等待时间从低到高排序如下：\n")
    for i, (wait, d2, d3) in enumerate(sorted_local_minima[:20]):
        marker = "👑 全局最优点" if i == 0 else f"   局部极小点 #{i+1}"
        print(f"{marker}:")
        print(f"  - 总等待时间: {wait:.4f} 分钟")
        print(f"  - L2 偏移 (δ₂): {d2:.4f}, L3 偏移 (δ₃): {d3:.4f}\n")
    if len(sorted_local_minima) > 20: print(f"... (只显示前20个) ...\n")

    print("-" * 30)
    global_min_wait, optimal_delta_2, optimal_delta_3 = sorted_local_minima[0]
    optimal_idx = np.unravel_index(np.argmin(total_wait_grid), total_wait_grid.shape)
    optimal_E_W2_val = E_W2_grid[optimal_idx[0]]
    optimal_E_W3_val = E_W3_grid[optimal_idx]

    print("全局最优条件下，各项期望等待时间分解:")
    print(f"  - E[W1] (L1等待): {E_W1:.4f} 分钟 (固定值)")
    print(f"  - E[W2] (L2等待): {optimal_E_W2_val:.4f} 分钟")
    print(f"  - E[W3] (L3等待): {optimal_E_W3_val:.4f} 分钟")
    print(f"  - 验证: E[W1]+E[W2]+E[W3] = {E_W1 + optimal_E_W2_val + optimal_E_W3_val:.4f} 分钟")
print("="*50)

# --- 8. 可视化 ---
fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(total_wait_grid.T, cmap='viridis_r', origin='lower',
               extent=[offsets_2[0], offsets_2[-1], offsets_3[0], offsets_3[-1]],
               aspect='auto')
cbar = fig.colorbar(im)
cbar.set_label('总期望等待时间 (分钟)', fontsize=12)
ax.set_xlabel('L2 相位偏移 ($\delta_2$) [分钟]', fontsize=12)
ax.set_ylabel('L3 相位偏移 ($\delta_3$) [分钟]', fontsize=12)
ax.set_title(f'总期望等待时间 E($\delta_2, \delta_3$) | 精确班次模型', fontsize=14, pad=20)

ax.scatter(local_min_delta2, local_min_delta3, 
           facecolors='none', 
           edgecolors='cyan',  # 使用更醒目的颜色，如青色
           s=40,               # 减小尺寸
           linewidths=1.0,     # 减小线宽
           alpha=0.6,          # 增加透明度
           label='局部极小值点')

# 全局最优点的标记依然保持突出
if sorted_local_minima:
    global_min_wait, optimal_delta_2_val, optimal_delta_3_val = sorted_local_minima[0]
    ax.plot(optimal_delta_2_val, optimal_delta_3_val, 'r*', markersize=18, markeredgecolor='white',
            label=f'全局最优点\n($\delta_2$={optimal_delta_2_val:.2f}, $\delta_3$={optimal_delta_3_val:.2f})')

ax.legend(loc='upper right', fontsize=11)
ax.grid(True, linestyle='--', alpha=0.5)
plt.show()


# --- 参数定义 ---
T1 = 6.0
T2 = 6.5
T3 = 7.0

N = 91 # LCM / T1
delta_2 = 0.0
delta_3 = 0.0

# --- L1发车时刻表 ---
t_depart_A = np.arange(1, N + 1) * T1

# --- 计算 E[W2] ---
# W2 = (delta_2 - t) % T2
wait_times_2 = (delta_2 - t_depart_A) % T2
E_W2 = np.mean(wait_times_2)

# --- 计算 E[W3] ---
# 到达的时刻 = L1出发时刻 + W2
t_arrival_C = t_depart_A + wait_times_2
# W3 = (delta_3 - t) % T3
wait_times_3 = (delta_3 - t_arrival_C) % T3
E_W3 = np.mean(wait_times_3)

# --- 计算总等待时间 ---
E_W1 = T1 / 2
E_total = E_W1 + E_W2 + E_W3

# --- 输出结果 ---
print("--- 数学模型精确验证 ---")
print(f"验证点: (δ₂, δ₃) = ({delta_2}, {delta_3})")
print("-" * 30)
print(f"E[W1] (理论值): {E_W1:.4f}")
print(f"E[W2] (数学计算): {E_W2:.4f}")
print(f"E[W3] (数学计算): {E_W3:.4f}")
print("-" * 30)
print(f"E[Total] (数学计算): {E_total:.4f}")
print(f"您的模拟结果: 9.2308")