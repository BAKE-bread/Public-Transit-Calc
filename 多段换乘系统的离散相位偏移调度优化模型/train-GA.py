# 遗传算法，仅寻找一组解

import pygad
import numpy as np
import math
from fractions import Fraction
import matplotlib.pyplot as plt

# 设置matplotlib以正确显示中文字符
plt.rcParams['font.sans-serif'] = ['SimHei']  # 'SimHei' 是黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题


# --- 辅助函数：计算浮点数的最小公倍数 ---
def float_lcm(numbers):
    if not numbers:
        return 0.0
    fractions = [Fraction(str(n)).limit_denominator() for n in numbers]
    denominators = [f.denominator for f in fractions]
    common_denominator = denominators[0]
    for i in range(1, len(denominators)):
        common_denominator = (common_denominator * denominators[i]) // math.gcd(common_denominator, denominators[i])
    numerators = [f.numerator * (common_denominator // f.denominator) for f in fractions]
    lcm_of_numerators = numerators[0]
    for i in range(1, len(numerators)):
        lcm_of_numerators = (lcm_of_numerators * numerators[i]) // math.gcd(lcm_of_numerators, numerators[i])
    return float(lcm_of_numerators / common_denominator)


# --- 核心逻辑：计算给定相位偏移下的总等待时间 (目标函数的核心) ---
def calculate_total_wait_time(deltas, T_values):
    """
    计算给定相位偏移向量下的总期望等待时间。
    Args:
        deltas (list or np.array): 包含 (δ2, δ3, ..., δn) 的一维数组。
        T_values (list): 包含 (T1, T2, ..., Tn) 的完整周期列表。
    Returns:
        float: 总期望等待时间 E_total。
    """
    num_lines = len(T_values)
    T1 = T_values[0]
    
    # 计算公共周期和L1班次数
    common_period = float_lcm(T_values)
    if T1 == 0: return float('inf')
    N1 = int(round(common_period / T1))
    if N1 == 0: return float('inf')
    
    # L1到达时刻
    k = np.arange(1, N1 + 1)
    current_arrival_times = (k * T1).astype(np.float64)
    
    E_W1 = T1 / 2.0
    total_E_wait = E_W1

    # 递归计算 L2, L3, ... Ln 的等待时间
    for j in range(num_lines - 1):
        T_current = T_values[j + 1]
        delta_current = deltas[j]
        wait_times = np.mod(delta_current - current_arrival_times, T_current)
        total_E_wait += np.mean(wait_times)
        current_arrival_times += wait_times
        
    return total_E_wait


# --- 遗传算法部分 ---

# 定义适应度函数 (Fitness Function)
# PyGAD默认执行最大化任务，而我们的目标是最小化等待时间。
# 因此，适应度函数可以定义为等待时间的倒数 (1 / E_total)。
# 适应度越高，代表等待时间越短，解的质量越好。
def fitness_func(ga_instance, solution, solution_idx):
    """
    PyGAD 的适应度函数包装器。
    Args:
        ga_instance: 当前的GA实例。
        solution (list): 一个候选解，即相位偏移向量 (δ2, δ3, ..., δn)。
        solution_idx (int): 该解在种群中的索引。
    Returns:
        float: 该解的适应度值。
    """
    # 从主程序中获取 T_values
    global T_values_example
    
    wait_time = calculate_total_wait_time(solution, T_values_example)
    
    # 避免除以零的错误
    if wait_time == 0:
        return float('inf')
    
    fitness = 1.0 / wait_time
    return fitness


# --- 适应度曲线绘图函数 ---
def plot_detailed_fitness_convergence(ga_instance):
    """
    绘制带有网格和收敛值标记的详细适应度收敛曲线。
    """
    plt.figure(figsize=(12, 7))
    
    # 1. 获取适应度历史数据
    fitness_history = ga_instance.best_solutions_fitness
    generations = range(len(fitness_history))
    
    # 2. 绘制主要的适应度曲线
    plt.plot(generations, fitness_history, lw=3, color='#66cc00', label='每代最优适应度')
    
    # 3. 找到并标记最终收敛值
    # ga_instance.best_solution() 返回的是整个运行过程中的最优解
    best_solution, best_fitness, best_idx = ga_instance.best_solution()
    best_generation_num = ga_instance.best_solution_generation
    
    # 绘制一条水平虚线，标示出收敛到的水平
    plt.axhline(y=best_fitness, color='red', linestyle='--', lw=2, 
                label=f'收敛适应度: {best_fitness:.5f}')
    
    # 在首次达到最优解的位置，绘制一个星形标记
    plt.plot(best_generation_num, best_fitness, 'r*', markersize=15, 
             label=f'在第 {best_generation_num} 代首次达到')
    
    # 4. 添加网格和各种标签
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.title('GA 适应度收敛曲线 (含详细标记)', fontsize=16)
    plt.xlabel('迭代代数', fontsize=12)
    plt.ylabel('适应度 (1 / 总等待时间)', fontsize=12)
    plt.legend(fontsize='large')
    plt.show()
    

# --- 增强的模拟时刻表可视化函数 ---
def visualize_optimized_schedule(T_values, best_deltas):
    """
    可视化优化后的时刻表，并自动寻找并绘制后续总换乘等待时间最短的“黄金路径”。
    """
    num_lines = len(T_values)
    T1 = T_values[0]
    
    # --- 核心改进：寻找“黄金路径” ---
    # 1. 计算在一个公共周期内，L1的总班次数
    common_period = float_lcm(T_values)
    if T1 == 0:
        print("错误：T1不能为零。")
        return
    N1 = int(round(common_period / T1))
    if N1 == 0:
        print(f"错误：根据周期 {T_values} 计算出L1班次数为0。")
        return

    min_path_wait_time = float('inf')
    best_start_time = -1

    # 2. 遍历所有L1到达时刻，计算每条路径的总换乘等待时间
    for k in range(1, N1 + 1):
        path_start_time = k * T1
        path_total_wait = 0
        
        current_t = path_start_time
        # 模拟换乘过程
        for i in range(num_lines - 1):
            T_next = T_values[i+1]
            delta_next = best_deltas[i]
            wait_time = (delta_next - current_t) % T_next
            path_total_wait += wait_time
            current_t += wait_time # 更新时刻为下一程的出发时刻
        
        # 如果当前路径的总等待时间更短，则记录下来
        if path_total_wait < min_path_wait_time:
            min_path_wait_time = path_total_wait
            best_start_time = path_start_time

    print(f"\nINFO: [可视化] 在所有{N1}个L1换乘路径中，找到的最佳路径始于 t={best_start_time:.2f}。")
    print(f"INFO: [可视化] 该最佳路径的总换乘等待时间为: {min_path_wait_time:.2f} 分钟。")

    # --- 开始绘图 ---
    plt.figure(figsize=(16, num_lines * 1.5))
    plot_duration = max(best_start_time + min_path_wait_time + max(T_values), max(T_values) * 2.5)
    
    # 1. 绘制各条线路的发车时刻表
    line_names = [f'L{i+1}' for i in range(num_lines)]
    line_positions = np.arange(num_lines, 0, -1)

    for i in range(num_lines):
        T = T_values[i]
        delta = 0 if i == 0 else best_deltas[i-1]
        departures = np.arange(0, plot_duration + T, T) + delta
        plt.eventplot(departures, lineoffsets=line_positions[i], linelengths=0.8, 
                      colors=f'C{i}', label=f'L{i+1} 发车时刻 (T={T})')

    # 2. 绘制找到的最佳路径
    t_arrival = best_start_time
    plt.scatter(t_arrival, line_positions[0], s=100, c='red', marker='v', zorder=5, label='乘客从L1到达 (最佳路径起点)')

    current_t = t_arrival
    total_wait_labels = []
    for i in range(num_lines - 1):
        T_next = T_values[i+1]
        delta_next = best_deltas[i]
        
        wait_time = (delta_next - current_t) % T_next
        t_depart_next = current_t + wait_time
        total_wait_labels.append(f'L{i+2} 等待: {wait_time:.2f} min')
        
        plt.plot([current_t, t_depart_next], [line_positions[i+1], line_positions[i+1]], 'r--', lw=2)
        plt.plot([current_t, current_t], [line_positions[i], line_positions[i+1]], 'k:', lw=1)
        plt.scatter(t_depart_next, line_positions[i+1], s=100, c=f'C{i+1}', marker='o', zorder=5)
        current_t = t_depart_next

    # 为等待路径创建一个统一的图例标签
    if total_wait_labels:
        plt.plot([], [], 'r--', lw=2, label='\n'.join(total_wait_labels))
    plt.plot([], [], marker='o', color='gray', linestyle='None', label='搭乘后续线路')

    # 3. 设置图表样式
    plt.title('优化后的时刻表与“黄金”换乘路径', fontsize=16)
    plt.xlabel('时间 (分钟)', fontsize=12)
    plt.ylabel('线路', fontsize=12)
    plt.yticks(line_positions, line_names)
    plt.grid(axis='x', linestyle=':', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='medium')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()


# --- 主程序入口 ---
if __name__ == '__main__':
    T_values_example = [6, 6.5, 7]
    num_variables_to_optimize = len(T_values_example) - 1

    print("="*40)
    print("使用遗传算法进行N段换乘系统优化")
    print("="*40)
    print(f"模型参数: 线路周期 (T1...T{len(T_values_example)}): {T_values_example} 分钟")

    # 配置遗传算法，加入早停策略
    ga_instance = pygad.GA(
        num_generations=500,
        stop_criteria="saturate_30",  # 30代适应度不提升则早停
        num_parents_mating=10,
        fitness_func=fitness_func,
        sol_per_pop=50,
        num_genes=num_variables_to_optimize,
        gene_space=[{'low': 0, 'high': T_values_example[i]} for i in range(1, len(T_values_example))],
        parent_selection_type="sss",
        keep_parents=5,
        crossover_type="single_point",
        mutation_type="random",
        mutation_num_genes=1 # 直接指定变异1个基因，避免警告
    )

    print("\n开始运行遗传算法进行优化 (带早停策略)...")
    ga_instance.run()
    print("优化完成。")

    # 输出结果，加入更多信息
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    
    print("\n--- 优化结果 ---")
    if ga_instance.best_solution_generation != -1:
        print(f"INFO: 最优解在第 {ga_instance.best_solution_generation} 代被发现 (总共运行了 {ga_instance.generations_completed} 代)。")
    
    min_wait_time = 1.0 / solution_fitness
    formatted_offsets = ", ".join([f"{d:.4f}" for d in solution])
    print(f"最优解的适应度值 (1/E_total): {solution_fitness:.4f}")
    print(f"计算出的最小总期望等待时间 E_total*: {min_wait_time:.4f} 分钟")
    print(f"对应的最优相位偏移 (δ2*...δ{len(T_values_example)}*): ({formatted_offsets})")

    # 可视化适应度曲线和新的时刻协调图
    # ga_instance.plot_fitness(title="GA 适应度收敛曲线", xlabel="迭代代数", ylabel="适应度 (1 / 总等待时间)")
    plot_detailed_fitness_convergence(ga_instance)
    visualize_optimized_schedule(T_values_example, solution)