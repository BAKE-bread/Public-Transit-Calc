# 数量少的线路组合的迭代精确解

import numpy as np
import math
from fractions import Fraction
from itertools import product

def float_lcm(numbers):
    """
    计算一个或多个浮点数的最小公倍数。
    
    Args:
        numbers (list or tuple): 包含浮点数或整数的列表。

    Returns:
        float: 所有输入数值的最小公倍数。
    """
    if not numbers:
        return 0.0

    # 将所有数字转换为最简分数
    fractions = [Fraction(str(n)).limit_denominator() for n in numbers]
    
    # 所有分母的最小公倍数作为新的公共分母
    denominators = [f.denominator for f in fractions]
    common_denominator = denominators[0]
    for i in range(1, len(denominators)):
        common_denominator = (common_denominator * denominators[i]) // math.gcd(common_denominator, denominators[i])
        
    # 将所有分数通分，并提取分子
    numerators = [f.numerator * (common_denominator // f.denominator) for f in fractions]
    
    # 计算所有分子的最小公倍数
    lcm_of_numerators = numerators[0]
    for i in range(1, len(numerators)):
        lcm_of_numerators = (lcm_of_numerators * numerators[i]) // math.gcd(lcm_of_numerators, numerators[i])
        
    # 最终的 LCM 是分子的 LCM 除以公共分母
    return float(lcm_of_numerators / common_denominator)


def optimize_n_line_schedule(T_values, steps=100):
    """
    根据N段换乘系统的离散相位偏移优化模型，计算最小总等待时间。

    Args:
        T_values (list or tuple): 包含所有线路发车周期的列表 (T1, T2, ..., Tn)。
                                  可以是整数或小数。单位需统一（如全为分钟或全为秒）。
        steps (int): 每个相位偏移量的离散化步数。步数越多结果越精确，但计算时间
                     会呈指数级增长 (复杂度 O(steps^(n-1)))。

    Returns:
        tuple: (最小总等待时间, 最优相位偏移向量元组)
               (min_total_wait_time, (best_delta_2, ..., best_delta_n))
    """
    num_lines = len(T_values)
    if num_lines < 2:
        raise ValueError("至少需要两条线路 (T1, T2) 才能进行换乘优化。")

    # --- 参数与变量 ---
    T1 = T_values[0]
    
    # 计算公共周期 P 和 L1 的发车班次数目 N1
    common_period = float_lcm(T_values)
    # 使用 round() 来避免浮点数精度问题
    N1 = int(round(common_period / T1))
    
    if N1 == 0:
         raise ValueError(f"计算出的 L1 班次数为0，请检查输入的周期 {T_values}")

    # --- 初始状态 (L1) ---
    # L1 乘客到达第一个换乘站的时刻
    k = np.arange(1, N1 + 1)
    # arrival_times_at_prev_station: 在每次迭代中，代表乘客到达下一个换乘站的时刻
    arrival_times_at_prev_station = k * T1
    E_W1 = T1 / 2.0

    min_total_wait_time = float('inf')
    best_deltas = None

    # --- 创建待优化的相位偏移量 (δ2, ..., δn) 的离散化搜索空间 ---
    # delta_ranges 是一个列表，每个元素是对应线路的相位偏移取值范围
    delta_ranges = [np.linspace(0, T_values[i], steps, endpoint=False) for i in range(1, num_lines)]
    
    # --- 优化目标 (求解) ---
    # 使用 itertools.product 高效遍历所有相位偏移组合
    # delta_offsets 将是一个元组 (δ2, δ3, ..., δn)
    num_combinations = steps ** (num_lines - 1)
    print(f"开始计算... 总计 {num_combinations} 种相位偏移组合。")
    
    for i, delta_offsets in enumerate(product(*delta_ranges)):
        
        current_arrival_times = arrival_times_at_prev_station.copy()
        current_total_E_wait = E_W1

        # --- 递归计算等待时间 ---
        # 遍历 L2, L3, ..., Ln
        for j in range(num_lines - 1):
            T_current = T_values[j + 1]
            delta_current = delta_offsets[j]
            
            # W_i,k = (δi - t_arr^(i-1)) mod Ti
            wait_times_current_line = np.mod(delta_current - current_arrival_times, T_current)
            
            # E[Wi]
            E_W_current = np.mean(wait_times_current_line)
            current_total_E_wait += E_W_current
            
            # t_arr^(i) = t_arr^(i-1) + W_i,k
            current_arrival_times = current_arrival_times + wait_times_current_line

        # 检查是否找到更优解
        if current_total_E_wait < min_total_wait_time:
            min_total_wait_time = current_total_E_wait
            best_deltas = delta_offsets
            
        if (i + 1) % (num_combinations // 10 or 1) == 0:
             print(f"  已完成 {((i + 1) / num_combinations) * 100:.0f}%...")

    return min_total_wait_time, best_deltas


# --- 示例：四条线路换乘系统 ---
if __name__ == '__main__':
    # 设置四条线路的发车周期，可以使用小数分钟
    # T1=5分钟, T2=7.5分钟, T3=4分钟, T4=12分钟
    T_values_example = [10, 6, 6.5, 8]
    
    # 离散化步数。注意：步数增加会导致计算时间急剧上升！
    # 对于n条线路会有n-1个偏移量
    discretization_steps = 60

    print("="*40)
    print("N段换乘系统优化模型")
    print("="*40)
    print(f"模型参数:")
    print(f"  线路周期 (T1...T{len(T_values_example)}): {T_values_example} 分钟")
    print(f"  相位偏移离散化步数: {discretization_steps}\n")

    # 执行优化计算
    min_wait_time, best_offsets = optimize_n_line_schedule(
        T_values_example, 
        steps=discretization_steps
    )
    
    print("\n--- 计算结果 ---")
    
    if best_offsets:
        formatted_offsets = ", ".join([f"{d:.4f}" for d in best_offsets])
        print(f"最优相位偏移 (δ2*...δ{len(T_values_example)}*): ({formatted_offsets})")
        print(f"对应的最小总期望等待时间 E_total*: {min_wait_time:.4f} 分钟")
    else:
        print("未能找到最优解。")