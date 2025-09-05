import math
from itertools import combinations, chain

def prime_factorize(n):
    """
    对 n 进行质因数分解。
    返回一个字典 {prime: exponent}。
    复杂度: O(sqrt(n))
    """
    factors = {}
    d = 2
    temp_n = n
    while d * d <= temp_n:
        while (temp_n % d) == 0:
            factors[d] = factors.get(d, 0) + 1
            temp_n //= d
        d += 1
    if temp_n > 1:
        factors[temp_n] = factors.get(temp_n, 0) + 1
    return factors

def get_num_divisors_from_factors(factors):
    """
    根据质因数分解字典计算约数个数。
    """
    if not factors:
        return 1
    num = 1
    for exp in factors.values():
        num *= (exp + 1)
    return num

def count_valid_sets_optimized(n):
    """
    使用基于容斥原理的优化算法计算满足条件的集合数量。
    复杂度: O(sqrt(n) + 2^(2*ω(n)))
    """
    if n <= 0:
        print("输入必须是正整数。")
        return 0
    if n == 1:
        return 1

    # 步骤 1: 对 n 进行质因数分解
    factors_of_n = prime_factorize(n)
    primes = list(factors_of_n.keys())
    r = len(primes)

    # 步骤 2: 定义所有的 2r 个“失败条件”
    # 'max' failure: 指数 < eᵢ  (等价于所有元素都是 n/pᵢ 的约数)
    # 'min' failure: 指数 > 0   (等价于所有元素都是 pᵢ 的倍数)
    failure_conditions = []
    for p in primes:
        failure_conditions.append({'p': p, 'type': 'max'})
        failure_conditions.append({'p': p, 'type': 'min'})

    total_valid_count = 0

    # 步骤 3: 遍历所有失败条件的组合 (从 0 到 2r 个)
    # 这对应容斥原理中的所有项
    for i in range(len(failure_conditions) + 1):
        for failure_subset in combinations(failure_conditions, i):
            
            # 对每个失败组合，计算满足条件的约数数量
            # d_p: 'min' 失败涉及的质数乘积
            # d_q: 'max' 失败涉及的质数乘积
            d_p, d_q = 1, 1
            
            for cond in failure_subset:
                if cond['type'] == 'min':
                    d_p *= cond['p']
                else: # 'max'
                    d_q *= cond['p']

            denominator = d_p * d_q
            
            if n % denominator != 0:
                # 如果分母不整除n，说明没有这样的约数
                num_divisors_for_term = 0
            else:
                # 计算 n / denominator 的质因数分解
                temp_factors = factors_of_n.copy()
                
                # 从分子中减去分母的质因子指数
                temp_denom = denominator
                for p, exp in temp_factors.items():
                    while temp_denom % p == 0:
                        temp_factors[p] -= 1
                        temp_denom //= p
                
                num_divisors_for_term = get_num_divisors_from_factors(temp_factors)
            
            # 容斥原理的项: (-1)^|F| * 2^(满足F条件的约数个数)
            term = 2**num_divisors_for_term
            sign = (-1)**len(failure_subset)
            
            total_valid_count += sign * term
            
    return total_valid_count

# --- 示例 ---

# 示例 1: n = 4, 结果应为 2
n1 = 4
count1 = count_valid_sets_optimized(n1)
print(f"对于 n = {n1}, 满足条件的集合数量是: {count1}")

# 示例 2: n = 12, 结果应为 8
n2 = 12
count2 = count_valid_sets_optimized(n2)
print(f"对于 n = {n2}, 满足条件的集合数量是: {count2}")

# 示例 3: n = 100, 结果应为 322
n3 = 100
count3 = count_valid_sets_optimized(n3)
print(f"对于 n = {n3}, 满足条件的集合数量是: {count3}")

# 示例 4: n = 720, 这是之前暴力算法无法计算的
# 720 = 2^4 * 3^2 * 5^1. ω(720)=3. 
# 暴力法依赖 d(720)=30, 优化算法依赖 ω(720)=3, 速度极快
n4 = 720720
count4 = count_valid_sets_optimized(n4)
print(f"对于 n = {n4}, 满足条件的集合数量是: {count4}")