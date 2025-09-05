import math
from functools import reduce
from itertools import combinations, chain

def get_divisors(n):
    """
    计算一个数 n 的所有正约数。
    返回一个包含所有约数的集合。
    """
    divs = {1, n}
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            divs.add(i)
            divs.add(n // i)
    return sorted(list(divs))

def calculate_gcd_for_set(numbers):
    """
    计算一个集合中所有数字的最大公约数。
    """
    if not numbers:
        return 0
    return reduce(math.gcd, numbers)

def calculate_lcm_for_set(numbers):
    """
    计算一个集合中所有数字的最小公倍数。
    """
    if not numbers:
        return 0
        
    def lcm(a, b):
        # 避免 a 或 b 为 0 的情况
        if a == 0 or b == 0:
            return 0
        return abs(a * b) // math.gcd(a, b) if a != 0 and b != 0 else 0

    return reduce(lcm, numbers, 1)

def count_valid_sets(n):
    """
    计算满足 gcd=1 且 lcm=n 的集合数量。
    
    参数:
        n (int): 目标整数，必须大于0。
        
    返回:
        int: 满足条件的集合数量。
    """
    if n <= 0:
        print("输入必须是正整数。")
        return 0
    
    # 特例：n=1，只有集合 {1} 满足条件 (gcd(1)=1, lcm(1)=1)
    if n == 1:
        return 1

    # 步骤 1: 找出 n 的所有约数
    divisors = get_divisors(n)
    
    # 步骤 2: 生成所有非空子集
    # chain.from_iterable 会将所有长度的组合连接成一个迭代器
    # combinations(divisors, r) 会生成长度为 r 的所有子集
    all_subsets = chain.from_iterable(combinations(divisors, r) for r in range(1, len(divisors) + 1))
    
    valid_sets_count = 0
    
    # 步骤 3 & 4: 遍历所有子集并检查条件
    for subset in all_subsets:
        # 跳过只有一个元素的集合，因为它的 gcd 和 lcm 都是它自己
        # 除非 n=1 (已处理)，否则不可能同时满足 gcd=1 和 lcm=n
        if len(subset) < 2:
            continue

        lcm_val = calculate_lcm_for_set(subset)
        
        # 优化：如果lcm已经不等于n，就没必要计算gcd了
        if lcm_val == n:
            gcd_val = calculate_gcd_for_set(subset)
            if gcd_val == 1:
                valid_sets_count += 1
                # 如果需要，可以取消下面的注释来查看是哪些集合满足条件
                # print(f"找到有效集合: {subset}")

    return valid_sets_count

# --- 示例 ---

# 示例 1: n = 4
# 我们分析过，结果应该是 2 ({1, 4} 和 {1, 2, 4})
n1 = 4
count1 = count_valid_sets(n1)
print(f"对于 n = {n1}, 满足条件的集合数量是: {count1}")

# 示例 2: n = 6
# 约数: [1, 2, 3, 6]
# 有效集合: {1, 6}, {2, 3}, {1, 2, 3}, {1, 2, 6}, {1, 3, 6}, {2, 3, 6}, {1, 2, 3, 6}
n2 = 6
count2 = count_valid_sets(n2)
print(f"对于 n = {n2}, 满足条件的集合数量是: {count2}")

# 示例 3: n = 12
# 约数: [1, 2, 3, 4, 6, 12]
# 有效集合包括 {1, 12}, {3, 4}, {1, 3, 4}, {1, 4, 6}, {3, 4, 6}, 等等...
n3 = 12
count3 = count_valid_sets(n3)
print(f"对于 n = {n3}, 满足条件的集合数量是: {count3}")

# 示例 4: n = 100
n4 = 100
count4 = count_valid_sets(n4)
print(f"对于 n = {n4}, 满足条件的集合数量是: {count4}")