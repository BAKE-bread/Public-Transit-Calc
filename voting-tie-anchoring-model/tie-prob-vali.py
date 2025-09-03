# Enhanced simulation and analysis code with corrected FFT implementation.
# This code will:
# 1. Run a fast Monte Carlo simulation for larger n.
# 2. Provide a greatly enhanced exact enumeration for small n (now including n=5),
#    which categorizes tie types and calculates expected tied pairs.
# 3. Implement a new FFT-based method to precisely calculate the single-pair
#    tie probability P(S_a = S_b) to diagnose lattice and Edgeworth effects.
# 4. Compare results for n=3, 4, 5, 6 to investigate the dip at n=5.

import numpy as np
import itertools
import time
from collections import Counter
from scipy.fft import next_fast_len

rng = np.random.default_rng(12345)

def fast_simulation(n, num_simulations=100000, batch_size=5000, rng=None):
    """
    Faster Monte Carlo simulation (vectorized in batches).
    """
    if rng is None: rng = np.random.default_rng()
    if n <= 1: return 0.0
    if n == 2: return 1.0

    base_scores = np.arange(n-1, 0, -1, dtype=int)
    collisions = 0
    sims_done = 0
    while sims_done < num_simulations:
        b = min(batch_size, num_simulations - sims_done)
        totals = np.zeros((b, n), dtype=int)

        for i in range(n):
            off_idx = [j for j in range(n) if j != i]
            perms = rng.permuted(np.tile(base_scores, (b, 1)), axis=1)
            totals[:, i] += n
            for pos, col_j in enumerate(off_idx):
                totals[:, col_j] += perms[:, pos]

        for row in range(b):
            if len(np.unique(totals[row])) != n:
                collisions += 1
        sims_done += b
    return collisions / num_simulations

def enhanced_exact_enumeration(n):
    """
    Enhanced exact enumeration for small n (now handles n<=5).
    It returns not just the overall tie probability, but also detailed statistics
    about the types of ties, directly testing the "combinatorial coincidence" hypothesis.
    """
    if n <= 1: 
        return {}
    if n == 2: 
        return {'p_any_tie': 1.0, 'E_tied_pairs': 1.0, 'tie_type_dist': {1: 1.0}}
    
    base_scores = list(range(n - 1, 0, -1))
    perms = list(itertools.permutations(base_scores))

    total_configs = len(perms)**n
    collision_configs = 0

    # Statistics to collect
    total_tied_pairs = 0
    tie_type_counts = Counter() # Counts configurations by number of distinct scores

    # For n>=5, this loop is computationally intensive.
    print(f"Starting exact enumeration for n={n}, total configurations to check: {total_configs}...")
    for choice in itertools.product(perms, repeat=n):
        totals = np.zeros(n, dtype=int)
        for i in range(n):
            totals[i] += n
            off_idx = [j for j in range(n) if j != i]
            for pos, col_j in enumerate(off_idx):
                totals[col_j] += choice[i][pos]

        unique_scores, counts_per_score = np.unique(totals, return_counts=True)
        num_distinct_scores = len(unique_scores)

        tie_type_counts[num_distinct_scores] += 1

        if num_distinct_scores != n:
            collision_configs += 1 
            # Calculate number of tied pairs for this configuration
            for count in counts_per_score:
                total_tied_pairs += count * (count - 1) // 2
    
    # Compile results
    results = {
        'p_any_tie': collision_configs / total_configs, 'E_tied_pairs': total_tied_pairs / total_configs,
        'tie_type_dist': {k: v / total_configs for k, v in tie_type_counts.items()},
        'collision_configs': collision_configs, 'total_configs': total_configs
    }
    return results

def get_pmf_array(vals_and_probs, min_support, max_support):
    """Helper to create a PMF array given its support range."""
    size = max_support - min_support + 1
    pmf = np.zeros(size)
    for val, prob in vals_and_probs.items():
        pmf[val - min_support] = prob
    return pmf

def calculate_pairwise_tie_pmf(n, use_fft=True):
    """
    Calculates the exact probability mass function (PMF) of the score difference
    D_ab = S_a - S_b using characteristic functions (via FFT).
    This directly tests for lattice/Edgeworth effects on the single-pair tie probability.
    This version handles support ranges and zero-offsets.
    """
    # Define score vector and sub-vector for non-first places
    w_sub = np.arange(n - 1, 0, -1)
    prob = 1.0 / (n - 1)

    # 1. Define PMFs and supports for each X_i
    # Case i=a: X_a = n - W
    min_Xa, max_Xa = n - w_sub.max(), n - w_sub.min()
    pmf_Xa = get_pmf_array({n - val: prob for val in w_sub}, min_Xa, max_Xa)

    # Case i=b: X_b = W' - n
    min_Xb, max_Xb = w_sub.min() - n, w_sub.max() - n
    pmf_Xb = get_pmf_array({val - n: prob for val in w_sub}, min_Xb, max_Xb)

    # Case i != a,b: X_i = U - V
    min_Xi, max_Xi = 1 - (n - 1), (n - 1) - 1
    vals_Xi = Counter(u - v for u in w_sub for v in w_sub if u != v)
    total_pairs = (n - 1) * (n - 2)
    pmf_Xi = get_pmf_array({val: count / total_pairs for val, count in vals_Xi.items()}, min_Xi, max_Xi)
    
    # 2. Calculate final support and required FFT length
    min_Dab = min_Xa + min_Xb + (n - 2) * min_Xi
    max_Dab = max_Xa + max_Xb + (n - 2) * max_Xi
    
    if not use_fft:
        # Direct convolution for validation
        final_pmf = np.convolve(pmf_Xa, pmf_Xb)
        for _ in range(n - 2):
            final_pmf = np.convolve(final_pmf, pmf_Xi)
    else:
        # FFT-based convolution
        final_len = len(pmf_Xa) + len(pmf_Xb) + (n-2)*(len(pmf_Xi)-1) -1 # Exact length needed
        fft_len = next_fast_len(final_len)

        cf_Xa = np.fft.fft(pmf_Xa, n=fft_len)
        cf_Xb = np.fft.fft(pmf_Xb, n=fft_len)
        cf_Xi = np.fft.fft(pmf_Xi, n=fft_len)
        
        total_cf = cf_Xa * cf_Xb * (cf_Xi ** (n - 2))
        final_pmf = np.fft.ifft(total_cf).real

    # 3. Extract P(D_ab = 0)
    # The probability of D_ab=k is at index k - min_Dab. So P(D_ab=0) is at -min_Dab.
    p_tie_single_pair = final_pmf[-min_Dab]
    return p_tie_single_pair, min_Dab, final_pmf

# --- Main Execution ---

# Step 1: Validation for n=3
print(f"\n{'='*10} Validation for n=3 {'='*10}")
p_fft, min_D_fft, _ = calculate_pairwise_tie_pmf(3, use_fft=True)
p_conv, min_D_conv, _ = calculate_pairwise_tie_pmf(3, use_fft=False)
print(f"  P(Sa=Sb) from FFT:      {p_fft:.7f}")
print(f"  P(Sa=Sb) from direct conv: {p_conv:.7f}")
print(f"  Theoretical P(Sa=Sb) from E[Tied Pairs]/3: {0.75/3:.7f}")
assert np.isclose(p_fft, p_conv) and np.isclose(p_fft, 0.25), "Validation failed for n=3"
print("  Validation Successful!")

# Step 2: Corrected FFT Analysis of Single-Pair Tie Probability for n=3 to 10
print(f"\n{'='*10} Corrected P(Sa=Sb) via FFT {'='*10}")
print("  n | P(Sa=Sb)  | E[Tied Pairs] approx. | Trend")
print("----|-----------|-----------------------|-------")
p_prev = -1
for n in range(3, 11):
    p_single_pair, _, _ = calculate_pairwise_tie_pmf(n, use_fft=True)
    expected_pairs_approx = (n * (n-1) / 2) * p_single_pair
    trend = '↓' if p_single_pair < p_prev else '↑' if p_prev!=-1 else '-'
    p_prev = p_single_pair
    print(f" {n:2d} | {p_single_pair:.7f} | {expected_pairs_approx:21.6f} |  {trend}")

# Step 3: Rerun Enumeration and Simulation to connect insights
print(f"\n{'='*10} Final Insights from Enumeration {'='*10}")
for n in range(3, 6):
    print(f"\n{'='*10} n = {n} (Exact Enumeration Results) {'='*10}")
    t0 = time.time()
    exact_results = enhanced_exact_enumeration(n)
    t1 = time.time()
    
    p_any_tie = exact_results['p_any_tie']
    E_tied_pairs = exact_results['E_tied_pairs']
    dist = exact_results['tie_type_dist']
    
    print(f"  -> Time elapsed: {t1-t0:.3f}s")
    print(f"  P(Any Tie): {p_any_tie:.6f}")
    print(f"  E[Tied Pairs]: {E_tied_pairs:.6f}")
    # Print distribution of tie types (how many distinct scores)
    print(f"  Distribution of distinct scores:")
    for num_distinct in sorted(dist.keys()):
        print(f"    - {num_distinct} distinct scores: {dist[num_distinct]:.6f} probability")