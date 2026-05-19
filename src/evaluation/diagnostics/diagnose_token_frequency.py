#!/usr/bin/env python3
"""
Diagnostic: Token-Level Frequency Distribution Comparison

Compares the MARGINAL token frequency distribution (token index vs frequency)
between synthetic and real data for both electricity and gas.

Motivation (Decision 039):
- With quantile binning, the real (training) token distribution is approximately
  uniform: each token 0-63 appears ~1/64 of the time.
- If synthetic data is biased high in Wh space, the synthetic token distribution
  must be skewed toward higher token indices.
- This diagnostic directly visualises and quantifies that skew.
- The results determine whether temperature scaling is an appropriate intervention:
  (a) If the synthetic distribution is mildly skewed upward relative to uniform
      (consistent with per-timestep probability leakage to upper tokens),
      temperature < 1.0 should help.
  (b) If the synthetic distribution is dramatically shifted or has a fundamentally
      different shape, temperature scaling may not be appropriate.

Both synthetic and real data are stored as decoded Wh values, not raw tokens.
This script re-tokenises them using the bin boundaries from the tokeniser file
to recover the token indices.

Usage:
    python src/evaluation/diagnostics/diagnose_token_frequency.py

Outputs:
    - token_frequency_comparison.csv: Per-token frequencies for synth and real
    - token_frequency_comparison.png: Bar chart comparison (token index vs freq)
    - token_frequency_cumulative.png: Cumulative distribution comparison
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Note: matplotlib not available, will skip plot generation")

# ============================================================================
# UPDATE THESE PATHS FOR YOUR TRE ENVIRONMENT
# ============================================================================

# Path to directory containing synthetic batch files (batch_NNNN_month_MM.pkl)
SYNTHETIC_DATA_DIR = "experiments/separate_models_inference_test/run_005/generated_data"

# Path to directory containing real tokenised household files
REAL_DATA_DIR = "experiments/separate_models_train_test/run_008/data/tokenised_data"

# Path to tokeniser file (required — needed to convert Wh values back to tokens)
TOKENISER_PATH = "experiments/separate_models_train_test/run_008/tokenisers/tokenisers.pkl"

# Number of real households to sample (None for all)
REAL_DATA_SAMPLE_SIZE = 100

# Output directory for results
OUTPUT_DIR = "experiments/separate_models_inference_test/run_005/diagnosis"

# ============================================================================


def load_bin_edges(tokeniser_path):
    """Load bin boundaries from the tokeniser file.

    Returns:
        tuple of (bin_edges_dict, value_transform)
        - bin_edges_dict: dict with keys 'electricity' and 'gas', each containing bin edges array
          (in log space if value_transform is "log1p")
        - value_transform: "none" or "log1p"
    """
    with open(tokeniser_path, 'rb') as f:
        tokenisers = pickle.load(f)

    print(f"Loaded tokenisers from: {tokeniser_path}")
    print(f"  Keys: {list(tokenisers.keys())}")

    value_transform = tokenisers.get('value_transform', 'none')
    print(f"  value_transform: {value_transform}")

    result = {}

    if 'elec_boundaries' in tokenisers:
        result['electricity'] = np.array(tokenisers['elec_boundaries'])
        result['gas'] = np.array(tokenisers['gas_boundaries'])
    else:
        # Try object-based format
        for key, fuel in [('elec_tokeniser', 'electricity'), ('electricity', 'electricity'),
                          ('gas_tokeniser', 'gas'), ('gas', 'gas')]:
            if key in tokenisers and tokenisers[key] is not None:
                obj = tokenisers[key]
                if hasattr(obj, 'bin_edges_'):
                    result[fuel] = np.array(obj.bin_edges_[0])
                elif hasattr(obj, 'boundaries'):
                    result[fuel] = np.array(obj.boundaries)

    for fuel in ['electricity', 'gas']:
        if fuel in result:
            n_bins = len(result[fuel]) - 1
            if value_transform == "log1p":
                # Boundaries are in log space — show both log and Wh ranges
                wh_low = np.expm1(result[fuel][0])
                wh_high = np.expm1(result[fuel][-1])
                print(f"  {fuel}: {n_bins} bins, log range [{result[fuel][0]:.2f}, {result[fuel][-1]:.2f}], "
                      f"Wh range [{wh_low:.1f}, {wh_high:.1f}]")
            else:
                print(f"  {fuel}: {n_bins} bins, range [{result[fuel][0]:.1f}, {result[fuel][-1]:.1f}]")

    return result, value_transform


def values_to_tokens(values, bin_edges, value_transform="none"):
    """Convert Wh values to token indices using bin edges.

    Uses np.digitize to assign each value to a bin. Values below the first
    edge get token 0, values above the last edge get the highest token.

    If value_transform is "log1p", applies log1p to values before digitizing
    (bin_edges are assumed to be in log space).
    """
    if value_transform == "log1p":
        values = np.log1p(np.maximum(values, 0))
    # np.digitize with bin_edges[1:-1] maps values into bins 0..n_bins-1
    tokens = np.digitize(values, bin_edges[1:-1])
    n_bins = len(bin_edges) - 1
    return np.clip(tokens, 0, n_bins - 1)


def bin_edges_to_wh(bin_edges, value_transform="none"):
    """Convert bin edges to Wh space (for display/annotation).

    If value_transform is "log1p", edges are in log space and need expm1.
    Returns edges in Wh space.
    """
    if value_transform == "log1p":
        return np.expm1(bin_edges)
    return bin_edges


def find_batch_files(data_dir):
    """Find all batch pickle files in the directory."""
    data_path = Path(data_dir)
    new_format = sorted(data_path.glob("batch_*_month_*.pkl"))
    if new_format:
        print(f"Found {len(new_format)} month files (new format) in {data_dir}")
        return new_format
    old_format = sorted(data_path.glob("batch_*.pkl"))
    old_format = [f for f in old_format if 'conditioning' not in f.name]
    print(f"Found {len(old_format)} batch files (old format) in {data_dir}")
    return old_format


def collect_synthetic_token_counts(batch_files, bin_edges_dict, value_transform="none"):
    """Accumulate token counts from synthetic batch files (streaming, low memory).

    Instead of collecting all raw Wh values, tokenises per-file and accumulates
    counts. Memory usage is O(n_bins) regardless of number of batch files.

    Values below the first bin edge are counted separately as "below-range"
    (these are typically zeros handled by zero-aware tokenization for gas).

    Returns:
        dict: {fuel: (counts_array, total_values, below_range_count)}
    """
    result = {}
    for fuel in ['electricity', 'gas']:
        if fuel in bin_edges_dict:
            n_bins = len(bin_edges_dict[fuel]) - 1
            result[fuel] = [np.zeros(n_bins, dtype=np.int64), 0, 0]  # counts, total, below_range

    print("\nLoading synthetic data...")
    for i, batch_path in enumerate(batch_files):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Processing {i+1}/{len(batch_files)}: {batch_path.name}")

        df = pd.read_pickle(batch_path)

        elec_cols = [c for c in df.columns if '_elec_net_Wh' in c or '_elec_Wh' in c]
        gas_cols = [c for c in df.columns if '_gas_Wh' in c and '_elec' not in c]

        for fuel, cols in [('electricity', elec_cols), ('gas', gas_cols)]:
            if fuel not in result:
                continue
            edges = bin_edges_dict[fuel]
            n_bins = len(edges) - 1
            # min_edge is in log space if value_transform == "log1p"
            # Convert to Wh space for the below-range comparison
            min_edge_wh = np.expm1(edges[0]) if value_transform == "log1p" else edges[0]
            for col in cols:
                vals = df[col].values.astype(float)
                vals = vals[~np.isnan(vals)]
                if len(vals) > 0:
                    # Separate values below first bin edge (zero-aware token territory)
                    below_mask = vals < min_edge_wh
                    n_below = below_mask.sum()
                    result[fuel][2] += n_below
                    result[fuel][1] += len(vals)

                    in_range = vals[~below_mask]
                    if len(in_range) > 0:
                        tokens = values_to_tokens(in_range, edges, value_transform)
                        result[fuel][0] += np.bincount(tokens.astype(int), minlength=n_bins)

    for fuel in result:
        counts, total, below = result[fuel]
        in_range = counts.sum()
        print(f"  Synthetic {fuel}: {total:,} total values, {below:,} below range "
              f"({below/total*100:.1f}%), {in_range:,} in-range across {len(counts)} bins")

    return result


def collect_real_token_counts(data_dir, bin_edges_dict, sample_size=None, value_transform="none"):
    """Accumulate token counts from real household files (streaming, low memory).

    Values below the first bin edge are counted separately as "below-range".

    Returns:
        dict: {fuel: (counts_array, total_values, below_range_count)}
    """
    household_files = sorted(Path(data_dir).glob("*_tokenised.pkl"))
    print(f"\nFound {len(household_files)} real household files in {data_dir}")

    if sample_size and len(household_files) > sample_size:
        import random
        random.seed(42)
        household_files = random.sample(household_files, sample_size)
        print(f"Sampled {sample_size} households")

    result = {}
    for fuel in ['electricity', 'gas']:
        if fuel in bin_edges_dict:
            n_bins = len(bin_edges_dict[fuel]) - 1
            result[fuel] = [np.zeros(n_bins, dtype=np.int64), 0, 0]

    # Map from possible column names to fuel type
    elec_candidates = ['Clean_elec_net_Wh', 'elec_Wh', 'elec_value']
    gas_candidates = ['Clean_gas_Wh', 'gas_Wh', 'gas_value']

    print("Loading real data...")
    for i, hh_path in enumerate(household_files):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  Processing household {i+1}/{len(household_files)}")

        try:
            df = pd.read_pickle(hh_path)
        except Exception as e:
            print(f"  Warning: Could not load {hh_path}: {e}")
            continue

        for fuel, candidates in [('electricity', elec_candidates), ('gas', gas_candidates)]:
            if fuel not in result:
                continue
            edges = bin_edges_dict[fuel]
            n_bins = len(edges) - 1
            min_edge_wh = np.expm1(edges[0]) if value_transform == "log1p" else edges[0]
            for col in candidates:
                if col in df.columns:
                    vals = df[col].values.astype(float)
                    vals = vals[~np.isnan(vals)]
                    if len(vals) > 0:
                        below_mask = vals < min_edge_wh
                        n_below = below_mask.sum()
                        result[fuel][2] += n_below
                        result[fuel][1] += len(vals)

                        in_range = vals[~below_mask]
                        if len(in_range) > 0:
                            tokens = values_to_tokens(in_range, edges, value_transform)
                            result[fuel][0] += np.bincount(tokens.astype(int), minlength=n_bins)
                    break

    for fuel in result:
        counts, total, below = result[fuel]
        in_range = counts.sum()
        print(f"  Real {fuel}: {total:,} total values, {below:,} below range "
              f"({below/total*100:.1f}%), {in_range:,} in-range across {len(counts)} bins")

    return result


def counts_to_freqs(counts):
    """Convert count array to frequency array."""
    total = counts.sum()
    if total == 0:
        return np.zeros_like(counts, dtype=float)
    return counts / total


def create_frequency_plot(synth_freqs, real_freqs, bin_edges_dict, output_dir, value_transform="none"):
    """Create bar chart comparing token frequencies for synth vs real.

    Layout: 2 rows x 2 columns
    - Top row: token frequency bars (synth vs real) for elec and gas
    - Bottom row: deviation from uniform (synth - 1/n_bins) for elec and gas
    """
    if not HAS_MATPLOTLIB:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    for col_idx, fuel in enumerate(['electricity', 'gas']):
        sf = synth_freqs.get(fuel)
        rf = real_freqs.get(fuel)
        if sf is None or rf is None:
            continue

        n_bins = len(sf)
        x = np.arange(n_bins)
        uniform = 1.0 / n_bins

        # Compute bin widths in Wh space for annotation
        bin_widths = None
        if fuel in bin_edges_dict:
            edges_wh = bin_edges_to_wh(bin_edges_dict[fuel], value_transform)
            bin_widths = np.diff(edges_wh)

        # --- Top row: raw frequencies ---
        ax = axes[0, col_idx]
        width = 0.35
        ax.bar(x - width/2, rf, width, label='Real (train)', color='steelblue', alpha=0.7)
        ax.bar(x + width/2, sf, width, label='Synthetic', color='coral', alpha=0.7)
        ax.axhline(uniform, color='black', linestyle='--', linewidth=1,
                    label=f'Uniform (1/{n_bins} = {uniform:.4f})')
        ax.set_xlabel('Token index')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{fuel.upper()}: Token Frequency Distribution', fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2, axis='y')

        # Annotate bin widths for a few tokens
        if bin_widths is not None:
            for t_idx in [0, n_bins//4, n_bins//2, 3*n_bins//4, n_bins-1]:
                ax.annotate(f'{bin_widths[t_idx]:.0f}Wh',
                            (t_idx, max(sf[t_idx], rf[t_idx])),
                            textcoords='offset points', xytext=(0, 8),
                            ha='center', fontsize=7, color='grey')

        # --- Bottom row: deviation from uniform ---
        ax = axes[1, col_idx]
        synth_dev = sf - uniform
        real_dev = rf - uniform

        ax.bar(x - width/2, real_dev * 100, width, label='Real deviation',
               color='steelblue', alpha=0.7)
        ax.bar(x + width/2, synth_dev * 100, width, label='Synthetic deviation',
               color='coral', alpha=0.7)
        ax.axhline(0, color='black', linewidth=1)
        ax.set_xlabel('Token index')
        ax.set_ylabel('Deviation from uniform (%)')
        ax.set_title(f'{fuel.upper()}: Deviation from Uniform Distribution', fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2, axis='y')

        # Annotate which half has excess mass
        lower_half = synth_dev[:n_bins//2].sum() * 100
        upper_half = synth_dev[n_bins//2:].sum() * 100
        ax.text(0.02, 0.98,
                f'Synth excess: lower half {lower_half:+.2f}%, upper half {upper_half:+.2f}%',
                transform=ax.transAxes, fontsize=9, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.suptitle(
        'Token Frequency Comparison: Synthetic vs Real (Train)\n'
        'With quantile binning, real data should be approximately uniform across all tokens',
        fontsize=13, fontweight='bold'
    )
    fig.tight_layout()

    path = os.path.join(output_dir, 'token_frequency_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Frequency comparison plot saved to: {path}")


def create_cumulative_plot(synth_freqs, real_freqs, output_dir):
    """Create cumulative distribution comparison.

    If synthetic is skewed toward upper tokens, its CDF will be below
    the real CDF (reaching each cumulative threshold at a higher token index).
    """
    if not HAS_MATPLOTLIB:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, fuel in zip(axes, ['electricity', 'gas']):
        sf = synth_freqs.get(fuel)
        rf = real_freqs.get(fuel)
        if sf is None or rf is None:
            continue

        n_bins = len(sf)
        x = np.arange(n_bins)
        uniform_cdf = np.cumsum(np.full(n_bins, 1.0 / n_bins))

        synth_cdf = np.cumsum(sf)
        real_cdf = np.cumsum(rf)

        ax.plot(x, real_cdf, color='steelblue', linewidth=2, label='Real (train)')
        ax.plot(x, synth_cdf, color='coral', linewidth=2, label='Synthetic')
        ax.plot(x, uniform_cdf, color='black', linewidth=1, linestyle='--', label='Uniform')

        # Shade the gap between synth and real
        ax.fill_between(x, real_cdf, synth_cdf, alpha=0.2, color='red',
                         label='Synth-Real gap')

        # Find max deviation
        max_dev_idx = np.argmax(np.abs(synth_cdf - real_cdf))
        max_dev = synth_cdf[max_dev_idx] - real_cdf[max_dev_idx]
        ax.annotate(f'Max gap: {max_dev:+.4f}\nat token {max_dev_idx}',
                    (max_dev_idx, (synth_cdf[max_dev_idx] + real_cdf[max_dev_idx]) / 2),
                    textcoords='offset points', xytext=(30, 0),
                    fontsize=9, arrowprops=dict(arrowstyle='->', color='red'),
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlabel('Token index')
        ax.set_ylabel('Cumulative frequency')
        ax.set_title(f'{fuel.upper()}: Cumulative Token Distribution', fontweight='bold')
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(True, alpha=0.3)

        # Interpretation
        if max_dev < 0:
            interp = 'Synth CDF below real → skewed toward UPPER tokens'
        else:
            interp = 'Synth CDF above real → skewed toward LOWER tokens'
        ax.text(0.98, 0.02, interp, transform=ax.transAxes,
                fontsize=9, ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    fig.suptitle(
        'Cumulative Token Distribution: Is Synthetic Skewed Relative to Real?',
        fontsize=13, fontweight='bold'
    )
    fig.tight_layout()

    path = os.path.join(output_dir, 'token_frequency_cumulative.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Cumulative distribution plot saved to: {path}")


def print_diagnosis(synth_freqs, real_freqs, bin_edges_dict, value_transform="none"):
    """Print textual diagnosis of token frequency distributions."""
    print("\n" + "=" * 70)
    print("TOKEN FREQUENCY DIAGNOSIS")
    print("=" * 70)

    for fuel in ['electricity', 'gas']:
        sf = synth_freqs.get(fuel)
        rf = real_freqs.get(fuel)
        if sf is None or rf is None:
            continue

        n_bins = len(sf)
        uniform = 1.0 / n_bins

        print(f"\n{fuel.upper()} ({n_bins} bins):")
        print(f"  Expected uniform frequency: {uniform:.4f} ({uniform*100:.2f}%)")

        # Check how uniform the real distribution actually is
        real_std = np.std(rf)
        real_range = rf.max() - rf.min()
        print(f"  Real distribution: std={real_std:.5f}, range={real_range:.5f}")
        if real_std < uniform * 0.1:
            print(f"  Real is approximately uniform (std < 10% of mean) ✓")
        else:
            print(f"  Real deviates from uniform (std = {real_std/uniform*100:.1f}% of mean)")

        # Compare synth vs real
        synth_std = np.std(sf)
        print(f"  Synthetic distribution: std={synth_std:.5f}")

        # Directional analysis: split into lower and upper halves
        lower = slice(0, n_bins // 2)
        upper = slice(n_bins // 2, n_bins)

        real_lower_mass = rf[lower].sum()
        real_upper_mass = rf[upper].sum()
        synth_lower_mass = sf[lower].sum()
        synth_upper_mass = sf[upper].sum()

        print(f"\n  Mass in lower half (tokens 0-{n_bins//2-1}):")
        print(f"    Real:      {real_lower_mass:.4f} ({real_lower_mass*100:.2f}%)")
        print(f"    Synthetic: {synth_lower_mass:.4f} ({synth_lower_mass*100:.2f}%)")
        print(f"    Difference: {(synth_lower_mass - real_lower_mass)*100:+.2f}%")

        print(f"  Mass in upper half (tokens {n_bins//2}-{n_bins-1}):")
        print(f"    Real:      {real_upper_mass:.4f} ({real_upper_mass*100:.2f}%)")
        print(f"    Synthetic: {synth_upper_mass:.4f} ({synth_upper_mass*100:.2f}%)")
        print(f"    Difference: {(synth_upper_mass - real_upper_mass)*100:+.2f}%")

        # Wh-weighted analysis: how much does the token skew translate to Wh bias?
        if fuel in bin_edges_dict:
            edges_wh = bin_edges_to_wh(bin_edges_dict[fuel], value_transform)
            bin_centers = (edges_wh[:-1] + edges_wh[1:]) / 2
            synth_wh_mean = np.sum(sf * bin_centers)
            real_wh_mean = np.sum(rf * bin_centers)
            print(f"\n  Implied Wh mean from token frequencies:")
            print(f"    Real:      {real_wh_mean:.1f} Wh")
            print(f"    Synthetic: {synth_wh_mean:.1f} Wh")
            print(f"    Bias:      {synth_wh_mean - real_wh_mean:+.1f} Wh ({(synth_wh_mean/real_wh_mean - 1)*100:+.1f}%)")

        # Per-token comparison table (show biggest deviations)
        diffs = sf - rf
        with np.errstate(divide='ignore', invalid='ignore'):
            diffs_pct = np.where(rf > 0, diffs / rf * 100, 0.0)
        sorted_idx = np.argsort(diffs)

        print(f"\n  Tokens with largest UNDERPRODUCTION (synth < real):")
        for idx in sorted_idx[:5]:
            print(f"    Token {idx:3d}: synth={sf[idx]:.5f}, real={rf[idx]:.5f}, "
                  f"diff={diffs[idx]*100:+.3f}% ({diffs_pct[idx]:+.1f}% relative)")

        print(f"  Tokens with largest OVERPRODUCTION (synth > real):")
        for idx in sorted_idx[-5:][::-1]:
            print(f"    Token {idx:3d}: synth={sf[idx]:.5f}, real={rf[idx]:.5f}, "
                  f"diff={diffs[idx]*100:+.3f}% ({diffs_pct[idx]:+.1f}% relative)")

        # Temperature scaling assessment
        print(f"\n  TEMPERATURE SCALING ASSESSMENT:")
        # Check if the skew is consistent (gradual shift) or spiky (a few outlier tokens)
        upper_overproduced = np.sum(diffs[upper] > 0)
        upper_total = n_bins // 2
        lower_underproduced = np.sum(diffs[lower] < 0)
        lower_total = n_bins // 2

        if synth_upper_mass > real_upper_mass + 0.005:
            print(f"    Synthetic IS skewed toward upper tokens.")
            print(f"    Upper half: {upper_overproduced}/{upper_total} tokens overproduced")
            print(f"    Lower half: {lower_underproduced}/{lower_total} tokens underproduced")

            if upper_overproduced > upper_total * 0.6 and lower_underproduced > lower_total * 0.6:
                print(f"    Pattern: BROAD SHIFT — most upper tokens overproduced, most lower underproduced")
                print(f"    → Temperature scaling MAY help (suppresses per-timestep leakage)")
                print(f"    → But could also indicate systematic top-1 bias (temperature would worsen)")
                print(f"    → Recommend: test empirically with small sample at T=0.9")
            else:
                print(f"    Pattern: SPIKY — overproduction concentrated on specific tokens")
                print(f"    → Temperature scaling less likely to help")
                print(f"    → May indicate specific token prediction failures")
        elif synth_upper_mass < real_upper_mass - 0.005:
            print(f"    Synthetic is skewed toward LOWER tokens (opposite of expected).")
            print(f"    → Temperature < 1.0 would worsen this.")
        else:
            print(f"    No significant directional skew detected.")
            print(f"    → Bias may arise from within-bin effects, not token-level skew.")


def main():
    print("=" * 70)
    print("TOKEN FREQUENCY DISTRIBUTION COMPARISON")
    print("=" * 70)

    # Validate paths
    if not os.path.exists(TOKENISER_PATH):
        print(f"ERROR: Tokeniser file not found: {TOKENISER_PATH}")
        print("The tokeniser is required to convert Wh values back to token indices.")
        sys.exit(1)

    if not os.path.exists(SYNTHETIC_DATA_DIR):
        print(f"ERROR: Synthetic data directory not found: {SYNTHETIC_DATA_DIR}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load bin edges and value_transform
    bin_edges, value_transform = load_bin_edges(TOKENISER_PATH)

    # Collect token counts (streaming — O(n_bins) memory, not O(n_values))
    synth_data = collect_synthetic_token_counts(find_batch_files(SYNTHETIC_DATA_DIR), bin_edges, value_transform)

    real_data = {}
    if REAL_DATA_DIR and os.path.exists(REAL_DATA_DIR):
        real_data = collect_real_token_counts(REAL_DATA_DIR, bin_edges, REAL_DATA_SAMPLE_SIZE, value_transform)
    else:
        print(f"\nNote: Real data directory not found: {REAL_DATA_DIR}")
        print("Proceeding with synthetic vs uniform comparison only.")

    # Compute frequencies and build CSV
    # Frequencies are computed over IN-RANGE values only (excluding below-range/zeros).
    # This is critical for gas where zero-aware tokenization handles zeros separately —
    # including zeros would put ~50%+ of mass in bin 0 and make all other bins look
    # uniformly low, obscuring the actual non-zero token distribution.
    synth_freqs = {}
    real_freqs = {}
    zero_rates = {}  # {fuel: {'synth': rate, 'real': rate}}
    all_rows = []

    for fuel in ['electricity', 'gas']:
        if fuel not in bin_edges:
            print(f"Warning: no bin edges for {fuel}, skipping")
            continue

        edges = bin_edges[fuel]
        n_bins = len(edges) - 1

        # Track zero/below-range rates
        zero_rates[fuel] = {}

        # Synthetic
        if fuel in synth_data and synth_data[fuel][1] > 0:
            synth_counts = synth_data[fuel][0]
            synth_total = synth_data[fuel][1]
            synth_below = synth_data[fuel][2]
            synth_freq = counts_to_freqs(synth_counts)
            synth_freqs[fuel] = synth_freq
            zero_rates[fuel]['synth'] = synth_below / synth_total if synth_total > 0 else 0
        else:
            continue

        # Real
        if fuel in real_data and real_data[fuel][1] > 0:
            real_counts = real_data[fuel][0]
            real_total = real_data[fuel][1]
            real_below = real_data[fuel][2]
            real_freq = counts_to_freqs(real_counts)
            real_freqs[fuel] = real_freq
            zero_rates[fuel]['real'] = real_below / real_total if real_total > 0 else 0
        else:
            # Use uniform as baseline
            real_freqs[fuel] = np.full(n_bins, 1.0 / n_bins)
            real_counts = np.zeros(n_bins, dtype=np.int64)
            zero_rates[fuel]['real'] = 0

        # Report zero rates
        zr = zero_rates[fuel]
        if zr.get('synth', 0) > 0.01 or zr.get('real', 0) > 0.01:
            print(f"\n  {fuel.upper()} below-range rate (zeros/near-zeros handled by zero-aware tokenization):")
            print(f"    Real:      {zr.get('real', 0)*100:.1f}%")
            print(f"    Synthetic: {zr.get('synth', 0)*100:.1f}%")
            print(f"    Token frequencies below are computed over IN-RANGE values only.")

        # Build CSV rows — always report Wh-space values for interpretability
        edges_wh = bin_edges_to_wh(edges, value_transform)
        bin_widths_wh = np.diff(edges_wh)
        bin_centers_wh = (edges_wh[:-1] + edges_wh[1:]) / 2
        for t in range(n_bins):
            all_rows.append({
                'fuel': fuel,
                'token_index': t,
                'bin_lower': edges_wh[t],
                'bin_upper': edges_wh[t + 1],
                'bin_width': bin_widths_wh[t],
                'bin_center': bin_centers_wh[t],
                'synth_count': synth_counts[t],
                'synth_freq': synth_freq[t],
                'real_count': real_counts[t],
                'real_freq': real_freq[t],
                'freq_diff': synth_freq[t] - real_freq[t],
            })

    # Save CSV
    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(OUTPUT_DIR, 'token_frequency_comparison.csv')
    df.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"\nToken frequency data saved to: {csv_path}")

    # Create plots
    create_frequency_plot(synth_freqs, real_freqs, bin_edges, OUTPUT_DIR, value_transform)
    create_cumulative_plot(synth_freqs, real_freqs, OUTPUT_DIR)

    # Print diagnosis
    print_diagnosis(synth_freqs, real_freqs, bin_edges, value_transform)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
