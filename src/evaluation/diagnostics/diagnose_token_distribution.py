#!/usr/bin/env python3
"""
Diagnostic: Investigate quantile binning median bias hypothesis.

This script tests whether the model is biased toward predicting median tokens,
which would explain the systematic low bias in decoded values for skewed data.

Key analyses:
1. Token distribution: Are synthetic tokens clustered around median (token ~32)?
2. Decoded value comparison: Is mean(decoded_synthetic) < mean(decoded_real)?
3. Bin width analysis: Show asymmetry in value-space coverage
4. Per-month token analysis: Does bias vary by season?

Usage:
    python scripts/diagnose_token_distribution.py

Outputs:
    - token_distribution_analysis.csv: Token frequency statistics
    - token_distribution_plot.png: Visualization of distributions
    - bin_width_analysis.png: Bin width asymmetry visualization
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Try to import optional dependencies
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Note: matplotlib not available, will skip plot generation")

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Note: scipy not available, will use simple skewness estimate")

# ============================================================================
# UPDATE THESE PATHS FOR YOUR TRE ENVIRONMENT
# ============================================================================

# Path to directory containing synthetic batch files (batch_0001.pkl, etc.)
# Example: "experiments/separate_models_inference_test/run_001/generated_data"
SYNTHETIC_DATA_DIR = "/path/to/generated_data"

# Path to directory containing real tokenised household files (*_tokenised.pkl)
# Example: "experiments/separate_models_inference_test/run_001/data/tokenised_data"
REAL_DATA_DIR = "/path/to/tokenised_data"

# Path to tokeniser file (to get bin boundaries and bin_means)
# Example: "experiments/separate_models_inference_test/run_001/tokenisers/tokenisers.pkl"
TOKENISER_PATH = "/path/to/tokenisers.pkl"

# Output directory for results (will be created if doesn't exist)
# Example: "experiments/separate_models_inference_test/run_001/token_analysis"
OUTPUT_DIR = "/path/to/output"

# Number of real households to sample (None for all, recommend 50-100 for speed)
REAL_DATA_SAMPLE_SIZE = 100

# ============================================================================


def load_tokenisers(tokeniser_path):
    """Load tokenisers and extract bin information.

    Bin widths and centers are always returned in Wh space (for display).
    If value_transform is "log1p", the raw bin edges are in log space and
    are converted to Wh space via expm1 before computing widths/centers.
    """
    with open(tokeniser_path, 'rb') as f:
        tokenisers = pickle.load(f)

    print(f"Loaded tokenisers from: {tokeniser_path}")
    print(f"  Keys: {list(tokenisers.keys())}")

    value_transform = tokenisers.get('value_transform', 'none')
    binning_strategy = tokenisers.get('binning_strategy', 'quantile')
    print(f"  value_transform: {value_transform}")

    # Handle different tokeniser formats
    elec_info = {'name': 'electricity'}
    gas_info = {'name': 'gas'}

    # Format 1: Direct boundaries and bin_means in dict (AI-EDOL format)
    if 'elec_boundaries' in tokenisers:
        elec_info['bin_edges'] = np.array(tokenisers['elec_boundaries'])
        gas_info['bin_edges'] = np.array(tokenisers['gas_boundaries'])
        if 'elec_bin_means' in tokenisers:
            elec_info['bin_means'] = np.array(tokenisers['elec_bin_means'])
            gas_info['bin_means'] = np.array(tokenisers['gas_bin_means'])

    # Format 2: Tokeniser objects with attributes
    else:
        elec_tokeniser = tokenisers.get('elec_tokeniser') or tokenisers.get('electricity')
        gas_tokeniser = tokenisers.get('gas_tokeniser') or tokenisers.get('gas')
        elec_info = extract_bin_info(elec_tokeniser, 'electricity')
        gas_info = extract_bin_info(gas_tokeniser, 'gas')

    # Calculate bin widths and centers in Wh space
    for info in [elec_info, gas_info]:
        info['binning_strategy'] = binning_strategy
        if info.get('bin_edges') is not None:
            # Convert edges to Wh space if log-transformed
            edges_wh = np.expm1(info['bin_edges']) if value_transform == "log1p" else info['bin_edges']
            info['bin_widths'] = np.diff(edges_wh)
            info['bin_centers'] = (edges_wh[:-1] + edges_wh[1:]) / 2
            if 'bin_means' not in info:
                info['bin_means'] = info['bin_centers']

    return elec_info, gas_info


def extract_bin_info(tokeniser, name):
    """Extract bin edges, widths, and means from a tokeniser object."""
    info = {'name': name}

    if tokeniser is None:
        print(f"  Warning: {name} tokeniser is None")
        return info

    # Try different attribute names for bin edges
    if hasattr(tokeniser, 'bin_edges_'):
        bin_edges = tokeniser.bin_edges_[0]  # KBinsDiscretizer format
    elif hasattr(tokeniser, 'boundaries'):
        bin_edges = tokeniser.boundaries
    elif hasattr(tokeniser, 'bin_edges'):
        bin_edges = tokeniser.bin_edges
    else:
        print(f"  Warning: Could not find bin edges for {name}")
        bin_edges = None

    info['bin_edges'] = np.array(bin_edges) if bin_edges is not None else None

    # Get bin means if available
    if hasattr(tokeniser, 'bin_means'):
        info['bin_means'] = np.array(tokeniser.bin_means)
    elif hasattr(tokeniser, 'bin_means_'):
        info['bin_means'] = np.array(tokeniser.bin_means_)

    return info


def find_batch_files(data_dir):
    """Find all batch pickle files in the directory.

    Supports both formats:
    - Old format: batch_NNNN.pkl (one file per batch with all 12 months)
    - New format: batch_NNNN_month_MM.pkl (one file per month per batch)
    """
    data_path = Path(data_dir)

    # Check for new format first (month-specific files)
    new_format_files = sorted(data_path.glob("batch_*_month_*.pkl"))

    if new_format_files:
        print(f"Found {len(new_format_files)} month files (new format)")
        return new_format_files

    # Fall back to old format
    old_format_files = sorted(data_path.glob("batch_*.pkl"))
    # Exclude conditioning_summary.pkl
    old_format_files = [f for f in old_format_files if 'conditioning' not in f.name]
    print(f"Found {len(old_format_files)} batch files (old format)")
    return old_format_files


def analyze_synthetic_tokens(synthetic_dir, num_bins=64):
    """Analyze token distribution from synthetic batch files."""
    batch_files = find_batch_files(synthetic_dir)
    print(f"Found {len(batch_files)} synthetic batch files")

    if len(batch_files) == 0:
        return None

    all_elec_tokens = []
    all_gas_tokens = []
    monthly_elec_tokens = defaultdict(list)
    monthly_gas_tokens = defaultdict(list)

    for batch_path in batch_files:
        df = pd.read_pickle(batch_path)

        # Find token columns (need to infer from column names)
        # Synthetic data has decoded values, not tokens - need to re-tokenise or find tokens
        # Check if there are token columns
        elec_cols = [c for c in df.columns if '_elec_net_Wh' in c]
        gas_cols = [c for c in df.columns if '_gas_Wh' in c]

        # For now, we'll work with decoded values and analyze the distribution
        # This is still useful for understanding the bias
        if 'month' in df.columns:
            for month in range(1, 13):
                month_mask = df['month'] == month
                month_data = df[month_mask]

                for col in elec_cols:
                    vals = month_data[col].values.astype(float)
                    vals = vals[~np.isnan(vals)]
                    all_elec_tokens.extend(vals)
                    monthly_elec_tokens[month].extend(vals)

                for col in gas_cols:
                    vals = month_data[col].values.astype(float)
                    vals = vals[~np.isnan(vals)]
                    all_gas_tokens.extend(vals)
                    monthly_gas_tokens[month].extend(vals)

    return {
        'elec_values': np.array(all_elec_tokens),
        'gas_values': np.array(all_gas_tokens),
        'monthly_elec': monthly_elec_tokens,
        'monthly_gas': monthly_gas_tokens,
    }


def analyze_real_tokens(real_dir, sample_size=None, puprn_filter=None):
    """Analyze token/value distribution from real tokenised files."""
    household_files = sorted(Path(real_dir).glob("*_tokenised.pkl"))
    print(f"Found {len(household_files)} real household files")

    if puprn_filter is not None:
        puprn_set = set(puprn_filter)
        household_files = [f for f in household_files
                          if f.stem.replace('_tokenised', '') in puprn_set]
        print(f"Filtered to {len(household_files)} train households")

    if sample_size and len(household_files) > sample_size:
        import random
        random.seed(42)
        household_files = random.sample(household_files, sample_size)
        print(f"Sampled {sample_size} households")

    all_elec_values = []
    all_gas_values = []
    all_elec_tokens = []
    all_gas_tokens = []
    monthly_elec_values = defaultdict(list)
    monthly_gas_values = defaultdict(list)

    for hh_path in household_files:
        df = pd.read_pickle(hh_path)

        # Get values
        if 'Clean_elec_net_Wh' in df.columns:
            elec_vals = df['Clean_elec_net_Wh'].values.astype(float)
            elec_vals = elec_vals[~np.isnan(elec_vals)]
            all_elec_values.extend(elec_vals)

            if 'month' in df.columns:
                for month in range(1, 13):
                    month_vals = df[df['month'] == month]['Clean_elec_net_Wh'].values.astype(float)
                    month_vals = month_vals[~np.isnan(month_vals)]
                    monthly_elec_values[month].extend(month_vals)

        if 'Clean_gas_Wh' in df.columns:
            gas_vals = df['Clean_gas_Wh'].values.astype(float)
            gas_vals = gas_vals[~np.isnan(gas_vals)]
            all_gas_values.extend(gas_vals)

            if 'month' in df.columns:
                for month in range(1, 13):
                    month_vals = df[df['month'] == month]['Clean_gas_Wh'].values.astype(float)
                    month_vals = month_vals[~np.isnan(month_vals)]
                    monthly_gas_values[month].extend(month_vals)

        # Get tokens if available
        if 'elec_token' in df.columns:
            tokens = df['elec_token'].values.astype(float)
            tokens = tokens[~np.isnan(tokens)]
            all_elec_tokens.extend(tokens)

        if 'gas_token' in df.columns:
            tokens = df['gas_token'].values.astype(float)
            tokens = tokens[~np.isnan(tokens)]
            all_gas_tokens.extend(tokens)

    return {
        'elec_values': np.array(all_elec_values),
        'gas_values': np.array(all_gas_values),
        'elec_tokens': np.array(all_elec_tokens) if all_elec_tokens else None,
        'gas_tokens': np.array(all_gas_tokens) if all_gas_tokens else None,
        'monthly_elec': monthly_elec_values,
        'monthly_gas': monthly_gas_values,
    }


def compute_statistics(values, name):
    """Compute key statistics for a value distribution."""
    if len(values) == 0:
        return {}

    mean_val = np.mean(values)
    median_val = np.median(values)
    std_val = np.std(values)

    # Compute skewness
    if HAS_SCIPY:
        skewness = scipy_stats.skew(values)
    else:
        # Simple Pearson's skewness estimate
        skewness = 3 * (mean_val - median_val) / std_val if std_val > 0 else 0

    # Zero rate (useful for gas)
    zero_rate = (values == 0).sum() / len(values) if len(values) > 0 else 0

    return {
        'name': name,
        'count': len(values),
        'mean': mean_val,
        'median': median_val,
        'std': std_val,
        'min': np.min(values),
        'max': np.max(values),
        'p10': np.percentile(values, 10),
        'p25': np.percentile(values, 25),
        'p75': np.percentile(values, 75),
        'p90': np.percentile(values, 90),
        'p99': np.percentile(values, 99),
        'skewness': skewness,
        'mean_median_ratio': mean_val / median_val if median_val > 0 else np.nan,
        'zero_rate': zero_rate,
    }


def values_to_tokens(values, bin_edges):
    """Convert values to tokens using bin edges."""
    if bin_edges is None:
        return None
    tokens = np.digitize(values, bin_edges[1:-1])  # Exclude first and last edge
    return np.clip(tokens, 0, len(bin_edges) - 2)


def print_comparison(synth_stats, real_stats, label):
    """Print comparison of synthetic vs real statistics."""
    print(f"\n{label}:")
    print("-" * 70)
    print(f"{'Metric':<20} {'Synthetic':<15} {'Real':<15} {'Diff':<20}")
    print("-" * 70)

    for metric in ['mean', 'median', 'std', 'p10', 'p25', 'p75', 'p90', 'skewness', 'mean_median_ratio', 'zero_rate']:
        synth_val = synth_stats.get(metric, np.nan)
        real_val = real_stats.get(metric, np.nan)
        if not np.isnan(synth_val) and not np.isnan(real_val):
            diff = synth_val - real_val
            if metric == 'zero_rate':
                print(f"{metric:<20} {synth_val*100:<14.1f}% {real_val*100:<14.1f}% {diff*100:+.1f}%")
            elif real_val != 0:
                diff_pct = (diff / real_val * 100)
                print(f"{metric:<20} {synth_val:<15.2f} {real_val:<15.2f} {diff:+.2f} ({diff_pct:+.1f}%)")
            else:
                print(f"{metric:<20} {synth_val:<15.2f} {real_val:<15.2f} {diff:+.2f}")
        else:
            print(f"{metric:<20} {synth_val:<15.2f} {real_val:<15.2f} N/A")

    # Highlight median bias analysis
    if 'mean' in synth_stats and 'median' in real_stats:
        synth_mean = synth_stats['mean']
        real_mean = real_stats['mean']
        real_median = real_stats['median']

        print()
        print("MEDIAN BIAS ANALYSIS:")
        print(f"  Real data:      mean={real_mean:.1f}, median={real_median:.1f}, skewness={real_stats.get('skewness', 0):.2f}")
        print(f"  Synthetic data: mean={synth_mean:.1f}, median={synth_stats.get('median', 0):.1f}, skewness={synth_stats.get('skewness', 0):.2f}")
        print()

        dist_to_real_mean = abs(synth_mean - real_mean)
        dist_to_real_median = abs(synth_mean - real_median)

        print(f"  Synthetic mean distance to real MEAN:   {dist_to_real_mean:.1f}")
        print(f"  Synthetic mean distance to real MEDIAN: {dist_to_real_median:.1f}")

        if dist_to_real_median < dist_to_real_mean:
            print()
            print("  >>> MEDIAN BIAS DETECTED: Synthetic mean is closer to real MEDIAN than real MEAN")
            bias_pct = (real_mean - synth_mean) / real_mean * 100
            print(f"  >>> Synthetic is {bias_pct:.1f}% below real mean")
        else:
            print()
            print("  >>> NO MEDIAN BIAS: Synthetic mean is closer to real MEAN")


def create_distribution_plot(synth_data, real_data, elec_info, gas_info, output_path):
    """Create visualization of value distributions."""
    if not HAS_MATPLOTLIB:
        print("Skipping plot (matplotlib not available)")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Electricity value distribution
    ax1 = axes[0, 0]
    if len(synth_data['elec_values']) > 0 and len(real_data['elec_values']) > 0:
        # Use same bins for both
        max_val = max(np.percentile(real_data['elec_values'], 99),
                      np.percentile(synth_data['elec_values'], 99))
        bins = np.linspace(0, max_val, 50)

        ax1.hist(real_data['elec_values'], bins=bins, alpha=0.5, label='Real', density=True, color='green')
        ax1.hist(synth_data['elec_values'], bins=bins, alpha=0.5, label='Synthetic', density=True, color='red')

        # Add vertical lines for mean and median
        real_mean = np.mean(real_data['elec_values'])
        real_median = np.median(real_data['elec_values'])
        synth_mean = np.mean(synth_data['elec_values'])

        ax1.axvline(real_mean, color='darkgreen', linestyle='-', linewidth=2, label=f'Real mean: {real_mean:.0f}')
        ax1.axvline(real_median, color='darkgreen', linestyle='--', linewidth=2, label=f'Real median: {real_median:.0f}')
        ax1.axvline(synth_mean, color='darkred', linestyle='-', linewidth=2, label=f'Synth mean: {synth_mean:.0f}')

        ax1.set_xlabel('Electricity (Wh)')
        ax1.set_ylabel('Density')
        ax1.set_title('Electricity Value Distribution')
        ax1.legend(fontsize=8)

    # Gas value distribution
    ax2 = axes[0, 1]
    if len(synth_data['gas_values']) > 0 and len(real_data['gas_values']) > 0:
        max_val = max(np.percentile(real_data['gas_values'], 99),
                      np.percentile(synth_data['gas_values'], 99))
        bins = np.linspace(0, max_val, 50)

        ax2.hist(real_data['gas_values'], bins=bins, alpha=0.5, label='Real', density=True, color='green')
        ax2.hist(synth_data['gas_values'], bins=bins, alpha=0.5, label='Synthetic', density=True, color='orange')

        real_mean = np.mean(real_data['gas_values'])
        real_median = np.median(real_data['gas_values'])
        synth_mean = np.mean(synth_data['gas_values'])

        ax2.axvline(real_mean, color='darkgreen', linestyle='-', linewidth=2, label=f'Real mean: {real_mean:.0f}')
        ax2.axvline(real_median, color='darkgreen', linestyle='--', linewidth=2, label=f'Real median: {real_median:.0f}')
        ax2.axvline(synth_mean, color='darkorange', linestyle='-', linewidth=2, label=f'Synth mean: {synth_mean:.0f}')

        ax2.set_xlabel('Gas (Wh)')
        ax2.set_ylabel('Density')
        ax2.set_title('Gas Value Distribution')
        ax2.legend(fontsize=8)

    # Electricity bin widths
    ax3 = axes[1, 0]
    if elec_info.get('bin_widths') is not None:
        bin_widths = elec_info['bin_widths']
        bin_centers = elec_info.get('bin_centers', np.arange(len(bin_widths)))  # noqa: F841

        ax3.bar(range(len(bin_widths)), bin_widths, color='steelblue', alpha=0.7)
        ax3.axhline(np.mean(bin_widths), color='red', linestyle='--', label=f'Mean width: {np.mean(bin_widths):.1f}')
        ax3.set_xlabel('Token (bin index)')
        ax3.set_ylabel('Bin Width (Wh)')
        min_width = min(bin_widths) if min(bin_widths) > 0 else 0.01  # Avoid div by zero
        ratio = max(bin_widths) / min_width
        strategy_label = elec_info.get('binning_strategy', 'quantile').capitalize() + ' Binning'
        ax3.set_title(f'Electricity Bin Widths ({strategy_label})\nMin: {min(bin_widths):.1f}, Max: {max(bin_widths):.1f}, Ratio: {ratio:.1f}x')
        ax3.legend()

    # Gas bin widths
    ax4 = axes[1, 1]
    if gas_info.get('bin_widths') is not None:
        bin_widths = gas_info['bin_widths']

        ax4.bar(range(len(bin_widths)), bin_widths, color='darkorange', alpha=0.7)
        ax4.axhline(np.mean(bin_widths), color='red', linestyle='--', label=f'Mean width: {np.mean(bin_widths):.1f}')
        ax4.set_xlabel('Token (bin index)')
        ax4.set_ylabel('Bin Width (Wh)')
        strategy_label = gas_info.get('binning_strategy', 'quantile').capitalize() + ' Binning'
        ax4.set_title(f'Gas Bin Widths ({strategy_label})\nMin: {min(bin_widths):.1f}, Max: {max(bin_widths):.1f}, Ratio: {max(bin_widths)/min(bin_widths):.1f}x')
        ax4.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()


def create_monthly_comparison_plot(synth_data, real_data, output_path):
    """Create monthly mean vs median comparison plot."""
    if not HAS_MATPLOTLIB:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    months = range(1, 13)

    # Electricity
    ax1 = axes[0]
    real_means = [np.mean(real_data['monthly_elec'][m]) if real_data['monthly_elec'][m] else np.nan for m in months]
    real_medians = [np.median(real_data['monthly_elec'][m]) if real_data['monthly_elec'][m] else np.nan for m in months]
    synth_means = [np.mean(synth_data['monthly_elec'][m]) if synth_data['monthly_elec'][m] else np.nan for m in months]

    ax1.plot(months, real_means, 'g-o', linewidth=2, markersize=8, label='Real MEAN')
    ax1.plot(months, real_medians, 'g--s', linewidth=2, markersize=8, label='Real MEDIAN')
    ax1.plot(months, synth_means, 'r-^', linewidth=2, markersize=8, label='Synthetic MEAN')

    ax1.set_xlabel('Month')
    ax1.set_ylabel('Electricity (Wh)')
    ax1.set_title('Electricity: Is Synthetic Mean Tracking Real Median?')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(months)

    # Gas
    ax2 = axes[1]
    real_means = [np.mean(real_data['monthly_gas'][m]) if real_data['monthly_gas'][m] else np.nan for m in months]
    real_medians = [np.median(real_data['monthly_gas'][m]) if real_data['monthly_gas'][m] else np.nan for m in months]
    synth_means = [np.mean(synth_data['monthly_gas'][m]) if synth_data['monthly_gas'][m] else np.nan for m in months]

    ax2.plot(months, real_means, 'g-o', linewidth=2, markersize=8, label='Real MEAN')
    ax2.plot(months, real_medians, 'g--s', linewidth=2, markersize=8, label='Real MEDIAN')
    ax2.plot(months, synth_means, 'r-^', linewidth=2, markersize=8, label='Synthetic MEAN')

    ax2.set_xlabel('Month')
    ax2.set_ylabel('Gas (Wh)')
    ax2.set_title('Gas: Is Synthetic Mean Tracking Real Median?')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(months)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Monthly comparison plot saved to: {output_path}")
    plt.close()


def print_diagnosis(synth_elec_stats, real_elec_stats, synth_gas_stats, real_gas_stats):
    """Print overall diagnosis."""
    print("\n" + "=" * 70)
    print("QUANTILE BINNING MEDIAN BIAS DIAGNOSIS")
    print("=" * 70)

    print("\nHYPOTHESIS: For skewed data, quantile binning causes model to predict")
    print("            tokens near the median, resulting in decoded values that")
    print("            track the MEDIAN rather than the MEAN.")
    print()

    # Electricity analysis
    if real_elec_stats and synth_elec_stats:
        real_mean = real_elec_stats['mean']
        real_median = real_elec_stats['median']
        synth_mean = synth_elec_stats['mean']

        dist_to_mean = abs(synth_mean - real_mean)
        dist_to_median = abs(synth_mean - real_median)

        print("ELECTRICITY:")
        print(f"  Real mean:      {real_mean:.1f} Wh")
        print(f"  Real median:    {real_median:.1f} Wh")
        print(f"  Synthetic mean: {synth_mean:.1f} Wh")
        print()
        print(f"  Distance to real mean:   {dist_to_mean:.1f} Wh")
        print(f"  Distance to real median: {dist_to_median:.1f} Wh")
        print()

        if dist_to_median < dist_to_mean:
            print("  >>> MEDIAN BIAS CONFIRMED: Synthetic is closer to real MEDIAN")
            bias_pct = (real_mean - synth_mean) / real_mean * 100
            print(f"  >>> Synthetic is {bias_pct:.1f}% below real mean")
        else:
            print("  >>> No clear median bias for electricity")

    print()

    # Gas analysis
    if real_gas_stats and synth_gas_stats:
        real_mean = real_gas_stats['mean']
        real_median = real_gas_stats['median']
        synth_mean = synth_gas_stats['mean']

        dist_to_mean = abs(synth_mean - real_mean)
        dist_to_median = abs(synth_mean - real_median)

        print("GAS:")
        print(f"  Real mean:      {real_mean:.1f} Wh")
        print(f"  Real median:    {real_median:.1f} Wh")
        print(f"  Synthetic mean: {synth_mean:.1f} Wh")
        print()
        print(f"  Distance to real mean:   {dist_to_mean:.1f} Wh")
        print(f"  Distance to real median: {dist_to_median:.1f} Wh")
        print()

        if dist_to_median < dist_to_mean:
            print("  >>> MEDIAN BIAS CONFIRMED: Synthetic is closer to real MEDIAN")
            bias_pct = (real_mean - synth_mean) / real_mean * 100
            print(f"  >>> Synthetic is {bias_pct:.1f}% below real mean")
        else:
            print("  >>> No clear median bias for gas")

    print()
    print("=" * 70)
    print("IMPLICATIONS:")
    print("=" * 70)
    print("""
If median bias is confirmed:
1. Quantile binning transforms skewed data to uniform tokens
2. Model predicts "safe" middle tokens when uncertain
3. Middle tokens decode to MEDIAN values, not MEAN values
4. For right-skewed energy data: median << mean
5. Result: Systematic low bias in synthetic data

Potential remedies:
- Per-household normalization before tokenization
- Value-weighted loss (penalize high-value bin errors more)
- Explicit consumption level conditioning
- Mean-preserving binning schemes
""")


def main(synthetic_dir=None, real_dir=None, output_dir=None, tokeniser_path=None, sample_size=None, train_puprns=None):
    """Main analysis function.

    Args:
        synthetic_dir: Path to generated_data directory (default: SYNTHETIC_DATA_DIR)
        real_dir: Path to tokenised_data directory (default: REAL_DATA_DIR)
        output_dir: Output directory for results (default: OUTPUT_DIR)
        tokeniser_path: Path to tokenisers.pkl file (default: TOKENISER_PATH)
        sample_size: Number of real households to sample (default: REAL_DATA_SAMPLE_SIZE)

    Returns:
        Dictionary with analysis results
    """
    # Use provided values or fall back to global defaults
    synthetic_data_dir = synthetic_dir if synthetic_dir else SYNTHETIC_DATA_DIR
    real_data_dir = real_dir if real_dir else REAL_DATA_DIR
    out_dir = output_dir if output_dir else OUTPUT_DIR
    tok_path = tokeniser_path if tokeniser_path else TOKENISER_PATH
    real_sample_size = sample_size if sample_size is not None else REAL_DATA_SAMPLE_SIZE

    # Capture all printed output to save to file
    import io
    log_buffer = io.StringIO()

    class TeeWriter:
        """Write to both stdout and a buffer."""
        def __init__(self, *writers):
            self.writers = writers
        def write(self, text):
            for w in self.writers:
                w.write(text)
        def flush(self):
            for w in self.writers:
                w.flush()

    original_stdout = sys.stdout
    sys.stdout = TeeWriter(original_stdout, log_buffer)

    print("=" * 70)
    print("TOKEN DISTRIBUTION & MEDIAN BIAS ANALYSIS")
    print("=" * 70)

    # Validate paths
    if not os.path.exists(synthetic_data_dir):
        print(f"ERROR: Synthetic data directory not found: {synthetic_data_dir}")
        return {'status': 'failed', 'error': 'synthetic_dir not found'}

    # Create output directory
    os.makedirs(out_dir, exist_ok=True)

    # Load tokenisers if available
    elec_info, gas_info = {}, {}
    if tok_path and os.path.exists(tok_path):
        try:
            elec_info, gas_info = load_tokenisers(tok_path)
        except Exception as e:
            print(f"Warning: Could not load tokenisers: {e}")
    else:
        print(f"Note: Tokeniser file not found: {tok_path}")
        print("Bin width analysis will be skipped.")

    # Analyze synthetic data
    print("\nAnalyzing synthetic data...")
    synth_data = analyze_synthetic_tokens(synthetic_data_dir)
    if synth_data is None:
        print("ERROR: No synthetic data found!")
        return {'status': 'failed', 'error': 'no synthetic data'}

    # Analyze real data (optional)
    real_data = {'elec_values': [], 'gas_values': []}
    if real_data_dir and os.path.exists(real_data_dir):
        print("\nAnalyzing real data...")
        real_data = analyze_real_tokens(real_data_dir, real_sample_size, puprn_filter=train_puprns)
    else:
        print(f"\nNote: Real data directory not found: {real_data_dir}")
        print("Proceeding with synthetic data analysis only.")

    # Compute statistics
    synth_elec_stats = compute_statistics(synth_data['elec_values'], 'Synthetic Electricity')
    synth_gas_stats = compute_statistics(synth_data['gas_values'], 'Synthetic Gas')
    real_elec_stats = compute_statistics(real_data['elec_values'], 'Real Electricity') if len(real_data['elec_values']) > 0 else {}
    real_gas_stats = compute_statistics(real_data['gas_values'], 'Real Gas') if len(real_data['gas_values']) > 0 else {}

    # Print comparisons (only if real data available)
    if real_elec_stats:
        print_comparison(synth_elec_stats, real_elec_stats, "ELECTRICITY VALUE COMPARISON")
    if real_gas_stats:
        print_comparison(synth_gas_stats, real_gas_stats, "GAS VALUE COMPARISON")

    # Create plots
    if HAS_MATPLOTLIB:
        plot_path = os.path.join(out_dir, "token_distribution_analysis.png")
        create_distribution_plot(synth_data, real_data, elec_info, gas_info, plot_path)

        monthly_plot_path = os.path.join(out_dir, "monthly_mean_vs_median.png")
        create_monthly_comparison_plot(synth_data, real_data, monthly_plot_path)

    # Save statistics to CSV
    stats_rows = [synth_elec_stats, synth_gas_stats]
    if real_elec_stats:
        stats_rows.append(real_elec_stats)
    if real_gas_stats:
        stats_rows.append(real_gas_stats)

    stats_df = pd.DataFrame(stats_rows)
    csv_path = os.path.join(out_dir, "token_distribution_stats.csv")
    stats_df.to_csv(csv_path, index=False)
    print(f"\nStatistics saved to: {csv_path}")

    # Print diagnosis
    if real_elec_stats and real_gas_stats:
        print_diagnosis(synth_elec_stats, real_elec_stats, synth_gas_stats, real_gas_stats)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    # Restore stdout and save captured output
    sys.stdout = original_stdout
    report_path = os.path.join(out_dir, "token_distribution_report.txt")
    with open(report_path, 'w') as f:
        f.write(log_buffer.getvalue())
    print(f"Full report saved to: {report_path}")

    return {'status': 'completed', 'synth_elec_stats': synth_elec_stats, 'synth_gas_stats': synth_gas_stats}


if __name__ == "__main__":
    main()
