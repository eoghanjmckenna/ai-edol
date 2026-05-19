#!/usr/bin/env python3
"""
Diagnostic: Detect autoregressive collapse by analyzing monthly trends.

If autoregressive collapse is occurring, we expect to see:
- Electricity values decreasing from month 1 → month 12
- Gas values remaining more stable (due to zero-aware tokenization anchor)

Each synthetic batch starts with month 1 and ends with month 12, so collapse
would manifest as a downward trend within each batch, and consistently across batches.

Usage:
    python scripts/diagnose_autoregressive_collapse.py

Outputs:
    - autoregressive_collapse_analysis.csv: Monthly statistics
    - autoregressive_collapse_plot.png: Visualization of trends
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Try to import matplotlib, provide helpful message if not available
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Note: matplotlib not available, will skip plot generation")

# ============================================================================
# UPDATE THESE PATHS FOR YOUR TRE ENVIRONMENT
# ============================================================================

# Path to directory containing synthetic batch files (batch_0001.pkl, etc.)
SYNTHETIC_DATA_DIR = "experiments/separate_models_inference_test/run_005/generated_data"

# Path to directory containing real tokenised household files (for comparison)
# Set to None to skip real data comparison
REAL_DATA_DIR = None

# Number of real households to sample for comparison (set to None for all)
REAL_DATA_SAMPLE_SIZE = 20

# Output directory for results (will be created if doesn't exist)
OUTPUT_DIR = "experiments/separate_models_inference_test/run_005/diagnosis"

# ============================================================================


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
        print(f"Found {len(new_format_files)} month files (new format) in {data_dir}")
        return new_format_files

    # Fall back to old format
    old_format_files = sorted(data_path.glob("batch_*.pkl"))
    # Exclude conditioning_summary.pkl
    old_format_files = [f for f in old_format_files if 'conditioning' not in f.name]
    print(f"Found {len(old_format_files)} batch files (old format) in {data_dir}")
    return old_format_files


def find_real_household_files(data_dir, sample_size=None, puprn_filter=None):
    """Find real tokenised household files.

    Args:
        data_dir: Directory containing tokenised household files
        sample_size: Max households to sample (None = all)
        puprn_filter: List of PUPRNs to include (None = no filter)
    """
    household_files = sorted(Path(data_dir).glob("*_tokenised.pkl"))
    print(f"Found {len(household_files)} real household files in {data_dir}")

    if puprn_filter is not None:
        puprn_set = set(puprn_filter)
        household_files = [f for f in household_files
                          if f.stem.replace('_tokenised', '') in puprn_set]
        print(f"Filtered to {len(household_files)} train households")

    if sample_size and len(household_files) > sample_size:
        import random
        random.seed(42)
        household_files = random.sample(household_files, sample_size)
        print(f"Sampled {sample_size} households for comparison")

    return household_files


def analyze_real_household(household_path, elec_boundaries=None):
    """
    Analyze a single real household file for monthly statistics.

    Returns dict with monthly means for electricity and gas.
    """
    df = pd.read_pickle(household_path)

    # Check if month column exists
    if 'month' not in df.columns:
        return None

    results = {
        'household_file': household_path.name,
        'num_timesteps': len(df),
        'monthly_elec': {},
        'monthly_gas': {},
    }

    # Determine which columns contain the values
    # Real data might have tokens or decoded values
    elec_col = None
    gas_col = None

    if 'Clean_elec_net_Wh' in df.columns:
        elec_col = 'Clean_elec_net_Wh'
    elif 'elec_value' in df.columns:
        elec_col = 'elec_value'
    elif 'elec_token' in df.columns and elec_boundaries is not None:
        # Need to decode tokens - skip for simplicity, use raw values if available
        pass

    if 'Clean_gas_Wh' in df.columns:
        gas_col = 'Clean_gas_Wh'
    elif 'gas_value' in df.columns:
        gas_col = 'gas_value'

    # Calculate monthly statistics
    for month in range(1, 13):
        month_mask = df['month'] == month
        month_data = df[month_mask]

        if len(month_data) == 0:
            continue

        # Electricity
        if elec_col and elec_col in month_data.columns:
            elec_values = month_data[elec_col].values.astype(float)
            elec_values = elec_values[~np.isnan(elec_values)]
            if len(elec_values) > 0:
                results['monthly_elec'][month] = np.mean(elec_values)

        # Gas
        if gas_col and gas_col in month_data.columns:
            gas_values = month_data[gas_col].values.astype(float)
            gas_values = gas_values[~np.isnan(gas_values)]
            if len(gas_values) > 0:
                results['monthly_gas'][month] = np.mean(gas_values)

    return results


def analyze_real_data(household_files):
    """Analyze all real household files and aggregate results."""
    all_results = []
    monthly_elec_all = defaultdict(list)
    monthly_gas_all = defaultdict(list)

    print("\nAnalyzing real households...")
    for i, household_path in enumerate(household_files):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Processing household {i+1}/{len(household_files)}: {household_path.name}")

        result = analyze_real_household(household_path)
        if result is None:
            continue

        all_results.append(result)

        for month, value in result['monthly_elec'].items():
            monthly_elec_all[month].append(value)
        for month, value in result['monthly_gas'].items():
            monthly_gas_all[month].append(value)

    return all_results, monthly_elec_all, monthly_gas_all


def create_real_summary_dataframe(monthly_elec_all, monthly_gas_all):
    """Create summary DataFrame for real data."""
    rows = []

    for month in range(1, 13):
        elec_values = monthly_elec_all.get(month, [])
        gas_values = monthly_gas_all.get(month, [])

        row = {
            'month': month,
            'elec_mean': np.mean(elec_values) if elec_values else np.nan,
            'elec_std': np.std(elec_values) if elec_values else np.nan,
            'gas_mean': np.mean(gas_values) if gas_values else np.nan,
            'gas_std': np.std(gas_values) if gas_values else np.nan,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def extract_household_columns(df):
    """Extract electricity and gas column names for each household."""
    elec_cols = [c for c in df.columns if '_elec_net_Wh' in c]
    gas_cols = [c for c in df.columns if '_gas_Wh' in c]
    return elec_cols, gas_cols


def analyze_batch(batch_path):
    """
    Analyze a single batch file for monthly electricity and gas statistics.

    Returns dict with monthly means for electricity and gas.
    """
    df = pd.read_pickle(batch_path)

    # Check if month column exists
    if 'month' not in df.columns:
        print(f"  Warning: 'month' column not found in {batch_path}")
        return None

    elec_cols, gas_cols = extract_household_columns(df)

    if len(elec_cols) == 0:
        print(f"  Warning: No electricity columns found in {batch_path}")
        return None

    results = {
        'batch_file': batch_path.name,
        'num_households': len(elec_cols),
        'num_timesteps': len(df),
        'monthly_elec': {},
        'monthly_gas': {},
        'monthly_elec_std': {},
        'monthly_gas_std': {},
    }

    # Calculate monthly statistics
    for month in range(1, 13):
        month_mask = df['month'] == month
        month_data = df[month_mask]

        if len(month_data) == 0:
            continue

        # Electricity: calculate mean per household, then std across households
        # This gives between-household variance (comparable to real data)
        hh_elec_means = []
        for col in elec_cols:
            vals = month_data[col].values.astype(float)
            vals = vals[~np.isnan(vals)]
            if len(vals) > 0:
                hh_elec_means.append(np.mean(vals))

        if len(hh_elec_means) > 0:
            results['monthly_elec'][month] = np.mean(hh_elec_means)
            results['monthly_elec_std'][month] = np.std(hh_elec_means)

        # Gas: calculate mean per household, then std across households
        if len(gas_cols) > 0:
            hh_gas_means = []
            for col in gas_cols:
                vals = month_data[col].values.astype(float)
                vals = vals[~np.isnan(vals)]
                if len(vals) > 0:
                    hh_gas_means.append(np.mean(vals))

            if len(hh_gas_means) > 0:
                results['monthly_gas'][month] = np.mean(hh_gas_means)
                results['monthly_gas_std'][month] = np.std(hh_gas_means)

    return results


def analyze_all_batches(batch_files):
    """Analyze all batch files and aggregate results."""
    all_results = []

    # Aggregated monthly data across all batches
    monthly_elec_all = defaultdict(list)
    monthly_gas_all = defaultdict(list)

    print("\nAnalyzing batches...")
    for i, batch_path in enumerate(batch_files):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Processing batch {i+1}/{len(batch_files)}: {batch_path.name}")

        result = analyze_batch(batch_path)
        if result is None:
            continue

        all_results.append(result)

        # Aggregate monthly values
        for month, value in result['monthly_elec'].items():
            monthly_elec_all[month].append(value)
        for month, value in result['monthly_gas'].items():
            monthly_gas_all[month].append(value)

    return all_results, monthly_elec_all, monthly_gas_all


def create_summary_dataframe(monthly_elec_all, monthly_gas_all):
    """Create summary DataFrame with monthly statistics."""
    rows = []

    for month in range(1, 13):
        elec_values = monthly_elec_all.get(month, [])
        gas_values = monthly_gas_all.get(month, [])

        row = {
            'month': month,
            'elec_mean': np.mean(elec_values) if elec_values else np.nan,
            'elec_std': np.std(elec_values) if elec_values else np.nan,
            'elec_min': np.min(elec_values) if elec_values else np.nan,
            'elec_max': np.max(elec_values) if elec_values else np.nan,
            'elec_n_batches': len(elec_values),
            'gas_mean': np.mean(gas_values) if gas_values else np.nan,
            'gas_std': np.std(gas_values) if gas_values else np.nan,
            'gas_min': np.min(gas_values) if gas_values else np.nan,
            'gas_max': np.max(gas_values) if gas_values else np.nan,
            'gas_n_batches': len(gas_values),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Calculate collapse ratio (month 12 / month 1)
    if df['elec_mean'].iloc[0] > 0:
        elec_collapse_ratio = df['elec_mean'].iloc[11] / df['elec_mean'].iloc[0]
    else:
        elec_collapse_ratio = np.nan

    if df['gas_mean'].iloc[0] > 0:
        gas_collapse_ratio = df['gas_mean'].iloc[11] / df['gas_mean'].iloc[0]
    else:
        gas_collapse_ratio = np.nan

    return df, elec_collapse_ratio, gas_collapse_ratio


def create_plot(df_summary, elec_collapse_ratio, gas_collapse_ratio, output_path,
                df_real=None):
    """Create visualization of monthly trends with optional real data comparison."""
    if not HAS_MATPLOTLIB:
        print("Skipping plot generation (matplotlib not available)")
        return

    _, axes = plt.subplots(1, 2, figsize=(14, 5))

    months = df_summary['month'].values

    # Electricity plot
    ax1 = axes[0]

    # Plot real data first (if available) as reference - make it prominent
    print(f"[DEBUG] In create_plot: df_real is {'NOT None' if df_real is not None else 'None'}")
    if df_real is not None:
        print(f"[DEBUG] Real elec range: {df_real['elec_mean'].min():.0f} - {df_real['elec_mean'].max():.0f}")
        print(f"[DEBUG] Synth elec range: {df_summary['elec_mean'].min():.0f} - {df_summary['elec_mean'].max():.0f}")
        print(f"[DEBUG] Real gas range: {df_real['gas_mean'].min():.0f} - {df_real['gas_mean'].max():.0f}")
        print(f"[DEBUG] Synth gas range: {df_summary['gas_mean'].min():.0f} - {df_summary['gas_mean'].max():.0f}")
        ax1.plot(months, df_real['elec_mean'], marker='s', color='darkgreen',
                 linewidth=3, markersize=10, label='Real data', linestyle='-', zorder=10)
        ax1.fill_between(months,
                         df_real['elec_mean'] - df_real['elec_std'],
                         df_real['elec_mean'] + df_real['elec_std'],
                         alpha=0.3, color='green', label='Real ±1 std')
        # Add mean labels for real data
        for m, val in zip(months, df_real['elec_mean']):
            if not np.isnan(val):
                ax1.annotate(f'{val:.0f}', (m, val), textcoords='offset points',
                             xytext=(0, 12), ha='center', fontsize=7, color='darkgreen', fontweight='bold')

    # Plot synthetic data
    ax1.plot(months, df_summary['elec_mean'], marker='o', color='red',
             linewidth=2, markersize=8, label='Synthetic data', linestyle='--')
    ax1.fill_between(months,
                     df_summary['elec_mean'] - df_summary['elec_std'],
                     df_summary['elec_mean'] + df_summary['elec_std'],
                     alpha=0.2, color='red', label='Synth ±1 std')

    # Add mean labels for synthetic data
    for m, val in zip(months, df_summary['elec_mean']):
        if not np.isnan(val):
            ax1.annotate(f'{val:.0f}', (m, val), textcoords='offset points',
                         xytext=(0, -15), ha='center', fontsize=7, color='red')

    ax1.set_xlabel('Month', fontsize=12)
    ax1.set_ylabel('Mean Electricity (Wh)', fontsize=12)
    ax1.set_title(f'Electricity by Month\nCollapse Ratio (M12/M1): {elec_collapse_ratio:.2%}', fontsize=14)
    ax1.set_xticks(months)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')

    # Gas plot
    ax2 = axes[1]

    # Plot real data first (if available) as reference - make it prominent
    if df_real is not None:
        ax2.plot(months, df_real['gas_mean'], marker='s', color='darkgreen',
                 linewidth=3, markersize=10, label='Real data', linestyle='-', zorder=10)
        ax2.fill_between(months,
                         df_real['gas_mean'] - df_real['gas_std'],
                         df_real['gas_mean'] + df_real['gas_std'],
                         alpha=0.3, color='green', label='Real ±1 std')
        # Add mean labels for real data
        for m, val in zip(months, df_real['gas_mean']):
            if not np.isnan(val):
                ax2.annotate(f'{val:.0f}', (m, val), textcoords='offset points',
                             xytext=(0, 12), ha='center', fontsize=7, color='darkgreen', fontweight='bold')

    # Plot synthetic data
    ax2.plot(months, df_summary['gas_mean'], marker='o', color='darkorange',
             linewidth=2, markersize=8, label='Synthetic data', linestyle='--')
    ax2.fill_between(months,
                     df_summary['gas_mean'] - df_summary['gas_std'],
                     df_summary['gas_mean'] + df_summary['gas_std'],
                     alpha=0.2, color='orange', label='Synth ±1 std')

    # Add mean labels for synthetic data
    for m, val in zip(months, df_summary['gas_mean']):
        if not np.isnan(val):
            ax2.annotate(f'{val:.0f}', (m, val), textcoords='offset points',
                         xytext=(0, -15), ha='center', fontsize=7, color='darkorange')

    ax2.set_xlabel('Month', fontsize=12)
    ax2.set_ylabel('Mean Gas (Wh)', fontsize=12)
    ax2.set_title(f'Gas by Month\nCollapse Ratio (M12/M1): {gas_collapse_ratio:.2%}', fontsize=14)
    ax2.set_xticks(months)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    plt.close()


def print_diagnosis(df_summary, elec_collapse_ratio, gas_collapse_ratio):
    """Print diagnosis based on the analysis."""
    print("\n" + "=" * 70)
    print("AUTOREGRESSIVE COLLAPSE DIAGNOSIS")
    print("=" * 70)

    print("\nMonthly Mean Values:")
    print("-" * 50)
    print(f"{'Month':<8} {'Electricity (Wh)':<20} {'Gas (Wh)':<20}")
    print("-" * 50)
    for _, row in df_summary.iterrows():
        print(f"{int(row['month']):<8} {row['elec_mean']:<20.1f} {row['gas_mean']:<20.1f}")

    print("\n" + "-" * 50)
    print("COLLAPSE RATIOS (Month 12 / Month 1):")
    print("-" * 50)
    print(f"  Electricity: {elec_collapse_ratio:.2%}")
    print(f"  Gas:         {gas_collapse_ratio:.2%}")

    print("\n" + "-" * 50)
    print("INTERPRETATION:")
    print("-" * 50)

    if elec_collapse_ratio < 0.5:
        print("  ELECTRICITY: SEVERE COLLAPSE DETECTED")
        print(f"    Month 12 values are only {elec_collapse_ratio:.0%} of Month 1 values.")
        print("    This indicates significant autoregressive collapse.")
    elif elec_collapse_ratio < 0.8:
        print("  ELECTRICITY: MODERATE COLLAPSE DETECTED")
        print(f"    Month 12 values are {elec_collapse_ratio:.0%} of Month 1 values.")
        print("    Some autoregressive drift is occurring.")
    elif elec_collapse_ratio < 1.2:
        print("  ELECTRICITY: STABLE (No collapse)")
        print(f"    Month 12 values are {elec_collapse_ratio:.0%} of Month 1 values.")
        print("    Values are consistent across months.")
    else:
        print("  ELECTRICITY: INCREASING TREND")
        print(f"    Month 12 values are {elec_collapse_ratio:.0%} of Month 1 values.")
        print("    Values increase over time (opposite of collapse).")

    print()

    if gas_collapse_ratio < 0.5:
        print("  GAS: SEVERE COLLAPSE DETECTED")
    elif gas_collapse_ratio < 0.8:
        print("  GAS: MODERATE COLLAPSE DETECTED")
    elif gas_collapse_ratio < 1.2:
        print("  GAS: STABLE (No collapse)")
        print(f"    Month 12 values are {gas_collapse_ratio:.0%} of Month 1 values.")
    else:
        print("  GAS: INCREASING TREND (expected seasonal pattern)")
        print(f"    Month 12 values are {gas_collapse_ratio:.0%} of Month 1 values.")
        print("    Higher gas in winter months is expected (heating).")


def main(synthetic_dir=None, real_dir=None, output_dir=None, sample_size=None, train_puprns=None):
    # Use provided values or fall back to global defaults
    synthetic_data_dir = synthetic_dir if synthetic_dir else SYNTHETIC_DATA_DIR
    real_data_dir = real_dir if real_dir else REAL_DATA_DIR
    out_dir = output_dir if output_dir else OUTPUT_DIR
    real_sample_size = sample_size if sample_size is not None else REAL_DATA_SAMPLE_SIZE

    print("=" * 70)
    print("AUTOREGRESSIVE COLLAPSE ANALYSIS")
    print("=" * 70)

    # Validate paths
    if not os.path.exists(synthetic_data_dir):
        print(f"ERROR: Synthetic data directory not found: {synthetic_data_dir}")
        print("Please update SYNTHETIC_DATA_DIR in the script or use --synthetic-dir.")
        sys.exit(1)

    # Create output directory
    os.makedirs(out_dir, exist_ok=True)

    # Find batch files
    batch_files = find_batch_files(synthetic_data_dir)
    if len(batch_files) == 0:
        print("ERROR: No batch files found!")
        sys.exit(1)

    # Analyze all batches
    all_results, monthly_elec_all, monthly_gas_all = analyze_all_batches(batch_files)

    if len(all_results) == 0:
        print("ERROR: No valid results from batch analysis!")
        sys.exit(1)

    # Create summary
    df_summary, elec_collapse_ratio, gas_collapse_ratio = create_summary_dataframe(
        monthly_elec_all, monthly_gas_all
    )

    # Analyze real data if path is provided
    df_real = None
    if real_data_dir and os.path.exists(real_data_dir):
        household_files = find_real_household_files(real_data_dir, real_sample_size, puprn_filter=train_puprns)
        if len(household_files) > 0:
            real_results, real_elec_all, real_gas_all = analyze_real_data(household_files)
            if len(real_results) > 0:
                df_real = create_real_summary_dataframe(real_elec_all, real_gas_all)

                # Save real data summary too
                real_csv_path = os.path.join(out_dir, "real_data_monthly_summary.csv")
                df_real.to_csv(real_csv_path, index=False)
                print(f"\nReal data CSV saved to: {real_csv_path}")
    elif real_data_dir:
        print(f"\nNote: Real data directory not found: {real_data_dir}")
        print("Skipping real data comparison.")

    # Save CSV
    csv_path = os.path.join(out_dir, "autoregressive_collapse_analysis.csv")
    df_summary.to_csv(csv_path, index=False)
    print(f"\nSynthetic data CSV saved to: {csv_path}")

    # Create plot (with real data if available)
    plot_path = os.path.join(out_dir, "autoregressive_collapse_plot.png")

    # Debug: Check if df_real is being passed correctly
    if df_real is not None:
        print(f"\n[DEBUG] df_real is NOT None, shape: {df_real.shape}")
        print(f"[DEBUG] df_real columns: {list(df_real.columns)}")
        print(f"[DEBUG] df_real elec_mean values: {df_real['elec_mean'].tolist()}")
    else:
        print(f"\n[DEBUG] df_real IS None - real data will not be plotted")

    create_plot(df_summary, elec_collapse_ratio, gas_collapse_ratio, plot_path,
                df_real=df_real)

    # Print diagnosis
    print_diagnosis(df_summary, elec_collapse_ratio, gas_collapse_ratio)

    # Also save detailed per-batch results
    detailed_rows = []
    for result in all_results:
        for month in range(1, 13):
            detailed_rows.append({
                'batch': result['batch_file'],
                'month': month,
                'elec_mean': result['monthly_elec'].get(month, np.nan),
                'gas_mean': result['monthly_gas'].get(month, np.nan),
            })

    df_detailed = pd.DataFrame(detailed_rows)
    detailed_csv_path = os.path.join(OUTPUT_DIR, "autoregressive_collapse_detailed.csv")
    df_detailed.to_csv(detailed_csv_path, index=False)
    print(f"Detailed CSV saved to: {detailed_csv_path}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect autoregressive collapse by analyzing monthly trends in synthetic data."
    )
    parser.add_argument(
        "--synthetic-dir", "-s",
        help="Path to directory containing synthetic batch files (overrides SYNTHETIC_DATA_DIR)"
    )
    parser.add_argument(
        "--real-dir", "-r",
        help="Path to directory containing real tokenised household files (overrides REAL_DATA_DIR)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        help="Path to output directory for results (overrides OUTPUT_DIR)"
    )
    parser.add_argument(
        "--sample-size", "-n",
        type=int,
        help="Number of real households to sample for comparison (overrides REAL_DATA_SAMPLE_SIZE)"
    )

    args = parser.parse_args()

    # Override global config with CLI arguments if provided
    if args.synthetic_dir:
        SYNTHETIC_DATA_DIR = args.synthetic_dir
    if args.real_dir:
        REAL_DATA_DIR = args.real_dir
    if args.output_dir:
        OUTPUT_DIR = args.output_dir
    if args.sample_size is not None:
        REAL_DATA_SAMPLE_SIZE = args.sample_size

    main()
