#!/usr/bin/env python3
"""
Diagnostic: Weekly Fidelity Degradation Analysis

Tests whether synthetic data fidelity degrades within each month as
autoregressive generation proceeds. Compares marginal distributions and
means of synthetic vs real (train) data for weeks 1-4 within each month.

If fidelity degrades with autoregressive steps, we expect:
- Week 1 synthetic distributions to closely match real data
- Later weeks to show increasing divergence (shifted means, different shapes)

Each month file contains ~1344-1488 half-hourly timesteps (28-31 days × 48).
We split positionally into weeks of 336 timesteps (7 days × 48 half-hours).

Usage:
    python src/evaluation/diagnostics/diagnose_weekly_fidelity.py

Outputs:
    - weekly_fidelity_summary.csv: Per-week statistics (mean, std, percentiles)
    - weekly_fidelity_distributions.csv: Raw pooled values for replotting
    - weekly_fidelity_marginals.png: Overlaid KDE distributions per week
    - weekly_fidelity_means.png: Mean comparison across weeks
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Note: matplotlib not available, will skip plot generation")

# ============================================================================
# UPDATE THESE PATHS FOR YOUR ENVIRONMENT
# ============================================================================

# Path to directory containing synthetic batch files (batch_NNNN_month_MM.pkl)
SYNTHETIC_DATA_DIR = "experiments/separate_models_inference_test/run_005/generated_data"

# Path to directory containing real tokenised household files
REAL_DATA_DIR = "experiments/separate_models_train_test/run_008/data/tokenised_data"

# Number of real households to sample (set to None for all)
REAL_DATA_SAMPLE_SIZE = 50

# Output directory for results
OUTPUT_DIR = "experiments/separate_models_inference_test/run_005/diagnosis"

# Number of weeks to analyse per month (4 is standard; most months have 4 full weeks)
NUM_WEEKS = 4

# Max half-hourly values to keep per week for distribution plots (controls memory)
MAX_SAMPLES_PER_WEEK = 200_000

# ============================================================================

TIMESTEPS_PER_WEEK = 48 * 7  # 336 half-hours


def find_batch_files(data_dir):
    """Find all batch pickle files in the directory."""
    data_path = Path(data_dir)

    new_format_files = sorted(data_path.glob("batch_*_month_*.pkl"))
    if new_format_files:
        print(f"Found {len(new_format_files)} month files (new format) in {data_dir}")
        return new_format_files

    old_format_files = sorted(data_path.glob("batch_*.pkl"))
    old_format_files = [f for f in old_format_files if 'conditioning' not in f.name]
    print(f"Found {len(old_format_files)} batch files (old format) in {data_dir}")
    return old_format_files


def find_real_household_files(data_dir, sample_size=None):
    """Find real tokenised household files."""
    household_files = sorted(Path(data_dir).glob("*_tokenised.pkl"))
    print(f"Found {len(household_files)} real household files in {data_dir}")

    if sample_size and len(household_files) > sample_size:
        import random
        random.seed(42)
        household_files = random.sample(household_files, sample_size)
        print(f"Sampled {sample_size} households for comparison")

    return household_files


def extract_household_columns(df):
    """Extract electricity and gas column names."""
    elec_cols = [c for c in df.columns if '_elec_net_Wh' in c or '_elec_Wh' in c]
    gas_cols = [c for c in df.columns if '_gas_Wh' in c and '_elec' not in c]
    return elec_cols, gas_cols


def collect_weekly_values_synthetic(batch_files, num_weeks=NUM_WEEKS):
    """
    Collect all half-hourly values per week from synthetic batch files.

    Returns:
        dict: {fuel: {week: np.array of values}}
    """
    # {fuel: {week: list of arrays}}
    weekly = {'electricity': defaultdict(list), 'gas': defaultdict(list)}

    print("\nLoading synthetic data by week...")
    for i, batch_path in enumerate(batch_files):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  Processing {i+1}/{len(batch_files)}: {batch_path.name}")

        df = pd.read_pickle(batch_path)
        df = df.reset_index(drop=True)

        elec_cols, gas_cols = extract_household_columns(df)

        for week in range(1, num_weeks + 1):
            start = (week - 1) * TIMESTEPS_PER_WEEK
            end = week * TIMESTEPS_PER_WEEK
            week_df = df.iloc[start:end]

            if len(week_df) == 0:
                continue

            # Electricity
            for col in elec_cols:
                vals = week_df[col].values.astype(float)
                vals = vals[~np.isnan(vals)]
                if len(vals) > 0:
                    weekly['electricity'][week].append(vals)

            # Gas
            for col in gas_cols:
                vals = week_df[col].values.astype(float)
                vals = vals[~np.isnan(vals)]
                if len(vals) > 0:
                    weekly['gas'][week].append(vals)

    # Concatenate arrays per week
    result = {}
    for fuel in ['electricity', 'gas']:
        result[fuel] = {}
        for week in sorted(weekly[fuel].keys()):
            arr = np.concatenate(weekly[fuel][week]) if weekly[fuel][week] else np.array([])
            # Subsample if too large
            if len(arr) > MAX_SAMPLES_PER_WEEK:
                rng = np.random.RandomState(42)
                arr = rng.choice(arr, MAX_SAMPLES_PER_WEEK, replace=False)
            result[fuel][week] = arr
            print(f"  Synthetic {fuel} week {week}: {len(arr):,} values")

    return result


def collect_weekly_values_real(household_files, num_weeks=NUM_WEEKS):
    """
    Collect all half-hourly values per week from real household files.

    Within each month's data, splits positionally into weeks.

    Returns:
        dict: {fuel: {week: np.array of values}}
    """
    weekly = {'electricity': defaultdict(list), 'gas': defaultdict(list)}

    print("\nLoading real data by week...")
    for i, hh_path in enumerate(household_files):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  Processing household {i+1}/{len(household_files)}")

        try:
            df = pd.read_pickle(hh_path)
        except Exception as e:
            print(f"  Warning: Could not load {hh_path}: {e}")
            continue

        # Determine month column
        if 'month' in df.columns:
            month_series = df['month']
        elif hasattr(df.index, 'month'):
            month_series = df.index.month
        else:
            continue

        # Determine fuel columns
        elec_col = None
        for col in ['Clean_elec_net_Wh', 'elec_Wh', 'elec_value']:
            if col in df.columns:
                elec_col = col
                break

        gas_col = None
        for col in ['Clean_gas_Wh', 'gas_Wh', 'gas_value']:
            if col in df.columns:
                gas_col = col
                break

        # Process each month separately, then split into weeks
        for month_val in sorted(month_series.unique()):
            month_mask = month_series == month_val
            month_df = df[month_mask].reset_index(drop=True)

            for week in range(1, num_weeks + 1):
                start = (week - 1) * TIMESTEPS_PER_WEEK
                end = week * TIMESTEPS_PER_WEEK
                week_df = month_df.iloc[start:end]

                if len(week_df) == 0:
                    continue

                if elec_col:
                    vals = week_df[elec_col].values.astype(float)
                    vals = vals[~np.isnan(vals)]
                    if len(vals) > 0:
                        weekly['electricity'][week].append(vals)

                if gas_col:
                    vals = week_df[gas_col].values.astype(float)
                    vals = vals[~np.isnan(vals)]
                    if len(vals) > 0:
                        weekly['gas'][week].append(vals)

    # Concatenate arrays per week
    result = {}
    for fuel in ['electricity', 'gas']:
        result[fuel] = {}
        for week in sorted(weekly[fuel].keys()):
            arr = np.concatenate(weekly[fuel][week]) if weekly[fuel][week] else np.array([])
            if len(arr) > MAX_SAMPLES_PER_WEEK:
                rng = np.random.RandomState(42)
                arr = rng.choice(arr, MAX_SAMPLES_PER_WEEK, replace=False)
            result[fuel][week] = arr
            print(f"  Real {fuel} week {week}: {len(arr):,} values")

    return result


def compute_weekly_stats(synth_weekly, real_weekly, num_weeks=NUM_WEEKS):
    """
    Compute summary statistics per week for both synthetic and real data.

    Returns:
        pd.DataFrame with columns: fuel, week, source, mean, std, median,
        p10, p25, p75, p90, n
    """
    rows = []

    for fuel in ['electricity', 'gas']:
        for week in range(1, num_weeks + 1):
            for source, data in [('synthetic', synth_weekly), ('real', real_weekly)]:
                vals = data[fuel].get(week, np.array([]))
                if len(vals) == 0:
                    continue

                rows.append({
                    'fuel': fuel,
                    'week': week,
                    'source': source,
                    'mean': np.mean(vals),
                    'std': np.std(vals),
                    'median': np.median(vals),
                    'p10': np.percentile(vals, 10),
                    'p25': np.percentile(vals, 25),
                    'p75': np.percentile(vals, 75),
                    'p90': np.percentile(vals, 90),
                    'n': len(vals),
                })

    return pd.DataFrame(rows)


def create_marginal_plots(synth_weekly, real_weekly, output_dir, num_weeks=NUM_WEEKS):
    """
    Create overlaid marginal distribution plots (KDE) for each week.

    One figure per fuel: 1 row × num_weeks columns, each panel showing
    synth vs real KDE overlay for that week.
    """
    if not HAS_MATPLOTLIB:
        print("Skipping marginal plots (matplotlib not available)")
        return

    for fuel in ['electricity', 'gas']:
        fig, axes = plt.subplots(1, num_weeks, figsize=(5 * num_weeks, 5), sharey=True)
        if num_weeks == 1:
            axes = [axes]

        fuel_label = 'Electricity (Wh)' if fuel == 'electricity' else 'Gas (Wh)'

        for week_idx, week in enumerate(range(1, num_weeks + 1)):
            ax = axes[week_idx]

            synth_vals = synth_weekly[fuel].get(week, np.array([]))
            real_vals = real_weekly[fuel].get(week, np.array([]))

            has_synth = len(synth_vals) > 0
            has_real = len(real_vals) > 0

            if not has_synth and not has_real:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes)
                ax.set_title(f'Week {week}')
                continue

            # Determine shared bin range
            all_vals = []
            if has_real:
                all_vals.append(real_vals)
            if has_synth:
                all_vals.append(synth_vals)
            combined = np.concatenate(all_vals)
            bin_min = np.percentile(combined, 1)
            bin_max = np.percentile(combined, 99)
            bins = np.linspace(bin_min, bin_max, 80)

            if has_real:
                ax.hist(real_vals, bins=bins, density=True, alpha=0.5,
                        color='steelblue', label='Real (train)')
            if has_synth:
                ax.hist(synth_vals, bins=bins, density=True, alpha=0.5,
                        color='coral', label='Synthetic')

            # Add mean lines
            if has_real:
                real_mean = np.mean(real_vals)
                ax.axvline(real_mean, color='steelblue', linestyle='--',
                           linewidth=2, label=f'Real mean: {real_mean:.0f}')
            if has_synth:
                synth_mean = np.mean(synth_vals)
                ax.axvline(synth_mean, color='coral', linestyle='--',
                           linewidth=2, label=f'Synth mean: {synth_mean:.0f}')

            ax.set_title(f'Week {week}', fontsize=12, fontweight='bold')
            ax.set_xlabel(fuel_label)
            if week_idx == 0:
                ax.set_ylabel('Density')
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, alpha=0.3)

        fig.suptitle(
            f'{fuel.upper()}: Marginal Distribution by Week Within Month\n'
            f'(Fidelity degradation test: later weeks should match if generation is stable)',
            fontsize=13, fontweight='bold'
        )
        fig.tight_layout()

        path = os.path.join(output_dir, f'weekly_fidelity_marginals_{fuel}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Marginal distribution plot saved to: {path}")


def create_mean_plot(stats_df, output_dir, num_weeks=NUM_WEEKS):
    """
    Create a line plot of means across weeks, showing divergence.

    Two subplots (electricity, gas), each with synth and real lines.
    """
    if not HAS_MATPLOTLIB:
        print("Skipping mean plot (matplotlib not available)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, fuel in zip(axes, ['electricity', 'gas']):
        fuel_stats = stats_df[stats_df['fuel'] == fuel]

        for source, color, marker in [('real', 'steelblue', 's'), ('synthetic', 'coral', 'o')]:
            src_data = fuel_stats[fuel_stats['source'] == source].sort_values('week')
            if len(src_data) == 0:
                continue

            weeks = src_data['week'].values
            means = src_data['mean'].values
            stds = src_data['std'].values

            label = f'{source.capitalize()}'
            ax.plot(weeks, means, marker=marker, color=color, linewidth=2,
                    markersize=8, label=label)
            ax.fill_between(weeks, means - stds, means + stds,
                            alpha=0.2, color=color)

            # Annotate means
            for w, m in zip(weeks, means):
                ax.annotate(f'{m:.0f}', (w, m), textcoords='offset points',
                            xytext=(0, 10), ha='center', fontsize=8,
                            color=color, fontweight='bold')

        # Add ratio annotation if both sources present
        synth_data = fuel_stats[fuel_stats['source'] == 'synthetic'].sort_values('week')
        real_data = fuel_stats[fuel_stats['source'] == 'real'].sort_values('week')
        if len(synth_data) > 0 and len(real_data) > 0:
            ratios = []
            for _, srow in synth_data.iterrows():
                rrow = real_data[real_data['week'] == srow['week']]
                if len(rrow) > 0 and rrow['mean'].values[0] != 0:
                    ratios.append(srow['mean'] / rrow['mean'].values[0])
                else:
                    ratios.append(np.nan)

            ratio_text = 'Synth/Real ratio: ' + ', '.join(
                f'W{w}={r:.3f}' for w, r in zip(synth_data['week'], ratios)
                if not np.isnan(r)
            )
            ax.text(0.02, 0.02, ratio_text, transform=ax.transAxes,
                    fontsize=9, va='bottom', ha='left',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        fuel_label = 'Electricity (Wh)' if fuel == 'electricity' else 'Gas (Wh)'
        ax.set_xlabel('Week within month', fontsize=12)
        ax.set_ylabel(f'Mean {fuel_label}', fontsize=12)
        ax.set_title(f'{fuel.upper()}: Mean by Week', fontsize=13, fontweight='bold')
        ax.set_xticks(range(1, num_weeks + 1))
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        'Weekly Fidelity: Does Synthetic Quality Degrade Within Each Month?',
        fontsize=14, fontweight='bold'
    )
    fig.tight_layout()

    path = os.path.join(output_dir, 'weekly_fidelity_means.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Mean comparison plot saved to: {path}")


def print_diagnosis(stats_df, num_weeks=NUM_WEEKS):
    """Print textual diagnosis of weekly fidelity."""
    print("\n" + "=" * 70)
    print("WEEKLY FIDELITY DIAGNOSIS")
    print("=" * 70)

    for fuel in ['electricity', 'gas']:
        fuel_stats = stats_df[stats_df['fuel'] == fuel]
        synth = fuel_stats[fuel_stats['source'] == 'synthetic'].sort_values('week')
        real = fuel_stats[fuel_stats['source'] == 'real'].sort_values('week')

        if len(synth) == 0:
            print(f"\n{fuel.upper()}: No synthetic data")
            continue

        print(f"\n{fuel.upper()}:")
        print(f"  {'Week':<6} {'Synth Mean':>12} {'Real Mean':>12} {'Ratio':>8} {'Synth Std':>12} {'Real Std':>12}")
        print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*8} {'-'*12} {'-'*12}")

        ratios = []
        for _, srow in synth.iterrows():
            week = int(srow['week'])
            rrow = real[real['week'] == week]
            real_mean = rrow['mean'].values[0] if len(rrow) > 0 else np.nan
            real_std = rrow['std'].values[0] if len(rrow) > 0 else np.nan
            ratio = srow['mean'] / real_mean if real_mean and not np.isnan(real_mean) and real_mean != 0 else np.nan
            ratios.append(ratio)

            real_mean_str = f"{real_mean:.1f}" if not np.isnan(real_mean) else "N/A"
            real_std_str = f"{real_std:.1f}" if not np.isnan(real_std) else "N/A"
            ratio_str = f"{ratio:.3f}" if not np.isnan(ratio) else "N/A"

            print(f"  {week:<6} {srow['mean']:>12.1f} {real_mean_str:>12} {ratio_str:>8} "
                  f"{srow['std']:>12.1f} {real_std_str:>12}")

        # Assess degradation
        valid_ratios = [r for r in ratios if not np.isnan(r)]
        if len(valid_ratios) >= 2:
            drift = abs(valid_ratios[-1] - valid_ratios[0])
            print(f"\n  Ratio drift (Week {num_weeks} - Week 1): {valid_ratios[-1] - valid_ratios[0]:+.4f}")

            if drift < 0.02:
                print(f"  Assessment: STABLE - negligible fidelity change across weeks")
            elif drift < 0.05:
                print(f"  Assessment: MINOR DRIFT - small fidelity change ({drift:.1%})")
            elif drift < 0.10:
                print(f"  Assessment: MODERATE DRIFT - noticeable fidelity degradation ({drift:.1%})")
            else:
                print(f"  Assessment: SIGNIFICANT DEGRADATION - fidelity degrades substantially ({drift:.1%})")


def main():
    print("=" * 70)
    print("WEEKLY FIDELITY DEGRADATION ANALYSIS")
    print("=" * 70)

    # Validate paths
    if not os.path.exists(SYNTHETIC_DATA_DIR):
        print(f"ERROR: Synthetic data directory not found: {SYNTHETIC_DATA_DIR}")
        print("Please update SYNTHETIC_DATA_DIR in the script.")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find files
    batch_files = find_batch_files(SYNTHETIC_DATA_DIR)
    if len(batch_files) == 0:
        print("ERROR: No batch files found!")
        sys.exit(1)

    # Collect synthetic weekly values
    synth_weekly = collect_weekly_values_synthetic(batch_files, NUM_WEEKS)

    # Collect real weekly values (if available)
    real_weekly = {'electricity': {}, 'gas': {}}
    if REAL_DATA_DIR and os.path.exists(REAL_DATA_DIR):
        household_files = find_real_household_files(REAL_DATA_DIR, REAL_DATA_SAMPLE_SIZE)
        if len(household_files) > 0:
            real_weekly = collect_weekly_values_real(household_files, NUM_WEEKS)
    else:
        print(f"\nNote: Real data directory not found: {REAL_DATA_DIR}")
        print("Proceeding with synthetic data only.")

    # Compute summary statistics
    stats_df = compute_weekly_stats(synth_weekly, real_weekly, NUM_WEEKS)
    stats_path = os.path.join(OUTPUT_DIR, 'weekly_fidelity_summary.csv')
    stats_df.to_csv(stats_path, index=False, float_format='%.4f')
    print(f"\nSummary statistics saved to: {stats_path}")

    # Save raw distribution samples for replotting
    dist_rows = []
    for source, data in [('synthetic', synth_weekly), ('real', real_weekly)]:
        for fuel in ['electricity', 'gas']:
            for week, vals in data[fuel].items():
                for v in vals:
                    dist_rows.append({
                        'fuel': fuel,
                        'week': week,
                        'source': source,
                        'value': v,
                    })

    if dist_rows:
        dist_df = pd.DataFrame(dist_rows)
        dist_path = os.path.join(OUTPUT_DIR, 'weekly_fidelity_distributions.csv')
        dist_df.to_csv(dist_path, index=False, float_format='%.4f')
        print(f"Distribution data saved to: {dist_path}")

    # Create plots
    create_marginal_plots(synth_weekly, real_weekly, OUTPUT_DIR, NUM_WEEKS)
    create_mean_plot(stats_df, OUTPUT_DIR, NUM_WEEKS)

    # Print diagnosis
    print_diagnosis(stats_df, NUM_WEEKS)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
