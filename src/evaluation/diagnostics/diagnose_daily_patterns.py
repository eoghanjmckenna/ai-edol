#!/usr/bin/env python3
"""
Diagnostic: Analyze daily temporal patterns (half-hourly profiles) across months.

This script compares the average daily consumption profile between synthetic
and real data, showing how patterns evolve from month 1 through month 12.

Visualization approach (Option A from design discussion):
- Top row: Heatmaps showing daily profile × month (overview of all months)
- Bottom row: Line plots for key months (Jan, Jul, Dec) with std bands

Usage:
    python scripts/diagnose_daily_patterns.py

Outputs:
    - daily_patterns_analysis.csv: Half-hourly statistics by month
    - daily_patterns_plot.png: Heatmaps + key month profiles
"""

import os
import sys
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
# UPDATE THESE PATHS FOR YOUR ENVIRONMENT
# ============================================================================

# Path to directory containing synthetic batch files (batch_0001.pkl, etc.)
SYNTHETIC_DATA_DIR = "/path/to/generated_data"

# Path to directory containing real tokenised household files (for comparison)
# Set to None to skip real data comparison
REAL_DATA_DIR = "/path/to/tokenised_data"

# Number of real households to sample for comparison (set to None for all)
REAL_DATA_SAMPLE_SIZE = 50

# Output directory for results (will be created if doesn't exist)
OUTPUT_DIR = "/path/to/output"

# Fuels to analyze (list of "electricity" and/or "gas")
FUELS = ["electricity", "gas"]

# Context length in timesteps (for marking on degradation plot)
# Set to None to skip marking. Common values: 336 (1 week), 672 (2 weeks), 1344 (1 month)
CONTEXT_LENGTH = 1344  # Update this to match your model's context length

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


def extract_household_columns(df, fuel="electricity"):
    """Extract column names for households based on fuel type."""
    if fuel == "electricity":
        cols = [c for c in df.columns if '_elec_net_Wh' in c]
    else:  # gas
        cols = [c for c in df.columns if '_gas_Wh' in c]
    return cols


def analyze_batch_daily_patterns(batch_path, fuel="electricity"):
    """
    Analyze a single batch file for daily patterns by month.

    Returns dict with shape {month: {half_hour: [values across households]}}
    Note: half_hour is 0-indexed (0-47) in output, regardless of input format.
    """
    df = pd.read_pickle(batch_path)

    # Check required columns
    if 'month' not in df.columns:
        print(f"  Warning: 'month' column not found in {batch_path}")
        return None

    # Handle different half-hour column names
    hh_col = None
    hh_offset = 0
    if 'half_hour' in df.columns:
        hh_col = 'half_hour'
        if df['half_hour'].min() == 1:
            hh_offset = 1
    elif 'hh' in df.columns:
        hh_col = 'hh'
        if df['hh'].min() == 1:
            hh_offset = 1

    if hh_col is None:
        print(f"  Warning: No half_hour/hh column found in {batch_path}")
        return None

    cols = extract_household_columns(df, fuel)
    if len(cols) == 0:
        print(f"  Warning: No {fuel} columns found in {batch_path}")
        return None

    # Structure: {month: {half_hour: [household means]}} (0-indexed half_hour)
    results = defaultdict(lambda: defaultdict(list))

    for month in range(1, 13):
        month_mask = df['month'] == month
        month_data = df[month_mask]

        if len(month_data) == 0:
            continue

        for hh in range(48):
            file_hh = hh + hh_offset
            hh_mask = month_data[hh_col] == file_hh
            hh_data = month_data[hh_mask]

            if len(hh_data) == 0:
                continue

            # Get mean value for each household at this month/half_hour
            for col in cols:
                vals = hh_data[col].values.astype(float)
                vals = vals[~np.isnan(vals)]
                if len(vals) > 0:
                    results[month][hh].append(np.mean(vals))

    return dict(results)


def analyze_real_household_daily_patterns(household_path, fuel="electricity"):
    """
    Analyze a single real household file for daily patterns by month.

    Returns dict with shape {month: {half_hour: value}}
    Note: half_hour is 0-indexed (0-47) in output, regardless of input format.
    """
    df = pd.read_pickle(household_path)

    # Check required columns
    if 'month' not in df.columns:
        return None

    # Handle different half-hour column names
    # Tokenised files use 'hh' (1-48), batch files use 'half_hour' (0-47 or 1-48)
    hh_col = None
    hh_offset = 0  # Offset to convert to 0-indexed
    if 'half_hour' in df.columns:
        hh_col = 'half_hour'
        # Check if 0-indexed or 1-indexed
        if df['half_hour'].min() == 1:
            hh_offset = 1
    elif 'hh' in df.columns:
        hh_col = 'hh'
        # 'hh' is typically 1-indexed (1-48)
        if df['hh'].min() == 1:
            hh_offset = 1

    if hh_col is None:
        return None

    # Determine value column
    if fuel == "electricity":
        if 'Clean_elec_net_Wh' in df.columns:
            val_col = 'Clean_elec_net_Wh'
        elif 'elec_value' in df.columns:
            val_col = 'elec_value'
        else:
            return None
    else:  # gas
        if 'Clean_gas_Wh' in df.columns:
            val_col = 'Clean_gas_Wh'
        elif 'gas_value' in df.columns:
            val_col = 'gas_value'
        else:
            return None

    # Structure: {month: {half_hour: mean_value}} (0-indexed half_hour)
    results = defaultdict(dict)

    for month in range(1, 13):
        month_mask = df['month'] == month
        month_data = df[month_mask]

        if len(month_data) == 0:
            continue

        for hh in range(48):
            # Convert to the indexing used in the file
            file_hh = hh + hh_offset
            hh_mask = month_data[hh_col] == file_hh
            hh_data = month_data[hh_mask]

            if len(hh_data) == 0:
                continue

            vals = hh_data[val_col].values.astype(float)
            vals = vals[~np.isnan(vals)]
            if len(vals) > 0:
                results[month][hh] = np.mean(vals)

    return dict(results)


# ============================================================================
# WEEKLY ANALYSIS FUNCTIONS (for degradation analysis)
# ============================================================================

def analyze_batch_daily_patterns_by_week(batch_path, fuel="electricity", num_weeks=8):
    """
    Analyze a single batch file for daily patterns by WEEK (not month).

    Returns dict with shape {week: {half_hour: [values across households]}}
    Week 1 = first 7 days (timesteps 0-335), Week 2 = next 7 days, etc.
    """
    df = pd.read_pickle(batch_path)

    # Handle different half-hour column names
    hh_col = None
    hh_offset = 0
    if 'half_hour' in df.columns:
        hh_col = 'half_hour'
        if df['half_hour'].min() == 1:
            hh_offset = 1
    elif 'hh' in df.columns:
        hh_col = 'hh'
        if df['hh'].min() == 1:
            hh_offset = 1

    if hh_col is None:
        print(f"  Warning: No half_hour/hh column found in {batch_path}")
        return None

    cols = extract_household_columns(df, fuel)
    if len(cols) == 0:
        print(f"  Warning: No {fuel} columns found in {batch_path}")
        return None

    # Add row index for week calculation
    df = df.reset_index(drop=True)

    # Structure: {week: {half_hour: [household means]}}
    results = defaultdict(lambda: defaultdict(list))

    timesteps_per_week = 48 * 7  # 336 half-hours per week

    for week in range(1, num_weeks + 1):
        start_idx = (week - 1) * timesteps_per_week
        end_idx = week * timesteps_per_week

        week_data = df.iloc[start_idx:end_idx] if end_idx <= len(df) else df.iloc[start_idx:]

        if len(week_data) == 0:
            continue

        for hh in range(48):
            file_hh = hh + hh_offset
            hh_mask = week_data[hh_col] == file_hh
            hh_data = week_data[hh_mask]

            if len(hh_data) == 0:
                continue

            # Get mean value for each household at this week/half_hour
            for col in cols:
                vals = hh_data[col].values.astype(float)
                vals = vals[~np.isnan(vals)]
                if len(vals) > 0:
                    results[week][hh].append(np.mean(vals))

    return dict(results)


def analyze_real_household_daily_patterns_by_week(household_path, fuel="electricity", num_weeks=8):
    """
    Analyze a single real household file for daily patterns by WEEK.

    Returns dict with shape {week: {half_hour: mean_value}}
    """
    df = pd.read_pickle(household_path)

    # Handle different half-hour column names
    hh_col = None
    hh_offset = 0
    if 'half_hour' in df.columns:
        hh_col = 'half_hour'
        if df['half_hour'].min() == 1:
            hh_offset = 1
    elif 'hh' in df.columns:
        hh_col = 'hh'
        if df['hh'].min() == 1:
            hh_offset = 1

    if hh_col is None:
        return None

    # Determine value column
    if fuel == "electricity":
        if 'Clean_elec_net_Wh' in df.columns:
            val_col = 'Clean_elec_net_Wh'
        elif 'elec_value' in df.columns:
            val_col = 'elec_value'
        else:
            return None
    else:  # gas
        if 'Clean_gas_Wh' in df.columns:
            val_col = 'Clean_gas_Wh'
        elif 'gas_value' in df.columns:
            val_col = 'gas_value'
        else:
            return None

    # Reset index for week calculation
    df = df.reset_index(drop=True)

    results = defaultdict(dict)
    timesteps_per_week = 48 * 7

    for week in range(1, num_weeks + 1):
        start_idx = (week - 1) * timesteps_per_week
        end_idx = week * timesteps_per_week

        week_data = df.iloc[start_idx:end_idx] if end_idx <= len(df) else df.iloc[start_idx:]

        if len(week_data) == 0:
            continue

        for hh in range(48):
            file_hh = hh + hh_offset
            hh_mask = week_data[hh_col] == file_hh
            hh_data = week_data[hh_mask]

            if len(hh_data) == 0:
                continue

            vals = hh_data[val_col].values.astype(float)
            vals = vals[~np.isnan(vals)]
            if len(vals) > 0:
                results[week][hh] = np.mean(vals)

    return dict(results)


def aggregate_weekly_results(all_results, num_weeks=8):
    """Aggregate weekly results from all batches/households."""
    combined = defaultdict(lambda: defaultdict(list))

    for result in all_results:
        if result is None:
            continue
        for week, hh_data in result.items():
            if isinstance(hh_data, dict):
                for hh, values in hh_data.items():
                    if isinstance(values, list):
                        combined[week][hh].extend(values)
                    else:
                        combined[week][hh].append(values)

    # Calculate statistics
    aggregated = {}
    for week in range(1, num_weeks + 1):
        aggregated[week] = {}
        for hh in range(48):
            values = combined[week][hh]
            if values:
                aggregated[week][hh] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                }
            else:
                aggregated[week][hh] = {'mean': np.nan, 'std': np.nan}

    return aggregated


def calculate_weekly_metrics(synth_data, real_data, num_weeks=8):
    """
    Calculate daily pattern metrics for each week.

    Returns list of dicts with metrics per week.
    """
    metrics = []

    for week in range(1, num_weeks + 1):
        week_metrics = {'week': week, 'timestep_start': (week - 1) * 336}

        # Synthetic profile
        synth_profile = np.array([
            synth_data[week][hh]['mean'] if hh in synth_data.get(week, {}) else np.nan
            for hh in range(48)
        ])

        # Real profile
        real_profile = np.array([
            real_data[week][hh]['mean'] if hh in real_data.get(week, {}) else np.nan
            for hh in range(48)
        ]) if real_data else None

        # Synthetic metrics
        if not np.all(np.isnan(synth_profile)):
            synth_mean = np.nanmean(synth_profile)
            synth_std = np.nanstd(synth_profile)
            week_metrics['synth_cv'] = synth_std / synth_mean if synth_mean > 0 else np.nan
            week_metrics['synth_peak'] = np.nanmax(synth_profile)
            week_metrics['synth_trough'] = np.nanmin(synth_profile)
            week_metrics['synth_pt_ratio'] = week_metrics['synth_peak'] / week_metrics['synth_trough'] if week_metrics['synth_trough'] > 0 else np.nan
        else:
            week_metrics['synth_cv'] = np.nan
            week_metrics['synth_pt_ratio'] = np.nan

        # Real metrics and comparison
        if real_profile is not None and not np.all(np.isnan(real_profile)):
            real_mean = np.nanmean(real_profile)
            real_std = np.nanstd(real_profile)
            week_metrics['real_cv'] = real_std / real_mean if real_mean > 0 else np.nan
            week_metrics['real_peak'] = np.nanmax(real_profile)
            week_metrics['real_trough'] = np.nanmin(real_profile)
            week_metrics['real_pt_ratio'] = week_metrics['real_peak'] / week_metrics['real_trough'] if week_metrics['real_trough'] > 0 else np.nan

            # CV ratio
            week_metrics['cv_ratio'] = week_metrics['synth_cv'] / week_metrics['real_cv'] if week_metrics['real_cv'] > 0 else np.nan

            # Correlation
            valid_mask = ~(np.isnan(synth_profile) | np.isnan(real_profile))
            if np.sum(valid_mask) > 10:
                week_metrics['correlation'] = np.corrcoef(synth_profile[valid_mask], real_profile[valid_mask])[0, 1]
            else:
                week_metrics['correlation'] = np.nan
        else:
            week_metrics['real_cv'] = np.nan
            week_metrics['cv_ratio'] = np.nan
            week_metrics['correlation'] = np.nan

        metrics.append(week_metrics)

    return metrics


def create_degradation_plot(weekly_metrics, output_path, fuel="electricity", context_length=None):
    """
    Create a plot showing how daily pattern fidelity degrades over weeks.

    Shows CV ratio, correlation, and marks context_length boundary.
    """
    if not HAS_MATPLOTLIB:
        print("Skipping degradation plot (matplotlib not available)")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    weeks = [m['week'] for m in weekly_metrics]
    timesteps = [m['timestep_start'] for m in weekly_metrics]

    # Convert context_length to week number for marking
    context_week = None
    if context_length:
        context_week = context_length / 336  # 336 timesteps per week

    # Plot 1: CV Ratio over weeks
    ax1 = axes[0, 0]
    cv_ratios = [m.get('cv_ratio', np.nan) for m in weekly_metrics]
    ax1.plot(weeks, cv_ratios, 'b-o', linewidth=2, markersize=8, label='CV Ratio')
    ax1.axhline(y=0.7, color='green', linestyle='--', alpha=0.7, label='Good threshold (0.7)')
    ax1.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Warning threshold (0.5)')
    if context_week:
        ax1.axvline(x=context_week, color='red', linestyle=':', linewidth=2, label=f'Context length ({context_length} ts)')
    ax1.set_xlabel('Week', fontsize=11)
    ax1.set_ylabel('CV Ratio (Synth/Real)', fontsize=11)
    ax1.set_title('Daily Variation Preservation', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, max(1.2, np.nanmax(cv_ratios) * 1.1) if cv_ratios else 1.2)
    ax1.set_xlim(0.5, len(weeks) + 0.5)

    # Plot 2: Correlation over weeks
    ax2 = axes[0, 1]
    correlations = [m.get('correlation', np.nan) for m in weekly_metrics]
    ax2.plot(weeks, correlations, 'g-o', linewidth=2, markersize=8, label='Profile Correlation')
    ax2.axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='Good threshold (0.8)')
    ax2.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Warning threshold (0.5)')
    if context_week:
        ax2.axvline(x=context_week, color='red', linestyle=':', linewidth=2, label=f'Context length')
    ax2.set_xlabel('Week', fontsize=11)
    ax2.set_ylabel('Correlation', fontsize=11)
    ax2.set_title('Daily Profile Shape Similarity', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.2, 1.1)
    ax2.set_xlim(0.5, len(weeks) + 0.5)

    # Plot 3: Synth CV over weeks (absolute, not ratio)
    ax3 = axes[1, 0]
    synth_cvs = [m.get('synth_cv', np.nan) for m in weekly_metrics]
    real_cvs = [m.get('real_cv', np.nan) for m in weekly_metrics]
    ax3.plot(weeks, synth_cvs, 'b-o', linewidth=2, markersize=8, label='Synthetic CV')
    ax3.plot(weeks, real_cvs, 'g-s', linewidth=2, markersize=8, label='Real CV')
    if context_week:
        ax3.axvline(x=context_week, color='red', linestyle=':', linewidth=2, label=f'Context length')
    ax3.set_xlabel('Week', fontsize=11)
    ax3.set_ylabel('Coefficient of Variation', fontsize=11)
    ax3.set_title('Daily Variation (CV = std/mean)', fontsize=12, fontweight='bold')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0.5, len(weeks) + 0.5)

    # Plot 4: Peak-to-Trough ratio over weeks
    ax4 = axes[1, 1]
    synth_pt = [m.get('synth_pt_ratio', np.nan) for m in weekly_metrics]
    real_pt = [m.get('real_pt_ratio', np.nan) for m in weekly_metrics]
    ax4.plot(weeks, synth_pt, 'b-o', linewidth=2, markersize=8, label='Synthetic P/T')
    ax4.plot(weeks, real_pt, 'g-s', linewidth=2, markersize=8, label='Real P/T')
    if context_week:
        ax4.axvline(x=context_week, color='red', linestyle=':', linewidth=2, label=f'Context length')
    ax4.set_xlabel('Week', fontsize=11)
    ax4.set_ylabel('Peak / Trough Ratio', fontsize=11)
    ax4.set_title('Daily Peak-to-Trough Amplitude', fontsize=12, fontweight='bold')
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0.5, len(weeks) + 0.5)

    # Overall title
    fig.suptitle(f'Daily Pattern Degradation Analysis - {fuel.title()} (First 8 Weeks)',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved degradation plot to: {output_path}")
    plt.close()


# ============================================================================
# ORIGINAL AGGREGATION FUNCTIONS
# ============================================================================

def aggregate_batch_results(all_batch_results):
    """
    Aggregate results from all batches.

    Returns: {month: {half_hour: {'mean': float, 'std': float, 'values': list}}}
    """
    # Combine all values: {month: {half_hour: [all values]}}
    combined = defaultdict(lambda: defaultdict(list))

    for batch_result in all_batch_results:
        if batch_result is None:
            continue
        for month, hh_data in batch_result.items():
            for hh, values in hh_data.items():
                combined[month][hh].extend(values)

    # Calculate statistics
    aggregated = {}
    for month in range(1, 13):
        aggregated[month] = {}
        for hh in range(48):
            values = combined[month][hh]
            if values:
                aggregated[month][hh] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'values': values
                }
            else:
                aggregated[month][hh] = {'mean': np.nan, 'std': np.nan, 'values': []}

    return aggregated


def aggregate_real_results(all_household_results):
    """
    Aggregate results from all real households.

    Returns: {month: {half_hour: {'mean': float, 'std': float}}}
    """
    # Combine all values: {month: {half_hour: [household means]}}
    combined = defaultdict(lambda: defaultdict(list))

    for hh_result in all_household_results:
        if hh_result is None:
            continue
        for month, hh_data in hh_result.items():
            for hh, value in hh_data.items():
                combined[month][hh].append(value)

    # Calculate statistics
    aggregated = {}
    for month in range(1, 13):
        aggregated[month] = {}
        for hh in range(48):
            values = combined[month][hh]
            if values:
                aggregated[month][hh] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                }
            else:
                aggregated[month][hh] = {'mean': np.nan, 'std': np.nan}

    return aggregated


def create_heatmap_matrix(aggregated_data, stat='mean'):
    """
    Create a 12×48 matrix for heatmap visualization.

    Rows: months (1-12)
    Cols: half-hours (0-47)
    Values: mean or std
    """
    matrix = np.full((12, 48), np.nan)

    for month in range(1, 13):
        for hh in range(48):
            if month in aggregated_data and hh in aggregated_data[month]:
                matrix[month-1, hh] = aggregated_data[month][hh].get(stat, np.nan)

    return matrix


def create_summary_dataframe(synth_data, real_data=None, fuel="electricity"):
    """Create summary DataFrame with half-hourly statistics by month."""
    rows = []

    for month in range(1, 13):
        for hh in range(48):
            row = {
                'month': month,
                'half_hour': hh,
                'time': f"{hh//2:02d}:{(hh%2)*30:02d}",
                'fuel': fuel,
            }

            # Synthetic data
            if month in synth_data and hh in synth_data[month]:
                row['synth_mean'] = synth_data[month][hh].get('mean', np.nan)
                row['synth_std'] = synth_data[month][hh].get('std', np.nan)
            else:
                row['synth_mean'] = np.nan
                row['synth_std'] = np.nan

            # Real data
            if real_data and month in real_data and hh in real_data[month]:
                row['real_mean'] = real_data[month][hh].get('mean', np.nan)
                row['real_std'] = real_data[month][hh].get('std', np.nan)
            else:
                row['real_mean'] = np.nan
                row['real_std'] = np.nan

            # Difference
            if not np.isnan(row.get('synth_mean', np.nan)) and not np.isnan(row.get('real_mean', np.nan)):
                row['diff'] = row['synth_mean'] - row['real_mean']
                row['diff_pct'] = (row['diff'] / row['real_mean'] * 100) if row['real_mean'] != 0 else np.nan
            else:
                row['diff'] = np.nan
                row['diff_pct'] = np.nan

            rows.append(row)

    return pd.DataFrame(rows)


def create_plot(synth_data, real_data, output_path, fuel="electricity"):
    """
    Create visualization with heatmaps and key month line plots.

    Layout:
    - Row 1: 3 heatmaps (Real, Synthetic, Difference)
    - Row 2: 3 line plots (January, July, December)
    """
    if not HAS_MATPLOTLIB:
        print("Skipping plot generation (matplotlib not available)")
        return

    fig = plt.figure(figsize=(16, 10))

    # Create grid: 2 rows, 3 columns
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.25)

    # Create heatmap matrices
    synth_matrix = create_heatmap_matrix(synth_data, 'mean')
    real_matrix = create_heatmap_matrix(real_data, 'mean') if real_data else None

    # Calculate common color scale for real and synthetic
    if real_matrix is not None:
        all_values = np.concatenate([
            synth_matrix[~np.isnan(synth_matrix)],
            real_matrix[~np.isnan(real_matrix)]
        ])
    else:
        all_values = synth_matrix[~np.isnan(synth_matrix)]

    vmin = np.percentile(all_values, 2) if len(all_values) > 0 else 0
    vmax = np.percentile(all_values, 98) if len(all_values) > 0 else 1

    # X-axis labels (every 4 half-hours = every 2 hours)
    x_ticks = np.arange(0, 48, 4)
    x_labels = [f"{h//2:02d}:00" for h in x_ticks]

    # Y-axis labels (months)
    y_ticks = np.arange(12)
    y_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # -------------------------------------------------------------------------
    # Row 1: Heatmaps
    # -------------------------------------------------------------------------

    # Panel 1: Real data heatmap
    ax1 = fig.add_subplot(gs[0, 0])
    if real_matrix is not None:
        im1 = ax1.imshow(real_matrix, aspect='auto', cmap='YlOrRd',
                         vmin=vmin, vmax=vmax, origin='upper')
        ax1.set_title(f'Real Data - Daily {fuel.title()} Profile', fontsize=11, fontweight='bold')
        plt.colorbar(im1, ax=ax1, label='Wh')
    else:
        ax1.text(0.5, 0.5, 'No real data', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Real Data (not available)', fontsize=11)

    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
    ax1.set_yticks(y_ticks)
    ax1.set_yticklabels(y_labels, fontsize=9)
    ax1.set_xlabel('Time of Day', fontsize=10)
    ax1.set_ylabel('Month', fontsize=10)

    # Panel 2: Synthetic data heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(synth_matrix, aspect='auto', cmap='YlOrRd',
                     vmin=vmin, vmax=vmax, origin='upper')
    ax2.set_title(f'Synthetic Data - Daily {fuel.title()} Profile', fontsize=11, fontweight='bold')
    plt.colorbar(im2, ax=ax2, label='Wh')
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
    ax2.set_yticks(y_ticks)
    ax2.set_yticklabels(y_labels, fontsize=9)
    ax2.set_xlabel('Time of Day', fontsize=10)
    ax2.set_ylabel('Month', fontsize=10)

    # Panel 3: Difference heatmap
    ax3 = fig.add_subplot(gs[0, 2])
    if real_matrix is not None:
        diff_matrix = synth_matrix - real_matrix
        # Use diverging colormap centered at 0
        diff_abs_max = np.nanmax(np.abs(diff_matrix))
        if diff_abs_max > 0:
            im3 = ax3.imshow(diff_matrix, aspect='auto', cmap='RdBu_r',
                             vmin=-diff_abs_max, vmax=diff_abs_max, origin='upper')
        else:
            im3 = ax3.imshow(diff_matrix, aspect='auto', cmap='RdBu_r', origin='upper')
        ax3.set_title('Difference (Synth - Real)', fontsize=11, fontweight='bold')
        plt.colorbar(im3, ax=ax3, label='Wh')
    else:
        ax3.text(0.5, 0.5, 'No real data for comparison',
                 ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Difference (not available)', fontsize=11)

    ax3.set_xticks(x_ticks)
    ax3.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
    ax3.set_yticks(y_ticks)
    ax3.set_yticklabels(y_labels, fontsize=9)
    ax3.set_xlabel('Time of Day', fontsize=10)
    ax3.set_ylabel('Month', fontsize=10)

    # -------------------------------------------------------------------------
    # Row 2: Line plots for key months (January, July, December)
    # -------------------------------------------------------------------------

    key_months = [1, 7, 12]
    month_names = {1: 'January', 7: 'July', 12: 'December'}

    half_hours = np.arange(48)

    for idx, month in enumerate(key_months):
        ax = fig.add_subplot(gs[1, idx])

        # Extract data for this month
        synth_means = [synth_data[month][hh]['mean'] if hh in synth_data.get(month, {}) else np.nan
                       for hh in half_hours]
        synth_stds = [synth_data[month][hh]['std'] if hh in synth_data.get(month, {}) else np.nan
                      for hh in half_hours]

        synth_means = np.array(synth_means)
        synth_stds = np.array(synth_stds)

        # Plot synthetic
        ax.plot(half_hours, synth_means, color='blue', linewidth=2,
                label='Synthetic', marker=None)
        ax.fill_between(half_hours,
                        synth_means - synth_stds,
                        synth_means + synth_stds,
                        alpha=0.2, color='blue')

        # Plot real if available
        if real_data:
            real_means = [real_data[month][hh]['mean'] if hh in real_data.get(month, {}) else np.nan
                          for hh in half_hours]
            real_stds = [real_data[month][hh]['std'] if hh in real_data.get(month, {}) else np.nan
                         for hh in half_hours]

            real_means = np.array(real_means)
            real_stds = np.array(real_stds)

            ax.plot(half_hours, real_means, color='darkgreen', linewidth=2,
                    label='Real', marker=None)
            ax.fill_between(half_hours,
                            real_means - real_stds,
                            real_means + real_stds,
                            alpha=0.2, color='green')

        ax.set_title(f'{month_names[month]} Daily Profile', fontsize=11, fontweight='bold')
        ax.set_xlabel('Time of Day', fontsize=10)
        ax.set_ylabel(f'{fuel.title()} (Wh)', fontsize=10)
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)

        # X-axis formatting
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
        ax.set_xlim(0, 47)

        # Y-axis: ensure 0 is included
        ax.set_ylim(bottom=0)

    # Overall title
    fig.suptitle(f'Daily Temporal Pattern Analysis - {fuel.title()}',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved plot to: {output_path}")
    plt.close()


def calculate_pattern_metrics(synth_data, real_data=None):
    """
    Calculate summary metrics for pattern quality.

    Returns dict with metrics like peak-to-trough ratio, pattern correlation, etc.
    """
    metrics = {}

    for month in range(1, 13):
        month_metrics = {'month': month}

        # Synthetic metrics
        synth_profile = [synth_data[month][hh]['mean'] if hh in synth_data.get(month, {}) else np.nan
                         for hh in range(48)]
        synth_profile = np.array(synth_profile)

        if not np.all(np.isnan(synth_profile)):
            month_metrics['synth_peak'] = np.nanmax(synth_profile)
            month_metrics['synth_trough'] = np.nanmin(synth_profile)
            month_metrics['synth_peak_hour'] = np.nanargmax(synth_profile)
            month_metrics['synth_trough_hour'] = np.nanargmin(synth_profile)

            if month_metrics['synth_trough'] > 0:
                month_metrics['synth_peak_trough_ratio'] = month_metrics['synth_peak'] / month_metrics['synth_trough']
            else:
                month_metrics['synth_peak_trough_ratio'] = np.nan

            # Flatness metric: std / mean (coefficient of variation)
            month_metrics['synth_cv'] = np.nanstd(synth_profile) / np.nanmean(synth_profile) if np.nanmean(synth_profile) > 0 else np.nan

        # Real metrics and comparison
        if real_data:
            real_profile = [real_data[month][hh]['mean'] if hh in real_data.get(month, {}) else np.nan
                            for hh in range(48)]
            real_profile = np.array(real_profile)

            if not np.all(np.isnan(real_profile)):
                month_metrics['real_peak'] = np.nanmax(real_profile)
                month_metrics['real_trough'] = np.nanmin(real_profile)
                month_metrics['real_peak_hour'] = np.nanargmax(real_profile)
                month_metrics['real_trough_hour'] = np.nanargmin(real_profile)

                if month_metrics['real_trough'] > 0:
                    month_metrics['real_peak_trough_ratio'] = month_metrics['real_peak'] / month_metrics['real_trough']
                else:
                    month_metrics['real_peak_trough_ratio'] = np.nan

                month_metrics['real_cv'] = np.nanstd(real_profile) / np.nanmean(real_profile) if np.nanmean(real_profile) > 0 else np.nan

                # Correlation between profiles
                valid_mask = ~(np.isnan(synth_profile) | np.isnan(real_profile))
                if np.sum(valid_mask) > 10:
                    corr = np.corrcoef(synth_profile[valid_mask], real_profile[valid_mask])[0, 1]
                    month_metrics['profile_correlation'] = corr
                else:
                    month_metrics['profile_correlation'] = np.nan

        metrics[month] = month_metrics

    return metrics


def print_summary(metrics, fuel="electricity"):
    """Print a summary of pattern quality metrics."""
    print(f"\n{'='*70}")
    print(f"Daily Pattern Analysis Summary - {fuel.title()}")
    print(f"{'='*70}")

    has_real = 'real_cv' in metrics.get(1, {})

    print(f"\n{'Month':<8} {'Synth CV':<12} {'Real CV':<12} {'Correlation':<12} {'Synth P/T':<12} {'Real P/T':<12}")
    print("-" * 70)

    for month in range(1, 13):
        m = metrics.get(month, {})
        month_name = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][month]

        synth_cv = f"{m.get('synth_cv', np.nan):.3f}" if not np.isnan(m.get('synth_cv', np.nan)) else "N/A"
        real_cv = f"{m.get('real_cv', np.nan):.3f}" if has_real and not np.isnan(m.get('real_cv', np.nan)) else "N/A"
        corr = f"{m.get('profile_correlation', np.nan):.3f}" if has_real and not np.isnan(m.get('profile_correlation', np.nan)) else "N/A"
        synth_pt = f"{m.get('synth_peak_trough_ratio', np.nan):.2f}" if not np.isnan(m.get('synth_peak_trough_ratio', np.nan)) else "N/A"
        real_pt = f"{m.get('real_peak_trough_ratio', np.nan):.2f}" if has_real and not np.isnan(m.get('real_peak_trough_ratio', np.nan)) else "N/A"

        print(f"{month_name:<8} {synth_cv:<12} {real_cv:<12} {corr:<12} {synth_pt:<12} {real_pt:<12}")

    print("-" * 70)
    print("CV = Coefficient of Variation (std/mean) - higher means more daily variation")
    print("P/T = Peak to Trough ratio - higher means stronger daily pattern")
    print("Correlation = similarity of daily profile shape between synth and real")

    # Summary assessment
    if has_real:
        avg_synth_cv = np.nanmean([metrics[m].get('synth_cv', np.nan) for m in range(1, 13)])
        avg_real_cv = np.nanmean([metrics[m].get('real_cv', np.nan) for m in range(1, 13)])
        avg_corr = np.nanmean([metrics[m].get('profile_correlation', np.nan) for m in range(1, 13)])

        print(f"\nOverall Assessment:")
        print(f"  Average Synth CV: {avg_synth_cv:.3f}")
        print(f"  Average Real CV:  {avg_real_cv:.3f}")
        print(f"  CV Ratio (Synth/Real): {avg_synth_cv/avg_real_cv:.2f}" if avg_real_cv > 0 else "  CV Ratio: N/A")
        print(f"  Average Profile Correlation: {avg_corr:.3f}")

        if avg_synth_cv < avg_real_cv * 0.5:
            print(f"\n  WARNING: Synthetic daily patterns appear FLATTENED (CV ratio < 0.5)")
            print(f"           This indicates loss of temporal granularity.")
        elif avg_corr < 0.7:
            print(f"\n  WARNING: Low profile correlation (<0.7) suggests pattern shape mismatch.")


def analyze_fuel(fuel, batch_files, household_files=None):
    """Analyze daily patterns for a single fuel type."""
    print(f"\n{'='*70}")
    print(f"Analyzing {fuel.upper()}")
    print(f"{'='*70}")

    # Analyze synthetic data
    print(f"\nAnalyzing synthetic batches for {fuel}...")
    all_batch_results = []
    for i, batch_path in enumerate(batch_files):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Processing batch {i+1}/{len(batch_files)}: {batch_path.name}")
        result = analyze_batch_daily_patterns(batch_path, fuel=fuel)
        all_batch_results.append(result)

    synth_data = aggregate_batch_results(all_batch_results)
    print(f"  Aggregated results from {len([r for r in all_batch_results if r])} batches")

    # Analyze real data (if available)
    real_data = None
    if household_files and len(household_files) > 0:
        print(f"\nAnalyzing real households for {fuel}...")
        all_real_results = []
        for i, hh_path in enumerate(household_files):
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  Processing household {i+1}/{len(household_files)}: {hh_path.name}")
            result = analyze_real_household_daily_patterns(hh_path, fuel=fuel)
            all_real_results.append(result)

        real_data = aggregate_real_results(all_real_results)
        print(f"  Aggregated results from {len([r for r in all_real_results if r])} households")

    # Create summary DataFrame
    print("\nCreating summary...")
    df_summary = create_summary_dataframe(synth_data, real_data, fuel=fuel)
    csv_path = os.path.join(OUTPUT_DIR, f"daily_patterns_analysis_{fuel}.csv")
    df_summary.to_csv(csv_path, index=False)
    print(f"Saved summary to: {csv_path}")

    # Calculate and print metrics
    metrics = calculate_pattern_metrics(synth_data, real_data)
    print_summary(metrics, fuel=fuel)

    # Create visualization
    plot_path = os.path.join(OUTPUT_DIR, f"daily_patterns_plot_{fuel}.png")
    create_plot(synth_data, real_data, plot_path, fuel=fuel)

    # Save metrics
    metrics_df = pd.DataFrame([metrics[m] for m in range(1, 13)])
    metrics_path = os.path.join(OUTPUT_DIR, f"daily_patterns_metrics_{fuel}.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved metrics to: {metrics_path}")

    # =========================================================================
    # WEEKLY DEGRADATION ANALYSIS (First 8 weeks)
    # =========================================================================
    print(f"\n--- Weekly Degradation Analysis (First 8 Weeks) ---")

    # Analyze synthetic data by week
    print(f"Analyzing synthetic batches by week for {fuel}...")
    all_weekly_synth = []
    for i, batch_path in enumerate(batch_files):
        result = analyze_batch_daily_patterns_by_week(batch_path, fuel=fuel, num_weeks=8)
        all_weekly_synth.append(result)

    synth_weekly = aggregate_weekly_results(all_weekly_synth, num_weeks=8)

    # Analyze real data by week (if available)
    real_weekly = None
    if household_files and len(household_files) > 0:
        print(f"Analyzing real households by week for {fuel}...")
        all_weekly_real = []
        for hh_path in household_files:
            result = analyze_real_household_daily_patterns_by_week(hh_path, fuel=fuel, num_weeks=8)
            all_weekly_real.append(result)

        real_weekly = aggregate_weekly_results(all_weekly_real, num_weeks=8)

    # Calculate weekly metrics
    weekly_metrics = calculate_weekly_metrics(synth_weekly, real_weekly, num_weeks=8)

    # Print weekly summary
    print(f"\nWeekly Pattern Fidelity - {fuel.upper()}:")
    print(f"{'Week':<6} {'Timesteps':<12} {'CV Ratio':<12} {'Correlation':<12} {'Synth P/T':<12}")
    print("-" * 54)
    for m in weekly_metrics:
        cv_str = f"{m['cv_ratio']:.3f}" if not np.isnan(m.get('cv_ratio', np.nan)) else "N/A"
        corr_str = f"{m['correlation']:.3f}" if not np.isnan(m.get('correlation', np.nan)) else "N/A"
        pt_str = f"{m['synth_pt_ratio']:.2f}" if not np.isnan(m.get('synth_pt_ratio', np.nan)) else "N/A"
        print(f"{m['week']:<6} {m['timestep_start']:<12} {cv_str:<12} {corr_str:<12} {pt_str:<12}")

    # Mark context length
    if CONTEXT_LENGTH:
        context_week = CONTEXT_LENGTH / 336
        print(f"\n  Context length: {CONTEXT_LENGTH} timesteps = {context_week:.1f} weeks")

    # Create degradation plot
    degradation_plot_path = os.path.join(OUTPUT_DIR, f"daily_patterns_degradation_{fuel}.png")
    create_degradation_plot(weekly_metrics, degradation_plot_path, fuel=fuel, context_length=CONTEXT_LENGTH)

    # Save weekly metrics
    weekly_metrics_df = pd.DataFrame(weekly_metrics)
    weekly_metrics_path = os.path.join(OUTPUT_DIR, f"daily_patterns_weekly_metrics_{fuel}.csv")
    weekly_metrics_df.to_csv(weekly_metrics_path, index=False)
    print(f"Saved weekly metrics to: {weekly_metrics_path}")

    return metrics, weekly_metrics


def main(synthetic_dir=None, real_dir=None, output_dir=None, sample_size=None, context_length=None, train_puprns=None):
    """Main analysis function.

    Args:
        synthetic_dir: Path to generated_data directory (default: SYNTHETIC_DATA_DIR)
        real_dir: Path to tokenised_data directory (default: REAL_DATA_DIR)
        output_dir: Output directory for results (default: OUTPUT_DIR)
        sample_size: Number of real households to sample (default: REAL_DATA_SAMPLE_SIZE)
        context_length: Context length in timesteps for degradation marking (default: CONTEXT_LENGTH)

    Returns:
        Dictionary with analysis results
    """
    # Update globals if parameters provided (allows analyze_fuel to use them)
    global OUTPUT_DIR, CONTEXT_LENGTH

    synthetic_data_dir = synthetic_dir if synthetic_dir else SYNTHETIC_DATA_DIR
    real_data_dir = real_dir if real_dir else REAL_DATA_DIR
    real_sample_size = sample_size if sample_size is not None else REAL_DATA_SAMPLE_SIZE

    # Update globals for functions that depend on them
    if output_dir:
        OUTPUT_DIR = output_dir
    if context_length is not None:
        CONTEXT_LENGTH = context_length

    print("="*70)
    print("Daily Temporal Pattern Analysis")
    print("="*70)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find synthetic batch files
    batch_files = find_batch_files(synthetic_data_dir)
    if len(batch_files) == 0:
        print("ERROR: No batch files found!")
        sys.exit(1)

    # Find real household files (if available)
    household_files = None
    if real_data_dir and Path(real_data_dir).exists():
        household_files = find_real_household_files(real_data_dir, real_sample_size, puprn_filter=train_puprns)
    else:
        print("\nNo real data directory specified or found, skipping comparison")

    # Analyze each fuel
    all_metrics = {}
    all_weekly_metrics = {}
    for fuel in FUELS:
        monthly_metrics, weekly_metrics = analyze_fuel(fuel, batch_files, household_files)
        all_metrics[fuel] = monthly_metrics
        all_weekly_metrics[fuel] = weekly_metrics

    # Print combined summary table
    print("\n" + "="*70)
    print("DAILY PATTERN FIDELITY SUMMARY")
    print("="*70)

    # Build summary data for table
    summary_rows = []
    for fuel in FUELS:
        metrics = all_metrics[fuel]
        has_real = 'real_cv' in metrics.get(1, {})

        row = {'fuel': fuel.upper()}

        if has_real:
            # Calculate averages
            avg_synth_cv = np.nanmean([metrics[m].get('synth_cv', np.nan) for m in range(1, 13)])
            avg_real_cv = np.nanmean([metrics[m].get('real_cv', np.nan) for m in range(1, 13)])
            avg_corr = np.nanmean([metrics[m].get('profile_correlation', np.nan) for m in range(1, 13)])

            avg_synth_pt = np.nanmean([metrics[m].get('synth_peak_trough_ratio', np.nan) for m in range(1, 13)])
            avg_real_pt = np.nanmean([metrics[m].get('real_peak_trough_ratio', np.nan) for m in range(1, 13)])

            row['cv_ratio'] = avg_synth_cv / avg_real_cv if avg_real_cv > 0 else np.nan
            row['correlation'] = avg_corr
            row['synth_pt'] = avg_synth_pt
            row['real_pt'] = avg_real_pt
            row['pt_ratio'] = avg_synth_pt / avg_real_pt if avg_real_pt > 0 else np.nan
        else:
            row['cv_ratio'] = np.nan
            row['correlation'] = np.nan
            row['synth_pt'] = np.nan
            row['real_pt'] = np.nan
            row['pt_ratio'] = np.nan

        summary_rows.append(row)

    # Print table header
    print(f"\n{'Fuel':<12} {'CV Ratio':<12} {'Correlation':<12} {'P/T Synth':<12} {'P/T Real':<12} {'P/T Ratio':<12}")
    print("-" * 72)

    for row in summary_rows:
        cv_str = f"{row['cv_ratio']:.2f}" if not np.isnan(row['cv_ratio']) else "N/A"
        corr_str = f"{row['correlation']:.3f}" if not np.isnan(row['correlation']) else "N/A"
        spt_str = f"{row['synth_pt']:.2f}" if not np.isnan(row['synth_pt']) else "N/A"
        rpt_str = f"{row['real_pt']:.2f}" if not np.isnan(row['real_pt']) else "N/A"
        ptr_str = f"{row['pt_ratio']:.2f}" if not np.isnan(row['pt_ratio']) else "N/A"
        print(f"{row['fuel']:<12} {cv_str:<12} {corr_str:<12} {spt_str:<12} {rpt_str:<12} {ptr_str:<12}")

    print("-" * 72)

    # Interpretation guide
    print("\nINTERPRETATION GUIDE:")
    print("  CV Ratio    = Synth daily variation / Real daily variation")
    print("                > 0.7 = Good  |  0.5-0.7 = Moderate  |  < 0.5 = FLAT patterns")
    print("  Correlation = Shape similarity of daily profile (synth vs real)")
    print("                > 0.8 = Good  |  0.5-0.8 = Moderate  |  < 0.5 = Pattern mismatch")
    print("  P/T Ratio   = Peak-to-Trough ratio (higher = stronger daily swing)")
    print("                ~1.0 = Good   |  < 0.7 = Reduced peaks/troughs")

    # Assessment
    print("\nASSESSMENT:")
    for row in summary_rows:
        fuel = row['fuel']
        issues = []

        if not np.isnan(row['cv_ratio']) and row['cv_ratio'] < 0.5:
            issues.append("FLAT daily patterns (CV Ratio < 0.5)")
        if not np.isnan(row['correlation']) and row['correlation'] < 0.5:
            issues.append("Pattern shape mismatch (Correlation < 0.5)")
        if not np.isnan(row['pt_ratio']) and row['pt_ratio'] < 0.7:
            issues.append("Reduced peak/trough amplitude (P/T Ratio < 0.7)")

        if issues:
            row['assessment'] = f"WARNING: {'; '.join(issues)}"
            print(f"  {fuel}: WARNING - {'; '.join(issues)}")
        else:
            row['assessment'] = "OK - Daily patterns captured adequately"
            print(f"  {fuel}: OK - Daily patterns captured adequately")

    # Save summary table to CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUTPUT_DIR, "daily_patterns_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary table to: {summary_path}")

    # =========================================================================
    # DEGRADATION SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("WEEKLY DEGRADATION SUMMARY (First 8 Weeks)")
    print("="*70)

    if CONTEXT_LENGTH:
        context_week = CONTEXT_LENGTH / 336
        print(f"\nContext length: {CONTEXT_LENGTH} timesteps = Week {context_week:.1f}")

    for fuel in FUELS:
        weekly = all_weekly_metrics[fuel]
        print(f"\n{fuel.upper()}:")

        # Find when correlation drops below threshold
        corr_drop_week = None
        cv_drop_week = None
        for m in weekly:
            if corr_drop_week is None and m.get('correlation', 1.0) < 0.7:
                corr_drop_week = m['week']
            if cv_drop_week is None and m.get('cv_ratio', 1.0) < 0.5:
                cv_drop_week = m['week']

        # Week 1 vs Week 8 comparison
        w1 = weekly[0] if weekly else {}
        w8 = weekly[7] if len(weekly) > 7 else weekly[-1] if weekly else {}

        w1_corr = w1.get('correlation', np.nan)
        w8_corr = w8.get('correlation', np.nan)
        w1_cv = w1.get('cv_ratio', np.nan)
        w8_cv = w8.get('cv_ratio', np.nan)

        print(f"  Week 1 → Week 8:")
        print(f"    Correlation: {w1_corr:.3f} → {w8_corr:.3f}" if not (np.isnan(w1_corr) or np.isnan(w8_corr)) else "    Correlation: N/A")
        print(f"    CV Ratio:    {w1_cv:.3f} → {w8_cv:.3f}" if not (np.isnan(w1_cv) or np.isnan(w8_cv)) else "    CV Ratio: N/A")

        if corr_drop_week:
            print(f"  Correlation drops below 0.7 at: Week {corr_drop_week} (timestep {(corr_drop_week-1)*336})")
        if cv_drop_week:
            print(f"  CV Ratio drops below 0.5 at: Week {cv_drop_week} (timestep {(cv_drop_week-1)*336})")

        if not corr_drop_week and not cv_drop_week:
            print(f"  Pattern fidelity maintained through Week 8")

    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)


if __name__ == "__main__":
    main()
