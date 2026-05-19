#!/usr/bin/env python3
"""
Diagnostic: Household-Month Fidelity Analysis

Analyzes synthetic data at the household-month level to determine:
a) If the marginal distribution of household-month means matches the real data
b) If consumption is appropriately differentiated by conditioning variables

Each synthetic household-month is independently generated, so the correct unit
of comparison is the household-month mean — not a stitched-together annual mean
which would artificially reduce between-household variance.

Usage:
    python scripts/diagnose_full_year_fidelity.py

Outputs:
    - household_month_marginal_comparison.csv: Quintiles and mean comparison
    - household_month_conditioning_comparison.csv: Mean by conditioning variable category
    - household_month_marginal_comparison.png: Marginal distribution comparison (2x2)
    - household_month_electricity_conditioning.png: Electricity by conditioning variables (2x2)
    - household_month_gas_conditioning.png: Gas by conditioning variables (2x2)
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# Try to import matplotlib
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Note: matplotlib not available, will skip plot generation")

# ============================================================================
# UPDATE THESE PATHS FOR YOUR TRE ENVIRONMENT
# ============================================================================

# Path to directory containing synthetic batch files (batch_0001_month_01.pkl, etc.)
SYNTHETIC_DATA_DIR = "experiments/separate_models_inference_test/run_005/generated_data"

# Path to directory containing real tokenised household files
REAL_DATA_DIR = "experiments/separate_models_train_test/run_008/data/tokenised_data"

# Number of real households to sample (set to None for all)
REAL_DATA_SAMPLE_SIZE = 100

# Output directory for results
OUTPUT_DIR = "experiments/separate_models_inference_test/run_005/diagnosis"

# Path to metadata catalog for real household conditioning variables
# Usually in run_path/data/shards/metadata_catalog_filtered.pkl
REAL_METADATA_PATH = "experiments/separate_models_train_test/run_008/data/shards/metadata_catalog_filtered.pkl"

# Conditioning variables to analyze (must exist in batch files)
CONDITIONING_VARIABLES = ["building_type", "age_built", "num_rooms", "num_occs"]

# Mapping from conditioning variable names to metadata catalog columns
# Note: building_type and age_built use descriptive versions (strings) to match synthetic data
METADATA_COLUMN_MAP = {
    'building_type': 'building_type_descriptive',
    'age_built': 'age_built_descriptive',
    'num_rooms': 'B5',
    'num_occs': 'C1_new',
}

# ============================================================================


def find_batch_files(data_dir: str) -> List[Path]:
    """Find all batch pickle files in the directory.

    Supports both old format (batch_NNNN.pkl) and new format (batch_NNNN_month_MM.pkl).

    Args:
        data_dir: Directory containing batch files

    Returns:
        List of batch file paths
    """
    data_path = Path(data_dir)

    # Check for new format first (month-specific files)
    new_format_files = sorted(data_path.glob("batch_*_month_*.pkl"))

    if new_format_files:
        print(f"Detected NEW format: {len(new_format_files)} month files (all months)")
        return new_format_files

    # Fall back to old format
    old_format_files = sorted(data_path.glob("batch_*.pkl"))
    # Exclude conditioning_summary.pkl
    old_format_files = [f for f in old_format_files if 'conditioning' not in f.name]
    print(f"Detected OLD format: {len(old_format_files)} batch files")
    return old_format_files


def find_real_household_files(data_dir: str, sample_size: Optional[int] = None,
                              puprn_filter: Optional[List[str]] = None) -> List[Path]:
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


def load_real_conditioning(metadata_path: str) -> Optional[pd.DataFrame]:
    """
    Load conditioning variables for real households from metadata catalog.

    Args:
        metadata_path: Path to metadata_catalog_filtered.pkl

    Returns:
        DataFrame with puprn and conditioning variables, or None if not found
    """
    if not os.path.exists(metadata_path):
        print(f"  Metadata file not found: {metadata_path}")
        return None

    try:
        df = pd.read_pickle(metadata_path)
        print(f"  Loaded metadata for {len(df)} households")

        # Extract and rename conditioning columns
        result_cols = ['puprn']
        for var_name, col_name in METADATA_COLUMN_MAP.items():
            if col_name in df.columns:
                df[var_name] = df[col_name]
                result_cols.append(var_name)
            else:
                print(f"  Warning: Column {col_name} not found for {var_name}")

        return df[result_cols].copy()

    except Exception as e:
        print(f"  Error loading metadata: {e}")
        return None


def extract_household_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Extract electricity and gas column names for each household."""
    elec_cols = [c for c in df.columns if '_elec_net_Wh' in c or '_elec_Wh' in c]
    gas_cols = [c for c in df.columns if '_gas_Wh' in c and '_elec' not in c]
    return elec_cols, gas_cols


def load_household_month_synthetic_data(batch_files: List[Path]) -> Dict[str, pd.DataFrame]:
    """
    Load synthetic data at the household-month level.

    Each batch file corresponds to one month for a set of households.
    Emits one record per household per month (no cross-month aggregation),
    because each synthetic household-month is independently generated.

    Supports both formats:
    - New format: batch_NNNN_month_MM.pkl files for each month
    - Old format: batch_NNNN.pkl (treated as single period, month=None)

    Returns:
        Dictionary with keys 'electricity', 'gas', 'conditioning'
        - electricity: DataFrame with columns [household_id, month, mean, batch]
        - gas: DataFrame with columns [household_id, month, mean, batch]
        - conditioning: DataFrame with conditioning variables per household
    """
    all_elec = []
    all_gas = []
    all_conditioning = []

    # Track unique households to avoid duplicate conditioning entries
    seen_households = set()

    print("\nLoading synthetic data at household-month level...")

    # Group files by batch for new format
    batch_files_by_batch = defaultdict(list)
    for f in batch_files:
        if '_month_' in f.name:
            batch_name = f.stem.split('_month_')[0]
            batch_files_by_batch[batch_name].append(f)
        else:
            batch_files_by_batch[f.stem].append(f)

    print(f"  Found {len(batch_files_by_batch)} unique batches")

    for batch_name, files in batch_files_by_batch.items():
        batch_conditioning = {}  # household_id -> conditioning dict

        for batch_path in files:
            df = pd.read_pickle(batch_path)

            # Extract month from filename (new format) or use None (old format)
            is_new_format = '_month_' in batch_path.name
            month = int(batch_path.stem.split('_month_')[1]) if is_new_format else None

            elec_cols, gas_cols = extract_household_columns(df)

            if len(elec_cols) == 0:
                continue

            # Extract conditioning variables (only need to do once per household)
            cond_cols = [c for c in CONDITIONING_VARIABLES if c in df.columns]

            for i, elec_col in enumerate(elec_cols):
                hh_id = elec_col.split('_')[0]
                full_hh_id = f"{batch_name}_{hh_id}"

                # Electricity: compute mean for this household-month
                elec_values = df[elec_col].dropna().values.astype(float)
                if len(elec_values) > 0:
                    all_elec.append({
                        'household_id': full_hh_id,
                        'month': month,
                        'mean': np.mean(elec_values),
                        'batch': batch_name,
                    })

                # Gas: find corresponding column
                gas_col = None
                for gc in gas_cols:
                    if gc.startswith(hh_id):
                        gas_col = gc
                        break

                if gas_col:
                    gas_values = df[gas_col].dropna().values.astype(float)
                    if len(gas_values) > 0:
                        all_gas.append({
                            'household_id': full_hh_id,
                            'month': month,
                            'mean': np.mean(gas_values),
                            'batch': batch_name,
                        })

                # Conditioning variables - only record once per household
                if full_hh_id not in batch_conditioning and cond_cols:
                    cond_row = {'household_id': full_hh_id, 'batch': batch_name}
                    for col in cond_cols:
                        cond_values = df[col].dropna()
                        if len(cond_values) > 0:
                            cond_row[col] = cond_values.iloc[0]
                    batch_conditioning[full_hh_id] = cond_row

        # Add conditioning for this batch
        for hh_id, cond_row in batch_conditioning.items():
            if hh_id not in seen_households:
                all_conditioning.append(cond_row)
                seen_households.add(hh_id)

        # Progress indicator
        n_months = len(files)
        n_hh_month = sum(1 for r in all_elec if r['batch'] == batch_name)
        print(f"  {batch_name}: {n_months} months, {n_hh_month} household-month records")

    # Convert to DataFrames
    elec_df = pd.DataFrame(all_elec)
    gas_df = pd.DataFrame(all_gas)
    cond_df = pd.DataFrame(all_conditioning)

    n_hh = len(elec_df['household_id'].unique()) if len(elec_df) > 0 else 0
    print(f"\nLoaded synthetic data:")
    print(f"  Electricity: {len(elec_df)} household-month records ({n_hh} households)")
    print(f"  Gas: {len(gas_df)} household-month records")
    print(f"  Conditioning variables: {list(cond_df.columns)}")

    return {
        'electricity': elec_df,
        'gas': gas_df,
        'conditioning': cond_df,
    }


def load_household_month_real_data(household_files: List[Path]) -> Dict[str, pd.DataFrame]:
    """
    Load real data at the household-month level.

    Groups each household's data by month and computes monthly means,
    producing one record per household per month.

    Returns:
        Dictionary with keys 'electricity', 'gas'
    """
    all_elec = []
    all_gas = []

    print("\nLoading real data at household-month level...")

    for i, hh_path in enumerate(household_files):
        if (i + 1) % 20 == 0:
            print(f"  Processing household {i+1}/{len(household_files)}")

        try:
            df = pd.read_pickle(hh_path)
        except Exception as e:
            print(f"  Warning: Could not load {hh_path}: {e}")
            continue

        hh_id = hh_path.stem.replace('_tokenised', '')

        # Determine month for each row
        if 'month' in df.columns:
            month_series = df['month']
        elif hasattr(df.index, 'month'):
            month_series = df.index.month
        else:
            # No month info available - treat as single period
            month_series = pd.Series(0, index=df.index)

        # Electricity
        elec_col = None
        for col in ['Clean_elec_net_Wh', 'elec_Wh', 'elec_value']:
            if col in df.columns:
                elec_col = col
                break

        if elec_col:
            for month, month_df in df.groupby(month_series):
                elec_values = month_df[elec_col].dropna().values.astype(float)
                if len(elec_values) > 0:
                    all_elec.append({
                        'household_id': hh_id,
                        'month': int(month),
                        'mean': np.mean(elec_values),
                    })

        # Gas
        gas_col = None
        for col in ['Clean_gas_Wh', 'gas_Wh', 'gas_value']:
            if col in df.columns:
                gas_col = col
                break

        if gas_col:
            for month, month_df in df.groupby(month_series):
                gas_values = month_df[gas_col].dropna().values.astype(float)
                if len(gas_values) > 0:
                    all_gas.append({
                        'household_id': hh_id,
                        'month': int(month),
                        'mean': np.mean(gas_values),
                    })

    elec_df = pd.DataFrame(all_elec)
    gas_df = pd.DataFrame(all_gas)

    n_hh = len(elec_df['household_id'].unique()) if len(elec_df) > 0 else 0
    print(f"\nLoaded real data:")
    print(f"  Electricity: {len(elec_df)} household-month records ({n_hh} households)")
    print(f"  Gas: {len(gas_df)} household-month records")

    return {
        'electricity': elec_df,
        'gas': gas_df,
    }


def calculate_distribution_stats(values: np.ndarray) -> Dict[str, float]:
    """Calculate quintiles and mean for a distribution."""
    return {
        'mean': np.mean(values),
        'std': np.std(values),
        'q20': np.percentile(values, 20),
        'q40': np.percentile(values, 40),
        'q60': np.percentile(values, 60),
        'q80': np.percentile(values, 80),
        'min': np.min(values),
        'max': np.max(values),
        'n': len(values),
    }


def compare_marginal_distributions(
    synth_data: Dict[str, pd.DataFrame],
    real_data: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Compare marginal distributions (quintiles and mean) between synthetic and real.

    Uses household-month means for comparison (one value per household per month).
    """
    print("\n" + "=" * 70)
    print("MARGINAL DISTRIBUTION COMPARISON (Household-Month)")
    print("=" * 70)

    results = []

    for fuel in ['electricity', 'gas']:
        print(f"\n--- {fuel.upper()} ---")

        # Get household-month means
        synth_means = synth_data[fuel]['mean'].values if len(synth_data[fuel]) > 0 else np.array([])
        real_means = real_data[fuel]['mean'].values if len(real_data[fuel]) > 0 else np.array([])

        if len(synth_means) == 0 or len(real_means) == 0:
            print(f"  Insufficient data for {fuel}")
            continue

        synth_stats = calculate_distribution_stats(synth_means)
        real_stats = calculate_distribution_stats(real_means)

        # Calculate ratios
        for stat in ['mean', 'q20', 'q40', 'q60', 'q80']:
            synth_val = synth_stats[stat]
            real_val = real_stats[stat]
            ratio = synth_val / real_val if real_val != 0 else np.nan
            diff_pct = 100 * (synth_val - real_val) / real_val if real_val != 0 else np.nan

            results.append({
                'fuel': fuel,
                'statistic': stat,
                'synthetic': synth_val,
                'real': real_val,
                'ratio': ratio,
                'diff_pct': diff_pct,
            })

            print(f"  {stat:>6}: Synth={synth_val:>10.1f}, Real={real_val:>10.1f}, "
                  f"Ratio={ratio:.3f}, Diff={diff_pct:+.1f}%")

        # Also add std for reference
        results.append({
            'fuel': fuel,
            'statistic': 'std',
            'synthetic': synth_stats['std'],
            'real': real_stats['std'],
            'ratio': synth_stats['std'] / real_stats['std'] if real_stats['std'] != 0 else np.nan,
            'diff_pct': np.nan,
        })

        print(f"\n  Std: Synth={synth_stats['std']:.1f}, Real={real_stats['std']:.1f}")
        print(f"  N: Synth={synth_stats['n']}, Real={real_stats['n']}")

    return pd.DataFrame(results)


def compare_conditioning_means(
    synth_data: Dict[str, pd.DataFrame],
    real_data: Dict[str, pd.DataFrame],
    real_conditioning: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Compare mean consumption by conditioning variable category.

    For synthetic data: uses conditioning variables from batch files.
    For real data: uses conditioning from metadata catalog.
    """
    print("\n" + "=" * 70)
    print("CONDITIONING VARIABLE ANALYSIS (Household-Month)")
    print("=" * 70)

    results = []

    synth_cond_df = synth_data['conditioning']

    if len(synth_cond_df) == 0:
        print("No conditioning data available in synthetic batches")
        return pd.DataFrame(results)

    # Get available conditioning variables
    cond_vars = [c for c in CONDITIONING_VARIABLES if c in synth_cond_df.columns]

    if not cond_vars:
        print(f"No conditioning variables found. Available columns: {list(synth_cond_df.columns)}")
        return pd.DataFrame(results)

    print(f"\nAnalyzing conditioning variables: {cond_vars}")

    # Prepare real data with conditioning if available
    real_merged = {}
    if real_conditioning is not None and len(real_conditioning) > 0:
        for fuel in ['electricity', 'gas']:
            fuel_df = real_data[fuel].copy()
            if len(fuel_df) > 0:
                # Match by household_id (PUPRN)
                merged = fuel_df.merge(real_conditioning, left_on='household_id', right_on='puprn', how='left')
                real_merged[fuel] = merged
                matched = merged['puprn'].notna().sum()
                print(f"  Real {fuel}: {matched}/{len(fuel_df)} household-month records matched to conditioning")

    for fuel in ['electricity', 'gas']:
        print(f"\n--- {fuel.upper()} ---")

        # SYNTHETIC DATA
        synth_fuel_df = synth_data[fuel].copy()
        if len(synth_fuel_df) == 0:
            continue

        synth_merged = synth_fuel_df.merge(synth_cond_df, on='household_id', how='left')

        for var in cond_vars:
            if var not in synth_merged.columns:
                continue

            print(f"\n  {var}:")

            # Get all unique categories from both synth and real
            all_categories = set(synth_merged[var].dropna().unique())
            if fuel in real_merged and var in real_merged[fuel].columns:
                all_categories.update(real_merged[fuel][var].dropna().unique())

            # SYNTHETIC: Group by category
            synth_grouped = synth_merged.groupby(var)['mean'].agg(['mean', 'std', 'count']).reset_index()
            synth_grouped.columns = [var, 'mean_consumption', 'std_consumption', 'n_household_months']
            synth_dict = {row[var]: row for _, row in synth_grouped.iterrows()}

            # REAL: Group by category (if available)
            real_dict = {}
            if fuel in real_merged and var in real_merged[fuel].columns:
                real_fuel_df = real_merged[fuel].dropna(subset=[var])
                if len(real_fuel_df) > 0:
                    real_grouped = real_fuel_df.groupby(var)['mean'].agg(['mean', 'std', 'count']).reset_index()
                    real_grouped.columns = [var, 'mean_consumption', 'std_consumption', 'n_household_months']
                    real_dict = {row[var]: row for _, row in real_grouped.iterrows()}

            # Print comparison table
            print(f"    {'Category':<15} {'Synth Mean':>12} {'Real Mean':>12} {'Ratio':>8} {'Synth N':>8} {'Real N':>8}")
            print(f"    {'-'*15} {'-'*12} {'-'*12} {'-'*8} {'-'*8} {'-'*8}")

            for category in sorted(all_categories, key=lambda x: (x is None, str(x))):
                synth_row = synth_dict.get(category)
                real_row = real_dict.get(category)

                synth_mean = synth_row['mean_consumption'] if synth_row is not None else np.nan
                synth_n = int(synth_row['n_household_months']) if synth_row is not None else 0
                real_mean = real_row['mean_consumption'] if real_row is not None else np.nan
                real_n = int(real_row['n_household_months']) if real_row is not None else 0

                ratio = synth_mean / real_mean if (real_mean and not np.isnan(real_mean) and real_mean != 0) else np.nan

                # Store results
                if synth_row is not None:
                    results.append({
                        'fuel': fuel,
                        'variable': var,
                        'category': category,
                        'source': 'synthetic',
                        'mean_consumption': synth_mean,
                        'std_consumption': synth_row['std_consumption'],
                        'n_household_months': synth_n,
                    })

                if real_row is not None:
                    results.append({
                        'fuel': fuel,
                        'variable': var,
                        'category': category,
                        'source': 'real',
                        'mean_consumption': real_mean,
                        'std_consumption': real_row['std_consumption'],
                        'n_household_months': real_n,
                    })

                # Print row
                synth_str = f"{synth_mean:.1f}" if not np.isnan(synth_mean) else "N/A"
                real_str = f"{real_mean:.1f}" if not np.isnan(real_mean) else "N/A"
                ratio_str = f"{ratio:.3f}" if not np.isnan(ratio) else "N/A"

                print(f"    {str(category):<15} {synth_str:>12} {real_str:>12} {ratio_str:>8} {synth_n:>8} {real_n:>8}")

    return pd.DataFrame(results)


def create_summary_plots(
    marginal_df: pd.DataFrame,
    conditioning_df: pd.DataFrame,
    output_dir: str,
):
    """
    Create three separate figures for Household-Month fidelity analysis:
    1. Marginal distribution comparison
    2. Electricity conditioning variable comparison
    3. Gas conditioning variable comparison
    """
    if not HAS_MATPLOTLIB:
        print("Skipping plot generation (matplotlib not available)")
        return

    # Determine conditioning variables with data
    cond_vars_with_data = []
    if len(conditioning_df) > 0:
        cond_vars_with_data = conditioning_df['variable'].unique().tolist()

    # =========================================================================
    # FIGURE 1: Marginal Distribution Comparison
    # =========================================================================
    fig1, axes1 = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 0,0: Electricity marginal distribution
    ax = axes1[0, 0]
    elec_data = marginal_df[marginal_df['fuel'] == 'electricity']
    if len(elec_data) > 0:
        stats = ['q20', 'q40', 'mean', 'q60', 'q80']
        plot_data = elec_data[elec_data['statistic'].isin(stats)]
        x = np.arange(len(stats))
        width = 0.35

        synth_vals = [plot_data[plot_data['statistic'] == s]['synthetic'].values[0]
                      for s in stats if s in plot_data['statistic'].values]
        real_vals = [plot_data[plot_data['statistic'] == s]['real'].values[0]
                     for s in stats if s in plot_data['statistic'].values]

        if synth_vals and real_vals:
            bars1 = ax.bar(x - width/2, synth_vals, width, label='Synthetic', color='coral')
            bars2 = ax.bar(x + width/2, real_vals, width, label='Real', color='steelblue')
            ax.set_xticks(x)
            ax.set_xticklabels(stats)
            ax.set_ylabel('Wh')
            ax.set_title('ELECTRICITY: Household-Month Means')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')

            for bar, val in zip(bars1, synth_vals):
                ax.annotate(f'{val:.0f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           ha='center', va='bottom', fontsize=8, color='coral')
            for bar, val in zip(bars2, real_vals):
                ax.annotate(f'{val:.0f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           ha='center', va='bottom', fontsize=8, color='steelblue')

    # Plot 0,1: Gas marginal distribution
    ax = axes1[0, 1]
    gas_data = marginal_df[marginal_df['fuel'] == 'gas']
    if len(gas_data) > 0:
        stats = ['q20', 'q40', 'mean', 'q60', 'q80']
        plot_data = gas_data[gas_data['statistic'].isin(stats)]
        x = np.arange(len(stats))

        synth_vals = [plot_data[plot_data['statistic'] == s]['synthetic'].values[0]
                      for s in stats if s in plot_data['statistic'].values]
        real_vals = [plot_data[plot_data['statistic'] == s]['real'].values[0]
                     for s in stats if s in plot_data['statistic'].values]

        if synth_vals and real_vals:
            bars1 = ax.bar(x - width/2, synth_vals, width, label='Synthetic', color='coral')
            bars2 = ax.bar(x + width/2, real_vals, width, label='Real', color='steelblue')
            ax.set_xticks(x)
            ax.set_xticklabels(stats)
            ax.set_ylabel('Wh')
            ax.set_title('GAS: Household-Month Means')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')

            for bar, val in zip(bars1, synth_vals):
                ax.annotate(f'{val:.0f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           ha='center', va='bottom', fontsize=8, color='coral')
            for bar, val in zip(bars2, real_vals):
                ax.annotate(f'{val:.0f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           ha='center', va='bottom', fontsize=8, color='steelblue')

    # Plot 1,0: Electricity ratio
    ax = axes1[1, 0]
    if len(marginal_df) > 0:
        stats = ['q20', 'q40', 'mean', 'q60', 'q80']
        elec_ratios = []
        for s in stats:
            elec_row = marginal_df[(marginal_df['fuel'] == 'electricity') &
                                   (marginal_df['statistic'] == s)]
            if len(elec_row) > 0:
                elec_ratios.append(elec_row['ratio'].values[0])
            else:
                elec_ratios.append(np.nan)

        x = np.arange(len(stats))
        bars = ax.bar(x, elec_ratios, color='coral', alpha=0.7)
        ax.axhline(y=1.0, color='black', linestyle='--', linewidth=2, label='Perfect match (1.0)')
        ax.set_xticks(x)
        ax.set_xticklabels(stats)
        ax.set_ylabel('Synthetic / Real Ratio')
        ax.set_title('ELECTRICITY: Distribution Ratio (1.0 = perfect match)')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        if elec_ratios:
            valid_ratios = [r for r in elec_ratios if not np.isnan(r)]
            if valid_ratios:
                ax.set_ylim(0, max(1.2, max(valid_ratios) * 1.1))

        for bar, val in zip(bars, elec_ratios):
            if not np.isnan(val):
                ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Plot 1,1: Gas ratio
    ax = axes1[1, 1]
    if len(marginal_df) > 0:
        stats = ['q20', 'q40', 'mean', 'q60', 'q80']
        gas_ratios = []
        for s in stats:
            gas_row = marginal_df[(marginal_df['fuel'] == 'gas') &
                                  (marginal_df['statistic'] == s)]
            if len(gas_row) > 0:
                gas_ratios.append(gas_row['ratio'].values[0])
            else:
                gas_ratios.append(np.nan)

        x = np.arange(len(stats))
        bars = ax.bar(x, gas_ratios, color='steelblue', alpha=0.7)
        ax.axhline(y=1.0, color='black', linestyle='--', linewidth=2, label='Perfect match (1.0)')
        ax.set_xticks(x)
        ax.set_xticklabels(stats)
        ax.set_ylabel('Synthetic / Real Ratio')
        ax.set_title('GAS: Distribution Ratio (1.0 = perfect match)')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        if gas_ratios:
            valid_ratios = [r for r in gas_ratios if not np.isnan(r)]
            if valid_ratios:
                ax.set_ylim(0, max(1.2, max(valid_ratios) * 1.1))

        for bar, val in zip(bars, gas_ratios):
            if not np.isnan(val):
                ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           ha='center', va='bottom', fontsize=9, fontweight='bold')

    fig1.suptitle('Household-Month Marginal Distribution Comparison', fontsize=14, fontweight='bold')
    fig1.tight_layout()
    path1 = os.path.join(output_dir, "household_month_marginal_comparison.png")
    fig1.savefig(path1, dpi=150, bbox_inches='tight')
    print(f"\nMarginal plot saved to: {path1}")
    plt.close(fig1)

    # =========================================================================
    # FIGURE 2: Electricity Conditioning Variables
    # =========================================================================
    if len(cond_vars_with_data) > 0:
        n_vars = len(cond_vars_with_data)
        n_cols = 2
        n_rows = (n_vars + 1) // 2

        fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
        if n_rows == 1:
            axes2 = axes2.reshape(1, -1)

        for i, var in enumerate(cond_vars_with_data):
            row_idx = i // 2
            col_idx = i % 2
            ax = axes2[row_idx, col_idx]

            var_data = conditioning_df[(conditioning_df['variable'] == var) &
                                       (conditioning_df['fuel'] == 'electricity')]

            if len(var_data) == 0:
                ax.text(0.5, 0.5, f'No data for {var}',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Mean by {var}')
                continue

            categories = sorted(var_data['category'].unique(), key=lambda x: str(x))

            synth_vals = []
            real_vals = []
            for cat in categories:
                synth_row = var_data[(var_data['category'] == cat) & (var_data['source'] == 'synthetic')]
                real_row = var_data[(var_data['category'] == cat) & (var_data['source'] == 'real')]
                synth_vals.append(synth_row['mean_consumption'].values[0] if len(synth_row) > 0 else 0)
                real_vals.append(real_row['mean_consumption'].values[0] if len(real_row) > 0 else 0)

            x = np.arange(len(categories))
            width = 0.35

            bars1 = ax.bar(x - width/2, synth_vals, width, label='Synthetic', color='coral')
            bars2 = ax.bar(x + width/2, real_vals, width, label='Real', color='steelblue')

            ax.set_xticks(x)
            ax.set_xticklabels([str(c)[:15] for c in categories], rotation=45, ha='right')
            ax.set_ylabel('Mean Consumption (Wh)')
            ax.set_title(f'Mean by {var}')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3, axis='y')

            for bar, val in zip(bars1, synth_vals):
                if val > 0:
                    ax.annotate(f'{val:.0f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                               ha='center', va='bottom', fontsize=7, color='coral')
            for bar, val in zip(bars2, real_vals):
                if val > 0:
                    ax.annotate(f'{val:.0f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                               ha='center', va='bottom', fontsize=7, color='steelblue')

        # Hide unused subplots
        for i in range(len(cond_vars_with_data), n_rows * n_cols):
            row_idx = i // 2
            col_idx = i % 2
            axes2[row_idx, col_idx].set_visible(False)

        fig2.suptitle('Household-Month ELECTRICITY: Conditioning Variable Comparison', fontsize=14, fontweight='bold')
        fig2.tight_layout()
        path2 = os.path.join(output_dir, "household_month_electricity_conditioning.png")
        fig2.savefig(path2, dpi=150, bbox_inches='tight')
        print(f"Electricity conditioning plot saved to: {path2}")
        plt.close(fig2)

    # =========================================================================
    # FIGURE 3: Gas Conditioning Variables
    # =========================================================================
    if len(cond_vars_with_data) > 0:
        fig3, axes3 = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
        if n_rows == 1:
            axes3 = axes3.reshape(1, -1)

        for i, var in enumerate(cond_vars_with_data):
            row_idx = i // 2
            col_idx = i % 2
            ax = axes3[row_idx, col_idx]

            var_data = conditioning_df[(conditioning_df['variable'] == var) &
                                       (conditioning_df['fuel'] == 'gas')]

            if len(var_data) == 0:
                ax.text(0.5, 0.5, f'No data for {var}',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Mean by {var}')
                continue

            categories = sorted(var_data['category'].unique(), key=lambda x: str(x))

            synth_vals = []
            real_vals = []
            for cat in categories:
                synth_row = var_data[(var_data['category'] == cat) & (var_data['source'] == 'synthetic')]
                real_row = var_data[(var_data['category'] == cat) & (var_data['source'] == 'real')]
                synth_vals.append(synth_row['mean_consumption'].values[0] if len(synth_row) > 0 else 0)
                real_vals.append(real_row['mean_consumption'].values[0] if len(real_row) > 0 else 0)

            x = np.arange(len(categories))
            width = 0.35

            bars1 = ax.bar(x - width/2, synth_vals, width, label='Synthetic', color='coral')
            bars2 = ax.bar(x + width/2, real_vals, width, label='Real', color='steelblue')

            ax.set_xticks(x)
            ax.set_xticklabels([str(c)[:15] for c in categories], rotation=45, ha='right')
            ax.set_ylabel('Mean Consumption (Wh)')
            ax.set_title(f'Mean by {var}')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3, axis='y')

            for bar, val in zip(bars1, synth_vals):
                if val > 0:
                    ax.annotate(f'{val:.0f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                               ha='center', va='bottom', fontsize=7, color='coral')
            for bar, val in zip(bars2, real_vals):
                if val > 0:
                    ax.annotate(f'{val:.0f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                               ha='center', va='bottom', fontsize=7, color='steelblue')

        # Hide unused subplots
        for i in range(len(cond_vars_with_data), n_rows * n_cols):
            row_idx = i // 2
            col_idx = i % 2
            axes3[row_idx, col_idx].set_visible(False)

        fig3.suptitle('Household-Month GAS: Conditioning Variable Comparison', fontsize=14, fontweight='bold')
        fig3.tight_layout()
        path3 = os.path.join(output_dir, "household_month_gas_conditioning.png")
        fig3.savefig(path3, dpi=150, bbox_inches='tight')
        print(f"Gas conditioning plot saved to: {path3}")
        plt.close(fig3)


def print_summary(marginal_df: pd.DataFrame):
    """Print a summary assessment of household-month fidelity."""
    print("\n" + "=" * 70)
    print("HOUSEHOLD-MONTH FIDELITY ASSESSMENT")
    print("=" * 70)

    for fuel in ['electricity', 'gas']:
        fuel_data = marginal_df[marginal_df['fuel'] == fuel]
        if len(fuel_data) == 0:
            continue

        mean_row = fuel_data[fuel_data['statistic'] == 'mean']
        if len(mean_row) > 0:
            ratio = mean_row['ratio'].values[0]
            diff_pct = mean_row['diff_pct'].values[0]

            print(f"\n{fuel.upper()}:")
            print(f"  Mean ratio (synth/real): {ratio:.3f}")
            print(f"  Mean difference: {diff_pct:+.1f}%")

            if 0.9 <= ratio <= 1.1:
                print(f"  Assessment: GOOD - within 10% of real data")
            elif 0.8 <= ratio <= 1.2:
                print(f"  Assessment: ACCEPTABLE - within 20% of real data")
            else:
                print(f"  Assessment: POOR - more than 20% deviation")

            # Check quintile alignment
            quintile_ratios = []
            for q in ['q20', 'q40', 'q60', 'q80']:
                q_row = fuel_data[fuel_data['statistic'] == q]
                if len(q_row) > 0:
                    quintile_ratios.append(q_row['ratio'].values[0])

            if quintile_ratios:
                min_r = min(quintile_ratios)
                max_r = max(quintile_ratios)
                print(f"  Quintile ratio range: {min_r:.3f} to {max_r:.3f}")

                if max_r - min_r < 0.2:
                    print(f"  Distribution shape: GOOD - consistent across quintiles")
                else:
                    print(f"  Distribution shape: SKEWED - quintile ratios vary significantly")


def main(synthetic_dir=None, real_dir=None, output_dir=None, metadata_path=None, sample_size=None, train_puprns=None):
    """Main analysis function.

    Args:
        synthetic_dir: Path to generated_data directory (default: SYNTHETIC_DATA_DIR)
        real_dir: Path to tokenised_data directory (default: REAL_DATA_DIR)
        output_dir: Output directory for results (default: OUTPUT_DIR)
        metadata_path: Path to metadata_catalog_filtered.pkl (default: REAL_METADATA_PATH)
        sample_size: Number of real households to sample (default: REAL_DATA_SAMPLE_SIZE)

    Returns:
        Dictionary with analysis results
    """
    # Use provided values or fall back to global defaults
    synthetic_data_dir = synthetic_dir if synthetic_dir else SYNTHETIC_DATA_DIR
    real_data_dir = real_dir if real_dir else REAL_DATA_DIR
    out_dir = output_dir if output_dir else OUTPUT_DIR
    meta_path = metadata_path if metadata_path else REAL_METADATA_PATH
    real_sample_size = sample_size if sample_size is not None else REAL_DATA_SAMPLE_SIZE

    print("=" * 70)
    print("HOUSEHOLD-MONTH FIDELITY ANALYSIS")
    print("=" * 70)

    # Validate paths
    if not os.path.exists(synthetic_data_dir):
        print(f"ERROR: Synthetic data directory not found: {synthetic_data_dir}")
        print("Please provide valid synthetic_dir parameter.")
        return {'status': 'failed', 'error': 'synthetic_dir not found'}

    # Create output directory
    os.makedirs(out_dir, exist_ok=True)

    # Find synthetic batch files
    batch_files = find_batch_files(synthetic_data_dir)
    if len(batch_files) == 0:
        print("ERROR: No batch files found!")
        return {'status': 'failed', 'error': 'no batch files'}

    # Find real household files (optional for this diagnostic)
    household_files = []
    real_data = {}
    if real_data_dir and os.path.exists(real_data_dir):
        household_files = find_real_household_files(real_data_dir, real_sample_size, puprn_filter=train_puprns)
        if len(household_files) > 0:
            real_data = load_household_month_real_data(household_files)
        else:
            print("Note: No real household files found, skipping real data comparison")
    else:
        print(f"Note: Real data directory not found: {real_data_dir}")
        print("Proceeding with synthetic data analysis only.")

    # Load real conditioning data from metadata catalog
    real_conditioning = None
    if meta_path and os.path.exists(meta_path):
        print(f"\nLoading real conditioning data from: {meta_path}")
        real_conditioning = load_real_conditioning(meta_path)
    else:
        print(f"\nNote: Metadata file not found: {meta_path}")
        print("Conditioning variable comparison will be limited to synthetic data only.")

    # Load household-month synthetic data
    synth_data = load_household_month_synthetic_data(batch_files)

    # Compare marginal distributions
    marginal_df = compare_marginal_distributions(synth_data, real_data)

    # Compare conditioning variable means
    conditioning_df = compare_conditioning_means(synth_data, real_data, real_conditioning)

    # Save results
    marginal_path = os.path.join(out_dir, "household_month_marginal_comparison.csv")
    marginal_df.to_csv(marginal_path, index=False)
    print(f"\nMarginal comparison saved to: {marginal_path}")

    if len(conditioning_df) > 0:
        conditioning_path = os.path.join(out_dir, "household_month_conditioning_comparison.csv")
        conditioning_df.to_csv(conditioning_path, index=False)
        print(f"Conditioning comparison saved to: {conditioning_path}")

    # Create visualizations (3 separate figures)
    create_summary_plots(marginal_df, conditioning_df, out_dir)

    # Print summary assessment
    print_summary(marginal_df)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    return {'status': 'completed', 'marginal_df': marginal_df, 'conditioning_df': conditioning_df}


if __name__ == "__main__":
    main()
