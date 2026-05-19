"""
Within-batch variance analysis functions for MapReduce evaluation pipeline.

These functions detect potential overfitting by analyzing variance within
synthetic batches - if households are too similar, it suggests the model
may be memorizing rather than generating diverse samples.

Migrated from: src/evaluation/evaluator.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
from scipy.spatial.distance import pdist, squareform
from collections import Counter
from datetime import datetime


# Within-Batch Variance Test Constants
OVERFITTING_THRESHOLDS = {
    'high_correlation': 0.95,      # Flag if any pair has r > 0.95
    'near_identical_distance': 0.01,  # Flag if distance < 1% of mean
    'peak_concentration': 0.8,      # Flag if >80% share same peak hour
    'mean_correlation': 0.7,        # Flag if mean correlation > 0.7
}


def compute_within_batch_correlations(batch_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute pairwise correlations between households in a batch.

    Args:
        batch_df: Wide format batch DataFrame

    Returns:
        Dictionary containing:
            - elec_correlations: N×N pairwise correlation matrix
            - gas_correlations: N×N pairwise correlation matrix
            - elec_mean_corr: Mean off-diagonal correlation (electricity)
            - gas_mean_corr: Mean off-diagonal correlation (gas)
            - elec_max_corr: Max off-diagonal correlation (electricity)
            - gas_max_corr: Max off-diagonal correlation (gas)
            - num_high_corr_pairs_elec: Count of pairs with r > 0.95 (electricity)
            - num_high_corr_pairs_gas: Count of pairs with r > 0.95 (gas)
            - batch_size: Number of households in batch
    """
    household_cols = [col for col in batch_df.columns
                     if col.startswith('hh') and '_' in col
                     and col not in ['half_hour']]
    household_prefixes = sorted(set([col.split('_')[0] for col in household_cols]))

    # Extract electricity and gas timeseries, filtering out constant series
    elec_series = []
    gas_series = []
    valid_households = []

    for hh in household_prefixes:
        elec = batch_df[f'{hh}_elec_net_Wh'].values
        gas = batch_df[f'{hh}_gas_Wh'].values

        # Convert to float and replace NaN with 0 for numeric operations
        elec = np.nan_to_num(np.array(elec, dtype=float), nan=0.0)
        gas = np.nan_to_num(np.array(gas, dtype=float), nan=0.0)

        # Skip households with zero variance (constant values) to avoid div-by-zero in corrcoef
        if np.std(elec) > 0 and np.std(gas) > 0:
            elec_series.append(elec)
            gas_series.append(gas)
            valid_households.append(hh)

    # Handle edge case: not enough valid households for correlation
    if len(valid_households) < 2:
        N = len(household_prefixes)
        return {
            'elec_correlations': np.full((N, N), np.nan),
            'gas_correlations': np.full((N, N), np.nan),
            'elec_mean_corr': np.nan,
            'gas_mean_corr': np.nan,
            'elec_max_corr': np.nan,
            'gas_max_corr': np.nan,
            'num_high_corr_pairs_elec': 0,
            'num_high_corr_pairs_gas': 0,
            'batch_size': N,
            'num_constant_households': N - len(valid_households)
        }

    # Compute pairwise correlations (only for non-constant households)
    elec_corr_matrix = np.corrcoef(elec_series)
    gas_corr_matrix = np.corrcoef(gas_series)

    # Extract off-diagonal elements (exclude self-correlation)
    N_valid = len(valid_households)
    mask = ~np.eye(N_valid, dtype=bool)
    elec_off_diag = elec_corr_matrix[mask]
    gas_off_diag = gas_corr_matrix[mask]

    return {
        'elec_correlations': elec_corr_matrix,
        'gas_correlations': gas_corr_matrix,
        'elec_mean_corr': float(np.nanmean(elec_off_diag)),
        'gas_mean_corr': float(np.nanmean(gas_off_diag)),
        'elec_max_corr': float(np.nanmax(elec_off_diag)),
        'gas_max_corr': float(np.nanmax(gas_off_diag)),
        'num_high_corr_pairs_elec': int(np.sum(elec_off_diag > OVERFITTING_THRESHOLDS['high_correlation'])),
        'num_high_corr_pairs_gas': int(np.sum(gas_off_diag > OVERFITTING_THRESHOLDS['high_correlation'])),
        'batch_size': len(household_prefixes),
        'num_valid_households': N_valid,
        'num_constant_households': len(household_prefixes) - N_valid
    }


def compute_within_batch_distances(batch_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute pairwise Euclidean distances between households (normalized).

    Args:
        batch_df: Wide format batch DataFrame

    Returns:
        Dictionary containing distance metrics
    """
    household_cols = [col for col in batch_df.columns
                     if col.startswith('hh') and '_' in col
                     and col not in ['half_hour']]
    household_prefixes = sorted(set([col.split('_')[0] for col in household_cols]))

    elec_series = []
    gas_series = []
    for hh in household_prefixes:
        elec = batch_df[f'{hh}_elec_net_Wh'].values
        gas = batch_df[f'{hh}_gas_Wh'].values

        # Convert to float and replace NaN with 0 for distance calculation
        elec = np.nan_to_num(np.array(elec, dtype=float), nan=0.0)
        gas = np.nan_to_num(np.array(gas, dtype=float), nan=0.0)

        # Normalize by mean to make distances comparable
        elec_mean = np.mean(np.abs(elec))
        gas_mean = np.mean(np.abs(gas))

        if elec_mean > 0:
            elec_series.append(elec / elec_mean)
        else:
            elec_series.append(elec)

        if gas_mean > 0:
            gas_series.append(gas / gas_mean)
        else:
            gas_series.append(gas)

    # Compute pairwise Euclidean distances
    elec_dist_matrix = squareform(pdist(elec_series, metric='euclidean'))
    gas_dist_matrix = squareform(pdist(gas_series, metric='euclidean'))

    # Extract off-diagonal
    N = len(household_prefixes)
    mask = ~np.eye(N, dtype=bool)
    elec_off_diag = elec_dist_matrix[mask]
    gas_off_diag = gas_dist_matrix[mask]

    # Identify near-identical pairs (distance < 1% of mean distance)
    elec_mean_dist = np.mean(elec_off_diag)
    gas_mean_dist = np.mean(gas_off_diag)

    return {
        'elec_distances': elec_dist_matrix,
        'gas_distances': gas_dist_matrix,
        'elec_mean_dist': float(elec_mean_dist),
        'gas_mean_dist': float(gas_mean_dist),
        'elec_min_dist': float(np.min(elec_off_diag)),
        'gas_min_dist': float(np.min(gas_off_diag)),
        'num_near_identical_elec': int(np.sum(elec_off_diag < OVERFITTING_THRESHOLDS['near_identical_distance'] * elec_mean_dist)),
        'num_near_identical_gas': int(np.sum(gas_off_diag < OVERFITTING_THRESHOLDS['near_identical_distance'] * gas_mean_dist)),
        'batch_size': N
    }


def compute_within_batch_peak_diversity(batch_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Check if households have identical peak timing patterns.

    Args:
        batch_df: Wide format batch DataFrame

    Returns:
        Dictionary containing peak diversity metrics
    """
    household_cols = [col for col in batch_df.columns
                     if col.startswith('hh') and '_' in col
                     and col not in ['half_hour']]
    household_prefixes = sorted(set([col.split('_')[0] for col in household_cols]))

    elec_peak_hours = []
    gas_peak_hours = []

    for hh in household_prefixes:
        elec = batch_df[f'{hh}_elec_net_Wh'].values
        gas = batch_df[f'{hh}_gas_Wh'].values

        # Convert to float and replace NaN with 0 for peak finding
        elec = np.nan_to_num(np.array(elec, dtype=float), nan=0.0)
        gas = np.nan_to_num(np.array(gas, dtype=float), nan=0.0)

        # Find peak hour (based on half-hour index)
        elec_peak_idx = np.argmax(elec)
        gas_peak_idx = np.argmax(gas)

        elec_peak_hours.append(elec_peak_idx)
        gas_peak_hours.append(gas_peak_idx)

    elec_counts = Counter(elec_peak_hours)
    gas_counts = Counter(gas_peak_hours)

    N = len(household_prefixes)

    return {
        'unique_elec_peak_hours': len(elec_counts),
        'unique_gas_peak_hours': len(gas_counts),
        'elec_peak_concentration': max(elec_counts.values()) / N if N > 0 else 0.0,
        'gas_peak_concentration': max(gas_counts.values()) / N if N > 0 else 0.0,
        'batch_size': N
    }


def flag_potential_overfitting(variance_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Identify potential overfitting issues based on within-batch variance.

    Args:
        variance_metrics: Combined metrics from correlation, distance, and peak diversity tests

    Returns:
        Dictionary containing:
            - flags: List of warning messages
            - severity: 'none', 'low', 'medium', or 'high'
            - metrics: Original variance metrics
    """
    flags = []

    # Check high correlations
    if variance_metrics.get('num_high_corr_pairs_elec', 0) > 0:
        flags.append(f"HIGH CORRELATION: {variance_metrics['num_high_corr_pairs_elec']} electricity pairs with r > {OVERFITTING_THRESHOLDS['high_correlation']}")

    if variance_metrics.get('num_high_corr_pairs_gas', 0) > 0:
        flags.append(f"HIGH CORRELATION: {variance_metrics['num_high_corr_pairs_gas']} gas pairs with r > {OVERFITTING_THRESHOLDS['high_correlation']}")

    # Check mean correlations
    if variance_metrics.get('elec_mean_corr', 0) > OVERFITTING_THRESHOLDS['mean_correlation']:
        flags.append(f"HIGH MEAN CORRELATION: Electricity mean r = {variance_metrics['elec_mean_corr']:.3f}")

    if variance_metrics.get('gas_mean_corr', 0) > OVERFITTING_THRESHOLDS['mean_correlation']:
        flags.append(f"HIGH MEAN CORRELATION: Gas mean r = {variance_metrics['gas_mean_corr']:.3f}")

    # Check near-identical pairs
    if variance_metrics.get('num_near_identical_elec', 0) > 0:
        flags.append(f"NEAR-IDENTICAL: {variance_metrics['num_near_identical_elec']} electricity pairs")

    if variance_metrics.get('num_near_identical_gas', 0) > 0:
        flags.append(f"NEAR-IDENTICAL: {variance_metrics['num_near_identical_gas']} gas pairs")

    # Check peak concentration
    if variance_metrics.get('elec_peak_concentration', 0) > OVERFITTING_THRESHOLDS['peak_concentration']:
        flags.append(f"PEAK CLUSTERING: {variance_metrics['elec_peak_concentration']*100:.1f}% households peak at same hour (electricity)")

    if variance_metrics.get('gas_peak_concentration', 0) > OVERFITTING_THRESHOLDS['peak_concentration']:
        flags.append(f"PEAK CLUSTERING: {variance_metrics['gas_peak_concentration']*100:.1f}% households peak at same hour (gas)")

    # Determine severity
    if len(flags) == 0:
        severity = 'none'
    elif len(flags) <= 2:
        severity = 'low'
    elif len(flags) <= 4:
        severity = 'medium'
    else:
        severity = 'high'

    return {
        'flags': flags,
        'severity': severity,
        'metrics': variance_metrics
    }


def generate_within_batch_variance_report(batch_results: List[Dict]) -> str:
    """
    Generate markdown report summarizing within-batch variance analysis.

    Args:
        batch_results: List of variance test results per batch

    Returns:
        Markdown formatted report
    """
    report = []
    report.append("# Within-Batch Variance Analysis")
    report.append(f"\n**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"\n**Total Batches Analyzed**: {len(batch_results)}")

    # Overall summary
    severities = [r.get('severity', 'unknown') for r in batch_results]
    severity_counts = Counter(severities)

    report.append("\n## Summary")
    report.append(f"- **No issues**: {severity_counts.get('none', 0)} batches")
    report.append(f"- **Low severity**: {severity_counts.get('low', 0)} batches")
    report.append(f"- **Medium severity**: {severity_counts.get('medium', 0)} batches")
    report.append(f"- **High severity**: {severity_counts.get('high', 0)} batches")

    # Flagged batches
    flagged_batches = [r for r in batch_results if r.get('severity', 'unknown') not in ['none', 'unknown']]
    if flagged_batches:
        report.append("\n## Flagged Batches")
        for result in flagged_batches:
            batch_id = result.get('batch_id', 'unknown')
            severity = result.get('severity', 'unknown')
            report.append(f"\n### {batch_id} - Severity: {severity.upper()}")
            for flag in result.get('flags', []):
                report.append(f"  - {flag}")
    else:
        report.append("\n## All Batches Passed")
        report.append("No within-batch variance issues detected.")

    return "\n".join(report)


def plot_within_batch_variance(batch_results: List[Dict], output_dir: str):
    """
    Create visualizations for within-batch variance analysis.

    Args:
        batch_results: List of variance test results per batch
        output_dir: Directory to save plots

    Creates:
        1. Correlation heatmaps (sample 4 batches)
        2. Distribution of mean correlations across batches
        3. Peak time diversity
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Correlation heatmaps (sample up to 4 batches)
    num_batches_to_plot = min(4, len(batch_results))
    if num_batches_to_plot > 0:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        for idx, result in enumerate(batch_results[:num_batches_to_plot]):
            ax = axes[idx]
            corr = result['metrics']['elec_correlations']
            sns.heatmap(corr, ax=ax, cmap='RdYlGn_r', vmin=0, vmax=1,
                       annot=True, fmt='.2f', cbar_kws={'label': 'Correlation'})
            ax.set_title(f"{result['batch_id']} - Electricity")

        # Hide unused subplots
        for idx in range(num_batches_to_plot, 4):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig(f"{output_dir}/within_batch_correlation_heatmaps.png", dpi=150)
        plt.close()

    # 2. Distribution of mean correlations across batches
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    elec_corrs = [r['metrics']['elec_mean_corr'] for r in batch_results]
    gas_corrs = [r['metrics']['gas_mean_corr'] for r in batch_results]

    ax1.hist(elec_corrs, bins=20, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(OVERFITTING_THRESHOLDS['mean_correlation'], color='red',
               linestyle='--', label=f"Threshold ({OVERFITTING_THRESHOLDS['mean_correlation']})")
    ax1.set_xlabel('Mean Within-Batch Correlation')
    ax1.set_ylabel('Number of Batches')
    ax1.set_title('Electricity: Within-Batch Correlation Distribution')
    ax1.legend()

    ax2.hist(gas_corrs, bins=20, alpha=0.7, color='orange', edgecolor='black')
    ax2.axvline(OVERFITTING_THRESHOLDS['mean_correlation'], color='red',
               linestyle='--', label=f"Threshold ({OVERFITTING_THRESHOLDS['mean_correlation']})")
    ax2.set_xlabel('Mean Within-Batch Correlation')
    ax2.set_ylabel('Number of Batches')
    ax2.set_title('Gas: Within-Batch Correlation Distribution')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/within_batch_correlation_distributions.png", dpi=150)
    plt.close()

    # 3. Peak time diversity
    if len(batch_results) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))

        elec_peak_conc = [r['metrics'].get('elec_peak_concentration', 0) for r in batch_results]
        gas_peak_conc = [r['metrics'].get('gas_peak_concentration', 0) for r in batch_results]

        x = np.arange(len(batch_results))
        width = 0.35

        ax.bar(x - width/2, elec_peak_conc, width, label='Electricity', alpha=0.7)
        ax.bar(x + width/2, gas_peak_conc, width, label='Gas', alpha=0.7)
        ax.axhline(OVERFITTING_THRESHOLDS['peak_concentration'], color='red',
                  linestyle='--', label=f"Threshold ({OVERFITTING_THRESHOLDS['peak_concentration']})")

        ax.set_xlabel('Batch Index')
        ax.set_ylabel('Peak Concentration (fraction)')
        ax.set_title('Peak Time Clustering Within Batches')
        ax.legend()

        plt.tight_layout()
        plt.savefig(f"{output_dir}/within_batch_peak_diversity.png", dpi=150)
        plt.close()
