#!/usr/bin/env python3
"""
Diagnostic: Weekly Token Frequency Cascade Analysis

Tests the autoregressive cascade hypothesis (Decision 039) by comparing token
frequency distributions across weeks 1-4 within each month. Decomposes the
total token distribution bias into two mechanisms:

  Mechanism 1 (Learned per-timestep bias): Week-1 synthetic deviation from real.
    Present before any cascade can accumulate — this is the model's inherent
    per-step distribution error.

  Mechanism 2 (Autoregressive cascade): Week-over-week drift (week 4 - week 1).
    Accumulates as generated tokens enter the context and influence subsequent
    predictions. If present, expect the distribution to become progressively
    more distorted from uniform (both tails fatten, middle hollows) — i.e.
    more U-shaped, NOT flatter.

Predictions if cascade is present:
  - Token frequency std should increase week-over-week (distribution becomes
    more non-uniform as mass concentrates at extremes)
  - Shannon entropy should decrease (distribution moves further from uniform)
  - Lower-THIRD and upper-THIRD mass should both increase, while middle-third
    decreases (both tails fatten, middle hollows)
  - Implied Wh mean may increase even if cascade is symmetric in token space,
    because upper tokens span wider Wh ranges (bin width amplification)

Note on temporal resolution: week 1 is NOT a cascade-free baseline — it
already contains 336 autoregressive steps. If cascade happens fast (within
the first few hundred steps), most of the cascade effect may already be
captured in week 1, making the week 1→4 drift appear small.

Electricity only (for now).

Usage:
    python src/evaluation/diagnostics/diagnose_weekly_token_cascade.py

Outputs:
    - weekly_token_cascade.csv: Per-token frequencies by week
    - weekly_token_cascade_frequency.png: 4-panel token frequency comparison
    - weekly_token_cascade_decomposition.png: Mechanism decomposition overlay
    - weekly_token_cascade_metrics.png: Summary metrics across weeks
    - weekly_token_cascade_report.txt: Full text report with diagnosis
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

TIMESTEPS_PER_WEEK = 48 * 7  # 336 half-hours
NUM_WEEKS = 4


def load_elec_bin_edges(tokeniser_path):
    """Load electricity bin boundaries from the tokeniser file.

    Returns:
        tuple of (bin_edges, value_transform)
        - bin_edges: numpy array of bin edges (in log space if value_transform is "log1p")
        - value_transform: "none" or "log1p"
    """
    with open(tokeniser_path, 'rb') as f:
        tokenisers = pickle.load(f)

    print(f"Loaded tokenisers from: {tokeniser_path}")

    value_transform = tokenisers.get('value_transform', 'none')
    print(f"  value_transform: {value_transform}")

    if 'elec_boundaries' in tokenisers:
        edges = np.array(tokenisers['elec_boundaries'])
    else:
        edges = None
        for key in ['elec_tokeniser', 'electricity']:
            if key in tokenisers and tokenisers[key] is not None:
                obj = tokenisers[key]
                if hasattr(obj, 'bin_edges_'):
                    edges = np.array(obj.bin_edges_[0])
                elif hasattr(obj, 'boundaries'):
                    edges = np.array(obj.boundaries)
                if edges is not None:
                    break

    if edges is None:
        raise ValueError("Could not find electricity bin edges in tokeniser file")

    n_bins = len(edges) - 1
    if value_transform == "log1p":
        wh_low = np.expm1(edges[0])
        wh_high = np.expm1(edges[-1])
        print(f"  Electricity: {n_bins} bins, log range [{edges[0]:.2f}, {edges[-1]:.2f}], "
              f"Wh range [{wh_low:.1f}, {wh_high:.1f}]")
    else:
        print(f"  Electricity: {n_bins} bins, range [{edges[0]:.1f}, {edges[-1]:.1f}]")
    return edges, value_transform


def values_to_tokens(values, bin_edges, value_transform="none"):
    """Convert Wh values to token indices using bin edges.

    If value_transform is "log1p", applies log1p to values before digitizing
    (bin_edges are assumed to be in log space).
    """
    if value_transform == "log1p":
        values = np.log1p(np.maximum(values, 0))
    tokens = np.digitize(values, bin_edges[1:-1])
    n_bins = len(bin_edges) - 1
    return np.clip(tokens, 0, n_bins - 1)


def bin_edges_to_wh(bin_edges, value_transform="none"):
    """Convert bin edges to Wh space (for display/annotation)."""
    if value_transform == "log1p":
        return np.expm1(bin_edges)
    return bin_edges


def collect_synthetic_token_counts_by_week(data_dir, bin_edges, value_transform="none"):
    """Accumulate token counts from synthetic batch files, split by week.

    Each batch file has rows=timesteps, columns=household features.
    Weeks are split by row position (0-335 = week 1, 336-671 = week 2, etc.).

    Returns:
        week_counts: {week: np.array of token counts}
        week_totals: {week: int total values}
    """
    n_bins = len(bin_edges) - 1
    week_counts = {w: np.zeros(n_bins, dtype=np.int64) for w in range(1, NUM_WEEKS + 1)}
    week_totals = {w: 0 for w in range(1, NUM_WEEKS + 1)}

    data_path = Path(data_dir)
    batch_files = sorted(data_path.glob("batch_*_month_*.pkl"))
    if not batch_files:
        batch_files = sorted(data_path.glob("batch_*.pkl"))
        batch_files = [f for f in batch_files if 'conditioning' not in f.name]
    print(f"Found {len(batch_files)} batch files in {data_dir}")

    print("\nLoading synthetic data by week...")
    for i, batch_path in enumerate(batch_files):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Processing {i+1}/{len(batch_files)}: {batch_path.name}")

        df = pd.read_pickle(batch_path)
        elec_cols = [c for c in df.columns if '_elec_net_Wh' in c or '_elec_Wh' in c]
        if not elec_cols:
            continue

        for week in range(1, NUM_WEEKS + 1):
            start = (week - 1) * TIMESTEPS_PER_WEEK
            end = week * TIMESTEPS_PER_WEEK
            week_df = df.iloc[start:end]

            if len(week_df) == 0:
                continue

            for col in elec_cols:
                vals = week_df[col].values.astype(float)
                vals = vals[~np.isnan(vals)]
                if len(vals) > 0:
                    tokens = values_to_tokens(vals, bin_edges, value_transform)
                    week_counts[week] += np.bincount(tokens.astype(int), minlength=n_bins)
                    week_totals[week] += len(vals)

    for w in sorted(week_counts):
        print(f"  Week {w}: {week_totals[w]:,} values across {n_bins} bins")

    return week_counts, week_totals


def collect_real_token_counts(data_dir, bin_edges, sample_size=None, value_transform="none"):
    """Accumulate token counts from real household files (all weeks combined).

    Real data is the training target — it doesn't change by week.
    """
    n_bins = len(bin_edges) - 1
    counts = np.zeros(n_bins, dtype=np.int64)
    total = 0

    household_files = sorted(Path(data_dir).glob("*_tokenised.pkl"))
    print(f"\nFound {len(household_files)} real household files in {data_dir}")

    if sample_size and len(household_files) > sample_size:
        import random
        random.seed(42)
        household_files = random.sample(household_files, sample_size)
        print(f"Sampled {sample_size} households")

    elec_candidates = ['Clean_elec_net_Wh', 'elec_Wh', 'elec_value']

    print("Loading real data...")
    for i, hh_path in enumerate(household_files):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  Processing household {i+1}/{len(household_files)}")

        try:
            df = pd.read_pickle(hh_path)
        except Exception:
            continue

        for col in elec_candidates:
            if col in df.columns:
                vals = df[col].values.astype(float)
                vals = vals[~np.isnan(vals)]
                if len(vals) > 0:
                    tokens = values_to_tokens(vals, bin_edges, value_transform)
                    counts += np.bincount(tokens.astype(int), minlength=n_bins)
                    total += len(vals)
                break

    print(f"  Real electricity: {total:,} values across {n_bins} bins")
    return counts, total


def counts_to_freqs(counts):
    """Convert count array to frequency array."""
    total = counts.sum()
    if total == 0:
        return np.zeros_like(counts, dtype=float)
    return counts / total


# ============================================================================
# Plotting
# ============================================================================

def create_weekly_frequency_plot(week_freqs, real_freq, bin_edges, output_dir):
    """4-panel plot: one per week, showing real vs synth token frequencies + uniform."""
    if not HAS_MATPLOTLIB:
        return

    n_bins = len(real_freq)
    x = np.arange(n_bins)
    uniform = 1.0 / n_bins

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    for idx, week in enumerate(sorted(week_freqs)):
        ax = axes[idx // 2, idx % 2]
        sf = week_freqs[week]

        width = 0.35
        ax.bar(x - width/2, real_freq, width, label='Real (train)', color='steelblue', alpha=0.7)
        ax.bar(x + width/2, sf, width, label=f'Synthetic week {week}', color='coral', alpha=0.7)
        ax.axhline(uniform, color='black', linestyle='--', linewidth=1,
                    label=f'Uniform (1/{n_bins})')

        ax.set_xlabel('Token index')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Week {week}', fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2, axis='y')

    fig.suptitle(
        'ELECTRICITY: Token Frequency by Week — Testing Autoregressive Cascade\n'
        'If cascade is present, expect progressive distortion from uniform (U-shape deepens)',
        fontsize=12, fontweight='bold'
    )
    fig.tight_layout()

    path = os.path.join(output_dir, 'weekly_token_cascade_frequency.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Weekly frequency plot saved to: {path}")


def create_deviation_overlay_plot(week_freqs, real_freq, bin_edges, output_dir):
    """Overlay plot showing deviation from real for all 4 weeks,
    plus mechanism decomposition bar chart."""
    if not HAS_MATPLOTLIB:
        return

    n_bins = len(real_freq)
    x = np.arange(n_bins)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # --- Top: deviation from real, overlaid for each week ---
    ax = axes[0]
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#F44336']
    for idx, week in enumerate(sorted(week_freqs)):
        dev = (week_freqs[week] - real_freq) * 100
        ax.plot(x, dev, color=colors[idx], linewidth=1.5,
                label=f'Week {week}', alpha=0.8)

    ax.axhline(0, color='black', linewidth=1)
    ax.set_xlabel('Token index')
    ax.set_ylabel('Deviation from real (percentage points)')
    ax.set_title('Per-Token Deviation from Real Distribution by Week', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.98,
            'If cascade present: lines should fan out from week 1 → 4,\n'
            'with tails becoming more overproduced and middle more underproduced',
            transform=ax.transAxes, fontsize=8, va='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # --- Bottom: mechanism decomposition ---
    ax = axes[1]
    week1_dev = week_freqs[1] - real_freq    # Mechanism 1: learned bias
    cascade_dev = week_freqs[4] - week_freqs[1]  # Mechanism 2: cascade drift
    total_dev = week_freqs[4] - real_freq    # Total week-4 deviation

    width = 0.25
    ax.bar(x - width, week1_dev * 100, width,
           label='Mechanism 1: Week-1 bias (learned)',
           color='steelblue', alpha=0.7)
    ax.bar(x, cascade_dev * 100, width,
           label='Mechanism 2: Week 1→4 drift (cascade)',
           color='coral', alpha=0.7)
    ax.bar(x + width, total_dev * 100, width,
           label='Total: Week-4 deviation from real',
           color='grey', alpha=0.5)

    ax.axhline(0, color='black', linewidth=1)
    ax.set_xlabel('Token index')
    ax.set_ylabel('Deviation (percentage points)')
    ax.set_title('Mechanism Decomposition: Learned Bias vs Autoregressive Cascade',
                 fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle('ELECTRICITY: Two-Mechanism Decomposition of Token Distribution Bias',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()

    path = os.path.join(output_dir, 'weekly_token_cascade_decomposition.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Decomposition plot saved to: {path}")


def create_metrics_by_week_plot(week_freqs, real_freq, bin_edges, output_dir, value_transform="none"):
    """Line plots of summary metrics across weeks — shows cascade progression."""
    if not HAS_MATPLOTLIB:
        return

    n_bins = len(real_freq)
    edges_wh = bin_edges_to_wh(bin_edges, value_transform)
    bin_centers = (edges_wh[:-1] + edges_wh[1:]) / 2

    weeks = sorted(week_freqs)

    # Compute metrics per week
    stds = []
    lower_mass = []
    upper_mass = []
    implied_means = []
    entropies = []

    for w in weeks:
        sf = week_freqs[w]
        stds.append(np.std(sf))
        lower_mass.append(sf[:n_bins//2].sum())
        upper_mass.append(sf[n_bins//2:].sum())
        implied_means.append(np.sum(sf * bin_centers))
        sf_safe = sf[sf > 0]
        entropies.append(-np.sum(sf_safe * np.log2(sf_safe)))

    # Real baselines
    real_std = np.std(real_freq)
    real_lower = real_freq[:n_bins//2].sum()
    real_upper = real_freq[n_bins//2:].sum()
    real_mean = np.sum(real_freq * bin_centers)
    rf_safe = real_freq[real_freq > 0]
    real_entropy = -np.sum(rf_safe * np.log2(rf_safe))
    max_entropy = np.log2(n_bins)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Token frequency std
    ax = axes[0, 0]
    ax.plot(weeks, stds, 'o-', color='coral', linewidth=2, markersize=8, label='Synthetic')
    ax.axhline(real_std, color='steelblue', linestyle='--', linewidth=1.5, label='Real')
    ax.set_xlabel('Week')
    ax.set_ylabel('Std of token frequencies')
    ax.set_title('Distribution Spread\n(↑ = more non-uniform = more distorted)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(weeks)

    # Lower vs upper half mass
    ax = axes[0, 1]
    ax.plot(weeks, [m * 100 for m in lower_mass], 'o-', color='steelblue',
            linewidth=2, markersize=8, label='Synth lower half')
    ax.plot(weeks, [m * 100 for m in upper_mass], 'o-', color='coral',
            linewidth=2, markersize=8, label='Synth upper half')
    ax.axhline(real_lower * 100, color='steelblue', linestyle='--', linewidth=1,
               alpha=0.5, label='Real lower')
    ax.axhline(real_upper * 100, color='coral', linestyle='--', linewidth=1,
               alpha=0.5, label='Real upper')
    ax.axhline(50, color='black', linestyle=':', linewidth=1, alpha=0.5)
    ax.set_xlabel('Week')
    ax.set_ylabel('Mass (%)')
    ax.set_title('Lower vs Upper Half Mass\n(use thirds plot in report for cascade test)',
                 fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(weeks)

    # Implied Wh mean
    ax = axes[1, 0]
    ax.plot(weeks, implied_means, 'o-', color='coral', linewidth=2, markersize=8,
            label='Synthetic')
    ax.axhline(real_mean, color='steelblue', linestyle='--', linewidth=1.5, label='Real')
    ax.set_xlabel('Week')
    ax.set_ylabel('Implied Wh mean')
    ax.set_title('Implied Mean from Token Frequencies\n(bin-width amplification makes upper drift dominate)',
                 fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(weeks)

    # Entropy
    ax = axes[1, 1]
    ax.plot(weeks, entropies, 'o-', color='coral', linewidth=2, markersize=8,
            label='Synthetic')
    ax.axhline(real_entropy, color='steelblue', linestyle='--', linewidth=1.5, label='Real')
    ax.axhline(max_entropy, color='black', linestyle=':', linewidth=1,
               label=f'Max (uniform) = {max_entropy:.2f}')
    ax.set_xlabel('Week')
    ax.set_ylabel('Shannon entropy (bits)')
    ax.set_title('Token Distribution Entropy\n(cascade distortion → entropy ↓)',
                 fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(weeks)

    fig.suptitle('ELECTRICITY: Summary Metrics by Week — Cascade Progression',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()

    path = os.path.join(output_dir, 'weekly_token_cascade_metrics.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Metrics plot saved to: {path}")


# ============================================================================
# Report generation
# ============================================================================

def generate_report(week_freqs, real_freq, bin_edges, week_totals, output_dir, value_transform="none"):
    """Generate and save a text summary report with two-mechanism decomposition."""
    n_bins = len(real_freq)
    uniform = 1.0 / n_bins
    edges_wh = bin_edges_to_wh(bin_edges, value_transform)
    bin_centers = (edges_wh[:-1] + edges_wh[1:]) / 2
    bin_widths = np.diff(edges_wh)

    weeks = sorted(week_freqs)

    lines = []
    lines.append("=" * 70)
    lines.append("WEEKLY TOKEN CASCADE ANALYSIS — ELECTRICITY")
    lines.append("=" * 70)
    lines.append("")

    # --- Real baseline ---
    real_std = np.std(real_freq)
    real_mean = np.sum(real_freq * bin_centers)
    rf_safe = real_freq[real_freq > 0]
    real_entropy = -np.sum(rf_safe * np.log2(rf_safe))
    max_entropy = np.log2(n_bins)
    real_lower = real_freq[:n_bins//2].sum()
    real_upper = real_freq[n_bins//2:].sum()

    lines.append(f"Real (train) baseline:")
    lines.append(f"  Bins: {n_bins}")
    lines.append(f"  Token freq std:  {real_std:.6f} ({real_std/uniform*100:.1f}% of uniform)")
    lines.append(f"  Lower-half mass: {real_lower*100:.2f}%")
    lines.append(f"  Upper-half mass: {real_upper*100:.2f}%")
    lines.append(f"  Implied Wh mean: {real_mean:.1f}")
    lines.append(f"  Entropy:         {real_entropy:.4f} bits (max = {max_entropy:.4f})")
    lines.append("")

    # --- Per-week metrics ---
    lines.append("Per-week synthetic metrics:")
    header = (f"{'Week':>6} {'Values':>12} {'Freq std':>12} {'Lower %':>10} "
              f"{'Upper %':>10} {'Wh mean':>10} {'Entropy':>10}")
    lines.append(header)
    lines.append("-" * len(header))

    week_metrics = {}
    for w in weeks:
        sf = week_freqs[w]
        std = np.std(sf)
        lower = sf[:n_bins//2].sum() * 100
        upper = sf[n_bins//2:].sum() * 100
        wh_mean = np.sum(sf * bin_centers)
        sf_safe = sf[sf > 0]
        entropy = -np.sum(sf_safe * np.log2(sf_safe))

        week_metrics[w] = {
            'std': std, 'lower': lower, 'upper': upper,
            'wh_mean': wh_mean, 'entropy': entropy
        }

        lines.append(f"{w:>6} {week_totals[w]:>12,} {std:>12.6f} {lower:>10.2f} "
                      f"{upper:>10.2f} {wh_mean:>10.1f} {entropy:>10.4f}")

    lines.append("")

    # --- Trend analysis ---
    lines.append("Week-over-week trends:")
    for metric, label in [('std', 'Freq std'), ('lower', 'Lower-half %'),
                           ('upper', 'Upper-half %'), ('wh_mean', 'Implied Wh mean'),
                           ('entropy', 'Entropy')]:
        vals = [week_metrics[w][metric] for w in weeks]
        direction = "INCREASING" if vals[-1] > vals[0] else "DECREASING" if vals[-1] < vals[0] else "STABLE"
        delta = vals[-1] - vals[0]
        fmt = '.6f' if metric == 'std' else '.2f' if metric in ('lower', 'upper') else '.1f' if metric == 'wh_mean' else '.4f'
        lines.append(f"  {label:>18}: week 1 = {vals[0]:{fmt}}, week 4 = {vals[-1]:{fmt}}, "
                      f"delta = {delta:+{fmt}} → {direction}")
    lines.append("")

    # --- Two-mechanism decomposition ---
    lines.append("=" * 70)
    lines.append("TWO-MECHANISM DECOMPOSITION")
    lines.append("=" * 70)
    lines.append("")
    lines.append("IMPORTANT CAVEAT: Week 1 is NOT a cascade-free baseline. By week 1,")
    lines.append("the model has already autoregressed through 336 timesteps. If cascade")
    lines.append("happens fast (within the first few hundred steps), most of its effect")
    lines.append("is already captured in week 1. The 'Mechanism 1' numbers below therefore")
    lines.append("represent an UPPER BOUND on learned bias (they include any fast cascade),")
    lines.append("and 'Mechanism 2' represents a LOWER BOUND on total cascade contribution.")
    lines.append("")

    w1 = week_metrics[1]
    w4 = week_metrics[4]

    # Mechanism 1: week-1 deviation from real
    m1_wh = w1['wh_mean'] - real_mean
    m1_std = w1['std'] - real_std
    m1_upper = w1['upper'] - (real_upper * 100)
    m1_entropy = w1['entropy'] - real_entropy

    # Mechanism 2: week-4 minus week-1
    m2_wh = w4['wh_mean'] - w1['wh_mean']
    m2_std = w4['std'] - w1['std']
    m2_upper = w4['upper'] - w1['upper']
    m2_entropy = w4['entropy'] - w1['entropy']

    # Total
    total_wh = w4['wh_mean'] - real_mean
    total_std = w4['std'] - real_std
    total_upper = w4['upper'] - (real_upper * 100)
    total_entropy = w4['entropy'] - real_entropy

    lines.append("Mechanism 1 (Learned per-timestep bias): Week-1 synth vs real")
    lines.append(f"  Implied Wh mean shift: {m1_wh:+.1f} Wh")
    lines.append(f"  Token freq std shift:  {m1_std:+.6f}")
    lines.append(f"  Upper-half mass shift: {m1_upper:+.2f} pp")
    lines.append(f"  Entropy shift:         {m1_entropy:+.4f} bits")
    lines.append("")

    lines.append("Mechanism 2 (Autoregressive cascade): Week-4 synth vs week-1 synth")
    lines.append(f"  Implied Wh mean drift: {m2_wh:+.1f} Wh")
    lines.append(f"  Token freq std drift:  {m2_std:+.6f}")
    lines.append(f"  Upper-half mass drift: {m2_upper:+.2f} pp")
    lines.append(f"  Entropy drift:         {m2_entropy:+.4f} bits")
    lines.append("")

    lines.append("Total deviation (Week-4 synth vs real):")
    lines.append(f"  Implied Wh mean shift: {total_wh:+.1f} Wh")
    lines.append(f"  Token freq std shift:  {total_std:+.6f}")
    lines.append(f"  Upper-half mass shift: {total_upper:+.2f} pp")
    lines.append(f"  Entropy shift:         {total_entropy:+.4f} bits")
    lines.append("")

    # Attribution percentages
    lines.append("Attribution (% of total week-4 deviation):")
    for metric, m1_val, m2_val, total_val, label in [
        ('wh_mean', m1_wh, m2_wh, total_wh, 'Wh mean shift'),
        ('std', m1_std, m2_std, total_std, 'Freq std shift'),
        ('upper', m1_upper, m2_upper, total_upper, 'Upper-half mass shift'),
    ]:
        if abs(total_val) > 1e-6:
            m1_pct = m1_val / total_val * 100
            m2_pct = m2_val / total_val * 100
            lines.append(f"  {label:>22}: Mechanism 1 = {m1_pct:+.1f}%, Mechanism 2 = {m2_pct:+.1f}%")
        else:
            lines.append(f"  {label:>22}: total deviation ≈ 0, attribution not meaningful")
    lines.append("")

    # --- Cascade symmetry test ---
    lines.append("=" * 70)
    lines.append("CASCADE SYMMETRY TEST")
    lines.append("=" * 70)
    lines.append("")
    lines.append("If cascade is symmetric (pushes both tails equally), the week 1→4")
    lines.append("drift should increase lower-THIRD and upper-THIRD token frequencies,")
    lines.append("with middle-third decreasing. Note: lower-HALF and upper-HALF sum")
    lines.append("to 100% so cannot both increase — use thirds to test tail fattening.")
    lines.append("")

    # Per-token drift from week 1 to week 4
    drift = week_freqs[4] - week_freqs[1]

    lower_third = n_bins // 3
    upper_third = 2 * n_bins // 3

    zones = [
        ('Lower', 0, lower_third),
        ('Middle', lower_third, upper_third),
        ('Upper', upper_third, n_bins),
    ]

    for zone_name, z_start, z_end in zones:
        zone_drift = drift[z_start:z_end]
        n_up = np.sum(zone_drift > 0)
        n_down = np.sum(zone_drift < 0)
        n_tokens = z_end - z_start
        net = zone_drift.sum() * 100

        lines.append(f"  {zone_name} third (tokens {z_start}-{z_end-1}):")
        lines.append(f"    Drift UP:   {n_up}/{n_tokens} tokens")
        lines.append(f"    Drift DOWN: {n_down}/{n_tokens} tokens")
        lines.append(f"    Net mass change: {net:+.3f} pp")
        lines.append("")

    # Symmetry verdict
    lower_zone_drift = drift[:lower_third].sum()
    upper_zone_drift = drift[upper_third:].sum()
    mid_zone_drift = drift[lower_third:upper_third].sum()

    if lower_zone_drift > 0 and upper_zone_drift > 0 and mid_zone_drift < 0:
        lines.append("  SYMMETRY VERDICT: SYMMETRIC cascade detected")
        lines.append("  Both tails gain mass, middle loses mass → classic variance inflation")
    elif upper_zone_drift > 0 and lower_zone_drift <= 0:
        lines.append("  SYMMETRY VERDICT: ASYMMETRIC cascade (upper-only)")
        lines.append("  Only upper tail gains mass → cascade compounds with learned upward bias")
    elif lower_zone_drift > 0 and upper_zone_drift <= 0:
        lines.append("  SYMMETRY VERDICT: ASYMMETRIC cascade (lower-only)")
        lines.append("  Only lower tail gains mass → unexpected pattern")
    else:
        lines.append("  SYMMETRY VERDICT: UNCLEAR pattern")
        lines.append(f"  Lower drift: {lower_zone_drift*100:+.3f} pp, "
                      f"Upper drift: {upper_zone_drift*100:+.3f} pp")
    lines.append("")

    # --- Top drifting tokens ---
    lines.append("Top 5 tokens with largest UPWARD drift (week 1 → 4):")
    sorted_up = np.argsort(drift)[::-1]
    for idx in sorted_up[:5]:
        lines.append(f"    Token {idx:3d}: week1={week_freqs[1][idx]:.5f}, "
                      f"week4={week_freqs[4][idx]:.5f}, "
                      f"drift={drift[idx]*100:+.4f} pp "
                      f"(bin {edges_wh[idx]:.0f}-{edges_wh[idx+1]:.0f} Wh)")

    lines.append("")
    lines.append("Top 5 tokens with largest DOWNWARD drift (week 1 → 4):")
    for idx in sorted_up[-5:][::-1]:
        lines.append(f"    Token {idx:3d}: week1={week_freqs[1][idx]:.5f}, "
                      f"week4={week_freqs[4][idx]:.5f}, "
                      f"drift={drift[idx]*100:+.4f} pp "
                      f"(bin {edges_wh[idx]:.0f}-{edges_wh[idx+1]:.0f} Wh)")
    lines.append("")

    # --- Overall interpretation ---
    lines.append("=" * 70)
    lines.append("INTERPRETATION")
    lines.append("=" * 70)
    lines.append("")

    # Assess cascade significance
    if abs(total_wh) > 0.1:
        m2_fraction = abs(m2_wh / total_wh)
    else:
        m2_fraction = 0

    if m2_fraction < 0.1:
        lines.append("CASCADE SIGNIFICANCE: NEGLIGIBLE (<10% of total Wh bias)")
        lines.append("  The bias is almost entirely from learned per-timestep distribution errors.")
        lines.append("  Autoregressive cascade is not a meaningful contributor.")
    elif m2_fraction < 0.3:
        lines.append("CASCADE SIGNIFICANCE: MINOR (10-30% of total Wh bias)")
        lines.append("  The primary bias is from learned per-timestep distribution errors.")
        lines.append("  Cascade provides noticeable but secondary amplification.")
    elif m2_fraction < 0.5:
        lines.append("CASCADE SIGNIFICANCE: MODERATE (30-50% of total Wh bias)")
        lines.append("  Both mechanisms contribute meaningfully.")
        lines.append("  Temperature scaling should address both (sharpens per-step + dampens cascade).")
    else:
        lines.append("CASCADE SIGNIFICANCE: MAJOR (>50% of total Wh bias)")
        lines.append("  Autoregressive cascade is the dominant contributor.")
        lines.append("  Temperature scaling may help by preventing initial extreme tokens.")
    lines.append("")

    # Entropy trend
    e1 = week_metrics[1]['entropy']
    e4 = week_metrics[4]['entropy']
    if e4 < e1 - 0.001:
        lines.append(f"ENTROPY TREND: DECREASING (week 1: {e1:.4f} → week 4: {e4:.4f} bits)")
        lines.append("  Consistent with cascade: distribution becomes more concentrated at extremes.")
    elif e4 > e1 + 0.001:
        lines.append(f"ENTROPY TREND: INCREASING (week 1: {e1:.4f} → week 4: {e4:.4f} bits)")
        lines.append("  Unexpected if cascade concentrates mass at extremes.")
        lines.append("  May indicate mass is spreading across more tokens (e.g., lower tail broadening).")
    else:
        lines.append(f"ENTROPY TREND: STABLE (week 1: {e1:.4f} → week 4: {e4:.4f} bits)")
        lines.append("  No significant entropy change detected.")
    lines.append("")

    # Temperature scaling prediction
    lines.append("TEMPERATURE SCALING PREDICTIONS:")
    lines.append("  If T=0.85 results are available, compare against this T=1.0 baseline:")
    lines.append("  - Week-1 bias should decrease (Mechanism 1 responsive to temperature)")
    lines.append("  - Week-over-week spread should decrease (Mechanism 2 dampened)")
    lines.append("  - If only week-over-week changes but week-1 doesn't improve:")
    lines.append("    → cascade is responsive but learned bias is not (may need retraining)")
    lines.append("  - If week-1 improves but spread doesn't change:")
    lines.append("    → per-step bias improved but cascade dynamics unchanged")

    report = "\n".join(lines)

    # Print to terminal
    print("\n" + report)

    # Save to file
    report_path = os.path.join(output_dir, 'weekly_token_cascade_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")

    return report


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("WEEKLY TOKEN CASCADE ANALYSIS — ELECTRICITY")
    print("=" * 70)

    # Validate paths
    if not os.path.exists(TOKENISER_PATH):
        print(f"ERROR: Tokeniser not found: {TOKENISER_PATH}")
        sys.exit(1)

    if not os.path.exists(SYNTHETIC_DATA_DIR):
        print(f"ERROR: Synthetic data not found: {SYNTHETIC_DATA_DIR}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load bin edges
    bin_edges, value_transform = load_elec_bin_edges(TOKENISER_PATH)
    n_bins = len(bin_edges) - 1

    # Collect synthetic token counts by week
    print("\n--- Synthetic data ---")
    week_counts, week_totals = collect_synthetic_token_counts_by_week(
        SYNTHETIC_DATA_DIR, bin_edges, value_transform
    )

    # Compute per-week frequencies
    week_freqs = {w: counts_to_freqs(week_counts[w]) for w in week_counts}

    # Collect real token counts (all weeks combined — real is the baseline)
    print("\n--- Real data ---")
    if REAL_DATA_DIR and os.path.exists(REAL_DATA_DIR):
        real_counts, _real_total = collect_real_token_counts(
            REAL_DATA_DIR, bin_edges, REAL_DATA_SAMPLE_SIZE, value_transform
        )
        real_freq = counts_to_freqs(real_counts)
    else:
        print("Real data not available, using uniform as baseline")
        real_freq = np.full(n_bins, 1.0 / n_bins)

    # Save CSV (bin edges in Wh space for readability)
    edges_wh = bin_edges_to_wh(bin_edges, value_transform)
    rows = []
    for w in sorted(week_freqs):
        for t in range(n_bins):
            rows.append({
                'week': w,
                'token_index': t,
                'bin_lower_wh': edges_wh[t],
                'bin_upper_wh': edges_wh[t + 1],
                'bin_width_wh': edges_wh[t + 1] - edges_wh[t],
                'synth_freq': week_freqs[w][t],
                'synth_count': week_counts[w][t],
                'real_freq': real_freq[t],
            })

    csv_path = os.path.join(OUTPUT_DIR, 'weekly_token_cascade.csv')
    pd.DataFrame(rows).to_csv(csv_path, index=False, float_format='%.6f')
    print(f"\nCSV saved to: {csv_path}")

    # Create plots
    create_weekly_frequency_plot(week_freqs, real_freq, bin_edges, OUTPUT_DIR)
    create_deviation_overlay_plot(week_freqs, real_freq, bin_edges, OUTPUT_DIR)
    create_metrics_by_week_plot(week_freqs, real_freq, bin_edges, OUTPUT_DIR, value_transform)

    # Generate report
    generate_report(week_freqs, real_freq, bin_edges, week_totals, OUTPUT_DIR, value_transform)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
