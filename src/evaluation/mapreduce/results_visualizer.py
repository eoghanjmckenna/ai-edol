"""
Results Visualizer for Parallel Evaluation Pipeline.

Generates visualizations and summary reports from parallel_evaluation_results.json.

Outputs (per evaluation run):
  evaluation/
    report.md              — structured fidelity evaluation report
    run_summary.json       — machine-readable single-row summary
    fidelity_flags.csv     — per-metric fidelity flags with NV references
    visualizations/
      {stem}_paper.png / {stem}_presentation.png  — dual-mode figures
      {stem}.csv                                   — companion data for every figure

Usage:
    visualizer = ResultsVisualizer(results_path, output_dir)
    visualizer.run_full_analysis()
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Shared style constants
try:
    from ..plot_style import (
        COLOUR_SYNTH, COLOUR_TRAIN, COLOUR_TEST, COLOUR_VAL,
        LABEL_SYNTH, LABEL_TRAIN, LABEL_TEST, LABEL_VAL,
        SPLIT_STYLES, STYLES, SDC_MIN_HOUSEHOLDS,
    )
except ImportError:
    # Fallback if called standalone
    COLOUR_SYNTH = "#0072B2"
    COLOUR_TRAIN = "#E69F00"
    COLOUR_TEST  = "#009E73"
    COLOUR_VAL   = "#CC79A7"
    LABEL_SYNTH = "Synthetic"
    LABEL_TRAIN = "Train"
    LABEL_TEST  = "Test"
    LABEL_VAL   = "Val"
    SPLIT_STYLES = {
        'synth': {'color': COLOUR_SYNTH, 'label': LABEL_SYNTH},
        'train': {'color': COLOUR_TRAIN, 'label': LABEL_TRAIN},
        'test':  {'color': COLOUR_TEST,  'label': LABEL_TEST},
        'val':   {'color': COLOUR_VAL,   'label': LABEL_VAL},
    }
    STYLES = {
        'paper': {
            'figsize_single': (6, 4), 'figsize_wide': (10, 4),
            'figsize_2x2': (10, 8), 'figsize_2x3': (12, 8),
            'dpi': 300,
            'rc': {'font.size': 9, 'lines.linewidth': 1.5,
                   'axes.titlesize': 9, 'axes.labelsize': 9, 'legend.fontsize': 8},
        },
        'presentation': {
            'figsize_single': (10, 6), 'figsize_wide': (16, 6),
            'figsize_2x2': (16, 10), 'figsize_2x3': (20, 12),
            'dpi': 150,
            'rc': {'font.size': 16, 'lines.linewidth': 2.5,
                   'axes.titlesize': 16, 'axes.labelsize': 14, 'legend.fontsize': 14},
        },
    }
    SDC_MIN_HOUSEHOLDS = 10

try:
    from .sdc_sanitiser import sanitise_file as _sdc_sanitise_file
except ImportError:
    _sdc_sanitise_file = None

def _rn(n: Optional[int]) -> Optional[int]:
    """Round a household count to the nearest 10 for SDC-compliant publication."""
    if n is None:
        return None
    return round(n / 10) * 10

# Optional matplotlib/seaborn imports
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, visualization will be skipped")

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


@dataclass
class FidelityFlag:
    """A fidelity assessment for a specific metric.

    For ratio metrics (mean_ratio, std_ratio, etc.), ``value`` is the raw ratio
    and ``natural_variation`` is in deviation-from-1.0 units.  Pass condition:
    |value − 1| ≤ natural_variation.

    For absolute metrics (RMSE, ACF diff, missingness %), ``value`` is the
    synth-train distance and ``natural_variation`` is in the same units.  Pass
    condition: value ≤ natural_variation.
    """
    metric: str
    status: str  # 'pass', 'warn', 'fail'
    value: float
    threshold: str
    message: str
    # Decision 047: real-vs-real empirical references (all in same units as deviation)
    ref_train_val: Optional[float] = None
    ref_train_test: Optional[float] = None
    ref_val_test: Optional[float] = None
    natural_variation: Optional[float] = None
    k: float = 3.0


class ResultsVisualizer:
    """
    Visualizer for parallel evaluation results.

    Reads results JSON and generates comprehensive visualizations,
    companion CSVs, and machine-readable summary files.
    """

    # Thresholds for fidelity assessment (legacy — used only for metrics without
    # real-vs-real references, e.g. correlation diffs, within_batch_variance)
    THRESHOLDS = {
        'mean_ratio': {'pass': (0.8, 1.2), 'warn': (0.5, 2.0)},
        'std_ratio': {'pass': (0.7, 1.3), 'warn': (0.5, 2.0)},
        'acf_diff': {'pass': 0.1, 'warn': 0.2},
        'profile_distance': {'pass': 50, 'warn': 100},
        'correlation_diff': {'pass': 0.1, 'warn': 0.2},
    }

    # Decision 047: tolerance multiplier k (warn zone = k × natural_variation)
    K_TOLERANCE: float = 3.0

    # ------------------------------------------------------------------ #
    # Decision 047: real-vs-real threshold helpers                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _ratio_dev(a: Optional[float], b: Optional[float]) -> Optional[float]:
        """Return |a/b − 1.0|, or None if inputs are invalid."""
        if a is None or b is None:
            return None
        try:
            fa, fb = float(a), float(b)
        except (TypeError, ValueError):
            return None
        if np.isnan(fa) or np.isnan(fb) or abs(fb) < 1e-10:
            return None
        return abs(fa / fb - 1.0)

    @staticmethod
    def _nv_from_refs(*refs: Optional[float]) -> Optional[float]:
        """Natural variation: max of non-None, non-NaN reference values."""
        valid = [float(r) for r in refs if r is not None and not np.isnan(r)]
        return float(max(valid)) if valid else None

    @staticmethod
    def _nv_status(synth_val: float, nv: Optional[float],
                   k: float = 3.0) -> str:
        """Apply Decision-047 threshold. Falls back to 'warn' when no reference."""
        if nv is None or np.isnan(nv):
            return 'warn'
        if synth_val <= nv:
            return 'pass'
        elif synth_val <= k * nv:
            return 'warn'
        return 'fail'

    @staticmethod
    def _nv_threshold_str(nv: Optional[float], k: float = 3.0) -> str:
        """Human-readable threshold description for a flag."""
        if nv is None:
            return 'data-driven (no reference)'
        return f'≤{nv:.4f} (pass), ≤{k * nv:.4f} (warn) [nat.var={nv:.4f}, k={k}]'

    def __init__(self, results_path: str, output_dir: str):
        """
        Initialize visualizer.

        Args:
            results_path: Path to parallel_evaluation_results.json
            output_dir: Directory for output plots and reports
        """
        self.results_path = Path(results_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load results
        with open(self.results_path) as f:
            self.results = json.load(f)

        self.flags: List[FidelityFlag] = []

        # Top-level household counts (from metadata)
        meta = self.results.get('metadata', {})
        self._n_households_synth = meta.get('n_households', 0)
        self._n_households_train = meta.get('n_households_train', 0)
        self._n_households_val = meta.get('n_households_val', 0)
        self._n_households_test = meta.get('n_households_test', 0)

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _save_in_modes(self, fig_factory, stem: str) -> None:
        """
        Render a figure in paper and presentation modes and save both.

        Args:
            fig_factory: Callable(style) -> Optional[Figure]. Called once per
                mode; should create a fresh figure using style['figsize_*'].
            stem: Filename stem (no suffix). Saved as {stem}_paper.png etc.
        """
        if not HAS_MATPLOTLIB:
            return
        for mode in ('paper', 'presentation'):
            style = STYLES[mode]
            with plt.rc_context(style['rc']):
                fig = fig_factory(style)
            if fig is None:
                continue
            fig.savefig(
                self.output_dir / f'{stem}_{mode}.png',
                dpi=style['dpi'],
                bbox_inches='tight',
            )
            plt.close(fig)

    def _save_companion_csv(self, rows: List[Dict], stem: str) -> Optional[str]:
        """Save companion CSV alongside figures. Returns path or None."""
        if not rows:
            return None
        df = pd.DataFrame(rows)
        path = self.output_dir / f'{stem}.csv'
        df.to_csv(path, index=False)
        return str(path)

    # ------------------------------------------------------------------ #
    # Main orchestration                                                   #
    # ------------------------------------------------------------------ #

    def run_full_analysis(self) -> Dict[str, Any]:
        """Run complete visualization and analysis pipeline."""
        print(f"[ResultsViz] Starting analysis of {self.results_path}")
        outputs = {}

        if HAS_MATPLOTLIB:
            # Marginal distribution
            csv_rows = self._extract_marginal_data()
            self._save_companion_csv(csv_rows, 'marginal_distribution_comparison')
            self._save_in_modes(self._render_marginal, 'marginal_distribution_comparison')
            outputs['marginal_plot'] = str(self.output_dir / 'marginal_distribution_comparison_paper.png')

            # Conditional distribution — one figure per fuel
            csv_rows = self._extract_conditional_data()
            self._save_companion_csv(csv_rows, 'conditional_distribution_comparison')
            for fuel in ['electricity', 'gas']:
                self._save_in_modes(
                    lambda style, f=fuel: self._render_conditional_fuel(style, f),
                    f'conditional_distribution_comparison_{fuel}',
                )
            outputs['conditional_plot'] = str(
                self.output_dir / 'conditional_distribution_comparison_electricity_paper.png')

            # ACF comparison
            csv_rows = self._extract_acf_data()
            self._save_companion_csv(csv_rows, 'acf_comparison')
            self._save_in_modes(self._render_acf, 'acf_comparison')
            outputs['acf_plot'] = str(self.output_dir / 'acf_comparison_paper.png')

            # Daily profiles
            csv_rows = self._extract_daily_profiles_data()
            self._save_companion_csv(csv_rows, 'daily_profiles_comparison')
            self._save_in_modes(self._render_daily_profiles, 'daily_profiles_comparison')
            outputs['profile_plot'] = str(self.output_dir / 'daily_profiles_comparison_paper.png')

            # Volatility analysis
            if self.results.get('temporal_patterns', {}).get('volatility'):
                self._save_companion_csv(self._extract_volatility_spike_data(), 'volatility_analysis_spike')
                self._save_companion_csv(self._extract_volatility_hist_data(),  'volatility_analysis_hist')
                self._save_in_modes(self._render_volatility, 'volatility_analysis')
                outputs['volatility_plot'] = str(self.output_dir / 'volatility_analysis_paper.png')

            # PTG comparison
            csv_rows = self._extract_ptg_data()
            self._save_companion_csv(csv_rows, 'ptg_comparison')
            self._save_in_modes(self._render_ptg, 'ptg_comparison')
            outputs['ptg_plot'] = str(self.output_dir / 'ptg_comparison_paper.png')

            # Integrated diagnostics
            if self.results.get('seasonal_analysis'):
                csv_rows = self._extract_seasonal_data()
                self._save_companion_csv(csv_rows, 'seasonal_analysis')
                self._save_in_modes(self._render_seasonal, 'seasonal_analysis')
                outputs['seasonal_plot'] = str(self.output_dir / 'seasonal_analysis_paper.png')

            if self.results.get('daily_pattern_diagnostics'):
                outputs['heatmap_plots'] = self._plot_daily_pattern_heatmaps()
                if 'weekly_degradation' in self.results.get('daily_pattern_diagnostics', {}):
                    outputs['degradation_plots'] = self._plot_daily_pattern_degradation()

            if self.results.get('household_month_fidelity'):
                outputs['hh_month_plots'] = self._plot_household_month_fidelity()

            if self.results.get('token_analysis'):
                token = self.results['token_analysis']
                for fuel in ['electricity', 'gas']:
                    fuel_data = token.get(fuel, {})
                    if fuel_data.get('frequency'):
                        outputs[f'token_freq_plot_{fuel}'] = self._plot_token_frequency(fuel)
                outputs['token_dist_plot'] = self._plot_token_distribution()

        # Diagnostic data CSVs (legacy diagnostics_data/ location)
        self._save_diagnostics_csvs()

        # Fidelity assessment
        self._assess_fidelity()

        # Save fidelity flags CSV (with household counts)
        outputs['flags_csv'] = self._save_flags_csv()

        # Markdown report (replaces fidelity_report.txt)
        outputs['report_md'] = self._generate_report_md()

        # Machine-readable run summary
        outputs['run_summary'] = self._save_run_summary()

        # SDC-compliant results JSON (safe to export from TRE)
        if _sdc_sanitise_file is not None:
            sdc_path = self.results_path.parent / 'parallel_evaluation_results_sdc.json'
            try:
                _sdc_sanitise_file(self.results_path, sdc_path)
                outputs['sdc_results'] = str(sdc_path)
                print(f"[ResultsViz] SDC-compliant results written to {sdc_path}")
            except Exception as e:
                print(f"[ResultsViz] WARNING: SDC sanitiser failed: {e}")

        print(f"[ResultsViz] Analysis complete. Outputs in {self.output_dir}")
        return outputs

    # ------------------------------------------------------------------ #
    # Data extraction (companion CSV sources)                             #
    # ------------------------------------------------------------------ #

    def _extract_marginal_data(self) -> List[Dict]:
        """Extract marginal distribution data for companion CSV."""
        marginal = self.results.get('marginal_distribution', {})
        rows = []
        for fuel in ['electricity', 'gas']:
            if fuel not in marginal or marginal[fuel].get('status') == 'no_data':
                continue
            data = marginal[fuel]
            for split, n_hh in [
                ('synth',  self._n_households_synth),
                ('train',  _rn(self._n_households_train)),
                ('val',    _rn(self._n_households_val)),
                ('test',   _rn(self._n_households_test)),
            ]:
                mean_val = data.get(f'{split}_mean')
                if mean_val is None:
                    continue
                row = {
                    'fuel': fuel,
                    'split': split,
                    'mean': mean_val,
                    'std': data.get(f'{split}_std'),
                    'n_obs': data.get(f'{split}_n', 0),
                    'n_households': n_hh,
                }
                q_data = data.get(f'{split}_quantiles', {})
                for q_key in ['p5', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95']:
                    row[q_key] = q_data.get(q_key)
                rows.append(row)
        return rows

    def _extract_conditional_data(self) -> List[Dict]:
        """Extract conditional distribution data for companion CSV (with SDC suppression)."""
        conditional = self.results.get('conditional_distribution', {})
        rows = []
        for cond_var, values in conditional.items():
            for cond_value, fuels in values.items():
                for fuel, stats in fuels.items():
                    n_hh_train = stats.get('n_households_train', 0)
                    n_hh_val   = stats.get('n_households_val', 0)
                    n_hh_test  = stats.get('n_households_test', 0)
                    suppressed = (n_hh_train > 0 and n_hh_train < SDC_MIN_HOUSEHOLDS) or \
                                 (n_hh_test > 0 and n_hh_test < SDC_MIN_HOUSEHOLDS)
                    row = {
                        'fuel': fuel,
                        'cond_var': cond_var,
                        'cond_value': cond_value,
                        'n_households_synth': stats.get('n_batches', 0),
                        'n_households_train': _rn(n_hh_train) if not suppressed else None,
                        'n_households_val':   _rn(n_hh_val)   if not suppressed else None,
                        'n_households_test':  _rn(n_hh_test)  if not suppressed else None,
                        'suppressed': suppressed,
                    }
                    if not suppressed:
                        row.update({
                            'synth_mean': stats.get('synth_mean'),
                            'synth_std': stats.get('synth_std'),
                            'train_mean': stats.get('train_mean'),
                            'train_std': stats.get('train_std'),
                            'train_ratio': stats.get('train_ratio'),
                            'val_mean': stats.get('val_mean'),
                            'val_std': stats.get('val_std'),
                            'val_ratio': stats.get('val_ratio'),
                            'test_mean': stats.get('test_mean'),
                            'test_std': stats.get('test_std'),
                            'test_ratio': stats.get('test_ratio'),
                        })
                    else:
                        row.update({
                            'synth_mean': None, 'synth_std': None,
                            'train_mean': None, 'train_std': None, 'train_ratio': None,
                            'val_mean':   None, 'val_std':   None, 'val_ratio':   None,
                            'test_mean':  None, 'test_std':  None, 'test_ratio':  None,
                        })
                    rows.append(row)
        return rows

    def _extract_acf_data(self) -> List[Dict]:
        """Extract ACF comparison data for companion CSV."""
        temporal = self.results.get('temporal_patterns', {})
        acf = temporal.get('acf', {})
        rows = []
        for fuel in ['electricity', 'gas']:
            if fuel not in acf:
                continue
            fuel_acf = acf[fuel]
            for lag, lag_data in fuel_acf.items():
                rows.append({
                    'fuel': fuel,
                    'lag': int(lag),
                    'synth': lag_data.get('synth'),
                    'train': lag_data.get('train'),
                    'val': lag_data.get('val'),
                    'test': lag_data.get('test'),
                    'train_diff': lag_data.get('train_diff'),
                    'n_households_synth': self._n_households_synth,
                    'n_households_train': _rn(self._n_households_train),
                    'n_households_val':   _rn(self._n_households_val),
                    'n_households_test':  _rn(self._n_households_test),
                })
        return rows

    def _extract_daily_profiles_data(self) -> List[Dict]:
        """Extract daily profiles comparison data for companion CSV."""
        temporal = self.results.get('temporal_patterns', {})
        profiles = temporal.get('daily_profiles', {})
        rows = []
        for fuel in ['electricity', 'gas']:
            for profile_type in ['all', 'weekday', 'weekend']:
                key = f'{fuel}_{profile_type}'
                if key not in profiles:
                    continue
                data = profiles[key]
                synth_profile = data.get('synth', [])
                train_profile = data.get('train') or []
                val_profile = data.get('val') or []
                test_profile = data.get('test') or []
                synth_median = data.get('synth_median') or []
                train_median = data.get('train_median') or []
                for hh in range(48):
                    rows.append({
                        'fuel': fuel,
                        'profile_type': profile_type,
                        'halfhour': hh + 1,
                        'synth_mean': synth_profile[hh] if hh < len(synth_profile) else None,
                        'train_mean': train_profile[hh] if hh < len(train_profile) else None,
                        'val_mean': val_profile[hh] if hh < len(val_profile) else None,
                        'test_mean': test_profile[hh] if hh < len(test_profile) else None,
                        'synth_median': synth_median[hh] if hh < len(synth_median) else None,
                        'train_median': train_median[hh] if hh < len(train_median) else None,
                        'n_households_synth': self._n_households_synth,
                        'n_households_train': _rn(self._n_households_train),
                        'n_households_val':   _rn(self._n_households_val),
                        'n_households_test':  _rn(self._n_households_test),
                    })
        return rows

    def _extract_volatility_spike_data(self) -> List[Dict]:
        """Extract |Δx| quantile data for companion spike CSV (p50–p90, no p95)."""
        volatility = self.results.get('temporal_patterns', {}).get('volatility', {})
        rows = []
        for fuel in ['electricity', 'gas']:
            if fuel not in volatility:
                continue
            data = volatility[fuel]
            for q_key in ['p50', 'p75', 'p90']:
                rows.append({
                    'fuel': fuel,
                    'quantile': q_key,
                    'synth': data.get('synth_abs_quantiles', {}).get(q_key),
                    'train': data.get('train_abs_quantiles', {}).get(q_key),
                    'test': data.get('test_abs_quantiles', {}).get(q_key),
                    'n_households_synth': self._n_households_synth,
                    'n_households_train': _rn(self._n_households_train),
                    'n_households_test':  _rn(self._n_households_test),
                })
        return rows

    def _extract_volatility_hist_data(self) -> List[Dict]:
        """Extract binned Δx histogram data for companion hist CSV.

        Uses the same bins as the plot. Bins where train_count < 10 are suppressed.
        """
        SDC_MIN_BIN_COUNT = 10
        volatility = self.results.get('temporal_patterns', {}).get('volatility', {})
        rows = []
        for fuel in ['electricity', 'gas']:
            data = volatility.get(fuel, {})
            synth_samples = data.get('synth_diff_samples', [])
            train_samples = data.get('train_diff_samples', [])
            test_samples  = data.get('test_diff_samples',  [])
            if not synth_samples:
                continue
            synth_arr = np.array(synth_samples)
            ref = synth_arr if len(synth_arr) else np.array([0])
            max_abs = float(np.percentile(np.abs(ref), 99)) if len(ref) > 0 else 1.0
            bins = np.linspace(-max_abs, max_abs, 80)

            synth_counts, _ = np.histogram(synth_arr, bins=bins)
            train_counts, _ = np.histogram(np.array(train_samples), bins=bins) if train_samples else (np.zeros(len(bins)-1, dtype=int), None)
            test_counts,  _ = np.histogram(np.array(test_samples),  bins=bins) if test_samples  else (np.zeros(len(bins)-1, dtype=int), None)

            synth_total = synth_counts.sum()
            train_total = train_counts.sum()
            test_total  = test_counts.sum()

            for i in range(len(bins) - 1):
                tc = int(train_counts[i])
                suppressed = tc < SDC_MIN_BIN_COUNT
                row = {
                    'fuel':               fuel,
                    'bin_left':           float(bins[i]),
                    'bin_right':          float(bins[i + 1]),
                    'bin_centre':         float((bins[i] + bins[i + 1]) / 2),
                    'n_households_synth': self._n_households_synth,
                    'n_households_train': _rn(self._n_households_train),
                    'n_households_test':  _rn(self._n_households_test),
                    'suppressed':         suppressed,
                    'synth_count':        int(synth_counts[i]) if not suppressed else None,
                    'synth_density':      float(synth_counts[i] / (synth_total * (bins[i+1] - bins[i]))) if not suppressed and synth_total > 0 else None,
                    'train_count':        _rn(tc) if not suppressed else None,
                    'train_density':      float(tc / (train_total * (bins[i+1] - bins[i]))) if not suppressed and train_total > 0 else None,
                    'test_count':         _rn(int(test_counts[i])) if not suppressed else None,
                    'test_density':       float(test_counts[i] / (test_total * (bins[i+1] - bins[i]))) if not suppressed and test_total > 0 else None,
                }
                rows.append(row)
        return rows

    def _extract_ptg_data(self) -> List[Dict]:
        """Extract PTG data for companion CSV."""
        ptg = self.results.get('ptg_analysis', {})
        rows = []
        for split_name, split_data in ptg.items():
            for fuel in ['electricity', 'gas']:
                fuel_data = split_data.get(fuel, {})
                months = fuel_data.get('months', [])
                temps = fuel_data.get('mean_temp', [])
                consump = fuel_data.get('mean_consumption', [])
                n_hh = {
                    'synth': self._n_households_synth,
                    'train': _rn(self._n_households_train),
                    'val':   _rn(self._n_households_val),
                    'test':  _rn(self._n_households_test),
                }.get(split_name, 0)
                for m, t, c in zip(months, temps, consump):
                    rows.append({
                        'fuel': fuel,
                        'split': split_name,
                        'month': m,
                        'mean_temperature': t,
                        'mean_consumption': c,
                        'n_households': n_hh,
                    })
        return rows

    def _extract_seasonal_data(self) -> List[Dict]:
        """Extract seasonal analysis data for companion CSV."""
        seasonal = self.results.get('seasonal_analysis', {})
        rows = []
        for fuel, fuel_data in seasonal.items():
            for split, monthly in fuel_data.items():
                if not isinstance(monthly, dict):
                    continue
                n_hh = {
                    'synth': self._n_households_synth,
                    'train': _rn(self._n_households_train),
                    'test':  _rn(self._n_households_test),
                }.get(split, 0)
                for month, stats in monthly.items():
                    if not isinstance(stats, dict):
                        continue
                    rows.append({
                        'fuel': fuel,
                        'split': split,
                        'month': month,
                        'mean': stats.get('mean'),
                        'median': stats.get('median'),
                        'std': stats.get('std'),
                        'ci_lower': stats.get('ci_lower'),
                        'ci_upper': stats.get('ci_upper'),
                        'n_households': n_hh,
                    })
        return rows

    # ------------------------------------------------------------------ #
    # Render functions (create fresh Figure per mode)                     #
    # ------------------------------------------------------------------ #

    def _render_marginal(self, style: Dict) -> Optional['plt.Figure']:
        """Render marginal distribution comparison."""
        marginal = self.results.get('marginal_distribution', {})
        if not marginal:
            return None

        figsize = style.get('figsize_2x2', (10, 8))
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        split_colors = {
            'synth': COLOUR_SYNTH,
            'train': COLOUR_TRAIN,
            'val':   COLOUR_VAL,
            'test':  COLOUR_TEST,
        }
        split_labels = {
            'synth': LABEL_SYNTH, 'train': LABEL_TRAIN,
            'val': LABEL_VAL, 'test': LABEL_TEST,
        }

        for idx, fuel in enumerate(['electricity', 'gas']):
            if fuel not in marginal:
                continue
            data = marginal[fuel]

            # Left: mean comparison bar chart
            ax1 = axes[idx, 0]
            means, ci_errors, labels, colors = [], [], [], []
            for prefix in ['synth', 'train', 'val', 'test']:
                mean_val = data.get(f'{prefix}_mean')
                if mean_val is not None and not (isinstance(mean_val, float) and np.isnan(mean_val)):
                    means.append(mean_val)
                    labels.append(split_labels[prefix])
                    colors.append(split_colors[prefix])
                    std_val = data.get(f'{prefix}_std', 0)
                    n_val = data.get(f'{prefix}_n', 0)
                    if std_val and n_val and n_val > 0:
                        ci_errors.append(1.96 * std_val / np.sqrt(n_val))
                    else:
                        ci_errors.append(0)

            if means:
                ax1.bar(labels, means, color=colors,
                        yerr=ci_errors, capsize=4, error_kw={'linewidth': 1.5})
                ax1.set_ylabel('Mean (Wh)')
                ax1.set_title(f'{fuel.title()} — Mean Comparison')
                if len(means) > 1:
                    ratio = means[0] / means[1] if means[1] != 0 else np.nan
                    ax1.annotate(f'Ratio: {ratio:.2f}', xy=(0.5, 0.95),
                                 xycoords='axes fraction', ha='center')

            # Right: quantile comparison
            ax2 = axes[idx, 1]
            synth_q = data.get('synth_quantiles', {})
            train_q = data.get('train_quantiles', {})
            val_q   = data.get('val_quantiles', {})
            test_q  = data.get('test_quantiles', {})

            if synth_q or train_q:
                q_labels = sorted(
                    set(list(synth_q.keys()) + list(train_q.keys())),
                    key=lambda q: float(q.replace('p', ''))
                    if q.replace('p', '').replace('.', '').isdigit() else q
                )
                x = np.arange(len(q_labels))
                width = 0.2
                if synth_q:
                    ax2.bar(x - 1.5 * width, [synth_q.get(q, np.nan) for q in q_labels],
                            width, label=LABEL_SYNTH, color=COLOUR_SYNTH)
                if train_q:
                    ax2.bar(x - 0.5 * width, [train_q.get(q, np.nan) for q in q_labels],
                            width, label=LABEL_TRAIN, color=COLOUR_TRAIN)
                if val_q:
                    ax2.bar(x + 0.5 * width, [val_q.get(q, np.nan) for q in q_labels],
                            width, label=LABEL_VAL, color=COLOUR_VAL)
                if test_q:
                    ax2.bar(x + 1.5 * width, [test_q.get(q, np.nan) for q in q_labels],
                            width, label=LABEL_TEST, color=COLOUR_TEST)
                ax2.set_xticks(x)
                ax2.set_xticklabels(q_labels)
                ax2.set_ylabel('Value (Wh)')
                ax2.set_title(f'{fuel.title()} — Quantile Comparison')
                ax2.legend()

        fig.tight_layout()
        return fig

    def _render_conditional_fuel(self, style: Dict, fuel: str) -> Optional['plt.Figure']:
        """Render conditional distribution grouped bar charts for one fuel."""
        conditional = self.results.get('conditional_distribution', {})
        if not conditional:
            return None

        COND_VAR_LABELS = {
            'building_type': 'Building type', 'num_rooms': 'Num. rooms',
            'num_occs': 'Num. occupants', 'region': 'Region', 'age_built': 'Year built',
        }

        data_rows = []
        for cond_var, values in conditional.items():
            for cond_value, fuels in values.items():
                stats = fuels.get(fuel)
                if stats is None:
                    continue
                n_hh_train = stats.get('n_households_train', 0)
                suppressed = n_hh_train > 0 and n_hh_train < SDC_MIN_HOUSEHOLDS
                synth_mean = stats.get('synth_mean')
                if synth_mean is not None and not suppressed:
                    data_rows.append({
                        'variable': cond_var, 'category': str(cond_value),
                        'synth_mean': synth_mean,
                        'synth_std': stats.get('synth_std'),
                        'synth_n': stats.get('synth_n', 0),
                        'train_mean': stats.get('train_mean'),
                        'train_std': stats.get('train_std'),
                        'train_n': stats.get('train_n', 0),
                        'val_mean': stats.get('val_mean'),
                        'val_std': stats.get('val_std'),
                        'val_n': stats.get('val_n', 0),
                        'test_mean': stats.get('test_mean'),
                        'test_std': stats.get('test_std'),
                        'test_n': stats.get('test_n', 0),
                    })

        if not data_rows:
            return None

        df = pd.DataFrame(data_rows)
        cond_vars = df['variable'].unique()
        n_vars = len(cond_vars)
        n_cols = 2
        n_rows = (n_vars + 1) // 2

        base = style.get('figsize_2x2', (10, 8))
        figsize = (base[0], base[1] / 2 * n_rows)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for i, cond_var in enumerate(cond_vars):
            ax = axes[i // 2, i % 2]
            var_df = df[df['variable'] == cond_var].copy()
            if len(var_df) == 0:
                ax.set_visible(False)
                continue

            categories = sorted(var_df['category'].tolist(), key=str)
            var_df = var_df.set_index('category').reindex(categories).reset_index()
            x = np.arange(len(categories))

            def _ci(var_df, mean_col, std_col, n_col):
                errs = []
                for _, row in var_df.iterrows():
                    m, s, n = row.get(mean_col), row.get(std_col), row.get(n_col)
                    if m is not None and s is not None and n and n > 0:
                        errs.append(1.96 * s / np.sqrt(n))
                    else:
                        errs.append(0)
                return errs

            splits = [
                ('synth_mean', 'synth_std', 'synth_n', LABEL_SYNTH, COLOUR_SYNTH),
                ('train_mean', 'train_std', 'train_n', LABEL_TRAIN, COLOUR_TRAIN),
                ('val_mean',   'val_std',   'val_n',   LABEL_VAL,   COLOUR_VAL),
                ('test_mean',  'test_std',  'test_n',  LABEL_TEST,  COLOUR_TEST),
            ]
            # Only include splits with at least one non-None value
            active = [(mc, sc, nc, lbl, col) for mc, sc, nc, lbl, col in splits
                      if any(v is not None and not (isinstance(v, float) and np.isnan(v))
                             for v in var_df[mc].values)]

            n_splits = len(active)
            width = 0.8 / n_splits
            offsets = np.linspace(-(n_splits - 1) / 2, (n_splits - 1) / 2, n_splits) * width

            for offset, (mc, sc, nc, lbl, col) in zip(offsets, active):
                vals = [v if (v is not None and not (isinstance(v, float) and np.isnan(v))) else 0
                        for v in var_df[mc].values]
                ax.bar(x + offset, vals, width, label=lbl, color=col, alpha=0.8,
                       yerr=_ci(var_df, mc, sc, nc), capsize=3, error_kw={'linewidth': 1})

            ax.set_xticks(x)
            ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=8)
            ax.set_ylabel('Mean (Wh)')
            var_label = COND_VAR_LABELS.get(cond_var, cond_var.replace('_', ' ').title())
            ax.set_title(f'{fuel.title()} by {var_label}')
            if i == 0:
                ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')

        for i in range(n_vars, n_rows * n_cols):
            axes[i // 2, i % 2].set_visible(False)

        fig.suptitle(f'{fuel.title()} Conditional Distribution Comparison', fontweight='bold')
        fig.tight_layout()
        return fig

    def _render_acf(self, style: Dict) -> Optional['plt.Figure']:
        """Render ACF comparison bar chart."""
        temporal = self.results.get('temporal_patterns', {})
        acf = temporal.get('acf', {})
        if not acf:
            return None

        figsize = style.get('figsize_wide', (10, 4))
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        for idx, fuel in enumerate(['electricity', 'gas']):
            ax = axes[idx]
            if fuel not in acf:
                ax.set_title(f'{fuel.title()} — No ACF data')
                continue

            fuel_acf = acf[fuel]
            lags = sorted([int(k) for k in fuel_acf.keys()])
            synth_vals, train_vals, val_vals, test_vals = [], [], [], []
            for lag in lags:
                lag_data = fuel_acf[str(lag)]
                synth_vals.append(lag_data.get('synth', np.nan))
                train_vals.append(lag_data.get('train', np.nan))
                val_vals.append(lag_data.get('val', np.nan))
                test_vals.append(lag_data.get('test', np.nan))

            x = np.arange(len(lags))
            width = 0.2
            ax.bar(x - 1.5 * width, synth_vals, width, label=LABEL_SYNTH, color=COLOUR_SYNTH)
            ax.bar(x - 0.5 * width, train_vals, width, label=LABEL_TRAIN, color=COLOUR_TRAIN)
            if any(v is not None and not (isinstance(v, float) and np.isnan(v)) for v in val_vals):
                ax.bar(x + 0.5 * width, val_vals, width, label=LABEL_VAL, color=COLOUR_VAL)
            if any(v is not None and not (isinstance(v, float) and np.isnan(v)) for v in test_vals):
                ax.bar(x + 1.5 * width, test_vals, width, label=LABEL_TEST, color=COLOUR_TEST)

            ax.set_xticks(x)
            ax.set_xticklabels([str(l) for l in lags])
            ax.set_xlabel('Lag (half-hours)')
            ax.set_ylabel('ACF')
            ax.set_title(f'{fuel.title()} — Autocorrelation Comparison')
            ax.legend()
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        fig.tight_layout()
        return fig

    def _render_daily_profiles(self, style: Dict) -> Optional['plt.Figure']:
        """Render daily profile overlays."""
        temporal = self.results.get('temporal_patterns', {})
        profiles = temporal.get('daily_profiles', {})
        if not profiles:
            return None

        figsize = style.get('figsize_2x3', (12, 8))
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        half_hours = np.arange(48)

        for row_idx, fuel in enumerate(['electricity', 'gas']):
            for col_idx, profile_type in enumerate(['all', 'weekday', 'weekend']):
                ax = axes[row_idx, col_idx]
                key = f'{fuel}_{profile_type}'
                if key not in profiles:
                    ax.set_title(f'{fuel.title()} — {profile_type} (No data)')
                    continue

                data = profiles[key]
                synth_p = data.get('synth', [])
                train_p = data.get('train', [])
                val_p   = data.get('val', [])
                test_p  = data.get('test', [])

                from matplotlib.lines import Line2D

                if fuel == 'electricity':
                    for split_name, profile, color in [
                        ('synth', synth_p, COLOUR_SYNTH),
                        ('train', train_p, COLOUR_TRAIN),
                        ('val',   val_p,   COLOUR_VAL),
                        ('test',  test_p,  COLOUR_TEST),
                    ]:
                        if profile:
                            ax.plot(half_hours, profile, '-', color=color, linewidth=1.5)
                        median_p = data.get(f'{split_name}_median', [])
                        if median_p:
                            ax.plot(half_hours, median_p, '--', color=color,
                                    linewidth=1.2, alpha=0.7)

                    legend_handles = []
                    for split_name, color, label in [
                        ('synth', COLOUR_SYNTH, LABEL_SYNTH),
                        ('train', COLOUR_TRAIN, LABEL_TRAIN),
                        ('val',   COLOUR_VAL,   LABEL_VAL),
                        ('test',  COLOUR_TEST,  LABEL_TEST),
                    ]:
                        p = data.get(split_name, []) or data.get(f'{split_name}_median', [])
                        if p:
                            legend_handles.append(
                                Line2D([0], [0], color=color, linewidth=2, label=label))
                    legend_handles.append(
                        Line2D([0], [0], color='gray', linestyle='-', linewidth=2, label='Mean'))
                    legend_handles.append(
                        Line2D([0], [0], color='gray', linestyle='--', linewidth=1.5, label='Median'))
                    ax.legend(handles=legend_handles, fontsize=7)
                else:
                    for profile, color, label in [
                        (synth_p, COLOUR_SYNTH, LABEL_SYNTH),
                        (train_p, COLOUR_TRAIN, LABEL_TRAIN),
                        (val_p,   COLOUR_VAL,   LABEL_VAL),
                        (test_p,  COLOUR_TEST,  LABEL_TEST),
                    ]:
                        if profile:
                            ax.plot(half_hours, profile, '-', color=color,
                                    label=f'{label} (mean)', linewidth=1.5)
                    if synth_p or train_p:
                        ax.legend(fontsize=7)

                ax.set_xlabel('Time of day')
                ax.set_ylabel('Consumption (Wh)')
                ax.set_title(f'{fuel.title()} — {profile_type.title()}')
                ax.set_xticks([0, 12, 24, 36, 47])
                ax.set_xticklabels(['00:00', '06:00', '12:00', '18:00', '23:30'])

                train_dist = data.get('train_distance') or data.get('distance')
                if train_dist is not None:
                    ax.annotate(f'RMSE: {train_dist:.1f}', xy=(0.02, 0.98),
                                xycoords='axes fraction', ha='left', va='top', fontsize=9,
                                bbox=dict(boxstyle='round', facecolor='wheat'))

        fig.tight_layout()
        return fig

    def _render_ptg(self, style: Dict) -> Optional['plt.Figure']:
        """Render Power-Temperature Gradient comparison."""
        ptg = self.results.get('ptg_analysis', {})
        if not ptg:
            return None

        split_colors = {
            'synth': (COLOUR_SYNTH, 'o'),
            'train': (COLOUR_TRAIN, 's'),
            'val':   (COLOUR_VAL,   '^'),
            'test':  (COLOUR_TEST,  'D'),
        }
        breakpoint_temp = 15.5
        figsize = style.get('figsize_wide', (10, 4))
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        for idx, fuel in enumerate(['electricity', 'gas']):
            ax = axes[idx]
            for split_name, (color, marker) in split_colors.items():
                split_data = ptg.get(split_name, {})
                fuel_data = split_data.get(fuel, {})
                temps = np.array(fuel_data.get('mean_temp', []))
                consump = np.array(fuel_data.get('mean_consumption', []))
                if len(temps) == 0 or len(consump) == 0:
                    continue

                heating_gradient = None
                for mask in [temps <= breakpoint_temp, temps > breakpoint_temp]:
                    if np.sum(mask) >= 2:
                        coeffs = np.polyfit(temps[mask], consump[mask], 1)
                        t_line = np.linspace(temps[mask].min(), temps[mask].max(), 50)
                        ax.plot(t_line, np.polyval(coeffs, t_line), color=color,
                                linewidth=1.5, alpha=0.7)
                        if np.sum(mask) >= 2 and not np.any(temps[mask] > breakpoint_temp):
                            heating_gradient = coeffs[0]

                # Recompute heating gradient clearly
                below = temps <= breakpoint_temp
                if np.sum(below) >= 2:
                    coeffs_h = np.polyfit(temps[below], consump[below], 1)
                    heating_gradient = coeffs_h[0]

                label = split_name.title()
                if heating_gradient is not None:
                    label += f' ({heating_gradient:+.0f} Wh/°C)'
                ax.scatter(temps, consump, c=color, marker=marker, s=60,
                           label=label, zorder=3, edgecolors='white', linewidths=0.5)

            ax.axvline(x=breakpoint_temp, color='gray', linestyle='--', alpha=0.5, label='15.5°C')
            ax.set_xlabel('Monthly Mean Temperature (°C)')
            ax.set_ylabel('Monthly Mean Consumption (Wh)')
            ax.set_title(f'{fuel.title()} — Power-Temperature Gradient')
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)

        fig.tight_layout()
        return fig

    def _render_volatility(self, style: Dict) -> Optional['plt.Figure']:
        """Render first-difference distributions and |Δx| quantile comparison."""
        volatility = self.results.get('temporal_patterns', {}).get('volatility', {})
        fuels = [f for f in ['electricity', 'gas'] if f in volatility
                 and volatility[f].get('synth_diff_samples')]
        if not fuels:
            return None

        n_cols = len(fuels)
        figsize = style.get('figsize_2x2', (10, 8))
        # Scale width by number of fuels
        figsize = (figsize[0] * n_cols / 2, figsize[1])
        fig, axes = plt.subplots(2, n_cols, figsize=figsize)
        if n_cols == 1:
            axes = axes.reshape(2, 1)

        split_styles = {
            'synth': (COLOUR_SYNTH, LABEL_SYNTH),
            'train': (COLOUR_TRAIN, LABEL_TRAIN),
            'test':  (COLOUR_TEST,  LABEL_TEST),
        }

        for col_idx, fuel in enumerate(fuels):
            data = volatility[fuel]
            arrays = {
                split: np.array(data.get(f'{split}_diff_samples', []))
                for split in ['synth', 'train', 'test']
            }

            # Top: histogram of Δx
            ax = axes[0, col_idx]
            ref = arrays['synth'] if len(arrays['synth']) else np.array([0])
            max_abs = float(np.percentile(np.abs(ref), 99)) if len(ref) > 0 else 1.0
            bins = np.linspace(-max_abs, max_abs, 80)
            for split, (color, label) in split_styles.items():
                arr = arrays[split]
                if len(arr) == 0:
                    continue
                kurt = data.get(f'{split}_kurtosis')
                kurt_str = f', kurt={kurt:.2f}' if kurt is not None else ''
                ax.hist(arr, bins=bins, density=True, alpha=0.45, color=color,
                        label=f'{label} (std={np.std(arr):.1f}{kurt_str})')
            ax.set_xlabel('Δx (Wh)')
            ax.set_ylabel('Density')
            ax.set_title(f'{fuel.title()} First Differences')
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

            # Bottom: |Δx| quantile bar chart (p50–p90 only)
            ax = axes[1, col_idx]
            q_keys = ['p50', 'p75', 'p90']
            x = np.arange(len(q_keys))
            width = 0.25
            offset = -width
            plotted = False
            for split, (color, label) in split_styles.items():
                q_data = data.get(f'{split}_abs_quantiles', {})
                if not q_data:
                    continue
                vals = [q_data.get(k, np.nan) for k in q_keys]
                ax.bar(x + offset, vals, width, label=label, color=color, alpha=0.8)
                offset += width
                plotted = True
            if plotted:
                ax.set_xticks(x)
                ax.set_xticklabels(['P50', 'P75', 'P90'])
                ax.set_ylabel('|Δx| (Wh)')
                ax.set_title(f'{fuel.title()} Spike Magnitude (|Δx| quantiles)')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3, axis='y')

        fig.suptitle('Volatility Analysis (First Differences)', fontweight='bold')
        fig.tight_layout()
        return fig

    def _render_seasonal(self, style: Dict) -> Optional['plt.Figure']:
        """Render monthly mean trends with 95% CI bands."""
        seasonal = self.results.get('seasonal_analysis', {})
        if not seasonal:
            return None

        fuels = [f for f in ['electricity', 'gas'] if f in seasonal]
        if not fuels:
            return None

        figsize_base = style.get('figsize_wide', (10, 4))
        figsize = (figsize_base[0] / 2 * len(fuels), figsize_base[1])
        fig, axes = plt.subplots(1, len(fuels), figsize=figsize)
        if len(fuels) == 1:
            axes = [axes]

        split_colors = {
            'synth': COLOUR_SYNTH,
            'train': COLOUR_TRAIN,
            'test':  COLOUR_TEST,
        }
        split_labels = {
            'synth': LABEL_SYNTH, 'train': LABEL_TRAIN, 'test': LABEL_TEST,
        }

        for idx, fuel in enumerate(fuels):
            ax = axes[idx]
            fuel_data = seasonal[fuel]
            for split in ['synth', 'train', 'test']:
                if split not in fuel_data:
                    continue
                monthly = fuel_data[split]
                color = split_colors[split]
                label = split_labels[split]
                months = sorted([int(m) for m in monthly.keys()])
                means = [monthly.get(str(m), monthly.get(m, {})).get('mean', np.nan) for m in months]
                ci_lo = [monthly.get(str(m), monthly.get(m, {})).get('ci_lower', np.nan) for m in months]
                ci_hi = [monthly.get(str(m), monthly.get(m, {})).get('ci_upper', np.nan) for m in months]
                ax.plot(months, means, 'o-', color=color, label=f'{label} mean', markersize=4)
                ax.fill_between(months, ci_lo, ci_hi, alpha=0.2, color=color)
                if fuel == 'electricity':
                    medians = [monthly.get(str(m), monthly.get(m, {})).get('median', np.nan) for m in months]
                    ax.plot(months, medians, 's--', color=color,
                            label=f'{label} median', markersize=3, alpha=0.7)

            ax.set_title(f'{fuel.title()} — Seasonal Analysis')
            ax.set_xlabel('Month')
            ax.set_ylabel('Mean Wh')
            ax.legend(fontsize=8)
            ax.set_xticks(range(1, 13))
            ax.grid(True, alpha=0.3)

        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------ #
    # Fidelity assessment                                                  #
    # ------------------------------------------------------------------ #

    def _assess_fidelity(self) -> None:
        """Assess fidelity metrics using Decision 047 real-vs-real empirical thresholds.

        For each metric, three real-real reference values are computed (train-val,
        train-test, val-test). natural_variation = max of these; pass if synth ≤ nv,
        warn if synth ≤ k×nv, fail otherwise.  k = K_TOLERANCE (3.0).

        Metrics without a real-vs-real analogue (correlation diffs, within_batch_variance)
        retain fixed thresholds.
        """
        self.flags = []
        k = self.K_TOLERANCE

        marginal = self.results.get('marginal_distribution', {})
        temporal = self.results.get('temporal_patterns', {})

        # ------------------------------------------------------------------ #
        # 1. Mean ratio (per fuel)                                             #
        # ------------------------------------------------------------------ #
        for fuel in ['electricity', 'gas']:
            if fuel not in marginal:
                continue
            data = marginal[fuel]
            s_mean = data.get('synth_mean')
            t_mean = data.get('train_mean')
            v_mean = data.get('val_mean')
            x_mean = data.get('test_mean')
            mean_ratio = data.get('mean_ratio')

            if mean_ratio is None or np.isnan(mean_ratio):
                continue

            synth_dev = abs(mean_ratio - 1.0)
            ref_tv = self._ratio_dev(t_mean, v_mean)
            ref_tt = self._ratio_dev(t_mean, x_mean)
            ref_vt = self._ratio_dev(v_mean, x_mean)
            nv = self._nv_from_refs(ref_tv, ref_tt, ref_vt)

            status = self._nv_status(synth_dev, nv, k)
            self.flags.append(FidelityFlag(
                metric=f'{fuel}_mean_ratio', status=status, value=mean_ratio,
                threshold=self._nv_threshold_str(nv, k),
                message=f'{fuel.title()} synth/train mean ratio (deviation: {synth_dev:.4f})',
                ref_train_val=ref_tv, ref_train_test=ref_tt, ref_val_test=ref_vt,
                natural_variation=nv, k=k))

        # ------------------------------------------------------------------ #
        # 2. Std ratio (per fuel)                                              #
        # ------------------------------------------------------------------ #
        for fuel in ['electricity', 'gas']:
            if fuel not in marginal:
                continue
            data = marginal[fuel]
            t_std = data.get('train_std')
            v_std = data.get('val_std')
            x_std = data.get('test_std')
            std_ratio = data.get('std_ratio')

            if std_ratio is None or np.isnan(std_ratio):
                continue

            synth_dev = abs(std_ratio - 1.0)
            ref_tv = self._ratio_dev(t_std, v_std)
            ref_tt = self._ratio_dev(t_std, x_std)
            ref_vt = self._ratio_dev(v_std, x_std)
            nv = self._nv_from_refs(ref_tv, ref_tt, ref_vt)

            status = self._nv_status(synth_dev, nv, k)
            self.flags.append(FidelityFlag(
                metric=f'{fuel}_std_ratio', status=status, value=std_ratio,
                threshold=self._nv_threshold_str(nv, k),
                message=f'{fuel.title()} synth/train std ratio',
                ref_train_val=ref_tv, ref_train_test=ref_tt, ref_val_test=ref_vt,
                natural_variation=nv, k=k))

        # ------------------------------------------------------------------ #
        # 3. ACF max diff (per fuel)                                           #
        # ------------------------------------------------------------------ #
        acf = temporal.get('acf', {})
        for fuel in ['electricity', 'gas']:
            if fuel not in acf:
                continue
            synth_max = 0.0
            ref_tv_max = ref_tt_max = ref_vt_max = 0.0
            has_any = False

            for lag_data in acf[fuel].values():
                t_val = lag_data.get('train')
                v_val = lag_data.get('val')
                x_val = lag_data.get('test')
                s_val = lag_data.get('synth')

                if s_val is not None and t_val is not None:
                    synth_max = max(synth_max, abs(s_val - t_val))
                    has_any = True
                if t_val is not None and v_val is not None:
                    ref_tv_max = max(ref_tv_max, abs(t_val - v_val))
                if t_val is not None and x_val is not None:
                    ref_tt_max = max(ref_tt_max, abs(t_val - x_val))
                if v_val is not None and x_val is not None:
                    ref_vt_max = max(ref_vt_max, abs(v_val - x_val))

            if not has_any:
                continue

            ref_tv = ref_tv_max if ref_tv_max > 0 else None
            ref_tt = ref_tt_max if ref_tt_max > 0 else None
            ref_vt = ref_vt_max if ref_vt_max > 0 else None
            nv = self._nv_from_refs(ref_tv, ref_tt, ref_vt)

            status = self._nv_status(synth_max, nv, k)
            self.flags.append(FidelityFlag(
                metric=f'{fuel}_acf_max_diff', status=status, value=synth_max,
                threshold=self._nv_threshold_str(nv, k),
                message=f'{fuel.title()} max ACF difference from train',
                ref_train_val=ref_tv, ref_train_test=ref_tt, ref_val_test=ref_vt,
                natural_variation=nv, k=k))

        # ------------------------------------------------------------------ #
        # 4. Daily profile RMSE (per profile key)                              #
        # ------------------------------------------------------------------ #
        profiles = temporal.get('daily_profiles', {})

        def _profile_rmse(a, b) -> Optional[float]:
            if a is None or b is None:
                return None
            a_arr = np.array(a, dtype=float)
            b_arr = np.array(b, dtype=float)
            if len(a_arr) != len(b_arr):
                return None
            mask = ~np.isnan(a_arr) & ~np.isnan(b_arr)
            return float(np.sqrt(np.mean((a_arr[mask] - b_arr[mask]) ** 2))) if np.any(mask) else None

        for key, data in profiles.items():
            train_distance = data.get('train_distance')
            if train_distance is None or np.isnan(train_distance):
                continue

            t_prof = data.get('train')
            v_prof = data.get('val')
            x_prof = data.get('test')

            ref_tv = _profile_rmse(t_prof, v_prof)
            ref_tt = _profile_rmse(t_prof, x_prof)
            ref_vt = _profile_rmse(v_prof, x_prof)
            nv = self._nv_from_refs(ref_tv, ref_tt, ref_vt)

            fuel_for_key = 'electricity' if key.startswith('electricity') else 'gas'
            train_mean = marginal.get(fuel_for_key, {}).get('train_mean', 0)
            relative_str = (f', {train_distance / train_mean:.2f}× train mean'
                            if train_mean and train_mean > 0 else '')

            status = self._nv_status(train_distance, nv, k)
            self.flags.append(FidelityFlag(
                metric=f'{key}_profile_rmse', status=status, value=train_distance,
                threshold=self._nv_threshold_str(nv, k),
                message=f'{key} daily profile RMSE vs train{relative_str}',
                ref_train_val=ref_tv, ref_train_test=ref_tt, ref_val_test=ref_vt,
                natural_variation=nv, k=k))

        # ------------------------------------------------------------------ #
        # 5. Seasonal RMSE (per fuel) — val split not stored; only train-test #
        # ------------------------------------------------------------------ #
        seasonal = self.results.get('seasonal_analysis', {})

        def _seasonal_rmse(months_a, months_b) -> Optional[float]:
            if not months_a or not months_b:
                return None
            shared = set(months_a) & set(months_b)
            if len(shared) < 3:
                return None
            diffs = []
            for m in shared:
                a_v = months_a[m].get('mean') if isinstance(months_a[m], dict) else months_a[m]
                b_v = months_b[m].get('mean') if isinstance(months_b[m], dict) else months_b[m]
                if a_v is not None and b_v is not None:
                    diffs.append((a_v - b_v) ** 2)
            return float(np.sqrt(np.mean(diffs))) if diffs else None

        for fuel in ['electricity', 'gas']:
            fuel_seasonal = seasonal.get(fuel, {})
            synth_months = fuel_seasonal.get('synth', {})
            train_months = fuel_seasonal.get('train', {})
            test_months = fuel_seasonal.get('test', {})

            synth_rmse = _seasonal_rmse(synth_months, train_months)
            if synth_rmse is None:
                continue

            # Val split not stored in seasonal_analysis; only train-test reference available
            ref_tt = _seasonal_rmse(train_months, test_months)
            nv = self._nv_from_refs(ref_tt)

            train_mean = marginal.get(fuel, {}).get('train_mean', 0)
            relative_str = (f', {synth_rmse / train_mean:.2f}× train mean'
                            if train_mean and train_mean > 0 else '')

            status = self._nv_status(synth_rmse, nv, k)
            self.flags.append(FidelityFlag(
                metric=f'{fuel}_seasonal_rmse', status=status, value=synth_rmse,
                threshold=self._nv_threshold_str(nv, k),
                message=f'{fuel.title()} seasonal pattern RMSE{relative_str}',
                ref_train_val=None, ref_train_test=ref_tt, ref_val_test=None,
                natural_variation=nv, k=k))

        # ------------------------------------------------------------------ #
        # 6. PTG slope diff (per fuel)                                         #
        # ------------------------------------------------------------------ #
        ptg = self.results.get('ptg_analysis', {})
        synth_ptg = ptg.get('synth', {})
        train_ptg = ptg.get('train', {})
        val_ptg = ptg.get('val', {})
        test_ptg = ptg.get('test', {})

        def _ptg_slope(fuel_data) -> Optional[float]:
            temps = fuel_data.get('mean_temp', [])
            consump = fuel_data.get('mean_consumption', [])
            if len(temps) < 3:
                return None
            try:
                return float(np.polyfit(temps, consump, 1)[0])
            except Exception:
                return None

        for fuel in ['electricity', 'gas']:
            s_slope = _ptg_slope(synth_ptg.get(fuel, {}))
            t_slope = _ptg_slope(train_ptg.get(fuel, {}))
            v_slope = _ptg_slope(val_ptg.get(fuel, {}))
            x_slope = _ptg_slope(test_ptg.get(fuel, {}))

            if s_slope is None or t_slope is None or abs(t_slope) < 1e-6:
                continue

            synth_val = abs(s_slope - t_slope) / abs(t_slope)
            ref_tv = abs(t_slope - v_slope) / abs(t_slope) if v_slope is not None else None
            ref_tt = abs(t_slope - x_slope) / abs(t_slope) if x_slope is not None else None
            ref_vt = (abs(v_slope - x_slope) / abs(t_slope)
                      if v_slope is not None and x_slope is not None else None)
            nv = self._nv_from_refs(ref_tv, ref_tt, ref_vt)

            status = self._nv_status(synth_val, nv, k)
            self.flags.append(FidelityFlag(
                metric=f'{fuel}_ptg_slope_diff', status=status, value=synth_val,
                threshold=self._nv_threshold_str(nv, k),
                message=(f'{fuel.title()} PTG: synth slope {s_slope:.2f}, '
                         f'train slope {t_slope:.2f}, relative diff {synth_val:.2f}'),
                ref_train_val=ref_tv, ref_train_test=ref_tt, ref_val_test=ref_vt,
                natural_variation=nv, k=k))

        # ------------------------------------------------------------------ #
        # 7. Conditional mean ratio (worst-case across groups, per fuel)       #
        # ------------------------------------------------------------------ #
        cond_dist = self.results.get('conditional_distribution', {})
        for fuel in ['electricity', 'gas']:
            synth_devs, ref_tv_devs, ref_tt_devs, ref_vt_devs = [], [], [], []
            min_ratio, max_ratio = float('inf'), 0.0
            n_groups = 0

            for cond_var, groups in cond_dist.items():
                if not isinstance(groups, dict):
                    continue
                for cond_val, group_data in groups.items():
                    if not isinstance(group_data, dict):
                        continue
                    fd = group_data.get(fuel, group_data)
                    n_train = fd.get('n_households_train', 0)
                    if n_train < SDC_MIN_HOUSEHOLDS:
                        continue
                    s_mean = fd.get('synth_mean')
                    t_mean = fd.get('train_mean')
                    v_mean = fd.get('val_mean')
                    x_mean = fd.get('test_mean')
                    n_val = fd.get('n_households_val', 0)
                    n_test = fd.get('n_households_test', 0)

                    if s_mean is not None and t_mean and t_mean > 0:
                        r = s_mean / t_mean
                        synth_devs.append(abs(r - 1.0))
                        min_ratio = min(min_ratio, r)
                        max_ratio = max(max_ratio, r)
                        n_groups += 1
                    d = self._ratio_dev(t_mean, v_mean)
                    if d is not None and n_val >= SDC_MIN_HOUSEHOLDS:
                        ref_tv_devs.append(d)
                    d = self._ratio_dev(t_mean, x_mean)
                    if d is not None and n_test >= SDC_MIN_HOUSEHOLDS:
                        ref_tt_devs.append(d)
                    d = self._ratio_dev(v_mean, x_mean)
                    if d is not None and n_val >= SDC_MIN_HOUSEHOLDS and n_test >= SDC_MIN_HOUSEHOLDS:
                        ref_vt_devs.append(d)

            if not synth_devs:
                continue

            worst_synth = max(synth_devs)
            ref_tv = max(ref_tv_devs) if ref_tv_devs else None
            ref_tt = max(ref_tt_devs) if ref_tt_devs else None
            ref_vt = max(ref_vt_devs) if ref_vt_devs else None
            nv = self._nv_from_refs(ref_tv, ref_tt, ref_vt)

            worst_ratio = max(max_ratio, 1 / min_ratio if min_ratio > 0 else float('inf'))
            status = self._nv_status(worst_synth, nv, k)
            self.flags.append(FidelityFlag(
                metric=f'{fuel}_conditional_mean_ratio', status=status, value=worst_ratio,
                threshold=self._nv_threshold_str(nv, k),
                message=(f'{fuel.title()} conditioning: worst group synth/train mean ratio '
                         f'(range {min_ratio:.2f}–{max_ratio:.2f} over {n_groups} groups)'),
                ref_train_val=ref_tv, ref_train_test=ref_tt, ref_val_test=ref_vt,
                natural_variation=nv, k=k))

        # ------------------------------------------------------------------ #
        # 8. Correlation diffs (real-vs-real framework)                        #
        # ------------------------------------------------------------------ #
        correlation = self.results.get('correlation_analysis', {})
        for pair_key, description in [
            ('elec_gas', 'Electricity-gas correlation difference'),
            ('temp_elec', 'Temperature-electricity correlation difference'),
            ('temp_gas', 'Temperature-gas correlation difference'),
        ]:
            corr_diff = correlation.get(f'{pair_key}_diff_synth_train')
            if corr_diff is None or np.isnan(corr_diff):
                continue
            synth_val = abs(corr_diff)

            t_corr = correlation.get(f'{pair_key}_corr_train')
            v_corr = correlation.get(f'{pair_key}_corr_val')
            x_corr = correlation.get(f'{pair_key}_corr_test')

            ref_tv = abs(t_corr - v_corr) if t_corr is not None and v_corr is not None else None
            ref_tt = abs(t_corr - x_corr) if t_corr is not None and x_corr is not None else None
            ref_vt = abs(v_corr - x_corr) if v_corr is not None and x_corr is not None else None
            nv = self._nv_from_refs(ref_tv, ref_tt, ref_vt)

            status = self._nv_status(synth_val, nv, k)
            self.flags.append(FidelityFlag(
                metric=f'{pair_key}_correlation_diff', status=status, value=corr_diff,
                threshold=self._nv_threshold_str(nv, k),
                message=description,
                ref_train_val=ref_tv, ref_train_test=ref_tt, ref_val_test=ref_vt,
                natural_variation=nv, k=k))

        # ------------------------------------------------------------------ #
        # 9. Missingness (per fuel)                                            #
        # ------------------------------------------------------------------ #
        missingness = self.results.get('missingness_analysis', {})
        for fuel in ['electricity', 'gas']:
            fuel_miss = missingness.get(fuel, {})
            if not fuel_miss:
                continue

            t_miss = fuel_miss.get('train_missing_pct')
            v_miss = fuel_miss.get('val_missing_pct')
            x_miss = fuel_miss.get('test_missing_pct')
            s_miss = fuel_miss.get('synth_missing_pct')

            if t_miss is not None and s_miss is not None:
                synth_val = abs(s_miss - t_miss)
                ref_tv = abs(t_miss - v_miss) if v_miss is not None else None
                ref_tt = abs(t_miss - x_miss) if x_miss is not None else None
                ref_vt = (abs(v_miss - x_miss)
                          if v_miss is not None and x_miss is not None else None)
                nv = self._nv_from_refs(ref_tv, ref_tt, ref_vt)
                status = self._nv_status(synth_val, nv, k)
                self.flags.append(FidelityFlag(
                    metric=f'{fuel}_missing_pct_diff', status=status,
                    value=fuel_miss.get('missing_pct_diff', synth_val),
                    threshold=self._nv_threshold_str(nv, k),
                    message=(f'{fuel.title()} missing %: synth {s_miss:.1f}%, '
                             f'train {t_miss:.1f}%'),
                    ref_train_val=ref_tv, ref_train_test=ref_tt, ref_val_test=ref_vt,
                    natural_variation=nv, k=k))

            t_zero = fuel_miss.get('train_zeros_pct')
            v_zero = fuel_miss.get('val_zeros_pct')
            x_zero = fuel_miss.get('test_zeros_pct')
            s_zero = fuel_miss.get('synth_zeros_pct')

            if t_zero is not None and s_zero is not None:
                synth_val = abs(s_zero - t_zero)
                ref_tv = abs(t_zero - v_zero) if v_zero is not None else None
                ref_tt = abs(t_zero - x_zero) if x_zero is not None else None
                ref_vt = (abs(v_zero - x_zero)
                          if v_zero is not None and x_zero is not None else None)
                nv = self._nv_from_refs(ref_tv, ref_tt, ref_vt)
                status = self._nv_status(synth_val, nv, k)
                self.flags.append(FidelityFlag(
                    metric=f'{fuel}_zeros_pct_diff', status=status,
                    value=fuel_miss.get('zeros_pct_diff', synth_val),
                    threshold=self._nv_threshold_str(nv, k),
                    message=(f'{fuel.title()} zeros %: synth {s_zero:.1f}%, '
                             f'train {t_zero:.1f}%'),
                    ref_train_val=ref_tv, ref_train_test=ref_tt, ref_val_test=ref_vt,
                    natural_variation=nv, k=k))

        # ------------------------------------------------------------------ #
        # 10. Volatility (per fuel)                                            #
        # ------------------------------------------------------------------ #
        volatility = temporal.get('volatility', {})
        for fuel in ['electricity', 'gas']:
            vol = volatility.get(fuel, {})
            if not vol:
                continue

            t_std = vol.get('train_std')
            v_std = vol.get('val_std')
            x_std = vol.get('test_std')
            std_ratio = vol.get('std_ratio')

            if std_ratio is not None and not np.isnan(std_ratio):
                synth_dev = abs(std_ratio - 1.0)
                ref_tv = self._ratio_dev(t_std, v_std)
                ref_tt = self._ratio_dev(t_std, x_std)
                ref_vt = self._ratio_dev(v_std, x_std)
                nv = self._nv_from_refs(ref_tv, ref_tt, ref_vt)
                status = self._nv_status(synth_dev, nv, k)
                self.flags.append(FidelityFlag(
                    metric=f'{fuel}_volatility_std_ratio', status=status, value=std_ratio,
                    threshold=self._nv_threshold_str(nv, k),
                    message=f'{fuel.title()} first-difference std ratio (synth/train)',
                    ref_train_val=ref_tv, ref_train_test=ref_tt, ref_val_test=ref_vt,
                    natural_variation=nv, k=k))

            t_q = vol.get('train_quantiles') or {}
            v_q = vol.get('val_quantiles') or {}
            x_q = vol.get('test_quantiles') or {}
            s_q = vol.get('synth_quantiles') or {}
            synth_p90 = s_q.get('p90')
            train_p90 = t_q.get('p90')
            val_p90 = v_q.get('p90')
            test_p90 = x_q.get('p90')

            if (synth_p90 is not None and train_p90 is not None
                    and not np.isnan(synth_p90) and not np.isnan(train_p90)
                    and abs(train_p90) > 1e-6):
                tail_ratio = synth_p90 / train_p90
                synth_dev = abs(tail_ratio - 1.0)
                ref_tv = self._ratio_dev(train_p90, val_p90)
                ref_tt = self._ratio_dev(train_p90, test_p90)
                ref_vt = self._ratio_dev(val_p90, test_p90)
                nv = self._nv_from_refs(ref_tv, ref_tt, ref_vt)
                status = self._nv_status(synth_dev, nv, k)
                self.flags.append(FidelityFlag(
                    metric=f'{fuel}_volatility_tail_ratio', status=status, value=tail_ratio,
                    threshold=self._nv_threshold_str(nv, k),
                    message=f'{fuel.title()} volatility p90 ratio (synth/train)',
                    ref_train_val=ref_tv, ref_train_test=ref_tt, ref_val_test=ref_vt,
                    natural_variation=nv, k=k))

        # ------------------------------------------------------------------ #
        # 11. Within-batch variance (excluded from NV framework per decision) #
        # ------------------------------------------------------------------ #
        variance = self.results.get('within_batch_variance', {})
        var_status = variance.get('overall_status', 'unknown')
        severity = variance.get('severity_counts', {})
        n_flagged = severity.get('medium', 0) + severity.get('high', 0)
        self.flags.append(FidelityFlag(
            metric='within_batch_variance', status=var_status, value=n_flagged,
            threshold='0 medium/high flagged [fixed]',
            message=f'Within-batch variance check ({n_flagged} batches flagged)'))

    def _check_ratio(self, ratio: float, metric_type: str) -> str:
        """Check ratio against thresholds."""
        thresholds = self.THRESHOLDS.get(metric_type, {})
        pass_range = thresholds.get('pass', (0.8, 1.2))
        warn_range = thresholds.get('warn', (0.5, 2.0))
        if pass_range[0] <= ratio <= pass_range[1]:
            return 'pass'
        elif warn_range[0] <= ratio <= warn_range[1]:
            return 'warn'
        return 'fail'

    # ------------------------------------------------------------------ #
    # Report + Summary outputs                                             #
    # ------------------------------------------------------------------ #

    def _generate_report_md(self) -> str:
        """Generate structured Markdown evaluation report with SDC checker guidance."""
        meta = self.results.get('metadata', {})
        timestamp = datetime.now().isoformat(timespec='seconds')

        run_id = self._derive_run_id()

        # Flags shared into both fuel sections (duplicated in display, counted per fuel)
        SHARED_METRICS = {'within_batch_variance', 'elec_gas_correlation_diff'}
        # Flags that belong to exactly one fuel section (not electricity_ or gas_ prefixed)
        ELEC_ONLY_METRICS = {'temp_elec_correlation_diff'}
        GAS_ONLY_METRICS  = {'temp_gas_correlation_diff'}

        def _flags_for_fuel(fuel_prefix: str) -> list:
            """All flags to show under a fuel section."""
            flags = [f for f in self.flags if f.metric.startswith(fuel_prefix)]
            if fuel_prefix == 'electricity':
                flags += [f for f in self.flags if f.metric in ELEC_ONLY_METRICS | SHARED_METRICS]
            else:
                flags += [f for f in self.flags if f.metric in GAS_ONLY_METRICS | SHARED_METRICS]
            return flags

        def _fuel_summary(fuel_prefix: str):
            fuel_flags = _flags_for_fuel(fuel_prefix)
            n_p = sum(1 for f in fuel_flags if f.status == 'pass')
            n_w = sum(1 for f in fuel_flags if f.status == 'warn')
            n_f = sum(1 for f in fuel_flags if f.status == 'fail')
            ov = 'FAIL' if n_f > 0 else ('WARN' if n_w > 0 else 'PASS')
            return ov, n_p, n_w, n_f

        elec_ov, elec_p, elec_w, elec_f = _fuel_summary('electricity')
        gas_ov,  gas_p,  gas_w,  gas_f  = _fuel_summary('gas')

        icon_map = {'pass': '✓', 'warn': '⚠', 'fail': '✗'}
        status_order = {'fail': 0, 'warn': 1, 'pass': 2}

        def _fmt(v: Optional[float], digits: int = 4) -> str:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return '—'
            return f'{v:.{digits}f}'

        def _flag_rows(flags):
            rows = []
            for flag in sorted(flags, key=lambda f: status_order.get(f.status, 3)):
                icon = icon_map.get(flag.status, '?')
                val_str = _fmt(flag.value) if isinstance(flag.value, float) else str(flag.value)
                nv_str = _fmt(flag.natural_variation)
                rtv = _fmt(flag.ref_train_val)
                rtt = _fmt(flag.ref_train_test)
                rvt = _fmt(flag.ref_val_test)
                rows.append(
                    f'| {flag.metric} | {icon} {flag.status.upper()} '
                    f'| {val_str} | {nv_str} | {rtv} | {rtt} | {rvt} |')
            return rows

        def _count_csv_suppressed(filename: str) -> Optional[int]:
            """Count rows where suppressed=True in a companion CSV. None if CSV absent."""
            path = self.output_dir / filename
            if not path.exists():
                return None
            try:
                df = pd.read_csv(path)
                return int(df['suppressed'].sum()) if 'suppressed' in df.columns else None
            except Exception:
                return None

        # ------------------------------------------------------------------ #
        # Title and preamble                                                   #
        # ------------------------------------------------------------------ #
        lines = [
            f'# Fidelity Evaluation Report — {run_id}',
            '',
            f'Generated: {timestamp}',
            '',
            '## About This Report',
            '',
            'This report is produced automatically by the AI-EDOL evaluation pipeline. '
            'It serves two purposes: (1) summarising the fidelity of the synthetic data against '
            'the real SERL Observatory household energy data, and (2) providing supporting '
            'information for Statistical Disclosure Control (SDC) review before outputs are '
            'exported from the secure data environment.',
            '',
            '**Real vs synthetic data.** Statistics in this report are derived from two sources. '
            'Curves or statistics labelled *Train*, *Val*, or *Test* are computed from real SERL '
            'household energy data and are subject to SDC review. Curves labelled *Synthetic* are '
            'computed from the AI-generated synthetic data; the synthetic data is the '
            'privacy-preserving output of the project and is not individually sensitive, but it '
            'appears alongside real-data curves in comparison plots.',
            '',
            '**Household counts.** The N values in the Summary table are the number of real '
            'households in each data split. All population-level statistics are aggregated over '
            'these full N values. Subgroup statistics (conditional distributions, household-month '
            'analyses) may be aggregated over smaller groups — see the SDC Compliance section for '
            'all suppression applied.',
            '',
            f'**SDC processing applied to outputs.** All outputs in this report have been '
            f'processed for SDC compliance before export. Two rules are applied: (1) any cell, '
            f'group, or histogram bin where the count of real households or observations falls '
            f'below {SDC_MIN_HOUSEHOLDS} is suppressed — both the statistic and its accompanying '
            f'count are set to null; (2) all real-data household counts (N values for train, val, '
            f'and test splits) and histogram bin observation counts are rounded to the nearest 10. '
            f'Synthetic data counts and statistics are not subject to these rules. Full suppression '
            f'detail is in the SDC Compliance section.',
            '',
            f'**Histogram outputs.** The volatility histogram and token distribution plots pool '
            f'individual-level observations across all households before binning — no per-household '
            f'statistic is visible in these outputs. This substantially reduces re-identification '
            f'risk. Nonetheless, bins where the number of pooled training observations falls below '
            f'{SDC_MIN_HOUSEHOLDS} are suppressed, and counts in remaining bins are rounded to the '
            f'nearest 10. SDC reviewers should verify that suppressed bins are absent from plots '
            f'and appear as null values in companion CSVs.',
            '',
            '**For SDC reviewers.** Work through the Diagnostics section in sequence, reviewing '
            'each plot (paper version) and its companion CSV. The SDC Compliance section lists all '
            'suppression rules and counts. The pre-sign-off checklist and reviewer declarations '
            'are at the end of this document.',
            '',
        ]

        # ------------------------------------------------------------------ #
        # Privacy Pre-Filter (only if applied)                                 #
        # ------------------------------------------------------------------ #
        pf = meta.get('prefilter_summary')
        if pf and pf.get('enabled'):
            lines += [
                '## Privacy Pre-Filter',
                '',
                'A privacy pre-filter was applied to exclude synthetic household-months '
                'flagged as too close to training data. This filtering was performed '
                'before computing all fidelity metrics in this report.',
                '',
                '| Metric | Value |',
                '|--------|-------|',
                f'| Exclusion CSV rows | {pf.get("csv_rows", 0):,} |',
                f'| Duplicate CSV rows (deduplicated) | {pf.get("csv_duplicates", 0):,} |',
                f'| Household-months excluded | {pf.get("n_household_months_excluded", 0):,} |',
                f'| Batches affected | {pf.get("n_batches_affected", 0):,} / {pf.get("n_batches_total", 0):,} |',
                f'| Households fully excluded (all months) | {pf.get("n_households_fully_excluded", 0):,} |',
                f'| Households partially excluded (some months) | {pf.get("n_households_partially_excluded", 0):,} |',
                f'| Batches emptied by filter | {pf.get("n_batches_emptied", 0):,} |',
                '',
                '_The synthetic household and value counts in the Summary table below '
                'reflect the post-filter population._',
                '',
            ]

        # ------------------------------------------------------------------ #
        # Summary                                                              #
        # ------------------------------------------------------------------ #
        lines += [
            '## Summary',
            '',
            '| Fuel | Result | Pass | Warn | Fail |',
            '|------|--------|------|------|------|',
            f'| Electricity | **{elec_ov}** | {elec_p} | {elec_w} | {elec_f} |',
            f'| Gas         | **{gas_ov}**  | {gas_p}  | {gas_w}  | {gas_f}  |',
            '',
        ]
        _n_synth   = meta.get('n_households', None)
        _n_batches = meta.get('n_batches', None)
        if isinstance(_n_synth, int) and isinstance(_n_batches, int) and _n_batches > 0:
            _batch_size  = _n_synth // _n_batches
            _synth_str   = f'{_n_synth:,} ({_n_batches} batches of {_batch_size:,} households)'
        else:
            _synth_str = f'{_n_synth:,}' if isinstance(_n_synth, int) else str(_n_synth or 'N/A')
        lines += [
            '| Metric | Value |',
            '|--------|-------|',
            f'| Synthetic households | {_synth_str} |',
            f'| Train households (rounded to nearest 10) | {_rn(meta["n_households_train"]) if "n_households_train" in meta else "N/A"} |',
            f'| Val households (rounded to nearest 10) | {_rn(meta["n_households_val"]) if "n_households_val" in meta else "N/A"} |',
            f'| Test households (rounded to nearest 10) | {_rn(meta["n_households_test"]) if "n_households_test" in meta else "N/A"} |',
            f'| Electricity values | {meta.get("n_values", {}).get("electricity", 0):,} |',
            f'| Gas values | {meta.get("n_values", {}).get("gas", 0):,} |',
            '',
            '_Note: synthetic data is generated in batches where households in each batch share '
            'the same conditioning variables (e.g. building type, number of occupants)._',
            '',
        ]

        # ------------------------------------------------------------------ #
        # Fidelity Checks                                                      #
        # ------------------------------------------------------------------ #
        lines += [
            '## Fidelity Checks',
            '',
            '_For ratio metrics (mean_ratio, std_ratio, …) pass criterion is '
            '|value − 1| ≤ Nat.var. '
            'For absolute metrics (RMSE, ACF diff, …) pass criterion is value ≤ Nat.var. '
            f'k = {self.K_TOLERANCE} (warn zone = k × Nat.var)._',
            '',
        ]

        for fuel_label, fuel_prefix in [('Electricity', 'electricity'), ('Gas', 'gas')]:
            fuel_flags = _flags_for_fuel(fuel_prefix)
            if fuel_flags:
                lines += [
                    f'### {fuel_label}',
                    '',
                    '| Check | Status | Value | Nat.var | ref(t-v) | ref(t-t) | ref(v-t) |',
                    '|-------|--------|-------|---------|----------|----------|----------|',
                ] + _flag_rows(fuel_flags) + ['']

        # ------------------------------------------------------------------ #
        # SDC Compliance                                                       #
        # ------------------------------------------------------------------ #
        lines += [
            '## SDC Compliance',
            '',
            'This section lists all statistical suppression applied to protect individual '
            'household privacy. Suppressed items are excluded from plots and appear as `null` '
            'values in companion CSVs (flagged via `suppressed=True`). SDC reviewers should '
            'verify that suppressed items are absent from the corresponding plots, and that no '
            'suppressed value is inferrable by subtraction from a visible total or subtotal.',
            '',
        ]

        # --- Conditional group suppression ---
        cond_dist = self.results.get('conditional_distribution', {})
        suppressed_rows = []
        for cond_var, groups in cond_dist.items():
            if not isinstance(groups, dict):
                continue
            for cond_val, group_data in groups.items():
                if not isinstance(group_data, dict):
                    continue
                for fuel in ['electricity', 'gas']:
                    fd = group_data.get(fuel, {})
                    if not isinstance(fd, dict):
                        continue
                    n_train = fd.get('n_households_train', 0)
                    n_val   = fd.get('n_households_val', 0)
                    n_test  = fd.get('n_households_test', 0)
                    if 0 < n_train < SDC_MIN_HOUSEHOLDS or 0 < n_test < SDC_MIN_HOUSEHOLDS:
                        suppressed_rows.append({
                            'cond_var': cond_var,
                            'cond_value': str(cond_val),
                            'n_train': n_train,
                            'n_val': n_val,
                            'n_test': n_test,
                        })
                        break  # one row per group (fuel-agnostic suppression)

        lines += ['### Conditional distribution group suppression', '']
        if suppressed_rows:
            lines += [
                f'**{len(suppressed_rows)} conditioning group(s) suppressed** '
                f'(n_households_train or n_households_test < {SDC_MIN_HOUSEHOLDS}). '
                f'These groups are excluded from the conditional distribution plots and CSVs. '
                f'See [conditional_distribution_comparison.csv]'
                f'(visualizations/conditional_distribution_comparison.csv) (`suppressed=True` rows).',
                '',
                '| Conditioning variable | Value | n_train | n_val | n_test |',
                '|-----------------------|-------|---------|-------|--------|',
            ]
            def _fmt_suppressed_n(n: int) -> str:
                return '0' if n == 0 else f'< {SDC_MIN_HOUSEHOLDS}'

            for r in sorted(suppressed_rows, key=lambda x: (x['cond_var'], x['cond_value'])):
                lines.append(
                    f'| {r["cond_var"]} | {r["cond_value"]} '
                    f'| {_fmt_suppressed_n(r["n_train"])} '
                    f'| {_fmt_suppressed_n(r["n_val"])} '
                    f'| {_fmt_suppressed_n(r["n_test"])} |')
            lines += ['']
        else:
            lines += [
                f'All conditioning groups meet the SDC threshold '
                f'(n_households ≥ {SDC_MIN_HOUSEHOLDS} for train and test splits). '
                'No suppression applied.',
                '',
            ]

        # --- Histogram / token suppression ---
        _SDC_HIST_MIN = 10  # threshold for histogram bin and token count suppression
        n_supp_vhist   = _count_csv_suppressed('volatility_analysis_hist.csv')
        n_supp_tfreq_e = _count_csv_suppressed('token_frequency_comparison_electricity.csv')
        n_supp_tfreq_g = _count_csv_suppressed('token_frequency_comparison_gas.csv')
        n_supp_tdist_e = _count_csv_suppressed('token_distribution_analysis_electricity.csv')
        n_supp_tdist_g = _count_csv_suppressed('token_distribution_analysis_gas.csv')

        def _supp_str(n: Optional[int]) -> str:
            return str(n) if n is not None else 'N/A (not generated)'

        lines += [
            '### Histogram bin and token count suppression',
            '',
            f'The following outputs apply per-bin or per-token suppression where the count of '
            f'real training data observations falls below {_SDC_HIST_MIN}. '
            f'Suppressed entries appear as `null` in count and density columns of each CSV. '
            f'Observation counts in non-suppressed bins are rounded to the nearest 10.',
            '',
            '| Output | Threshold | Suppressed items | CSV |',
            '|--------|-----------|------------------|-----|',
            f'| Volatility histogram | train observations < {_SDC_HIST_MIN} per bin'
            f' | {_supp_str(n_supp_vhist)} bins'
            f' | [volatility_analysis_hist.csv](visualizations/volatility_analysis_hist.csv) |',
            f'| Token frequency (electricity) | train token count < {_SDC_HIST_MIN}'
            f' | {_supp_str(n_supp_tfreq_e)} tokens'
            f' | [token_frequency_comparison_electricity.csv](visualizations/token_frequency_comparison_electricity.csv) |',
            f'| Token frequency (gas) | train token count < {_SDC_HIST_MIN}'
            f' | {_supp_str(n_supp_tfreq_g)} tokens'
            f' | [token_frequency_comparison_gas.csv](visualizations/token_frequency_comparison_gas.csv) |',
            f'| Token distribution (electricity) | train observations < {_SDC_HIST_MIN} per bin'
            f' | {_supp_str(n_supp_tdist_e)} bins'
            f' | [token_distribution_analysis_electricity.csv](visualizations/token_distribution_analysis_electricity.csv) |',
            f'| Token distribution (gas) | train observations < {_SDC_HIST_MIN} per bin'
            f' | {_supp_str(n_supp_tdist_g)} bins'
            f' | [token_distribution_analysis_gas.csv](visualizations/token_distribution_analysis_gas.csv) |',
            '',
        ]

        # ------------------------------------------------------------------ #
        # Diagnostics                                                          #
        # ------------------------------------------------------------------ #
        lines += ['## Diagnostics', '']

        # Each entry: stem, label, list of (link_label, csv_name), description,
        #             data_source, sdc_note.
        diag_entries = [
            {
                'stem': 'marginal_distribution_comparison',
                'label': 'Marginal Distribution',
                'csvs': [('data', 'marginal_distribution_comparison.csv')],
                'description': (
                    'Population-level distribution of half-hourly energy values '
                    '(mean, std, quantiles) for electricity and gas, comparing synthetic '
                    'to train/val/test.'
                ),
                'data_source': 'All train, val, and test households vs all synthetic households.',
                'sdc_note': (
                    f'Population aggregate — N is the full household count per split (see Summary). '
                    f'No subgroup breakdown; no suppression applied. '
                    f'For each reported quantile, check that the implied tail group is not smaller '
                    f'than the SDC threshold: a quantile at percentile p over N households implies '
                    f'min(p, 1−p) × N households in the smaller tail. For example, p95 over '
                    f'N=100 implies 5 households above the line — below the threshold of '
                    f'{SDC_MIN_HOUSEHOLDS}. Verify that for every reported quantile and every '
                    f'split, min(p, 1−p) × N ≥ {SDC_MIN_HOUSEHOLDS}.'
                ),
            },
            {
                'stem': 'daily_profiles_comparison',
                'label': 'Daily Profiles',
                'csvs': [('data', 'daily_profiles_comparison.csv')],
                'description': (
                    'Mean energy consumption at each of the 48 half-hours of the day, '
                    'averaged across all households, for electricity and gas.'
                ),
                'data_source': 'All train, val, and test households vs all synthetic households.',
                'sdc_note': (
                    'Population aggregate; 48 time bins. '
                    'Check n_households column in CSV equals the full split N. '
                    'No individual household traces shown.'
                ),
            },
            {
                'stem': 'acf_comparison',
                'label': 'Autocorrelation (ACF)',
                'csvs': [('data', 'acf_comparison.csv')],
                'description': (
                    'Mean autocorrelation at fixed lags from 30 minutes to 4 weeks, '
                    'measuring temporal persistence in energy use patterns.'
                ),
                'data_source': 'All train, val, and test households vs all synthetic households.',
                'sdc_note': (
                    'Population aggregate; 8 lag values. '
                    'Each value is the mean autocorrelation across all households. '
                    'Check that values represent population averages, not individual ACF curves.'
                ),
            },
            {
                'stem': 'conditional_distribution_comparison_electricity',
                'label': 'Conditional Distribution (Electricity)',
                'csvs': [('data', 'conditional_distribution_comparison.csv')],
                'description': (
                    'Mean electricity consumption broken down by each conditioning variable '
                    '(building type, number of rooms, region, age of build, number of occupants), '
                    'comparing synthetic to real splits.'
                ),
                'data_source': (
                    'Train and test households per conditioning group vs synthetic; '
                    'val included where available.'
                ),
                'sdc_note': (
                    f'Per-conditioning-group aggregate. Groups with n_households_train or '
                    f'n_households_test < {SDC_MIN_HOUSEHOLDS} are suppressed '
                    f'(see SDC Compliance). Verify suppressed groups are absent from the plot. '
                    f'Check n_households_train and n_households_test columns in CSV for each '
                    f'visible bar.'
                ),
            },
            {
                'stem': 'conditional_distribution_comparison_gas',
                'label': 'Conditional Distribution (Gas)',
                'csvs': [('data', 'conditional_distribution_comparison.csv')],
                'description': (
                    'Mean gas consumption broken down by each conditioning variable, '
                    'comparing synthetic to real splits.'
                ),
                'data_source': (
                    'Train and test households per conditioning group vs synthetic; '
                    'val included where available.'
                ),
                'sdc_note': (
                    f'Per-conditioning-group aggregate. Same suppression rules as the electricity '
                    f'conditional distribution (groups with n < {SDC_MIN_HOUSEHOLDS} suppressed). '
                    f'Verify suppressed groups are absent.'
                ),
            },
            {
                'stem': 'volatility_analysis',
                'label': 'Volatility Analysis',
                'csvs': [
                    ('spike data', 'volatility_analysis_spike.csv'),
                    ('histogram data', 'volatility_analysis_hist.csv'),
                ],
                'description': (
                    'Two views of energy volatility: (left panels) P50/P75/P90 quantiles of '
                    'half-hourly energy changes; (right panels) histogram of half-hourly energy '
                    'changes. Shown for electricity and gas separately.'
                ),
                'data_source': 'Train and test households vs synthetic.',
                'sdc_note': (
                    'Spike CSV: population quantiles, no suppression. '
                    'Check tail quantiles are not dominated by one or two unusual households. '
                    'Histogram CSV: bins with train observations < 10 suppressed '
                    '(count in SDC Compliance). Verify suppressed bins show null values.'
                ),
            },
            {
                'stem': 'ptg_comparison',
                'label': 'Power Temperature Gradient (PTG)',
                'csvs': [('data', 'ptg_comparison.csv')],
                'description': (
                    'Mean energy consumption as a function of daily mean temperature, comparing '
                    'synthetic to train/val/test. Captures the temperature sensitivity (gradient) '
                    'of electricity and gas demand.'
                ),
                'data_source': 'All train, val, and test households vs all synthetic households.',
                'sdc_note': (
                    'Population aggregate; 48 time bins. No suppression applied. '
                    'Check that no subgroup or individual household breakdown is visible.'
                ),
            },
            {
                'stem': 'seasonal_analysis',
                'label': 'Seasonal Analysis',
                'csvs': [('data', 'seasonal_analysis.csv')],
                'description': (
                    'Monthly mean energy and a collapse ratio showing how well the synthetic '
                    'model captures seasonal variation, for electricity and gas.'
                ),
                'data_source': 'Train and test monthly means vs synthetic.',
                'sdc_note': (
                    '12 monthly data points per fuel. '
                    'Check n_households per month in the companion CSV; '
                    'verify no month has N below the SDC threshold.'
                ),
            },
            {
                'stem': 'daily_pattern_heatmap_electricity',
                'label': 'Daily Pattern Heatmap (Electricity)',
                'csvs': [('data', 'daily_pattern_heatmap_electricity.csv')],
                'description': (
                    'Month × half-hour-of-day heatmap of mean electricity consumption '
                    '(12 months × 48 time points = 576 cells), for each split and synthetic.'
                ),
                'data_source': 'Train, val, and test households vs synthetic.',
                'sdc_note': (
                    'Each cell is a mean across all households for that month–time combination. '
                    'n_households per split is in the CSV. '
                    'Check no split has N below the SDC threshold.'
                ),
            },
            {
                'stem': 'daily_pattern_heatmap_gas',
                'label': 'Daily Pattern Heatmap (Gas)',
                'csvs': [('data', 'daily_pattern_heatmap_gas.csv')],
                'description': (
                    'Month × half-hour-of-day heatmap of mean gas consumption, '
                    'for each split and synthetic.'
                ),
                'data_source': 'Train, val, and test households vs synthetic.',
                'sdc_note': (
                    'Same as electricity heatmap. '
                    'n_households per split is in the CSV; '
                    'check no split has N below the SDC threshold.'
                ),
            },
            {
                'stem': 'daily_pattern_degradation_electricity',
                'label': 'Daily Pattern Degradation (Electricity)',
                'csvs': [('data', 'daily_pattern_degradation_electricity.csv')],
                'description': (
                    'Mean electricity consumption per week within the autoregressive generation '
                    'window (weeks 1–8), showing whether synthetic data quality changes as the '
                    'context window fills.'
                ),
                'data_source': 'Train, val, and test households vs synthetic, aggregated by week.',
                'sdc_note': (
                    'Weekly aggregate. n_households per split and week is in the companion CSV. '
                    'Check N is not below the SDC threshold for any week.'
                ),
            },
            {
                'stem': 'daily_pattern_degradation_gas',
                'label': 'Daily Pattern Degradation (Gas)',
                'csvs': [('data', 'daily_pattern_degradation_gas.csv')],
                'description': (
                    'Mean gas consumption per week within the autoregressive generation window.'
                ),
                'data_source': 'Train, val, and test households vs synthetic, aggregated by week.',
                'sdc_note': (
                    'Same as electricity degradation. '
                    'n_households per split and week is in the companion CSV.'
                ),
            },
            {
                'stem': 'household_month_fidelity_marginal',
                'label': 'Household-Month Fidelity (Marginal)',
                'csvs': [('data', 'household_month_fidelity_marginal.csv')],
                'description': (
                    'Quintile distribution of per-household-monthly mean energy values, '
                    'comparing synthetic to train/val/test. Assesses whether the synthetic data '
                    'reproduces the spread of consumption levels across households.'
                ),
                'data_source': (
                    'Train, val, and test households vs synthetic, '
                    'at per-household-month level.'
                ),
                'sdc_note': (
                    'Population aggregate at household-month level. '
                    'N values (n_households and n_household_months) are in the companion CSV. '
                    'Check these are not below the SDC threshold.'
                ),
            },
            {
                'stem': 'household_month_fidelity_conditioning_electricity',
                'label': 'Household-Month Fidelity by Conditioning Group (Electricity)',
                'csvs': [('data', 'household_month_fidelity_conditioning_electricity.csv')],
                'description': (
                    'Mean per-household-monthly electricity consumption broken down by '
                    'conditioning variable (building type, rooms, region, age, occupants), '
                    'comparing synthetic to train/val/test.'
                ),
                'data_source': (
                    'Train, val, and test households per conditioning group vs synthetic, '
                    'at per-household-month level.'
                ),
                'sdc_note': (
                    f'Per-conditioning-group aggregate. Groups with n_households < '
                    f'{SDC_MIN_HOUSEHOLDS} are suppressed. Verify suppressed groups are absent. '
                    f'Check n_households_train and n_households_test for each visible bar in '
                    f'the companion CSV.'
                ),
            },
            {
                'stem': 'household_month_fidelity_conditioning_gas',
                'label': 'Household-Month Fidelity by Conditioning Group (Gas)',
                'csvs': [('data', 'household_month_fidelity_conditioning_gas.csv')],
                'description': (
                    'Mean per-household-monthly gas consumption broken down by conditioning '
                    'variable, comparing synthetic to train/val/test.'
                ),
                'data_source': (
                    'Train, val, and test households per conditioning group vs synthetic, '
                    'at per-household-month level.'
                ),
                'sdc_note': (
                    f'Per-conditioning-group aggregate. Same suppression rules as the '
                    f'electricity conditioning plot (groups with n < {SDC_MIN_HOUSEHOLDS} '
                    f'suppressed). Verify suppressed groups are absent from the plot.'
                ),
            },
            {
                'stem': 'token_frequency_comparison_electricity',
                'label': 'Token Frequency (Electricity)',
                'csvs': [('data', 'token_frequency_comparison_electricity.csv')],
                'description': (
                    'Frequency of each discrete energy token in the model vocabulary for '
                    'electricity, comparing the token distribution in synthetic vs real data.'
                ),
                'data_source': 'All train and test electricity data vs all synthetic electricity data.',
                'sdc_note': (
                    'Token-level counts across the full dataset. '
                    'Tokens with train_count < 10 are suppressed (count in SDC Compliance). '
                    'Verify suppressed tokens show null frequency values. '
                    'Check that remaining token counts are not dominated by a small unusual subgroup.'
                ),
            },
            {
                'stem': 'token_frequency_comparison_gas',
                'label': 'Token Frequency (Gas)',
                'csvs': [('data', 'token_frequency_comparison_gas.csv')],
                'description': (
                    'Frequency of each discrete gas token, comparing synthetic to real.'
                ),
                'data_source': 'All train and test gas data vs all synthetic gas data.',
                'sdc_note': (
                    'Same rules as electricity token frequency. '
                    'Tokens with train_count < 10 are suppressed (count in SDC Compliance).'
                ),
            },
            {
                'stem': 'token_distribution_analysis',
                'label': 'Token Distribution Analysis',
                'csvs': [
                    ('electricity data', 'token_distribution_analysis_electricity.csv'),
                    ('gas data', 'token_distribution_analysis_gas.csv'),
                ],
                'description': (
                    'Histogram of energy token values (Wh magnitudes), comparing the '
                    'distribution of values represented by synthetic vs real tokens, '
                    'for electricity and gas.'
                ),
                'data_source': 'Sample of real training energy values vs sample of synthetic values.',
                'sdc_note': (
                    'Histogram bins with train_count < 10 are suppressed '
                    '(count in SDC Compliance). '
                    'Verify suppressed bins show null density values in both CSVs.'
                ),
            },
        ]

        for entry in diag_entries:
            stem = entry['stem']
            paper_path = self.output_dir / f'{stem}_paper.png'
            if not paper_path.exists():
                continue

            # Build plot + CSV links on one line
            csv_links = []
            for csv_label, csv_name in entry.get('csvs', []):
                if (self.output_dir / csv_name).exists():
                    csv_links.append(f'[{csv_label}](visualizations/{csv_name})')
            link_line = (
                f'[{entry["label"]}](visualizations/{stem}_paper.png)'
                + ((' | ' + ' | '.join(csv_links)) if csv_links else '')
            )

            lines += [
                f'- **{link_line}**',
                f'  *{entry["description"]}*  ',
                f'  Data: {entry["data_source"]}  ',
                f'  SDC: {entry["sdc_note"]}',
                '',
            ]

        # ------------------------------------------------------------------ #
        # Data Sources                                                         #
        # ------------------------------------------------------------------ #
        lines += [
            '## Data Sources',
            '',
            'All fidelity check inputs are verifiable in `parallel_evaluation_results.json`.',
            '',
            '| Fidelity check | JSON path |',
            '|---|---|',
            '| `{fuel}_mean_ratio` | `marginal_distribution[fuel].{synth_mean, train_mean, val_mean, test_mean}` |',
            '| `{fuel}_std_ratio` | `marginal_distribution[fuel].{synth_std, train_std, val_std, test_std}` |',
            '| `{fuel}_acf_max_diff` | `temporal_patterns.acf[fuel][lag].{synth, train, val, test}` |',
            '| `{fuel}_{type}_profile_rmse` | `temporal_patterns.daily_profiles[fuel_type].{synth, train, val, test, *_distance}` |',
            '| `{fuel}_seasonal_rmse` | `seasonal_analysis[fuel].{synth, train, test}` (monthly means; val not stored) |',
            '| `{fuel}_ptg_slope_diff` | `ptg_analysis.{synth, train, val, test}[fuel].{mean_temp, mean_consumption}` |',
            '| `{fuel}_conditional_mean_ratio` | `conditional_distribution[cond_var][value][fuel].{synth_mean, train_mean, val_mean, test_mean, n_households_*}` |',
            '| `{fuel}_missing_pct_diff` | `missingness_analysis[fuel].{synth_missing_pct, train_missing_pct, val_missing_pct, test_missing_pct}` |',
            '| `{fuel}_zeros_pct_diff` | `missingness_analysis[fuel].{synth_zeros_pct, train_zeros_pct, val_zeros_pct, test_zeros_pct}` |',
            '| `{fuel}_volatility_std_ratio` | `temporal_patterns.volatility[fuel].{std_ratio, train_std, val_std, test_std}` |',
            '| `{fuel}_volatility_tail_ratio` | `temporal_patterns.volatility[fuel].{synth_quantiles, train_quantiles, val_quantiles, test_quantiles}.p90` |',
            '| `elec_gas_correlation_diff` | `correlation_analysis.{elec_gas_corr_synth, elec_gas_corr_train, elec_gas_corr_val, elec_gas_corr_test}` |',
            '| `temp_elec_correlation_diff` | `correlation_analysis.{temp_elec_corr_synth, temp_elec_corr_train, temp_elec_corr_val, temp_elec_corr_test}` |',
            '| `temp_gas_correlation_diff` | `correlation_analysis.{temp_gas_corr_synth, temp_gas_corr_train, temp_gas_corr_val, temp_gas_corr_test}` |',
            '| `within_batch_variance` | `within_batch_variance.{severity_counts, flagged_batches, overall_status}` |',
            '',
        ]

        # ------------------------------------------------------------------ #
        # SDC Sign-off                                                         #
        # ------------------------------------------------------------------ #
        lines += [
            '## SDC Sign-off',
            '',
            '### Pre-sign-off checklist',
            '',
            'Before signing, each reviewer should confirm all of the following:',
            '',
            '1. [ ] I have reviewed every plot listed in the Diagnostics section (paper versions).',
            '2. [ ] I have reviewed every companion CSV listed in the Diagnostics section.',
            '3. [ ] I have verified that all conditioning groups listed as suppressed in the '
            'SDC Compliance section are absent from the conditional distribution plots and CSVs.',
            '4. [ ] I have verified that suppressed rows in all histogram-based CSVs '
            '(volatility, token frequency, token distribution) show `null` values for count '
            'and density columns.',
            '5. [ ] I have checked that no visible data point in any plot represents fewer than '
            '5 real households.',
            '6. [ ] I have checked that no individual household energy time series is visible '
            'in any plot.',
            '7. [ ] I have checked that no combination of visible values (e.g. subtraction '
            'across rows or against a visible total) could reveal a suppressed cell value.',
            '',
            '### Reviewer 1',
            '',
            '| Field | Value |',
            '|-------|-------|',
            '| Name | |',
            '| Organisation / Role | |',
            '| Date of review | |',
            '| Decision | Approved / Approved with conditions / Rejected |',
            '| Conditions or observations | |',
            '| Signature | |',
            '',
            '### Reviewer 2',
            '',
            '| Field | Value |',
            '|-------|-------|',
            '| Name | |',
            '| Organisation / Role | |',
            '| Date of review | |',
            '| Decision | Approved / Approved with conditions / Rejected |',
            '| Conditions or observations | |',
            '| Signature | |',
            '',
            '### Lead researcher counter-signature',
            '',
            '_Required only if either reviewer raised conditions or rejected._',
            '',
            '| Field | Value |',
            '|-------|-------|',
            '| Name | |',
            '| Date | |',
            '| Response to conditions | |',
            '| Signature | |',
        ]

        report_text = '\n'.join(lines)
        output_path = self.output_dir.parent / 'report.md'
        with open(output_path, 'w') as f:
            f.write(report_text)

        print(f'[ResultsViz] Saved report to {output_path}')
        return str(output_path)

    def _derive_run_id(self) -> str:
        """Derive a human-readable run_id from results metadata.

        Priority:
        1. experiment_name + run_id from metadata (injected by run_experiment.py)
        2. experiment_name + model config + epoch (when training config present)
        3. Filename stem fallback
        """
        meta = self.results.get('metadata', {})
        experiment = meta.get('experiment_name', '')
        run_id = meta.get('run_id', '')
        n_embd = meta.get('n_embd', 'unknown')
        n_head = meta.get('n_head', 'unknown')
        n_layer = meta.get('n_layer', 'unknown')
        epoch = meta.get('epoch', 'unknown')

        if n_embd != 'unknown':
            # Full model config available — include architecture
            base = f'{experiment}__{run_id}' if experiment and run_id else (experiment or run_id)
            return f'{base}__{n_embd}x{n_head}x{n_layer}__ep{epoch}'
        elif experiment and run_id:
            return f'{experiment}__{run_id}'
        elif experiment:
            return experiment
        # Fallback: use results filename stem
        return self.results_path.stem

    def _save_run_summary(self) -> str:
        """Save machine-readable run_summary.json for cross-run joining."""
        meta = self.results.get('metadata', {})

        pass_count  = sum(1 for f in self.flags if f.status == 'pass')
        warn_count  = sum(1 for f in self.flags if f.status == 'warn')
        fail_count  = sum(1 for f in self.flags if f.status == 'fail')
        overall = 'fail' if fail_count > 0 else ('warn' if warn_count > 0 else 'pass')

        checks = {}
        for flag in self.flags:
            entry = {'status': flag.status, 'value': flag.value}
            if flag.natural_variation is not None:
                entry['natural_variation'] = flag.natural_variation
                entry['k'] = flag.k
                entry['ref_train_val'] = flag.ref_train_val
                entry['ref_train_test'] = flag.ref_train_test
                entry['ref_val_test'] = flag.ref_val_test
            checks[flag.metric] = entry

        # Key metrics from marginal distribution
        key_metrics = {}
        marginal = self.results.get('marginal_distribution', {})
        for fuel in ['electricity', 'gas']:
            if fuel in marginal:
                m = marginal[fuel]
                for metric in ['mean_ratio', 'std_ratio', 'quantile_mae']:
                    val = m.get(metric)
                    if val is not None:
                        key_metrics[f'{fuel}_{metric}'] = val

        temporal = self.results.get('temporal_patterns', {})
        for fuel in ['electricity', 'gas']:
            acf = temporal.get('acf', {}).get(fuel, {})
            if acf:
                diffs = [
                    d.get('train_diff') for d in acf.values()
                    if d.get('train_diff') is not None
                ]
                if diffs:
                    key_metrics[f'{fuel}_acf_max_diff'] = max(diffs)

        summary = {
            'run_id': self._derive_run_id(),
            'model': {
                'n_embd': meta.get('n_embd', 'unknown'),
                'n_head': meta.get('n_head', 'unknown'),
                'n_layer': meta.get('n_layer', 'unknown'),
            },
            'epoch': meta.get('epoch', 'unknown'),
            'tokeniser_version': meta.get('tokeniser_version', 'unknown'),
            'n_batches': meta.get('n_batches', 0),
            'n_households': {
                'synth': meta.get('n_households', 0),
                'train': meta.get('n_households_train', 0),
                'val':   meta.get('n_households_val', 0),
                'test':  meta.get('n_households_test', 0),
            },
            'n_values': meta.get('n_values', {}),
            'fidelity': {
                'n_pass': pass_count,
                'n_warn': warn_count,
                'n_fail': fail_count,
                'overall': overall,
                'checks': checks,
            },
            'key_metrics': key_metrics,
            'generated_at': datetime.now().isoformat(timespec='seconds'),
        }

        output_path = self.output_dir.parent / 'run_summary.json'
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        print(f'[ResultsViz] Saved run summary to {output_path}')
        return str(output_path)

    def _save_flags_csv(self) -> str:
        """Save fidelity flags to CSV (with household counts and NV references)."""
        rows = []
        for flag in self.flags:
            rows.append({
                'metric': flag.metric,
                'status': flag.status,
                'value': flag.value,
                'natural_variation': flag.natural_variation,
                'k': flag.k if flag.natural_variation is not None else None,
                'ref_train_val': flag.ref_train_val,
                'ref_train_test': flag.ref_train_test,
                'ref_val_test': flag.ref_val_test,
                'threshold': flag.threshold,
                'message': flag.message,
                'n_households_synth': self._n_households_synth,
                'n_households_train': _rn(self._n_households_train),
            })
        df = pd.DataFrame(rows)
        output_path = self.output_dir.parent / 'fidelity_flags.csv'
        df.to_csv(output_path, index=False)
        print(f'[ResultsViz] Saved fidelity flags to {output_path}')
        return str(output_path)

    # ------------------------------------------------------------------ #
    # Diagnostic visualizations (complex layout — inline dual-mode save) #
    # ------------------------------------------------------------------ #

    def _plot_daily_pattern_heatmaps(self) -> List[str]:
        """Plot 12×48 heatmaps (real, synth, diff) + key-month profile comparisons."""
        daily = self.results.get('daily_pattern_diagnostics', {})
        paths = []

        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        key_months = [1, 7, 12]
        key_month_names = ['January', 'July', 'December']
        time_labels = [f'{h:02d}:00' for h in range(0, 24, 2)]
        time_ticks = list(range(0, 48, 4))

        for fuel in ['electricity', 'gas']:
            fuel_data = daily.get(fuel, {})
            synth_data = fuel_data.get('synth', {})
            if not synth_data or 'mean' not in synth_data:
                continue

            def _build_matrix(data_dict):
                matrix = np.full((12, 48), np.nan)
                mean_data = data_dict.get('mean', {})
                for month_str, hh_dict in mean_data.items():
                    m = int(month_str) - 1
                    if 0 <= m < 12:
                        for hh_str, val in hh_dict.items():
                            hh = int(hh_str)
                            if 0 <= hh < 48:
                                matrix[m, hh] = val
                return matrix

            def _build_se_matrix(data_dict):
                matrix = np.full((12, 48), np.nan)
                se_data = data_dict.get('se', {})
                for month_str, hh_dict in se_data.items():
                    m = int(month_str) - 1
                    if 0 <= m < 12:
                        for hh_str, val in hh_dict.items():
                            hh = int(hh_str)
                            if 0 <= hh < 48:
                                matrix[m, hh] = val
                return matrix

            synth_matrix = _build_matrix(synth_data)
            synth_se = _build_se_matrix(synth_data)

            real_matrix = real_se = real_split = None
            for split in ['train', 'test']:
                split_data = fuel_data.get(split, {})
                if split_data and 'mean' in split_data:
                    real_matrix = _build_matrix(split_data)
                    real_se = _build_se_matrix(split_data)
                    real_split = split
                    break

            has_real = real_matrix is not None
            n_rows = 2 if has_real else 1

            for mode in ('paper', 'presentation'):
                style = STYLES[mode]
                base = style.get('figsize_2x3', (12, 8))
                figsize = (base[0], base[1] * n_rows / 2 + 1)
                with plt.rc_context(style['rc']):
                    fig = plt.figure(figsize=figsize)
                    gs = fig.add_gridspec(n_rows, 3, hspace=0.35, wspace=0.3)

                    vmin = np.nanmin(synth_matrix)
                    vmax = np.nanmax(synth_matrix)
                    if has_real:
                        vmin = min(vmin, np.nanmin(real_matrix))
                        vmax = max(vmax, np.nanmax(real_matrix))

                    ax_real  = fig.add_subplot(gs[0, 0])
                    ax_synth = fig.add_subplot(gs[0, 1])
                    ax_diff  = fig.add_subplot(gs[0, 2])

                    if has_real:
                        im = ax_real.imshow(real_matrix, aspect='auto', cmap='YlOrRd',
                                            vmin=vmin, vmax=vmax)
                        ax_real.set_title(f'Real ({real_split}) — {fuel.title()} Daily Profile')
                    else:
                        im = ax_real.imshow(synth_matrix, aspect='auto', cmap='YlOrRd',
                                            vmin=vmin, vmax=vmax)
                        ax_real.set_title(f'Synthetic — {fuel.title()} Daily Profile')

                    for _ax in [ax_real, ax_synth]:
                        _ax.set_yticks(range(12))
                        _ax.set_yticklabels(month_labels)
                        _ax.set_xticks(time_ticks)
                        _ax.set_xticklabels(time_labels, rotation=45, ha='right', fontsize=7)
                    fig.colorbar(im, ax=ax_real, shrink=0.8, label='Wh')

                    im2 = ax_synth.imshow(synth_matrix, aspect='auto', cmap='YlOrRd',
                                          vmin=vmin, vmax=vmax)
                    ax_synth.set_title(f'Synthetic — {fuel.title()} Daily Profile')
                    fig.colorbar(im2, ax=ax_synth, shrink=0.8, label='Wh')

                    if has_real:
                        diff = synth_matrix - real_matrix
                        abs_max = max(np.nanmax(np.abs(diff)), 1.0)
                        im3 = ax_diff.imshow(diff, aspect='auto', cmap='RdBu_r',
                                             vmin=-abs_max, vmax=abs_max)
                        ax_diff.set_title('Difference (Synth − Real)')
                    else:
                        im3 = ax_diff.imshow(np.zeros_like(synth_matrix), aspect='auto',
                                             cmap='RdBu_r', vmin=-1, vmax=1)
                        ax_diff.set_title('Difference (no real data)')
                    ax_diff.set_yticks(range(12))
                    ax_diff.set_yticklabels(month_labels)
                    ax_diff.set_xticks(time_ticks)
                    ax_diff.set_xticklabels(time_labels, rotation=45, ha='right', fontsize=7)
                    fig.colorbar(im3, ax=ax_diff, shrink=0.8, label='Wh')

                    if has_real:
                        shared_ymax = 0
                        for km in key_months:
                            for matrix, se_m in [(synth_matrix, synth_se), (real_matrix, real_se)]:
                                upper = matrix[km - 1, :] + 1.96 * se_m[km - 1, :]
                                peak = np.nanmax(upper)
                                if not np.isnan(peak):
                                    shared_ymax = max(shared_ymax, peak)
                        shared_ymax *= 1.05

                        for col_idx, (km, km_name) in enumerate(zip(key_months, key_month_names)):
                            ax_line = fig.add_subplot(gs[1, col_idx])
                            x = np.arange(48)
                            sp = synth_matrix[km - 1, :]
                            ss = synth_se[km - 1, :]
                            ax_line.plot(x, sp, color=COLOUR_SYNTH, linewidth=1.5, label=LABEL_SYNTH)
                            ax_line.fill_between(x, sp - 1.96 * ss, sp + 1.96 * ss,
                                                 alpha=0.2, color=COLOUR_SYNTH)
                            rp = real_matrix[km - 1, :]
                            rs = real_se[km - 1, :]
                            ax_line.plot(x, rp, color=COLOUR_TRAIN, linewidth=1.5, label=LABEL_TRAIN)
                            ax_line.fill_between(x, rp - 1.96 * rs, rp + 1.96 * rs,
                                                 alpha=0.2, color=COLOUR_TRAIN)
                            ax_line.set_title(f'{km_name} Daily Profile')
                            ax_line.set_xlabel('Time of Day')
                            ax_line.set_ylabel(f'{fuel.title()} (Wh)')
                            ax_line.set_xticks(time_ticks)
                            ax_line.set_xticklabels(time_labels, rotation=45, ha='right', fontsize=7)
                            ax_line.legend(fontsize=8)
                            ax_line.set_ylim(0, shared_ymax)
                            ax_line.grid(True, alpha=0.3)

                    fig.suptitle(f'Daily Temporal Pattern Analysis — {fuel.title()}',
                                 fontweight='bold')
                    path = self.output_dir / f'daily_pattern_heatmap_{fuel}_{mode}.png'
                    fig.savefig(path, dpi=style['dpi'], bbox_inches='tight')
                    plt.close(fig)
                    if mode == 'paper':
                        paths.append(str(path))
                    print(f'[ResultsViz] Saved heatmap {mode} to {path}')

        return paths

    def _plot_daily_pattern_degradation(self) -> List[str]:
        """Plot weekly degradation analysis."""
        daily = self.results.get('daily_pattern_diagnostics', {})
        degradation = daily.get('weekly_degradation', {})
        paths = []

        split_styles_deg = {
            'synth': {'color': COLOUR_SYNTH, 'marker': 'o', 'label': LABEL_SYNTH},
            'train': {'color': COLOUR_TRAIN, 'marker': 's', 'label': LABEL_TRAIN},
            'test':  {'color': COLOUR_TEST,  'marker': '^', 'label': LABEL_TEST},
        }

        for fuel in ['electricity', 'gas']:
            fuel_data = degradation.get(fuel, {})
            synth_metrics = fuel_data.get('synth', {}).get('degradation_metrics', [])
            if not synth_metrics:
                synth_metrics = fuel_data.get('degradation_metrics', [])
            if not synth_metrics:
                continue

            for mode in ('paper', 'presentation'):
                style = STYLES[mode]
                with plt.rc_context(style['rc']):
                    fig, axes = plt.subplots(2, 2, figsize=style.get('figsize_2x2', (10, 8)))
                    metric_configs = [
                        (axes[0, 0], 'cv', 'Coefficient of Variation', None),
                        (axes[0, 1], 'correlation_to_week1', 'Correlation to Week 1', (0, 1.05)),
                        (axes[1, 0], 'peak_to_trough', 'Peak-to-Trough Ratio', None),
                        (axes[1, 1], 'mean', 'Mean Consumption (Wh)', None),
                    ]

                    for split_name, s in split_styles_deg.items():
                        split_data = fuel_data.get(split_name, {})
                        metrics = split_data.get('degradation_metrics', [])
                        if not metrics and split_name == 'synth':
                            metrics = fuel_data.get('degradation_metrics', [])
                        if not metrics:
                            continue
                        weeks = [m['week'] for m in metrics]
                        for ax, metric_key, title, ylim in metric_configs:
                            values = [m.get(metric_key, np.nan) for m in metrics]
                            ax.plot(weeks, values, f'{s["marker"]}-',
                                    color=s['color'], label=s['label'],
                                    linewidth=1.5, markersize=5, alpha=0.85)

                    for ax, metric_key, title, ylim in metric_configs:
                        ax.set_title(title)
                        ax.set_xlabel('Week')
                        ax.grid(True, alpha=0.3)
                        if ylim:
                            ax.set_ylim(*ylim)
                        ax.legend(fontsize=8)

                    fig.suptitle(f'{fuel.title()} — Weekly Degradation Analysis', fontweight='bold')
                    fig.tight_layout()
                    path = self.output_dir / f'daily_pattern_degradation_{fuel}_{mode}.png'
                    fig.savefig(path, dpi=style['dpi'])
                    plt.close(fig)
                    if mode == 'paper':
                        paths.append(str(path))

        return paths

    def _plot_household_month_fidelity(self) -> List[str]:
        """Plot household-month fidelity: marginal quintiles + conditioning grids."""
        hh_fidelity = self.results.get('household_month_fidelity', {})
        paths = []

        for mode in ('paper', 'presentation'):
            style = STYLES[mode]
            with plt.rc_context(style['rc']):
                fig1, axes1 = plt.subplots(2, 2, figsize=style.get('figsize_2x2', (10, 8)))

                for col_idx, fuel in enumerate(['electricity', 'gas']):
                    fuel_data = hh_fidelity.get(fuel, {})
                    synth = fuel_data.get('synth', {})
                    splits = [
                        ('synth', synth, LABEL_SYNTH, COLOUR_SYNTH),
                        ('train', fuel_data.get('train', {}), LABEL_TRAIN, COLOUR_TRAIN),
                        ('val',   fuel_data.get('val',   {}), LABEL_VAL,   COLOUR_VAL),
                        ('test',  fuel_data.get('test',  {}), LABEL_TEST,  COLOUR_TEST),
                    ]
                    # Only include splits that have data
                    splits = [(n, d, l, c) for n, d, l, c in splits if d]

                    ax = axes1[0, col_idx]
                    stats_order = ['Q20', 'Q40', 'mean', 'Q60', 'Q80']
                    x_labels = ['q20', 'q40', 'mean', 'q60', 'q80']
                    x = np.arange(len(x_labels))
                    n_splits = len(splits)
                    width = 0.8 / n_splits
                    offsets = np.linspace(-(n_splits - 1) / 2, (n_splits - 1) / 2, n_splits) * width

                    split_vals = {}  # split_name -> list of values (for ratio panel)
                    for offset, (split_name, split_data, label, colour) in zip(offsets, splits):
                        vals, cis = [], []
                        for stat in stats_order:
                            if stat == 'mean':
                                m = split_data.get('mean', {})
                                vals.append(m.get('mean', 0) if isinstance(m, dict) else 0)
                                cis.append(1.96 * m.get('se', 0) if isinstance(m, dict) else 0)
                            else:
                                vals.append(split_data.get(stat, 0))
                                cis.append(0)
                        split_vals[split_name] = vals
                        ax.bar(x + offset, vals, width, yerr=cis, label=label,
                               color=colour, capsize=3, error_kw={'linewidth': 1}, alpha=0.85)

                    ax.set_xticks(x)
                    ax.set_xticklabels(x_labels)
                    ax.set_ylabel('Wh')
                    ax.set_title(f'{fuel.upper()}: Household-Month Means')
                    ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.3, axis='y')

                    # Ratio panel: synth / each real split
                    ax_ratio = axes1[1, col_idx]
                    synth_vals = split_vals.get('synth', [])
                    real_splits = [(n, d, l, c) for n, d, l, c in splits if n != 'synth']
                    if synth_vals and real_splits:
                        n_real = len(real_splits)
                        r_width = 0.8 / n_real
                        r_offsets = np.linspace(-(n_real - 1) / 2, (n_real - 1) / 2, n_real) * r_width
                        for r_offset, (split_name, _, label, colour) in zip(r_offsets, real_splits):
                            real_vals = split_vals.get(split_name, [])
                            if not real_vals:
                                continue
                            ratios = [sv / rv if rv != 0 else np.nan
                                      for sv, rv in zip(synth_vals, real_vals)]
                            bars = ax_ratio.bar(x + r_offset, ratios, r_width,
                                                label=f'synth/{label}', color=colour, alpha=0.7)
                            for bar, val in zip(bars, ratios):
                                if not np.isnan(val):
                                    ax_ratio.annotate(
                                        f'{val:.2f}',
                                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                                        ha='center', va='bottom', fontsize=7, fontweight='bold')
                        ax_ratio.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5)
                        ax_ratio.set_xticks(x)
                        ax_ratio.set_xticklabels(x_labels)
                        ax_ratio.set_ylabel('Synthetic / Real Ratio')
                        ax_ratio.set_title(f'{fuel.upper()}: Distribution Ratio')
                        ax_ratio.legend(fontsize=7)
                        ax_ratio.grid(True, alpha=0.3, axis='y')

                fig1.suptitle('Household-Month Marginal Distribution Comparison', fontweight='bold')
                fig1.tight_layout()
                path1 = self.output_dir / f'household_month_fidelity_marginal_{mode}.png'
                fig1.savefig(path1, dpi=style['dpi'], bbox_inches='tight')
                plt.close(fig1)
                if mode == 'paper':
                    paths.append(str(path1))

            # Conditioning per fuel
            for fuel in ['electricity', 'gas']:
                fuel_data = hh_fidelity.get(fuel, {})
                conditioning = fuel_data.get('conditioning', {})
                cond_vars = [v for v in conditioning.keys() if conditioning[v]]
                if not cond_vars:
                    continue
                n_vars = len(cond_vars)
                n_cols = 2
                n_rows = (n_vars + 1) // 2
                with plt.rc_context(style['rc']):
                    fig, axes = plt.subplots(n_rows, n_cols,
                                             figsize=style.get('figsize_2x2', (10, 8)))
                    if n_rows == 1:
                        axes = axes.reshape(1, -1)
                    for i, cond_var in enumerate(cond_vars):
                        ax = axes[i // 2, i % 2]
                        cond_data = conditioning[cond_var]
                        synth_groups = cond_data.get('synth', {})
                        train_groups = cond_data.get('train', {})
                        val_groups   = cond_data.get('val',   {})
                        test_groups  = cond_data.get('test',  {})
                        all_categories = sorted(
                            set(list(synth_groups.keys()) + list(train_groups.keys())), key=str)

                        # Apply SDC suppression: exclude categories where train n_households < threshold
                        categories = [
                            c for c in all_categories
                            if train_groups.get(c, {}).get('n_households', 0) >= SDC_MIN_HOUSEHOLDS
                        ]
                        n_suppressed = len(all_categories) - len(categories)

                        if not categories:
                            ax.text(0.5, 0.5, 'All groups suppressed (SDC)',
                                    ha='center', va='center', transform=ax.transAxes,
                                    fontsize=9, color='grey')
                            ax.set_title(f'Mean by {cond_var}')
                            continue

                        x = np.arange(len(categories))
                        split_groups = [
                            (synth_groups, LABEL_SYNTH, COLOUR_SYNTH),
                            (train_groups, LABEL_TRAIN, COLOUR_TRAIN),
                        ]
                        if val_groups:
                            split_groups.append((val_groups, LABEL_VAL, COLOUR_VAL))
                        if test_groups:
                            split_groups.append((test_groups, LABEL_TEST, COLOUR_TEST))

                        n_splits = len(split_groups)
                        width = 0.8 / n_splits
                        offsets = np.linspace(-(n_splits - 1) / 2, (n_splits - 1) / 2, n_splits) * width

                        for offset, (groups, label, colour) in zip(offsets, split_groups):
                            means = [groups.get(c, {}).get('mean', 0) for c in categories]
                            errs  = [1.96 * groups.get(c, {}).get('se', 0) for c in categories]
                            ax.bar(x + offset, means, width, yerr=errs, label=label,
                                   color=colour, capsize=3, alpha=0.85)

                        ax.set_xticks(x)
                        ax.set_xticklabels([str(c)[:15] for c in categories],
                                           rotation=45, ha='right')
                        ax.set_ylabel('Mean Consumption (Wh)')
                        title = f'Mean by {cond_var}'
                        if n_suppressed:
                            title += f' ({n_suppressed} suppressed)'
                        ax.set_title(title)
                        ax.legend(loc='upper right', fontsize=8)
                        ax.grid(True, alpha=0.3, axis='y')
                    for i in range(n_vars, n_rows * n_cols):
                        axes[i // 2, i % 2].set_visible(False)
                    fig.suptitle(f'HH-Month {fuel.upper()}: Conditioning Comparison',
                                 fontweight='bold')
                    fig.tight_layout()
                    path = self.output_dir / f'household_month_fidelity_conditioning_{fuel}_{mode}.png'
                    fig.savefig(path, dpi=style['dpi'], bbox_inches='tight')
                    plt.close(fig)
                    if mode == 'paper':
                        paths.append(str(path))

        return paths

    def _plot_token_frequency(self, fuel: str) -> str:
        """Plot token frequency bars (synth/real) and CDF comparison."""
        token = self.results.get('token_analysis', {}).get(fuel, {}).get('frequency', {})
        if not token:
            return ''

        for mode in ('paper', 'presentation'):
            style = STYLES[mode]
            with plt.rc_context(style['rc']):
                fig, axes = plt.subplots(1, 2, figsize=style.get('figsize_wide', (10, 4)))
                freqs = token.get('frequencies', [])
                n_bins = len(freqs)
                x = np.arange(n_bins)

                ax = axes[0]
                ax.bar(x, freqs, alpha=0.7, color=COLOUR_SYNTH, label=LABEL_SYNTH, width=0.8)
                train_freqs = token.get('train_frequencies')
                if train_freqs:
                    ax.step(x, train_freqs, where='mid', color=COLOUR_TRAIN,
                            linewidth=1.5, label=LABEL_TRAIN)
                ax.set_title('Token Frequency Distribution')
                ax.set_xlabel('Token Index')
                ax.set_ylabel('Frequency')
                ax.legend(fontsize=7)

                ax = axes[1]
                cdf = token.get('cdf', [])
                ax.plot(x, cdf, color=COLOUR_SYNTH, label=LABEL_SYNTH)
                train_cdf = token.get('train_cdf')
                if train_cdf:
                    ax.plot(x, train_cdf, color=COLOUR_TRAIN, label=LABEL_TRAIN)
                    max_gap = token.get('train_max_cdf_gap')
                    ax.set_title(f'CDF Comparison (max gap: {max_gap:.4f})'
                                 if max_gap else 'CDF Comparison')
                else:
                    ax.set_title('CDF')
                ax.set_xlabel('Token Index')
                ax.set_ylabel('Cumulative Frequency')
                ax.legend(fontsize=7)

                fig.suptitle(f'{fuel.title()} — Token Frequency Analysis', fontweight='bold')
                fig.tight_layout()
                path = self.output_dir / f'token_frequency_comparison_{fuel}_{mode}.png'
                fig.savefig(path, dpi=style['dpi'])
                plt.close(fig)

        return str(self.output_dir / f'token_frequency_comparison_{fuel}_paper.png')

    def _plot_token_distribution(self) -> str:
        """Plot value distribution histograms (real vs synth) per fuel."""
        token = self.results.get('token_analysis', {})
        elec = token.get('electricity', {})
        gas = token.get('gas', {})
        has_elec = bool(elec.get('value_samples_synth'))
        has_gas = bool(gas.get('value_samples_synth'))
        if not has_elec and not has_gas:
            return ''

        for mode in ('paper', 'presentation'):
            style = STYLES[mode]
            with plt.rc_context(style['rc']):
                fig, axes = plt.subplots(1, 2, figsize=style.get('figsize_wide', (10, 4)))
                for fuel_data, ax in [(elec, axes[0]), (gas, axes[1])]:
                    fuel = 'electricity' if fuel_data is elec else 'gas'
                    synth_vals = fuel_data.get('value_samples_synth', [])
                    real_vals  = fuel_data.get('value_samples_real', [])
                    if synth_vals and real_vals:
                        synth_arr = np.array(synth_vals)
                        real_arr  = np.array(real_vals)
                        max_val = max(np.percentile(real_arr, 99), np.percentile(synth_arr, 99))
                        bins = np.linspace(0, max_val, 50)
                        ax.hist(real_arr, bins=bins, alpha=0.5, label=LABEL_TRAIN,
                                density=True, color=COLOUR_TRAIN)
                        ax.hist(synth_arr, bins=bins, alpha=0.5, label=LABEL_SYNTH,
                                density=True, color=COLOUR_SYNTH)
                        ax.axvline(float(np.mean(real_arr)), color=COLOUR_TRAIN,
                                   linestyle='-', linewidth=2,
                                   label=f'{LABEL_TRAIN} mean: {np.mean(real_arr):.0f}')
                        ax.axvline(float(np.mean(synth_arr)), color=COLOUR_SYNTH,
                                   linestyle='-', linewidth=2,
                                   label=f'{LABEL_SYNTH} mean: {np.mean(synth_arr):.0f}')
                        ax.set_xlabel(f'{fuel.title()} (Wh)')
                        ax.set_ylabel('Density')
                        ax.set_title(f'{fuel.title()} Value Distribution')
                        ax.legend(fontsize=8)
                    else:
                        ax.set_title(f'{fuel.title()} Value Distribution (no data)')

                fig.suptitle('Token Distribution Analysis', fontweight='bold')
                fig.tight_layout()
                path = self.output_dir / f'token_distribution_analysis_{mode}.png'
                fig.savefig(path, dpi=style['dpi'], bbox_inches='tight')
                plt.close(fig)

        return str(self.output_dir / 'token_distribution_analysis_paper.png')

    # ------------------------------------------------------------------ #
    # Diagnostics CSVs — saved alongside their corresponding figures     #
    # ------------------------------------------------------------------ #

    def _save_diagnostics_csvs(self) -> None:
        """Save diagnostic data CSVs alongside their corresponding figures.

        Each CSV uses the same stem as the plot it accompanies so the
        files sit next to each other in the visualizations/ directory.
        seasonal_analysis.csv is skipped here — it is already produced
        by _extract_seasonal_data() with the main figures.
        """
        # Token frequency — alongside token_frequency_comparison_{fuel}_{mode}.png
        SDC_MIN_TOKEN_COUNT = 10
        token = self.results.get('token_analysis', {})
        if token:
            for fuel, fuel_data in token.items():
                freq = fuel_data.get('frequency', {})
                if not freq or not freq.get('frequencies'):
                    continue
                synth_counts  = freq.get('counts', [])
                train_counts  = freq.get('train_counts', [])
                test_counts   = freq.get('test_counts', [])
                n_synth_total = freq.get('n_tokens_total', 0)
                n_train_total = freq.get('train_n_tokens_total', 0)
                n_test_total  = freq.get('test_n_tokens_total', 0)
                rows = []
                for i, (synth_f, dev) in enumerate(zip(
                    freq['frequencies'],
                    freq.get('deviation_from_uniform', [0] * len(freq['frequencies']))
                )):
                    train_count = train_counts[i] if i < len(train_counts) else None
                    suppressed  = train_count is not None and train_count < SDC_MIN_TOKEN_COUNT
                    row = {
                        'fuel':              fuel,
                        'token_index':       i,
                        'n_households_synth': self._n_households_synth,
                        'n_households_train': _rn(self._n_households_train),
                        'n_households_test':  _rn(self._n_households_test),
                        'suppressed':        suppressed,
                        'synth_count':       synth_counts[i] if i < len(synth_counts) else None,
                        'synth_frequency':   synth_f if not suppressed else None,
                        'deviation_from_uniform': dev if not suppressed else None,
                        'train_count':       _rn(train_count) if not suppressed else None,
                        'train_frequency':   freq['train_frequencies'][i] if not suppressed and i < len(freq.get('train_frequencies', [])) else None,
                        'test_count':        _rn(test_counts[i]) if not suppressed and i < len(test_counts) else None,
                        'test_frequency':    freq['test_frequencies'][i] if not suppressed and i < len(freq.get('test_frequencies', [])) else None,
                    }
                    rows.append(row)
                if rows:
                    pd.DataFrame(rows).to_csv(
                        self.output_dir / f'token_frequency_comparison_{fuel}.csv', index=False)

            # Token distribution — alongside token_distribution_analysis_{mode}.png
            # CSV mirrors the plot: binned histogram from value_samples, not monthly_decoded.
            # Bins where train_count < 10 are suppressed (SDC precaution).
            SDC_MIN_BIN_COUNT = 10
            for fuel in ['electricity', 'gas']:
                fuel_data = token.get(fuel, {})
                synth_vals = fuel_data.get('value_samples_synth', [])
                real_vals  = fuel_data.get('value_samples_real', [])
                if not synth_vals or not real_vals:
                    continue
                synth_arr = np.array(synth_vals)
                real_arr  = np.array(real_vals)
                max_val = max(np.percentile(real_arr, 99), np.percentile(synth_arr, 99))
                bins = np.linspace(0, max_val, 50)
                synth_counts, _ = np.histogram(synth_arr, bins=bins)
                real_counts,  _ = np.histogram(real_arr,  bins=bins)
                synth_total = synth_counts.sum()
                real_total  = real_counts.sum()
                rows = []
                for i in range(len(bins) - 1):
                    train_count = int(real_counts[i])
                    suppressed  = train_count < SDC_MIN_BIN_COUNT
                    row = {
                        'fuel':              fuel,
                        'bin_left':          float(bins[i]),
                        'bin_right':         float(bins[i + 1]),
                        'bin_centre':        float((bins[i] + bins[i + 1]) / 2),
                        'n_households_synth': self._n_households_synth,
                        'n_households_train': _rn(self._n_households_train),
                        'suppressed':        suppressed,
                        'train_count':       _rn(train_count) if not suppressed else None,
                        'train_density':     float(real_counts[i] / (real_total * (bins[i+1] - bins[i]))) if not suppressed and real_total > 0 else None,
                        'synth_count':       int(synth_counts[i]) if not suppressed else None,
                        'synth_density':     float(synth_counts[i] / (synth_total * (bins[i+1] - bins[i]))) if not suppressed and synth_total > 0 else None,
                    }
                    rows.append(row)
                pd.DataFrame(rows).to_csv(
                    self.output_dir / f'token_distribution_analysis_{fuel}.csv', index=False)

        # Monthly heatmap profiles — alongside daily_pattern_heatmap_{fuel}_{mode}.png
        daily = self.results.get('daily_pattern_diagnostics', {})
        if daily:
            n_hh_map = {
                'synth': self._n_households_synth,
                'train': _rn(self._n_households_train),
                'val':   _rn(self._n_households_val),
                'test':  _rn(self._n_households_test),
            }
            for fuel in ['electricity', 'gas']:
                fuel_data = daily.get(fuel, {})
                rows = []
                for split in ['synth', 'train', 'val', 'test']:
                    split_data = fuel_data.get(split, {})
                    mean_data = split_data.get('mean', {})
                    if mean_data:
                        try:
                            df_heat = pd.DataFrame(mean_data).T
                            df_heat.insert(0, 'n_households', n_hh_map.get(split, 0))
                            df_heat.insert(0, 'split', split)
                            rows.append(df_heat)
                        except Exception:
                            pass
                if rows:
                    try:
                        combined = pd.concat(rows)
                        combined.index.name = 'month'
                        combined.to_csv(
                            self.output_dir / f'daily_pattern_heatmap_{fuel}.csv', index=True)
                    except Exception:
                        pass

        # Weekly degradation — alongside daily_pattern_degradation_{fuel}_{mode}.png
        weekly_deg = daily.get('weekly_degradation', {}) if daily else {}
        if weekly_deg:
            n_hh_map = {
                'synth': self._n_households_synth,
                'train': _rn(self._n_households_train),
                'val':   _rn(self._n_households_val),
                'test':  _rn(self._n_households_test),
            }
            for fuel in ['electricity', 'gas']:
                fuel_deg = weekly_deg.get(fuel, {})
                rows = []
                for split in ['synth', 'train', 'val', 'test']:
                    split_data = fuel_deg.get(split, {})
                    metrics = split_data.get('degradation_metrics', [])
                    if not metrics and split == 'synth':
                        metrics = fuel_deg.get('degradation_metrics', [])
                    for m in metrics:
                        rows.append({'split': split,
                                     'n_households': n_hh_map.get(split, 0),
                                     **m})
                if rows:
                    pd.DataFrame(rows).to_csv(
                        self.output_dir / f'daily_pattern_degradation_{fuel}.csv', index=False)

        # Household-month fidelity — alongside household_month_fidelity_marginal_{mode}.png
        hh = self.results.get('household_month_fidelity', {})
        if hh:
            rows = []
            for fuel, fuel_data in hh.items():
                for split in ['synth', 'train', 'val', 'test']:
                    split_data = fuel_data.get(split, {})
                    if isinstance(split_data, dict):
                        _raw_n = split_data.get('n_households')
                        _n = _rn(_raw_n) if split in ('train', 'val', 'test') else _raw_n
                        row = {'fuel': fuel, 'split': split, 'n_households': _n}
                        for k, v in split_data.items():
                            if k == 'n_households':
                                continue
                            if isinstance(v, dict):
                                for sk, sv in v.items():
                                    row[f'{k}_{sk}'] = sv
                            else:
                                row[k] = v
                        rows.append(row)
            if rows:
                pd.DataFrame(rows).to_csv(
                    self.output_dir / 'household_month_fidelity_marginal.csv', index=False)

            # Household-month conditioning — one CSV per fuel alongside the conditioning PNGs
            for fuel, fuel_data in hh.items():
                conditioning = fuel_data.get('conditioning', {})
                if not conditioning:
                    continue
                cond_rows = []
                for cond_var, cond_data in conditioning.items():
                    synth_groups = cond_data.get('synth', {})
                    train_groups = cond_data.get('train', {})
                    val_groups   = cond_data.get('val',   {})
                    test_groups  = cond_data.get('test',  {})
                    all_categories = sorted(
                        set(list(synth_groups.keys()) + list(train_groups.keys())), key=str)
                    for category in all_categories:
                        sg = synth_groups.get(category, {})
                        tg = train_groups.get(category, {})
                        vg = val_groups.get(category, {})
                        xg = test_groups.get(category, {})
                        train_n_hh = tg.get('n_households', 0)
                        val_n_hh   = vg.get('n_households', 0)
                        test_n_hh  = xg.get('n_households', 0)
                        suppressed = (0 < train_n_hh < SDC_MIN_HOUSEHOLDS) or \
                                     (0 < val_n_hh   < SDC_MIN_HOUSEHOLDS) or \
                                     (0 < test_n_hh  < SDC_MIN_HOUSEHOLDS)
                        row = {
                            'fuel': fuel,
                            'cond_var': cond_var,
                            'category': category,
                            'synth_n_households': sg.get('n_households'),
                            'train_n_households': _rn(train_n_hh) if not suppressed and train_n_hh else None,
                            'val_n_households':   _rn(vg.get('n_households')) if not suppressed else None,
                            'test_n_households':  _rn(xg.get('n_households')) if not suppressed else None,
                            'suppressed': suppressed,
                        }
                        if not suppressed:
                            row.update({
                                'synth_mean': sg.get('mean'),
                                'synth_se': sg.get('se'),
                                'synth_n_household_months': sg.get('n'),
                                'train_mean': tg.get('mean'),
                                'train_se': tg.get('se'),
                                'train_n_household_months': tg.get('n'),
                                'val_mean': vg.get('mean') if vg else None,
                                'val_se': vg.get('se') if vg else None,
                                'val_n_household_months': vg.get('n') if vg else None,
                                'test_mean': xg.get('mean') if xg else None,
                                'test_se': xg.get('se') if xg else None,
                                'test_n_household_months': xg.get('n') if xg else None,
                            })
                        else:
                            row.update({
                                'synth_mean': None, 'synth_se': None, 'synth_n_household_months': None,
                                'train_mean': None, 'train_se': None, 'train_n_household_months': None,
                                'val_mean': None, 'val_se': None, 'val_n_household_months': None,
                                'test_mean': None, 'test_se': None, 'test_n_household_months': None,
                            })
                        cond_rows.append(row)
                if cond_rows:
                    pd.DataFrame(cond_rows).to_csv(
                        self.output_dir / f'household_month_fidelity_conditioning_{fuel}.csv',
                        index=False)


def visualize_results(results_path: str, output_dir: str) -> Dict[str, Any]:
    """
    Convenience function to run full visualization pipeline.

    Args:
        results_path: Path to parallel_evaluation_results.json
        output_dir: Directory for output files

    Returns:
        Dictionary with output file paths
    """
    visualizer = ResultsVisualizer(results_path, output_dir)
    return visualizer.run_full_analysis()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Visualize parallel evaluation results')
    parser.add_argument('results_path', help='Path to parallel_evaluation_results.json')
    parser.add_argument('output_dir', help='Directory for output files')
    args = parser.parse_args()
    visualize_results(args.results_path, args.output_dir)
