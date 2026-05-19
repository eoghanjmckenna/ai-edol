"""
SDC Sanitiser for parallel_evaluation_results.json
====================================================

Produces an SDC-compliant version of the raw evaluation results that can be
exported from the TRE alongside the visualisations/  folder.

Rules applied
-------------
1. **Suppress** any conditioning group or household-month group where the number
   of real households falls below SDC_MIN_HOUSEHOLDS.  Both the statistics *and*
   the accompanying N values are set to null; a ``suppressed: true`` flag is
   added to the group.
2. **Round** all real-data household counts (train / val / test N values) to the
   nearest 10.
3. **Round** token and histogram observation counts (train_counts / test_counts
   arrays) to the nearest 10; tokens with a rounded train_count below the
   threshold are also suppressed.
4. **Remove** raw observation sample arrays (``*_diff_samples``,
   ``value_samples_real``, ``value_samples_synth``) — these are only used
   internally to render histogram plots and must not leave the TRE.

Everything else is passed through unchanged.  Synthetic-data counts and
statistics are not subject to suppression or rounding.

Usage
-----
Typically called from ResultsVisualizer.visualize(); can also be run standalone::

    python -m src.evaluation.mapreduce.sdc_sanitiser \\
        path/to/parallel_evaluation_results.json \\
        path/to/parallel_evaluation_results_sdc.json
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from src.evaluation.plot_style import SDC_MIN_HOUSEHOLDS
except ImportError:
    SDC_MIN_HOUSEHOLDS = 10  # fallback


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rn(n: Any) -> Optional[int]:
    """Round a household count to the nearest 10 for SDC-compliant publication."""
    if n is None:
        return None
    try:
        return round(int(n) / 10) * 10
    except (TypeError, ValueError):
        return n


def _suppressed_n(n: Any) -> bool:
    """Return True if n is a real, positive count below the SDC threshold."""
    try:
        v = int(n)
        return 0 < v < SDC_MIN_HOUSEHOLDS
    except (TypeError, ValueError):
        return False


def _null_group_stats(group: Dict) -> Dict:
    """Replace all numeric stats in a leaf group dict with None, add suppressed flag."""
    result = {}
    for k, v in group.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            result[k] = None
        else:
            result[k] = v
    result['suppressed'] = True
    return result


# ---------------------------------------------------------------------------
# Per-section sanitisers
# ---------------------------------------------------------------------------

def _sanitise_metadata(meta: Dict) -> Dict:
    meta = dict(meta)
    for key in ('n_households_train', 'n_households_val', 'n_households_test'):
        if key in meta:
            meta[key] = _rn(meta[key])
    return meta


def _sanitise_marginal_distribution(marg: Dict) -> Dict:
    """Round per-split N values (train_n, val_n, test_n) in marginal distribution."""
    marg = copy.deepcopy(marg)
    for fuel_data in marg.values():
        if not isinstance(fuel_data, dict):
            continue
        for key in ('train_n', 'val_n', 'test_n'):
            if key in fuel_data:
                fuel_data[key] = _rn(fuel_data[key])
    return marg


def _sanitise_conditional_distribution(cond: Dict) -> Dict:
    """Suppress small groups; round N values in non-suppressed groups."""
    cond = copy.deepcopy(cond)
    for cond_var, groups in cond.items():
        if not isinstance(groups, dict):
            continue
        for cond_val, fuels in groups.items():
            if not isinstance(fuels, dict):
                continue
            # Determine suppression from any fuel (they share the same households)
            suppressed = False
            for fuel_stats in fuels.values():
                if not isinstance(fuel_stats, dict):
                    continue
                n_train = fuel_stats.get('n_households_train', 0)
                n_test  = fuel_stats.get('n_households_test',  0)
                if _suppressed_n(n_train) or _suppressed_n(n_test):
                    suppressed = True
                    break
            for fuel, fuel_stats in fuels.items():
                if not isinstance(fuel_stats, dict):
                    continue
                if suppressed:
                    fuels[fuel] = _null_group_stats(fuel_stats)
                else:
                    for key in ('n_households_train', 'n_households_val',
                                'n_households_test'):
                        if key in fuel_stats:
                            fuel_stats[key] = _rn(fuel_stats[key])
    return cond


def _sanitise_temporal_patterns(tp: Dict) -> Dict:
    """Remove raw sample arrays from volatility; leave daily_profiles and acf intact."""
    tp = copy.deepcopy(tp)
    volatility = tp.get('volatility', {})
    for fuel_data in volatility.values():
        if not isinstance(fuel_data, dict):
            continue
        for key in ('synth_diff_samples', 'train_diff_samples', 'test_diff_samples'):
            fuel_data.pop(key, None)
    return tp


def _sanitise_seasonal_analysis(seasonal: Dict) -> Dict:
    """Round household N per month entry for real-data splits."""
    seasonal = copy.deepcopy(seasonal)
    for fuel_data in seasonal.values():
        if not isinstance(fuel_data, dict):
            continue
        for split, monthly in fuel_data.items():
            if split in ('train', 'val', 'test') and isinstance(monthly, dict):
                for month_data in monthly.values():
                    if isinstance(month_data, dict) and 'n' in month_data:
                        month_data['n'] = _rn(month_data['n'])
    return seasonal


def _sanitise_household_month_fidelity(hmf: Dict) -> Dict:
    """Round N for top-level splits; suppress/round N for conditioning groups."""
    hmf = copy.deepcopy(hmf)
    for fuel_data in hmf.values():
        if not isinstance(fuel_data, dict):
            continue

        # Top-level split data (train/val/test): round n_households
        for split in ('train', 'val', 'test'):
            split_data = fuel_data.get(split)
            if isinstance(split_data, dict):
                if 'n_households' in split_data:
                    split_data['n_households'] = _rn(split_data['n_households'])

        # Conditioning groups: structure is [cond_var][split][cond_val]
        conditioning = fuel_data.get('conditioning', {})
        for cond_var, cv_data in conditioning.items():
            if not isinstance(cv_data, dict):
                continue
            # Collect all cond_vals that appear in any real-data split
            train_groups = cv_data.get('train', {})
            val_groups   = cv_data.get('val',   {})
            test_groups  = cv_data.get('test',  {})
            all_cond_vals = set(train_groups) | set(val_groups) | set(test_groups)

            for cond_val in all_cond_vals:
                n_train = (train_groups.get(cond_val) or {}).get('n_households', 0)
                n_val   = (val_groups.get(cond_val)   or {}).get('n_households', 0)
                n_test  = (test_groups.get(cond_val)  or {}).get('n_households', 0)
                suppressed = (_suppressed_n(n_train) or _suppressed_n(n_val)
                              or _suppressed_n(n_test))

                for split in ('synth', 'train', 'val', 'test'):
                    group = cv_data.get(split, {}).get(cond_val)
                    if not isinstance(group, dict):
                        continue
                    if suppressed:
                        cv_data[split][cond_val] = _null_group_stats(group)
                    else:
                        if 'n_households' in group:
                            group['n_households'] = _rn(group['n_households'])
    return hmf


def _sanitise_token_analysis(tok: Dict) -> Dict:
    """
    Remove raw sample arrays; round and suppress token/bin counts.
    weekly_cascade contains only synthetic data and is passed through unchanged.
    """
    tok = copy.deepcopy(tok)
    for fuel_data in tok.values():
        if not isinstance(fuel_data, dict):
            continue

        # Remove raw observation sample arrays
        fuel_data.pop('value_samples_real',  None)
        fuel_data.pop('value_samples_synth', None)

        # Token frequency: round train/test counts; suppress low-count tokens
        freq = fuel_data.get('frequency', {})
        if freq:
            train_counts = freq.get('train_counts', [])
            test_counts  = freq.get('test_counts',  [])

            rounded_train = [_rn(c) if c is not None else None for c in train_counts]
            rounded_test  = [_rn(c) if c is not None else None for c in test_counts]

            # Null out all per-token real-data values for suppressed tokens
            # (token suppressed if original train_count < threshold)
            suppressed_mask = [
                c is not None and 0 < int(c) < SDC_MIN_HOUSEHOLDS
                for c in train_counts
            ]

            freq['train_counts']      = [None if suppressed_mask[i] else rounded_train[i]
                                          for i in range(len(rounded_train))]
            freq['test_counts']       = [None if suppressed_mask[i] else rounded_test[i]
                                          for i in range(len(rounded_test))]

            # Null corresponding real-data frequency/cdf entries
            for key in ('train_frequencies', 'train_cdf',
                        'test_frequencies',  'test_cdf'):
                arr = freq.get(key, [])
                if arr:
                    freq[key] = [None if i < len(suppressed_mask) and suppressed_mask[i]
                                 else v for i, v in enumerate(arr)]

    return tok


def _sanitise_weekly_fidelity(wf: Dict) -> Dict:
    """Round n_households per week entry for real-data splits."""
    wf = copy.deepcopy(wf)
    for fuel_data in wf.values():
        if not isinstance(fuel_data, dict):
            continue
        for split, weekly in fuel_data.items():
            if split in ('train', 'test') and isinstance(weekly, dict):
                for week_data in weekly.values():
                    if isinstance(week_data, dict) and 'n_households' in week_data:
                        week_data['n_households'] = _rn(week_data['n_households'])
    return wf


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def sanitise(results: Dict) -> Dict:
    """
    Return an SDC-compliant deep copy of a raw parallel_evaluation_results dict.

    The input dict is not modified.
    """
    r = dict(results)  # shallow copy at top level; sections replaced below

    r['metadata']                = _sanitise_metadata(r.get('metadata', {}))
    r['marginal_distribution']   = _sanitise_marginal_distribution(
                                       r.get('marginal_distribution', {}))
    r['conditional_distribution'] = _sanitise_conditional_distribution(
                                       r.get('conditional_distribution', {}))
    r['temporal_patterns']       = _sanitise_temporal_patterns(
                                       r.get('temporal_patterns', {}))
    r['seasonal_analysis']       = _sanitise_seasonal_analysis(
                                       r.get('seasonal_analysis', {}))
    r['household_month_fidelity'] = _sanitise_household_month_fidelity(
                                       r.get('household_month_fidelity', {}))
    r['token_analysis']          = _sanitise_token_analysis(
                                       r.get('token_analysis', {}))
    r['weekly_fidelity']         = _sanitise_weekly_fidelity(
                                       r.get('weekly_fidelity', {}))

    # Sections not requiring transformation are carried through unchanged:
    # within_batch_variance, distance_analysis, missingness_analysis,
    # distribution_comparison, correlation_analysis, ptg_analysis,
    # daily_pattern_diagnostics.

    return r


def sanitise_file(input_path: Path, output_path: Path) -> None:
    """Read raw results JSON, sanitise, write SDC-compliant JSON."""
    with open(input_path) as f:
        results = json.load(f)
    sdc = sanitise(results)
    with open(output_path, 'w') as f:
        json.dump(sdc, f, indent=2)
    print(f'SDC-compliant results written to {output_path}')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python -m src.evaluation.mapreduce.sdc_sanitiser '
              '<input_results.json> <output_results_sdc.json>')
        sys.exit(1)
    sanitise_file(Path(sys.argv[1]), Path(sys.argv[2]))
