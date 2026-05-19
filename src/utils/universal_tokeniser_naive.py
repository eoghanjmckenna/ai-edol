#!/usr/bin/env python3

"""
Universal tokeniser Utility

This module provides quantile-based tokenization for household energy consumption data,
converting continuous electricity and gas usage values into discrete tokens for GPT model training.

Token Allocation Scheme:
-----------------------
Each modality (electricity, gas, temperature, solar) uses the same token allocation:
- Tokens 0 to (num_bins-1): Quantile buckets for actual values
- Token num_bins: Missing/NaN values
- Token (num_bins+1): SOS (Start of Sequence) token for autoregressive generation

For default num_bins=256:
- Tokens 0-255: Value quantiles
- Token 256: Missing/NaN token
- Token 257: SOS token (used at position 0 during training/inference)
- Total vocabulary size per modality: num_bins + 2 = 258

num_bins can be configured per modality for different granularities:
- electricity: High granularity (e.g., 256) for fine-grained consumption patterns
- gas: High granularity (e.g., 256) for fine-grained consumption patterns
- temperature: Lower granularity (e.g., 64) may be sufficient for conditioning
- solar: Lower granularity (e.g., 64) may be sufficient for conditioning

Config format:
  num_bins: 256  # Single value for all modalities (backward compatible)

  # OR per-modality:
  num_bins:
    electricity: 256
    gas: 256
    temperature: 64
    solar: 64

Usage Scenarios:
----------------
1) Create new tokenisers, optionally save them, and optionally tokenise:
   python universal_tokeniser.py \
       --data_folder /path/to/data \
       --create_tokenisers \
       --max_households_create 10 \
       --num_bins 256 \
       --save_tokenisers /path/to/tokenisers.pkl \
       --do_tokenise \
       --max_households_tokenise 10 \
       --output_folder /path/to/tokenised_files \
       --seed_create 42 \
       --seed_tokenise 99

2) Load existing tokenisers and tokenise:
   python universal_tokeniser.py \
       --data_folder /path/to/data \
       --load_tokenisers /path/to/tokenisers.pkl \
       --do_tokenise \
       --max_households_tokenise 10 \
       --output_folder /path/to/tokenised_files \
       --seed_tokenise 99

3) Load as a module in Python:
   import universal_tokeniser as ut
   elec_boundaries, gas_boundaries, temp_boundaries, solar_boundaries = ut.build_tokenisers_for_files([...])
   
4) decode tokenized data (e.g. after inference):
    import pickle
    import pandas as pd
    import universal_tokeniser_naive as ut
    in_path = 'S:\SERL_Observatory_DSH1\Individual Working files\EM\synserl/experiments/tokenisers/tokenisers.pkl'
    elec_boundaries, gas_boundaries, temp_boundaries, solar_boundaries = ut.load_tokenisers(in_path)
    # now load an example df
    in_path = 'S:\SERL_Observatory_DSH1\Individual Working files\EM\synserl/experiments/data/9GR5HHX1_tokenised.pkl'
    with open(in_path, "rb") as f:
        df = pickle.load(f)
    df.head()
    df_decode = ut.detokenise_data(df, elec_boundaries=elec_boundaries, gas_boundaries=gas_boundaries, 
                                   temp_boundaries=temp_boundaries, solar_boundaries=solar_boundaries)
    df_decode.head()
   ...
"""

import os
import glob
import argparse
import random
import pickle
import yaml
import json
import math
from collections import Counter
from bisect import bisect_left
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import time
import multiprocessing as mp

import numpy as np
import pandas as pd


###############################################################################
# 0. TokenVocabulary Class - Single Source of Truth for Token Schemes
###############################################################################

@dataclass
class TokenVocabulary:
    """
    Token allocation scheme for a single modality.

    Encapsulates the token vocabulary configuration and provides
    encode/decode methods for converting between values and tokens.

    Token allocation (standard modalities - elec, temp, solar):
    - Tokens 0 to num_bins-1: Quantile bins
    - Token num_bins: Missing/NaN
    - Token num_bins+1: SOS (Start of Sequence)

    Token allocation (zero-aware - gas):
    - Token 0: Zero value
    - Tokens 1 to num_bins-1: Non-zero quantile bins
    - Token num_bins: Missing/NaN
    - Token num_bins+1: SOS (Start of Sequence)

    Attributes:
        boundaries: List of quantile boundary values
        bin_means: Optional list of empirical bin means for decoding
        zero_aware: If True, token 0 is reserved for zero values (used for gas)
        num_bins: Total number of value bins (computed)
        missing_token: Token ID for missing/NaN values (computed)
        sos_token: Token ID for Start of Sequence (computed)
        vocab_size: Total vocabulary size including special tokens (computed)

    Example:
        >>> vocab = TokenVocabulary(boundaries=[100, 200, 300], zero_aware=False)
        >>> vocab.num_bins
        4
        >>> vocab.missing_token
        4
        >>> vocab.sos_token
        5
        >>> vocab.vocab_size
        6
    """
    boundaries: List[float]
    bin_means: Optional[List[float]] = None
    zero_aware: bool = False  # True for gas (token 0 = zero)
    value_transform: str = "none"  # "none" or "log1p" (Decision 040)

    # Computed fields
    num_bins: int = field(init=False)
    missing_token: int = field(init=False)
    sos_token: int = field(init=False)
    vocab_size: int = field(init=False)

    def __post_init__(self):
        if self.zero_aware:
            # Gas: boundaries are for non-zero values only
            # Token 0 = zero, tokens 1-N = non-zero bins
            # boundaries has N-1 values for N non-zero bins
            # Total bins = 1 (zero) + N (non-zero) = len(boundaries) + 2
            self.num_bins = len(self.boundaries) + 2
        else:
            # Standard: boundaries has N-1 values for N bins
            self.num_bins = len(self.boundaries) + 1

        self.missing_token = self.num_bins
        self.sos_token = self.num_bins + 1
        self.vocab_size = self.num_bins + 2

    def encode(self, value: float) -> int:
        """
        Encode a value to its token representation.

        Args:
            value: The numeric value to encode

        Returns:
            Token ID (0 to num_bins-1 for values, missing_token for NaN)
        """
        if self.zero_aware:
            return get_gas_token(value, self.boundaries, self.missing_token,
                                 value_transform=self.value_transform)
        else:
            return get_token(value, self.boundaries, self.missing_token,
                             value_transform=self.value_transform)

    def decode(self, token: int) -> Optional[float]:
        """
        Decode a token back to its approximate value.

        Uses bin_means if available for better fidelity, otherwise
        falls back to midpoint interpolation. If value_transform is "log1p",
        applies expm1() to convert from log space back to original space.

        Args:
            token: Token ID to decode

        Returns:
            Decoded value, or None for special tokens (missing, SOS)
        """
        if self.zero_aware:
            return get_gas_value_from_token(
                token, self.boundaries,
                self.missing_token, self.sos_token,
                self.bin_means,
                value_transform=self.value_transform
            )
        else:
            return get_value_from_token(
                token, self.boundaries,
                self.missing_token, self.sos_token,
                self.bin_means,
                value_transform=self.value_transform
            )

    def is_special_token(self, token: int) -> bool:
        """Check if token is a special token (missing or SOS)."""
        return token == self.missing_token or token == self.sos_token

    def is_valid_value_token(self, token: int) -> bool:
        """Check if token represents a valid value (not special)."""
        return 0 <= token < self.num_bins

    def encode_array(self, values: np.ndarray) -> np.ndarray:
        """
        Vectorized encoding - 10-100x faster than row-by-row apply.

        Uses numpy searchsorted for vectorized bisect operations instead of
        iterating with pandas .apply().

        If value_transform is "log1p", applies log1p to values before binning
        (with negative clipping for standard modalities, zero-check-first for gas).

        Args:
            values: Numpy array of values to encode

        Returns:
            Numpy array of token IDs (int32)

        Example:
            >>> vocab = TokenVocabulary(boundaries=[100, 200, 300])
            >>> vocab.encode_array(np.array([50, 150, 250, np.nan]))
            array([0, 1, 2, 4], dtype=int32)  # 4 is missing_token
        """
        values = np.asarray(values)
        is_nan = pd.isna(values)

        # Round to 0.1 precision (matching scalar get_token behavior)
        rounded = np.round(values, 1)

        if self.zero_aware:
            # Gas zero-aware tokenization:
            # Token 0 = exactly zero, tokens 1-N = non-zero bins
            tokens = np.zeros(len(values), dtype=np.int32)

            # Find zeros (not NaN) — zero check on raw rounded values BEFORE transform
            is_zero = (rounded == 0.0) & ~is_nan

            # Find non-zero, non-NaN values
            is_nonzero = ~is_zero & ~is_nan

            # Token 0 for zeros (already set by np.zeros)

            # For non-zero values: use searchsorted on boundaries, then add 1
            if is_nonzero.any():
                nonzero_vals = rounded[is_nonzero]
                # Apply log transform to non-zero values if configured
                if self.value_transform == "log1p":
                    nonzero_vals = np.round(np.log1p(nonzero_vals), 4)
                # searchsorted returns 0 to len(boundaries), we add 1 to get 1 to len(boundaries)+1
                nonzero_tokens = np.searchsorted(self.boundaries, nonzero_vals, side='left')
                # Clamp to valid non-zero range [0, num_non_zero_bins-1] then add 1
                num_non_zero_bins = len(self.boundaries) + 1
                nonzero_tokens = np.clip(nonzero_tokens, 0, num_non_zero_bins - 1)
                tokens[is_nonzero] = nonzero_tokens + 1  # Offset by 1 for zero bin

            # Set NaN values to missing_token
            tokens[is_nan] = self.missing_token

        else:
            # Apply log transform if configured (clip negatives first)
            if self.value_transform == "log1p":
                rounded = np.round(np.log1p(np.maximum(rounded, 0)), 4)

            # Standard tokenization: searchsorted directly gives bin index
            tokens = np.searchsorted(self.boundaries, rounded, side='left')

            # Clamp to valid range [0, num_bins-1]
            tokens = np.clip(tokens, 0, self.num_bins - 1)

            # Set NaN values to missing_token
            tokens[is_nan] = self.missing_token

            tokens = tokens.astype(np.int32)

        return tokens


###############################################################################
# 0.1 Filtering Functions for Tokeniser Creation
###############################################################################

def calculate_household_missingness(file_path: str) -> dict:
    """
    Calculate missingness rates for a single household file.

    Args:
        file_path: Path to household pickle file

    Returns:
        dict with 'puprn', 'elec_missingness', 'gas_missingness', 'total_rows'
    """
    try:
        df = pd.read_pickle(file_path)
        total_rows = len(df)

        if total_rows == 0:
            return {
                'puprn': os.path.basename(file_path).replace('.pkl', ''),
                'elec_missingness': 1.0,
                'gas_missingness': 1.0,
                'total_rows': 0
            }

        elec_missing = df['Clean_elec_net_Wh'].isna().sum() / total_rows
        gas_missing = df['Clean_gas_Wh'].isna().sum() / total_rows

        return {
            'puprn': os.path.basename(file_path).replace('.pkl', ''),
            'elec_missingness': elec_missing,
            'gas_missingness': gas_missing,
            'total_rows': total_rows
        }
    except Exception as e:
        print(f"[WARNING] Could not calculate missingness for {file_path}: {e}")
        return {
            'puprn': os.path.basename(file_path).replace('.pkl', ''),
            'elec_missingness': 1.0,
            'gas_missingness': 1.0,
            'total_rows': 0
        }


def load_exporter_puprns(exporter_path: str) -> set:
    """
    Load list of PV exporter PUPRNs from file.

    Args:
        exporter_path: Path to CSV file with exporter PUPRNs

    Returns:
        Set of PUPRN strings that are exporters

    Raises:
        FileNotFoundError: If the exporter PUPRNs file does not exist.
    """
    if not os.path.exists(exporter_path):
        raise FileNotFoundError(f"Exporter PUPRNs file not found: {exporter_path}")

    try:
        df = pd.read_csv(exporter_path)
        # Handle different column naming conventions
        if 'puprn' in df.columns:
            return set(df['puprn'].astype(str).values)
        elif 'PUPRN' in df.columns:
            return set(df['PUPRN'].astype(str).values)
        else:
            # Assume first column contains PUPRNs
            return set(df.iloc[:, 0].astype(str).values)
    except Exception as e:
        print(f"[WARNING] Could not load exporter PUPRNs: {e}")
        return set()


def apply_tokeniser_filters(file_paths: list, filtering_config: dict) -> tuple:
    """
    Apply filters to household files for tokeniser creation.

    This ensures tokeniser bins are built from the same population that will
    be used for training, preventing distribution mismatch.

    Args:
        file_paths: List of all household file paths
        filtering_config: Dict with filtering parameters:
            - max_missingness_rate: float (0.0-1.0)
            - exclude_pv_households: bool
            - path_to_list_of_exporter_puprns: str

    Returns:
        Tuple of (filtered_file_paths, filter_results_dict)
    """
    print("\n" + "=" * 70, flush=True)
    print("TOKENISER FILTERING", flush=True)
    print("=" * 70, flush=True)

    max_missingness = filtering_config.get('max_missingness_rate', 0.20)
    exclude_pv = filtering_config.get('exclude_pv_households', True)
    exporter_path = filtering_config.get('path_to_list_of_exporter_puprns', '')

    results = {
        'initial_count': len(file_paths),
        'stages': [],
        'excluded_households': {
            'pv_exporters': [],
            'high_missingness': []
        }
    }

    # Stage 0: Initial
    print(f"\nStage 0: Initial count = {len(file_paths)} households")
    results['stages'].append({
        'stage': 0,
        'name': 'Initial (all households)',
        'count': len(file_paths),
        'removed': 0
    })

    current_paths = file_paths.copy()

    # Stage 1: Exclude PV households
    if exclude_pv and exporter_path:
        exporter_puprns = load_exporter_puprns(exporter_path)
        if exporter_puprns:
            before_count = len(current_paths)
            filtered_paths = []

            for fp in current_paths:
                puprn = os.path.basename(fp).replace('.pkl', '')
                if puprn not in exporter_puprns:
                    filtered_paths.append(fp)
                else:
                    results['excluded_households']['pv_exporters'].append(puprn)

            current_paths = filtered_paths
            removed = before_count - len(current_paths)

            print(f"\nStage 1: Exclude PV households")
            print(f"  Exporter PUPRNs loaded: {len(exporter_puprns)}")
            print(f"  Removed: {removed} ({100*removed/before_count:.1f}%)")
            print(f"  Remaining: {len(current_paths)}")

            results['stages'].append({
                'stage': 1,
                'name': 'Exclude PV households',
                'count': len(current_paths),
                'removed': removed,
                'exporter_puprns_loaded': len(exporter_puprns)
            })
        else:
            print(f"\nStage 1: Exclude PV households - SKIPPED (no exporter list)")
    else:
        print(f"\nStage 1: Exclude PV households - SKIPPED (disabled or no path)")

    # Stage 2: Filter by missingness (parallelized for speed)
    if max_missingness < 1.0:
        before_count = len(current_paths)
        # Determine number of workers: use config value if provided, otherwise auto-detect
        # Hard limit of 8 workers (diminishing returns for I/O-bound work)
        available_cores = mp.cpu_count()
        configured_workers = filtering_config.get('num_workers')
        if configured_workers is not None:
            num_workers = max(1, min(configured_workers, 8))
        else:
            num_workers = max(1, min(available_cores - 1, 8))
        print(f"\nStage 2: Calculate missingness (processing {before_count} files with {num_workers}/{available_cores} workers)...", flush=True)

        # Use multiprocessing for parallel missingness calculation
        start_time = time.time()
        with mp.Pool(processes=num_workers) as pool:
            missingness_data = pool.map(calculate_household_missingness, current_paths)
        elapsed = time.time() - start_time
        print(f"  Completed in {elapsed:.1f}s ({before_count/elapsed:.0f} files/sec)", flush=True)

        # Filter based on missingness results
        filtered_paths = []
        for fp, miss_info in zip(current_paths, missingness_data):
            if (miss_info['elec_missingness'] < max_missingness and
                miss_info['gas_missingness'] < max_missingness):
                filtered_paths.append(fp)
            else:
                results['excluded_households']['high_missingness'].append({
                    'puprn': miss_info['puprn'],
                    'elec_missingness': miss_info['elec_missingness'],
                    'gas_missingness': miss_info['gas_missingness']
                })

        current_paths = filtered_paths
        removed = before_count - len(current_paths)

        print(f"\nStage 2: Missingness < {max_missingness*100:.0f}%")
        print(f"  Removed: {removed} ({100*removed/before_count:.1f}%)")
        print(f"  Remaining: {len(current_paths)}")

        results['stages'].append({
            'stage': 2,
            'name': f'Missingness < {max_missingness*100:.0f}%',
            'count': len(current_paths),
            'removed': removed,
            'threshold': max_missingness,
            'processing_time_seconds': elapsed
        })
        results['missingness_data'] = missingness_data

    # Final summary
    results['final_count'] = len(current_paths)
    results['total_excluded'] = results['initial_count'] - results['final_count']
    results['retention_rate'] = len(current_paths) / results['initial_count'] if results['initial_count'] > 0 else 0

    print(f"\n" + "=" * 70, flush=True)
    print(f"FILTERING COMPLETE", flush=True)
    print(f"  Initial: {results['initial_count']} households", flush=True)
    print(f"  Final: {results['final_count']} households", flush=True)
    print(f"  Excluded: {results['total_excluded']} ({100*(1-results['retention_rate']):.1f}%)", flush=True)
    print(f"  Retention rate: {results['retention_rate']*100:.1f}%", flush=True)
    print("=" * 70 + "\n", flush=True)

    return current_paths, results


def calculate_distribution_statistics(file_paths: list, sample_size: int = None, seed: int = None) -> dict:
    """
    Calculate distribution statistics for electricity and gas consumption.

    Args:
        file_paths: List of household file paths
        sample_size: If provided, randomly sample this many files (for speed)
        seed: Random seed for reproducible sampling (default: None)

    Returns:
        Dict with statistics for elec and gas (mean, median, std, percentiles)
    """
    if sample_size and len(file_paths) > sample_size:
        if seed is not None:
            random.seed(seed)
        file_paths = random.sample(file_paths, sample_size)

    elec_values = []
    gas_values = []

    for fp in file_paths:
        try:
            df = pd.read_pickle(fp)
            elec_values.extend(df['Clean_elec_net_Wh'].dropna().values)
            gas_values.extend(df['Clean_gas_Wh'].dropna().values)
        except Exception:
            continue

    def calc_stats(values):
        if len(values) == 0:
            return {'count': 0}
        arr = np.array(values)
        return {
            'count': len(arr),
            'mean': float(np.mean(arr)),
            'median': float(np.median(arr)),
            'std': float(np.std(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'p5': float(np.percentile(arr, 5)),
            'p25': float(np.percentile(arr, 25)),
            'p75': float(np.percentile(arr, 75)),
            'p95': float(np.percentile(arr, 95))
        }

    return {
        'electricity': calc_stats(elec_values),
        'gas': calc_stats(gas_values),
        'num_households': len(file_paths)
    }


def generate_distribution_report(all_file_paths: list,
                                  filtered_file_paths: list,
                                  filter_results: dict,
                                  output_path: str,
                                  sample_size: int = 500,
                                  seed: int = None) -> dict:
    """
    Generate a comparison report of distributions before and after filtering.

    Args:
        all_file_paths: All household file paths (before filtering)
        filtered_file_paths: Filtered household file paths
        filter_results: Results dict from apply_tokeniser_filters
        output_path: Path to save JSON report
        sample_size: Max files to sample for statistics (for speed)
        seed: Random seed for reproducible sampling (default: None)

    Returns:
        Report dict (also saved to output_path)
    """
    print("\n[REPORT] Calculating distribution statistics...")
    print(f"  Sampling up to {sample_size} households for statistics...")

    # Calculate stats for both sets (use same seed for both to ensure comparable samples)
    all_stats = calculate_distribution_statistics(all_file_paths, sample_size, seed)
    filtered_stats = calculate_distribution_statistics(filtered_file_paths, sample_size, seed)

    # Calculate differences
    def calc_diff(all_s, filt_s, key):
        if key not in all_s or key not in filt_s:
            return None
        return filt_s[key] - all_s[key]

    report = {
        'summary': {
            'all_households': len(all_file_paths),
            'filtered_households': len(filtered_file_paths),
            'excluded_households': len(all_file_paths) - len(filtered_file_paths),
            'retention_rate': len(filtered_file_paths) / len(all_file_paths) if len(all_file_paths) > 0 else 0
        },
        'distribution_comparison': {
            'electricity': {
                'all_data': all_stats['electricity'],
                'filtered_data': filtered_stats['electricity'],
                'differences': {
                    'mean_shift': calc_diff(all_stats['electricity'], filtered_stats['electricity'], 'mean'),
                    'median_shift': calc_diff(all_stats['electricity'], filtered_stats['electricity'], 'median'),
                    'std_change': calc_diff(all_stats['electricity'], filtered_stats['electricity'], 'std'),
                }
            },
            'gas': {
                'all_data': all_stats['gas'],
                'filtered_data': filtered_stats['gas'],
                'differences': {
                    'mean_shift': calc_diff(all_stats['gas'], filtered_stats['gas'], 'mean'),
                    'median_shift': calc_diff(all_stats['gas'], filtered_stats['gas'], 'median'),
                    'std_change': calc_diff(all_stats['gas'], filtered_stats['gas'], 'std'),
                }
            }
        },
        'filtering_stages': filter_results.get('stages', []),
        'excluded_counts': {
            'pv_exporters': len(filter_results.get('excluded_households', {}).get('pv_exporters', [])),
            'high_missingness': len(filter_results.get('excluded_households', {}).get('high_missingness', []))
        }
    }

    # Print summary
    print("\n" + "=" * 70)
    print("DISTRIBUTION COMPARISON REPORT")
    print("=" * 70)

    for var in ['electricity', 'gas']:
        all_d = report['distribution_comparison'][var]['all_data']
        filt_d = report['distribution_comparison'][var]['filtered_data']
        diffs = report['distribution_comparison'][var]['differences']

        print(f"\n{var.upper()}:")
        print(f"  {'Metric':<15} {'All Data':>15} {'Filtered':>15} {'Difference':>15}")
        print(f"  {'-'*60}")

        for metric in ['mean', 'median', 'std', 'p5', 'p95']:
            all_val = all_d.get(metric, 0)
            filt_val = filt_d.get(metric, 0)
            diff = filt_val - all_val
            pct_diff = (diff / all_val * 100) if all_val != 0 else 0
            print(f"  {metric:<15} {all_val:>15.2f} {filt_val:>15.2f} {diff:>+10.2f} ({pct_diff:>+5.1f}%)")

    print("=" * 70)

    # Check for significant distribution shift
    elec_mean_shift_pct = abs(report['distribution_comparison']['electricity']['differences']['mean_shift'] or 0) / (all_stats['electricity'].get('mean', 1)) * 100
    gas_mean_shift_pct = abs(report['distribution_comparison']['gas']['differences']['mean_shift'] or 0) / (all_stats['gas'].get('mean', 1)) * 100

    if elec_mean_shift_pct > 5 or gas_mean_shift_pct > 5:
        print("\n[WARNING] Significant distribution shift detected!")
        print(f"  Electricity mean shift: {elec_mean_shift_pct:.1f}%")
        print(f"  Gas mean shift: {gas_mean_shift_pct:.1f}%")
        print("  This indicates filtering is important for tokeniser accuracy.")

    # Save report
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n[REPORT] Saved distribution report to: {output_path}")

    return report


###############################################################################
# 1. Core Functions for Building & tokenising
###############################################################################

def build_quantile_boundaries(data_iterable, num_bins=256):
    """
    Build quantile boundaries by:
      1. Counting unique values + frequencies (values rounded to nearest 0.1 Wh)
      2. Sorting by numeric value
      3. Computing cumulative distribution (CDF)
      4. Determining each quantile cutoff

    Returns: A list of bin edges (length = num_bins - 1).
    """
    counts = Counter()
    for val in data_iterable:
        if not pd.isna(val):
            # Round to nearest 0.1 Wh to reduce float noise
            rounded_val = round(val, 1)
            counts[rounded_val] += 1

    if not counts:
        return []

    # Sort by numeric value
    unique_items = sorted(counts.items(), key=lambda x: x[0])  # [(value, freq), ...]
    unique_values = [item[0] for item in unique_items]
    freqs = [item[1] for item in unique_items]

    # Build CDF
    cdf = []
    running_sum = 0
    for f in freqs:
        running_sum += f
        cdf.append(running_sum)
    total_count = cdf[-1]

    # Determine boundary for each quantile
    boundaries = []
    for i in range(1, num_bins):
        frac = i / num_bins
        threshold = frac * total_count
        idx = bisect_left(cdf, threshold)
        boundaries.append(unique_values[idx])

    return boundaries


def build_tokenisers_for_files(file_paths, num_bins=256):
    """
    DEPRECATED: Legacy batch loading approach with high memory usage.

    This function loads ALL values from ALL files into memory before processing,
    which causes OOM errors for large datasets (>500 households).

    Memory usage: ~12 GB for 1,000 households, scales linearly with dataset size.

    Use build_tokenisers_incremental() instead for memory-efficient processing.

    Returns: (elec_boundaries, gas_boundaries, temp_boundaries, solar_boundaries)
    """
    print("[WARNING] Using legacy batch loading approach - may cause OOM for large datasets!")
    print("[WARNING] Consider using build_tokenisers_incremental() instead.")

    elec_values = []
    gas_values = []
    temp_values = []
    solar_values = []

    for fp in file_paths:
        df = pd.read_pickle(fp)
        elec_values.extend(df['Clean_elec_net_Wh'].values)
        gas_values.extend(df['Clean_gas_Wh'].values)
        temp_values.extend(df['temp_C'].values)
        solar_values.extend(df['surface_solar_radiation_downwards'].values)

    print(f"[BUILD] Found {len(elec_values)} electric, {len(gas_values)} gas, "
          f"{len(temp_values)} temperature, and {len(solar_values)} solar values in {len(file_paths)} files.")

    elec_boundaries = build_quantile_boundaries(elec_values, num_bins=num_bins)
    gas_boundaries = build_quantile_boundaries(gas_values, num_bins=num_bins)
    temp_boundaries = build_quantile_boundaries(temp_values, num_bins=num_bins)
    solar_boundaries = build_quantile_boundaries(solar_values, num_bins=num_bins)

    return elec_boundaries, gas_boundaries, temp_boundaries, solar_boundaries

def build_boundaries_from_counter(counter, num_bins=256):
    """Helper function to build quantile boundaries from a Counter object."""
    if not counter:
        return []

    # Sort by numeric value
    unique_items = sorted(counter.items(), key=lambda x: x[0])  # [(value, freq), ...]
    unique_values = [item[0] for item in unique_items]
    freqs = [item[1] for item in unique_items]

    # Build CDF
    cdf = []
    running_sum = 0
    for f in freqs:
        running_sum += f
        cdf.append(running_sum)
    total_count = cdf[-1]

    # Determine (right edge) boundary for each quantile
    boundaries = []
    for i in range(1, num_bins):
        frac = i / num_bins
        threshold = frac * total_count
        idx = bisect_left(cdf, threshold) # <--- returns leftmost insertion point that would keep list ordered
        boundaries.append(unique_values[idx])

    return boundaries


def build_uniform_boundaries_from_counter(counter, num_bins=256,
                                           clip_lower=0.0,
                                           clip_upper_percentile=99.5,
                                           clip_upper=None):
    """
    Build equal-width boundaries from a Counter object.

    Creates num_bins - 1 equally-spaced boundaries between clip_lower and the
    upper bound. Values below clip_lower fall into the first bin; values above
    the upper bound fall into the last bin.

    Upper bound resolution order:
      1. clip_upper (absolute value) — preferred for physically bounded variables
         like temperature, where the full range matters and there are no statistical
         outliers. Use this when the physical maximum is known (e.g. 35°C for UK).
      2. clip_upper_percentile — preferred for heavy-tailed variables like energy
         consumption, where extreme outliers would stretch all bin widths.

    This eliminates the extreme bin width ratios of quantile binning (39,000x)
    by ensuring all bins have identical width within the clipped range.

    Args:
        counter: Counter object with {rounded_value: frequency}
        num_bins: Total number of bins (boundaries returned = num_bins - 1)
        clip_lower: Lower bound of the bin range (absolute value, default 0)
        clip_upper_percentile: Upper bound as a data percentile (default 99.5).
                                Used only if clip_upper is None.
        clip_upper: Upper bound as an absolute value (default None).
                    When set, overrides clip_upper_percentile entirely.

    Returns:
        List of num_bins - 1 boundary values (equally spaced)
    """
    if not counter:
        return []

    if clip_upper is not None:
        # Absolute upper bound — no CDF computation needed
        clip_upper_value = clip_upper
        upper_desc = f"{clip_upper_value:.1f} (absolute)"
    else:
        # Sort by numeric value and build CDF to find the upper percentile value
        unique_items = sorted(counter.items(), key=lambda x: x[0])
        unique_values = [item[0] for item in unique_items]
        freqs = [item[1] for item in unique_items]

        cdf = []
        running_sum = 0
        for f in freqs:
            running_sum += f
            cdf.append(running_sum)
        total_count = cdf[-1]

        threshold = (clip_upper_percentile / 100.0) * total_count
        idx = bisect_left(cdf, threshold)
        idx = min(idx, len(unique_values) - 1)
        clip_upper_value = unique_values[idx]
        upper_desc = f"{clip_upper_value:.1f} (p{clip_upper_percentile})"

    # Defensive: ensure valid range
    if clip_upper_value <= clip_lower:
        print(f"[WARNING] clip_upper ({clip_upper_value}) <= clip_lower ({clip_lower}). "
              f"Falling back to quantile binning.")
        return build_boundaries_from_counter(counter, num_bins)

    # Create equally-spaced boundaries
    step = (clip_upper_value - clip_lower) / num_bins
    boundaries = [clip_lower + step * i for i in range(1, num_bins)]

    print(f"[BUILD] Uniform bins: range [{clip_lower}, {upper_desc}], "
          f"width={step:.2f}, {num_bins} bins")

    return boundaries


def compute_bin_means_from_counter(counter, boundaries, modality="unknown"):
    """
    Compute the mean value for each bin using the Counter data.

    This uses the Counter (value -> frequency) to compute weighted means
    for each bin, avoiding a second pass through the raw data files.

    Args:
        counter: Counter object with {rounded_value: frequency}
        boundaries: List of quantile boundaries (length = num_bins - 1)
        modality: Name of the modality for logging (e.g., "elec", "gas", "temp")

    Returns:
        List of bin means (length = num_bins = len(boundaries) + 1)
    """
    if not counter or not boundaries:
        return []

    num_bins = len(boundaries) + 1
    bin_sums = [0.0] * num_bins
    bin_counts = [0] * num_bins

    for value, freq in counter.items():
        # Determine which bin this value belongs to
        # bisect_left returns the bin index (0 to num_bins-1)
        idx = bisect_left(boundaries, value)
        # Clamp to valid range (shouldn't be needed, but defensive)
        idx = min(idx, num_bins - 1)

        bin_sums[idx] += value * freq
        bin_counts[idx] += freq

    # Compute means, handling empty bins
    bin_means = []
    empty_bin_count = 0
    for i in range(num_bins):
        if bin_counts[i] > 0:
            bin_means.append(bin_sums[i] / bin_counts[i])
        else:
            # Fallback for empty bins: use midpoint
            if i == 0:
                bin_means.append(boundaries[0])
            elif i == num_bins - 1:
                bin_means.append(boundaries[-1])
            else:
                bin_means.append((boundaries[i-1] + boundaries[i]) / 2)
            empty_bin_count += 1

    # Summary warning instead of per-bin warnings
    if empty_bin_count > 0:
        print(f"[WARNING] {modality}: {empty_bin_count}/{num_bins} bins empty (using fallback values). "
              f"Consider reducing num_bins if unique values ({len(counter)}) < bins ({num_bins}).")

    return bin_means

def build_tokenisers_incremental(file_paths, num_bins=256, num_bins_elec=None,
                                  num_bins_gas=None, num_bins_temp=None, num_bins_solar=None,
                                  value_transform="none",
                                  binning_strategy="quantile",
                                  bin_clip_lower=0.0,
                                  bin_clip_upper_percentile=99.5,
                                  temperature_clip_lower=-10.0,
                                  temperature_clip_upper_percentile=99.5,
                                  temperature_clip_upper=None,
                                  solar_clip_lower=0.0,
                                  solar_clip_upper_percentile=99.5):
    """
    Memory-efficient tokenizer creation using incremental Counter aggregation.

    Instead of loading all values into memory, this function:
    1. Processes one file at a time
    2. Updates Counter objects with rounded values
    3. Releases DataFrame memory after each file
    4. Computes bin means from the counters (no second pass needed)

    If value_transform is "log1p", applies log1p to electricity and gas values
    before counting. Boundaries and bin means are then in log space.

    Memory usage: ~50 MB (vs ~12 GB for batch approach)

    Args:
        file_paths: List of paths to household pickle files
        num_bins: Default number of bins for all modalities (used if per-modality not specified)
        num_bins_elec: Number of bins for electricity (overrides num_bins)
        num_bins_gas: Number of bins for gas (overrides num_bins). Note: gas uses num_bins-1
                      for non-zero values due to zero-aware tokenisation.
        num_bins_temp: Number of bins for temperature (overrides num_bins)
        num_bins_solar: Number of bins for solar radiation (overrides num_bins)
        value_transform: "none" or "log1p". If "log1p", applies log(x+1) to
                         electricity and gas values before binning.
        binning_strategy: "quantile" (equal-frequency) or "uniform" (equal-width with clipping).
                          Only affects electricity and gas.
        bin_clip_lower: Lower bound for uniform electricity/gas bins (default 0).
        bin_clip_upper_percentile: Upper bound percentile for uniform electricity/gas bins (default 99.5).
        temperature_clip_lower: Lower bound (°C) for uniform temperature bins (default -10.0).
                                 Temperature always uses uniform binning — quantile binning was found
                                 to produce extreme open-ended bins 17–22x wider than inner bins,
                                 preventing the model from distinguishing cold demand days (Decision 045).
        temperature_clip_upper_percentile: Upper bound percentile for uniform temperature bins (default 99.5).
                                            Used only when temperature_clip_upper is None.
        temperature_clip_upper: Absolute upper bound (°C) for uniform temperature bins (default None).
                                 When set, overrides temperature_clip_upper_percentile. Preferred over
                                 percentile clipping for temperature because temperature has physical
                                 bounds (35°C covers UK climate envelope) unlike energy consumption
                                 which has statistical outliers (Decision 045).
        solar_clip_lower: Lower bound (J/m²) for uniform solar bins (default 0.0).
                          Solar radiation has a physical floor at 0 J/m². The SERL variable
                          `surface_solar_radiation_downwards` is from ERA5 and stored as
                          accumulated energy per 30-minute timestep (J/m²), not instantaneous
                          irradiance (W/m²). Typical UK range: 0–~2,900,000 J/m².
        solar_clip_upper_percentile: Upper bound percentile for uniform solar bins (default 99.5).
                                      Solar uses percentile clipping (not absolute) because peak radiation
                                      varies by location and season — no universal physical ceiling
                                      like temperature (Decision 046).

    Returns:
        Tuple of (elec_boundaries, gas_boundaries, temp_boundaries, solar_boundaries,
                  elec_bin_means, gas_bin_means)
    """
    # Resolve per-modality num_bins (use default if not specified)
    n_bins_elec = num_bins_elec if num_bins_elec is not None else num_bins
    n_bins_gas = num_bins_gas if num_bins_gas is not None else num_bins
    n_bins_temp = num_bins_temp if num_bins_temp is not None else num_bins
    n_bins_solar = num_bins_solar if num_bins_solar is not None else num_bins

    print(f"[BUILD] Bins per modality - Elec: {n_bins_elec}, Gas: {n_bins_gas}, "
          f"Temp: {n_bins_temp}, Solar: {n_bins_solar}")
    # Initialise counters for aggregation
    elec_counter = Counter()
    gas_zero_count = 0
    gas_non_zero_counter = Counter()
    temp_counter = Counter()
    solar_counter = Counter()

    total_files = len(file_paths)

    for i, fp in enumerate(file_paths):
        # Progress logging every 100 files
        if i % 100 == 0:
            print(f"[BUILD] Processing file {i+1}/{total_files}...")

        try:
            df = pd.read_pickle(fp)

            # Vectorized approach: process entire column at once (FAST!)
            # Filter out NaN values and round
            elec_vals = df['Clean_elec_net_Wh'].dropna().values
            if value_transform == "log1p":
                # Clip negatives (safety), apply log1p, round to 4dp for log-space precision
                elec_vals = np.log1p(np.maximum(elec_vals, 0))
                for val in np.round(elec_vals, 4):
                    elec_counter[val] += 1
            else:
                for val in np.round(elec_vals, 1):
                    elec_counter[val] += 1

            gas_vals = df['Clean_gas_Wh'].dropna().values
            for val in np.round(gas_vals, 1):
                if val == 0.0:
                    gas_zero_count += 1
                else:
                    if value_transform == "log1p":
                        gas_non_zero_counter[round(np.log1p(val), 4)] += 1
                    else:
                        gas_non_zero_counter[val] += 1

            temp_vals = df['temp_C'].dropna().values
            for val in np.round(temp_vals, 1):
                temp_counter[val] += 1

            solar_vals = df['surface_solar_radiation_downwards'].dropna().values
            for val in np.round(solar_vals, 1):
                solar_counter[val] += 1

            # DataFrame is garbage collected here automatically

        except Exception as e:
            print(f"[WARNING] Skipping file {fp}: {e}")
            continue

    print(f"[BUILD] Processed {total_files} files. Building quantile boundaries...")
    print(f"[BUILD] Unique values - Elec: {len(elec_counter)}, Gas non-zero: {len(gas_non_zero_counter)}, "
          f"Gas zeros: {gas_zero_count}, Temp: {len(temp_counter)}, Solar: {len(solar_counter)}")

    # Build boundaries from aggregated counters (using per-modality bins)
    if binning_strategy == "uniform":
        print(f"[BUILD] Using UNIFORM binning strategy (clip_lower={bin_clip_lower}, "
              f"clip_upper_percentile={bin_clip_upper_percentile})")
        elec_boundaries = build_uniform_boundaries_from_counter(
            elec_counter, num_bins=n_bins_elec,
            clip_lower=bin_clip_lower,
            clip_upper_percentile=bin_clip_upper_percentile
        )
        # Gas: n_bins_gas total bins = 1 (zero bin) + (n_bins_gas-1) non-zero bins
        gas_boundaries = build_uniform_boundaries_from_counter(
            gas_non_zero_counter, num_bins=n_bins_gas-1,
            clip_lower=bin_clip_lower,
            clip_upper_percentile=bin_clip_upper_percentile
        )
    else:
        # Quantile binning (legacy default)
        elec_boundaries = build_boundaries_from_counter(elec_counter, num_bins=n_bins_elec)
        # Gas: n_bins_gas total bins = 1 (zero bin) + (n_bins_gas-1) non-zero bins
        gas_boundaries = build_boundaries_from_counter(gas_non_zero_counter, num_bins=n_bins_gas-1)

    # Temperature: always uniform binning (Decision 045).
    # Quantile binning produced extreme open-ended bins 17-22x wider than inner bins,
    # preventing the model from distinguishing cold demand days.
    temp_boundaries = build_uniform_boundaries_from_counter(
        temp_counter, num_bins=n_bins_temp,
        clip_lower=temperature_clip_lower,
        clip_upper_percentile=temperature_clip_upper_percentile,
        clip_upper=temperature_clip_upper
    )

    # Solar: uniform binning with percentile clipping (Decision 046).
    # Mirrors temperature approach but uses percentile upper bound (not absolute)
    # because peak solar radiation varies by location and season.
    solar_boundaries = build_uniform_boundaries_from_counter(
        solar_counter, num_bins=n_bins_solar,
        clip_lower=solar_clip_lower,
        clip_upper_percentile=solar_clip_upper_percentile
    )

    # Compute bin means from counters (uses the already-aggregated data, no second pass)
    print(f"[BUILD] Computing bin means for decoding...")
    elec_bin_means = compute_bin_means_from_counter(elec_counter, elec_boundaries, modality="elec")
    # Gas bin means: token 0 = zero, tokens 1-N = non-zero quantile bins
    temp_gas_bin_means = compute_bin_means_from_counter(gas_non_zero_counter, gas_boundaries, modality="gas")
    gas_bin_means = [0.0] + temp_gas_bin_means  # Prepend 0.0 for token 0

    print(f"[BUILD] Tokeniser creation complete.")
    print(f"[BUILD]   Elec: {n_bins_elec} bins, {len(elec_bin_means)} bin means")
    print(f"[BUILD]   Gas: {n_bins_gas} bins ({n_bins_gas-1} non-zero), {len(gas_bin_means)} bin means")
    print(f"[BUILD]   Temp: {n_bins_temp} bins, {len(temp_boundaries)+1} total tokens")
    print(f"[BUILD]   Solar: {n_bins_solar} bins, {len(solar_boundaries)+1} total tokens")

    return elec_boundaries, gas_boundaries, temp_boundaries, solar_boundaries, elec_bin_means, gas_bin_means

def get_token(value, boundaries, missing_token, value_transform="none"):
    """
    Convert a numeric value into a token index based on quantile boundaries.

    If value_transform is "log1p", applies log1p to the value before binning
    (boundaries are assumed to be in log space).

    Token assignment using bisect_left:
    - Token 0: values <= boundaries[0] (first bin, includes values below first quantile)
    - Token i: boundaries[i-1] < values <= boundaries[i]
    - Token num_bins-1: values > boundaries[-1] (last bin, includes values above last quantile)

    Args:
        value: Energy value to tokenize
        boundaries: List of quantile boundary values (length = num_bins - 1)
        missing_token: Token ID for NaN/missing values (usually num_bins)
        value_transform: "none" or "log1p"

    Returns:
        Token ID (0 to num_bins-1 for valid values, missing_token for NaN)
    """
    if pd.isna(value):
        return missing_token

    if len(boundaries) == 0:  # No boundaries => return missing
        return missing_token

    rounded_val = round(value, 1)

    # Apply log transform if configured (clip negatives for safety)
    if value_transform == "log1p":
        rounded_val = round(math.log1p(max(0, rounded_val)), 4)

    # Use bisect to find bin index (returns leftmost insertion position to keep the list sorted)
    idx = bisect_left(boundaries, rounded_val)

    # Clamp to valid token range [0, num_bins-1]
    # This ensures values below first boundary -> token 0
    # and values above last boundary -> token num_bins-1
    num_bins = len(boundaries) + 1
    idx = max(0, min(idx, num_bins - 1))

    return idx


def get_gas_token(value, boundaries, missing_token, value_transform="none"):
    """
    Zero-aware gas tokenisation.

    Gas data has many exact zeros (no gas usage). This function reserves
    token 0 for zero values, with tokens 1-255 for non-zero quantile bins.

    If value_transform is "log1p", applies log1p to non-zero values before
    binning (zero check happens first, boundaries are in log space).

    Token allocation:
    - Token 0: exactly 0.0 (zero gas usage)
    - Tokens 1-255: non-zero values in 255 quantile bins
    - Token 256: missing/NaN values
    - Token 257: SOS token (used during training/inference)

    Args:
        value: Gas value to tokenize (Wh)
        boundaries: List of quantile boundaries for NON-ZERO values only
                   (length = 254 for 255 non-zero bins)
        missing_token: Token ID for NaN/missing values (256)
        value_transform: "none" or "log1p"

    Returns:
        Token ID (0 for zero, 1-255 for non-zero values, 256 for missing)
    """
    if pd.isna(value):
        return missing_token

    rounded_val = round(value, 1)

    # Zero gets dedicated token 0 — checked BEFORE any transform
    if rounded_val == 0.0:
        return 0

    if len(boundaries) == 0:  # No non-zero data => return missing
        return missing_token

    # Apply log transform to non-zero values if configured
    if value_transform == "log1p":
        rounded_val = round(math.log1p(rounded_val), 4)

    # Non-zero values: use bisect on non-zero boundaries, then offset by 1
    idx = bisect_left(boundaries, rounded_val)

    # Clamp to valid non-zero token range [1, 255]
    # bisect_left returns 0 to len(boundaries), we add 1 to get 1 to len(boundaries)+1
    num_non_zero_bins = len(boundaries) + 1  # 255 bins from 254 boundaries
    idx = max(0, min(idx, num_non_zero_bins - 1))

    return idx + 1  # Offset by 1 since token 0 is reserved for zeros

def get_value_from_token(token, boundaries, missing_token, sos_token=None, bin_means=None,
                         value_transform="none"):
    """
    Convert token back to approximate value.

    If bin_means is provided, uses the empirical mean of values in that bin.
    Otherwise falls back to midpoint interpolation between boundaries.
    If value_transform is "log1p", applies expm1() to convert from log space.

    Args:
        token: Token ID to decode
        boundaries: List of quantile boundaries
        missing_token: Token ID for missing values
        sos_token: Token ID for SOS (Start of Sequence) - returns None if encountered
        bin_means: Optional list of empirical bin means (length = num_bins)
        value_transform: "none" or "log1p"

    Returns:
        Decoded value, or None for missing/SOS tokens
    """
    if token == missing_token:
        return None
    if sos_token is not None and token == sos_token:
        return None

    # Validate token is in range
    num_bins = len(boundaries) + 1
    if token < 0 or token >= num_bins:
        return None

    if bin_means is not None and len(bin_means) > token:
        decoded = bin_means[token]
    else:
        # Fallback to midpoint calculation
        if token == 0:
            decoded = boundaries[0]
        elif token == len(boundaries):
            decoded = boundaries[-1]
        else:
            decoded = (boundaries[token-1] + boundaries[token]) / 2

    # Apply inverse transform: exp(x) - 1 to convert from log space
    if value_transform == "log1p":
        decoded = math.expm1(decoded)

    return decoded


def get_gas_value_from_token(token, boundaries, missing_token, sos_token=None, bin_means=None,
                             value_transform="none"):
    """
    Zero-aware gas detokenisation.

    Decodes gas tokens back to values, handling the zero-aware token scheme:
    - Token 0: zero gas usage (returns 0.0) — no transform applied
    - Tokens 1-255: non-zero quantile bins
    - Token 256: missing (returns None)
    - Token 257: SOS (returns None)

    If value_transform is "log1p", applies expm1() to non-zero decoded values.

    Args:
        token: Token ID to decode
        boundaries: List of quantile boundaries for NON-ZERO values only
                   (length = 254 for 255 non-zero bins)
        missing_token: Token ID for missing values (256)
        sos_token: Token ID for SOS (Start of Sequence) - returns None if encountered
        bin_means: Optional list of empirical bin means (length = 256)
                   where bin_means[0] = 0.0 for zeros, bin_means[1:] for non-zero bins
        value_transform: "none" or "log1p"

    Returns:
        Decoded value, or None for missing/SOS tokens
    """
    if token == missing_token:
        return None
    if sos_token is not None and token == sos_token:
        return None

    # Token 0 is always exactly zero — no transform needed
    if token == 0:
        return 0.0

    # Total gas bins = 256 (1 zero bin + 255 non-zero bins)
    total_bins = len(boundaries) + 2  # 254 + 2 = 256
    if token < 0 or token >= total_bins:
        return None

    if bin_means is not None and len(bin_means) > token:
        decoded = bin_means[token]
    else:
        # Fallback to midpoint calculation for non-zero tokens
        # Adjust token index for boundaries (token 1 -> boundary index 0)
        adj_token = token - 1

        if adj_token == 0:
            decoded = boundaries[0]
        elif adj_token == len(boundaries):
            decoded = boundaries[-1]
        else:
            decoded = (boundaries[adj_token-1] + boundaries[adj_token]) / 2

    # Apply inverse transform for non-zero values
    if value_transform == "log1p":
        decoded = math.expm1(decoded)

    return decoded


def detokenise_data(tokens_df, elec_boundaries, gas_boundaries,
                    temp_boundaries=None, solar_boundaries=None,
                    elec_bin_means=None, gas_bin_means=None,
                    value_transform="none"):
    """
    Convert tokenised dataframe back to approximate values.

    If bin_means are provided, uses empirical bin means for better fidelity.
    Otherwise falls back to midpoint interpolation between boundaries.
    If value_transform is "log1p", applies expm1() after decoding to convert
    from log space back to original Wh space.

    Args:
        tokens_df: DataFrame with token columns (elec_token, gas_token, etc.)
        elec_boundaries: Electricity quantile boundaries
        gas_boundaries: Gas quantile boundaries
        temp_boundaries: Optional temperature boundaries
        solar_boundaries: Optional solar boundaries
        elec_bin_means: Optional electricity bin means for improved decoding
        gas_bin_means: Optional gas bin means for improved decoding
        value_transform: "none" or "log1p" — applied to electricity and gas only

    Returns:
        DataFrame with decoded value columns added
    """
    elec_usage_bins = len(elec_boundaries) + 1
    elec_missing_token = elec_usage_bins      # Token 256 for num_bins=256
    elec_sos_token = elec_usage_bins + 1      # Token 257 for num_bins=256

    # Gas uses zero-aware tokenisation: token 0 = zero, tokens 1-255 = non-zero
    # gas_boundaries has 254 values (for 255 non-zero bins), plus 1 zero bin = 256 total
    gas_usage_bins = len(gas_boundaries) + 2  # 254 + 2 = 256
    gas_missing_token = gas_usage_bins        # Token 256
    gas_sos_token = gas_usage_bins + 1        # Token 257

    df = tokens_df.copy()

    df['elec_value'] = df['elec_token'].apply(
        lambda x: get_value_from_token(
            x, elec_boundaries,
            elec_missing_token, elec_sos_token,
            bin_means=elec_bin_means,
            value_transform=value_transform
        )
    )

    # Gas decoding uses gas_bin_means which has 256 values: [0.0, bin1_mean, ..., bin255_mean]
    df['gas_value'] = df['gas_token'].apply(
        lambda x: get_gas_value_from_token(
            x, gas_boundaries,
            gas_missing_token, gas_sos_token,
            bin_means=gas_bin_means,
            value_transform=value_transform
        )
    )

    # Add temp and solar detokenization if boundaries are provided
    # (no bin_means for temp/solar - using midpoint fallback)
    if temp_boundaries is not None and 'temp_token' in df.columns:
        temp_usage_bins = len(temp_boundaries) + 1
        temp_missing_token = temp_usage_bins
        temp_sos_token = temp_usage_bins + 1

        df['temp_C'] = df['temp_token'].apply(
            lambda x: get_value_from_token(
                x, temp_boundaries,
                temp_missing_token, temp_sos_token
            )
        )

    if solar_boundaries is not None and 'solar_token' in df.columns:
        solar_usage_bins = len(solar_boundaries) + 1
        solar_missing_token = solar_usage_bins
        solar_sos_token = solar_usage_bins + 1

        df['surface_solar_radiation_downwards'] = df['solar_token'].apply(
            lambda x: get_value_from_token(
                x, solar_boundaries,
                solar_missing_token, solar_sos_token
            )
        )

    return df

def _tokenise_single_file(args):
    """
    Worker function to tokenise a single file. Used for parallel processing.

    Uses vectorized encoding via TokenVocabulary.encode_array() for 10-100x speedup
    compared to row-by-row .apply().

    Args:
        args: Tuple of (file_path, elec_vocab, gas_vocab, temp_vocab, solar_vocab, output_folder)
              where each vocab is a TokenVocabulary object (or dict representation for pickling)

    Returns:
        Tuple of (success: bool, file_path: str, error_msg: str or None)
    """
    (fp, elec_vocab_dict, gas_vocab_dict, temp_vocab_dict, solar_vocab_dict,
     output_folder) = args

    try:
        # Reconstruct TokenVocabulary objects from dicts (multiprocessing pickle workaround)
        elec_vocab = TokenVocabulary(**elec_vocab_dict)
        gas_vocab = TokenVocabulary(**gas_vocab_dict)
        temp_vocab = TokenVocabulary(**temp_vocab_dict)
        solar_vocab = TokenVocabulary(**solar_vocab_dict)

        df = pd.read_pickle(fp)

        # Vectorized tokenization - much faster than .apply()
        df['elec_token'] = elec_vocab.encode_array(df['Clean_elec_net_Wh'].values)
        df['gas_token'] = gas_vocab.encode_array(df['Clean_gas_Wh'].values)
        df['temp_token'] = temp_vocab.encode_array(df['temp_C'].values)
        df['solar_token'] = solar_vocab.encode_array(df['surface_solar_radiation_downwards'].values)

        # Add temporal features required for training
        if 'hh' not in df.columns:
            if 'Readings_from_midnight_local' in df.columns:
                df['hh'] = df['Readings_from_midnight_local']
            else:
                return (False, fp, f"Cannot derive 'hh': 'Readings_from_midnight_local' not found")

        if 'dow' not in df.columns:
            df['dow'] = pd.to_datetime(df.index).dayofweek

        if 'month' not in df.columns:
            df['month'] = pd.to_datetime(df.index).month

        if output_folder:
            base_name = os.path.basename(fp)
            out_fp = os.path.join(output_folder,
                                  base_name.replace(".pkl", "_tokenised.pkl"))
            df.to_pickle(out_fp)

        return (True, fp, None)

    except Exception as e:
        return (False, fp, str(e))


def tokenise_files(file_paths, elec_boundaries, gas_boundaries, temp_boundaries, solar_boundaries,
                   output_folder=None, num_workers=None, value_transform="none"):
    """
    Applies the tokenisers to each file in file_paths for all four modalities. Optionally saves output.

    Uses multiprocessing for parallel file processing and vectorized encoding via
    TokenVocabulary.encode_array() for maximum performance.

    Token allocation for each modality (elec, temp, solar):
    - Tokens 0 to len(boundaries): Quantile buckets for actual values (num_bins tokens)
    - Token num_bins: Missing/NaN values
    - Token num_bins+1: SOS token (used during training/inference, not during tokenisation)

    Gas uses zero-aware tokenisation:
    - Token 0: Zero gas usage
    - Tokens 1-255: Non-zero quantile bins
    - Token 256: Missing/NaN values
    - Token 257: SOS token (used during training/inference, not during tokenisation)

    Args:
        file_paths: List of paths to household pickle files
        elec_boundaries: Electricity quantile boundaries
        gas_boundaries: Gas quantile boundaries
        temp_boundaries: Temperature quantile boundaries
        solar_boundaries: Solar quantile boundaries
        output_folder: Folder to write tokenised files (optional)
        num_workers: Number of parallel workers. None = auto (cpu_count - 1, capped at 8).
                     Set to 1 for sequential processing.
        value_transform: "none" or "log1p" — passed to TokenVocabulary for encoding
    """
    # Create TokenVocabulary objects for vectorized encoding
    # Convert to dicts for multiprocessing pickling
    elec_vocab_dict = {'boundaries': list(elec_boundaries), 'zero_aware': False, 'value_transform': value_transform}
    gas_vocab_dict = {'boundaries': list(gas_boundaries), 'zero_aware': True, 'value_transform': value_transform}
    temp_vocab_dict = {'boundaries': list(temp_boundaries), 'zero_aware': False}
    solar_vocab_dict = {'boundaries': list(solar_boundaries), 'zero_aware': False}

    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    # Determine number of workers
    available_cores = mp.cpu_count()
    if num_workers is None:
        # Auto: use cpu_count - 1, capped at 8 (diminishing returns for I/O-bound work)
        num_workers = max(1, min(available_cores - 1, 8))
    else:
        num_workers = max(1, min(num_workers, available_cores))

    total_files = len(file_paths)

    # For small number of files or single worker, use sequential processing
    if num_workers == 1 or total_files <= 2:
        print(f"[tokenise] Processing {total_files} files sequentially (vectorized encoding)...")
        start_time = time.time()

        success_count = 0
        error_count = 0

        for i, fp in enumerate(file_paths):
            args = (fp, elec_vocab_dict, gas_vocab_dict, temp_vocab_dict, solar_vocab_dict,
                    output_folder)
            success, _, error_msg = _tokenise_single_file(args)

            if success:
                success_count += 1
            else:
                error_count += 1
                print(f"[tokenise] ERROR processing {fp}: {error_msg}")

            if (i + 1) % 100 == 0 or (i + 1) == total_files:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"[tokenise] Progress: {i+1}/{total_files} files ({rate:.1f} files/sec)")

    else:
        # Parallel processing with vectorized encoding
        print(f"[tokenise] Processing {total_files} files with {num_workers}/{available_cores} workers (vectorized encoding)...")
        start_time = time.time()

        # Prepare arguments for each file (pass vocab dicts, not raw boundaries)
        work_items = [
            (fp, elec_vocab_dict, gas_vocab_dict, temp_vocab_dict, solar_vocab_dict,
             output_folder)
            for fp in file_paths
        ]

        # Process in parallel
        success_count = 0
        error_count = 0

        with mp.Pool(processes=num_workers) as pool:
            # Use imap for progress reporting
            for i, (success, fp, error_msg) in enumerate(pool.imap(
                _tokenise_single_file, work_items, chunksize=max(1, total_files // (num_workers * 4))
            )):
                if success:
                    success_count += 1
                else:
                    error_count += 1
                    print(f"[tokenise] ERROR processing {fp}: {error_msg}")

                # Progress logging every 100 files
                if (i + 1) % 100 == 0 or (i + 1) == total_files:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed if elapsed > 0 else 0
                    print(f"[tokenise] Progress: {i+1}/{total_files} files ({rate:.1f} files/sec)")

    elapsed = time.time() - start_time
    rate = total_files / elapsed if elapsed > 0 else 0
    print(f"[tokenise] Completed: {success_count} succeeded, {error_count} failed in {elapsed:.1f}s ({rate:.1f} files/sec)")


###############################################################################
# 2. Saving & Loading the Boundaries
###############################################################################

def save_tokenisers(elec_boundaries, gas_boundaries, temp_boundaries, solar_boundaries,
                    out_path, elec_bin_means=None, gas_bin_means=None, value_transform="none",
                    binning_strategy="quantile", bin_clip_lower=0.0, bin_clip_upper_percentile=99.5,
                    temperature_clip_lower=-10.0, temperature_clip_upper_percentile=99.5,
                    temperature_clip_upper=None,
                    solar_clip_lower=0.0, solar_clip_upper_percentile=99.5):
    """
    Save tokeniser data to a pickle file.

    Args:
        elec_boundaries: Electricity boundaries
        gas_boundaries: Gas boundaries
        temp_boundaries: Temperature boundaries
        solar_boundaries: Solar boundaries
        out_path: Output file path
        elec_bin_means: Optional electricity bin means for improved decoding
        gas_bin_means: Optional gas bin means for improved decoding
        value_transform: "none" or "log1p" — stored so decode knows which transform was used
        binning_strategy: "quantile" or "uniform" — applies to electricity and gas
        bin_clip_lower: Lower clip bound for electricity/gas uniform bins
        bin_clip_upper_percentile: Upper clip percentile for electricity/gas uniform bins
        temperature_clip_lower: Lower clip bound for temperature uniform bins (default -10.0°C)
        temperature_clip_upper_percentile: Upper clip percentile for temperature uniform bins
        temperature_clip_upper: Absolute upper bound for temperature uniform bins (default None).
                                 When set, overrides temperature_clip_upper_percentile.
        solar_clip_lower: Lower clip bound for solar uniform bins (default 0.0 J/m²)
        solar_clip_upper_percentile: Upper clip percentile for solar uniform bins (default 99.5)
    """
    data = {
        "elec_boundaries": elec_boundaries,
        "gas_boundaries": gas_boundaries,
        "temp_boundaries": temp_boundaries,
        "solar_boundaries": solar_boundaries,
        "value_transform": value_transform,
        "binning_strategy": binning_strategy,
        "bin_clip_lower": bin_clip_lower,
        "bin_clip_upper_percentile": bin_clip_upper_percentile,
        "temperature_clip_lower": temperature_clip_lower,
        "temperature_clip_upper_percentile": temperature_clip_upper_percentile,
        "temperature_clip_upper": temperature_clip_upper,
        "solar_clip_lower": solar_clip_lower,
        "solar_clip_upper_percentile": solar_clip_upper_percentile,
        "version": 7,  # Version 7: solar uniform binning (Decision 046)
    }

    # Add bin means if provided
    if elec_bin_means is not None:
        data["elec_bin_means"] = elec_bin_means
    if gas_bin_means is not None:
        data["gas_bin_means"] = gas_bin_means

    with open(out_path, "wb") as f:
        pickle.dump(data, f)

    has_bin_means = elec_bin_means is not None or gas_bin_means is not None
    temp_upper_str = f"abs={temperature_clip_upper}" if temperature_clip_upper is not None else f"p{temperature_clip_upper_percentile}"
    print(f"[SAVE] Tokenisers saved to {out_path} (version 7, bin_means={has_bin_means}, "
          f"value_transform={value_transform}, binning_strategy={binning_strategy}, "
          f"temp_clip=[{temperature_clip_lower}, {temp_upper_str}], "
          f"solar_clip=[{solar_clip_lower}, p{solar_clip_upper_percentile}])")


def load_tokenisers(in_path):
    """
    Load tokeniser data from a pickle file.

    Returns:
        Tuple of (elec_boundaries, gas_boundaries, temp_boundaries, solar_boundaries,
                  elec_bin_means, gas_bin_means, value_transform)

        Note: bin_means may be None for older tokeniser files (version 1)
        Note: value_transform defaults to "none" for older files (version <3)
    """
    with open(in_path, "rb") as f:
        data = pickle.load(f)

    elec_boundaries = data["elec_boundaries"]
    gas_boundaries = data["gas_boundaries"]
    temp_boundaries = data.get("temp_boundaries", [])  # Backward compatibility
    solar_boundaries = data.get("solar_boundaries", [])  # Backward compatibility

    # Load bin means if available (version 2+)
    elec_bin_means = data.get("elec_bin_means", None)
    gas_bin_means = data.get("gas_bin_means", None)

    # Load value transform if available (version 3+)
    value_transform = data.get("value_transform", "none")

    # Load binning strategy metadata if available (version 4+)
    binning_strategy = data.get("binning_strategy", "quantile")

    # Load temperature clip parameters if available (version 5+)
    temperature_clip_lower = data.get("temperature_clip_lower", -10.0)
    temperature_clip_upper_percentile = data.get("temperature_clip_upper_percentile", 99.5)

    # Load absolute temperature upper bound if available (version 6+)
    temperature_clip_upper = data.get("temperature_clip_upper", None)

    # Load solar clip parameters if available (version 7+)
    solar_clip_lower = data.get("solar_clip_lower", 0.0)
    solar_clip_upper_percentile = data.get("solar_clip_upper_percentile", 99.5)

    version = data.get("version", 1)
    has_bin_means = elec_bin_means is not None or gas_bin_means is not None

    temp_upper_str = f"abs={temperature_clip_upper}" if temperature_clip_upper is not None else f"p{temperature_clip_upper_percentile}"
    print(f"[LOAD] Tokenisers loaded from {in_path} (version {version}, bin_means={has_bin_means}, "
          f"value_transform={value_transform}, binning_strategy={binning_strategy}, "
          f"temp_clip=[{temperature_clip_lower}, {temp_upper_str}], "
          f"solar_clip=[{solar_clip_lower}, p{solar_clip_upper_percentile}])")

    return elec_boundaries, gas_boundaries, temp_boundaries, solar_boundaries, elec_bin_means, gas_bin_means, value_transform


def load_tokenisers_as_vocab(in_path: str) -> Dict[str, TokenVocabulary]:
    """
    Load tokenisers and return as TokenVocabulary objects.

    This is the preferred way to load tokenisers for new code. It provides
    a cleaner interface with computed special tokens and encode/decode methods.

    Args:
        in_path: Path to the tokenisers pickle file

    Returns:
        Dict with keys: 'electricity', 'gas', 'temperature', 'solar'
        Each value is a TokenVocabulary instance.

    Example:
        >>> vocab = load_tokenisers_as_vocab('tokenisers.pkl')
        >>> vocab['electricity'].vocab_size
        258
        >>> vocab['gas'].decode(0)  # Zero token
        0.0
        >>> token = vocab['electricity'].encode(150.5)
    """
    (elec_boundaries, gas_boundaries, temp_boundaries, solar_boundaries,
     elec_bin_means, gas_bin_means, value_transform) = load_tokenisers(in_path)

    return {
        'electricity': TokenVocabulary(
            boundaries=list(elec_boundaries),
            bin_means=list(elec_bin_means) if elec_bin_means else None,
            zero_aware=False,
            value_transform=value_transform
        ),
        'gas': TokenVocabulary(
            boundaries=list(gas_boundaries),
            bin_means=list(gas_bin_means) if gas_bin_means else None,
            zero_aware=True,
            value_transform=value_transform
        ),
        'temperature': TokenVocabulary(
            boundaries=list(temp_boundaries) if temp_boundaries else [],
            bin_means=None,
            zero_aware=False
            # No value_transform for temperature — only elec/gas
        ),
        'solar': TokenVocabulary(
            boundaries=list(solar_boundaries) if solar_boundaries else [],
            bin_means=None,
            zero_aware=False
            # No value_transform for solar — only elec/gas
        ),
    }


###############################################################################
# 3. Utility Functions for Gathering File Paths with Optional Random Sampling
###############################################################################

def get_file_paths(folder_path, max_households=None, seed=None):
    """
    Returns a list of .pkl file paths in `folder_path`.
    If max_households < total number of files, randomly sample that many.
    """
    files = glob.glob(os.path.join(folder_path, "*.pkl"))
    total_files = len(files)
    print(f"[FILES] Found {total_files} pkl files in {folder_path}.")

    if seed is not None:
        random.seed(seed)

    if max_households is not None and max_households < len(files):
        files = random.sample(files, max_households)
        print(f"[FILES] Randomly selected {len(files)} out of {total_files} with seed={seed}.")

    return files


###############################################################################
# 4. Main (CLI) - Putting It All Together
###############################################################################

def load_config_from_yaml(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config.get('tokeniser', {})

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Universal tokeniser Utility")
    
    # Configuration file option
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML configuration file")
    
    # Flags for optional actions
    parser.add_argument("--create_tokenisers", action="store_true",
                        help="If set, build new tokenisers from data.")
    parser.add_argument("--do_tokenise", action="store_true",
                        help="If set, tokenise the data using the tokenisers (created or loaded).")

    # Folder for data
    parser.add_argument("--data_folder", type=str, default=None,
                        help="Folder containing .pkl files.")

    # Options for building tokenisers
    parser.add_argument("--max_households_create", type=int, default=None,
                        help="If creating tokenisers, randomly sample this many files.")
    parser.add_argument("--seed_create", type=int, default=None,
                        help="Random seed for selecting files for building tokenisers.")
    parser.add_argument("--num_bins", type=int, default=None,
                        help="Number of bins for quantile splitting when creating tokenisers.")

    # Options for loading tokenisers
    parser.add_argument("--load_tokenisers", type=str, default=None,
                        help="Path to an existing .pkl tokenisers file to load (skip creation).")

    # Options for saving tokenisers
    parser.add_argument("--save_tokenisers", type=str, default=None,
                        help="Path to save created tokenisers (if create_tokenisers is true).")

    # Options for tokenising
    parser.add_argument("--max_households_tokenise", type=int, default=None,
                        help="If tokenising, randomly sample this many files.")
    parser.add_argument("--seed_tokenise", type=int, default=None,
                        help="Random seed for selecting files for tokenisation.")
    parser.add_argument("--output_folder", type=str, default=None,
                        help="Folder to write tokenised .pkl files.")

    args = parser.parse_args()
    
    # Load configuration
    config = {}
    if args.config:
        config = load_config_from_yaml(args.config)
        print(f"[CONFIG] Loaded configuration from: {args.config}")
    
    # Override config with command line arguments (if provided)
    create_tokenisers = args.create_tokenisers if args.create_tokenisers else config.get('create_tokenisers', False)
    do_tokenise = args.do_tokenise if args.do_tokenise else config.get('do_tokenise', False)
    data_folder = args.data_folder if args.data_folder is not None else config.get('data_folder')
    max_households_create = args.max_households_create if args.max_households_create is not None else config.get('max_households_create')
    max_households_tokenise = args.max_households_tokenise if args.max_households_tokenise is not None else config.get('max_households_tokenise')
    seed_create = args.seed_create if args.seed_create is not None else config.get('seed_create')
    seed_tokenise = args.seed_tokenise if args.seed_tokenise is not None else config.get('seed_tokenise')

    # Parse num_bins - supports both single int and per-modality dict
    # CLI --num_bins overrides config for backward compatibility (applies to all modalities)
    num_bins_config = config.get('num_bins', 256)
    if args.num_bins is not None:
        # CLI override: use single value for all modalities
        num_bins_default = args.num_bins
        num_bins_elec = args.num_bins
        num_bins_gas = args.num_bins
        num_bins_temp = args.num_bins
        num_bins_solar = args.num_bins
    elif isinstance(num_bins_config, dict):
        # Per-modality config: {electricity: X, gas: Y, temperature: Z, solar: W}
        num_bins_default = num_bins_config.get('default', 256)
        num_bins_elec = num_bins_config.get('electricity', num_bins_default)
        num_bins_gas = num_bins_config.get('gas', num_bins_default)
        num_bins_temp = num_bins_config.get('temperature', num_bins_default)
        num_bins_solar = num_bins_config.get('solar', num_bins_default)
    else:
        # Single int config (backward compatible)
        num_bins_default = num_bins_config
        num_bins_elec = num_bins_config
        num_bins_gas = num_bins_config
        num_bins_temp = num_bins_config
        num_bins_solar = num_bins_config

    load_tokenisers_path = args.load_tokenisers if args.load_tokenisers is not None else config.get('load_tokenisers')
    save_tokenisers_path = args.save_tokenisers if args.save_tokenisers is not None else config.get('save_tokenisers')
    output_folder = args.output_folder if args.output_folder is not None else config.get('output_folder')
    num_workers_tokenise = config.get('num_workers_tokenise')  # null = auto
    value_transform = config.get('value_transform', 'none')  # "none" or "log1p" (Decision 040)
    binning_strategy = config.get('binning_strategy', 'quantile')  # "quantile" or "uniform" (Decision 041)
    bin_clip_lower = config.get('bin_clip_lower', 0.0)
    bin_clip_upper_percentile = config.get('bin_clip_upper_percentile', 99.5)
    temperature_clip_lower = config.get('temperature_clip_lower', -10.0)  # Decision 045
    temperature_clip_upper_percentile = config.get('temperature_clip_upper_percentile', 99.5)
    temperature_clip_upper = config.get('temperature_clip_upper', None)  # Decision 045 (absolute)
    solar_clip_lower = config.get('solar_clip_lower', 0.0)              # Decision 046
    solar_clip_upper_percentile = config.get('solar_clip_upper_percentile', 99.5)
    print(f"[CONFIG] Value transform: {value_transform}")
    print(f"[CONFIG] Binning strategy (elec/gas): {binning_strategy}")
    if binning_strategy == "uniform":
        print(f"[CONFIG] Uniform clip (elec/gas): lower={bin_clip_lower}, upper_percentile={bin_clip_upper_percentile}")
    temp_upper_str = f"abs={temperature_clip_upper}" if temperature_clip_upper is not None else f"p{temperature_clip_upper_percentile}"
    print(f"[CONFIG] Temperature binning: uniform (always), clip=[{temperature_clip_lower}, {temp_upper_str}]")
    print(f"[CONFIG] Solar binning: uniform (always), clip=[{solar_clip_lower}, p{solar_clip_upper_percentile}]")

    # Ensure data_folder is provided
    if not data_folder:
        raise ValueError("Data folder must be specified either via --data_folder or in config file")

    # Get filtering configuration
    filtering_config = config.get('filtering', {})
    filtering_enabled = filtering_config.get('enabled', False)

    # 1. Decide how to get the tokenisers
    elec_boundaries, gas_boundaries, temp_boundaries, solar_boundaries = None, None, None, None
    elec_bin_means, gas_bin_means = None, None  # For improved decoding
    tokeniser_puprns = set()  # Track which households were used to create tokenisers

    if load_tokenisers_path:
        # If user provided a path to load from, do that first
        (elec_boundaries, gas_boundaries, temp_boundaries, solar_boundaries,
         elec_bin_means, gas_bin_means, value_transform) = load_tokenisers(load_tokenisers_path)

        # Try to load associated PUPRN list if it exists
        puprn_list_path = load_tokenisers_path.replace('.pkl', '_puprns.json')
        if os.path.exists(puprn_list_path):
            with open(puprn_list_path, 'r') as f:
                tokeniser_puprns = set(json.load(f))
            print(f"[LOAD] Loaded {len(tokeniser_puprns)} PUPRNs used for tokeniser creation")

    elif create_tokenisers:
        # Get ALL file paths first (no max limit yet)
        # max_households_create is applied AFTER filtering to ensure we get N filtered households
        all_file_paths = get_file_paths(
            folder_path=data_folder,
            max_households=None,  # Get all files first
            seed=None
        )

        # Apply filtering if enabled
        if filtering_enabled:
            print(f"\n[FILTERING] Tokeniser filtering ENABLED")
            filtered_file_paths, filter_results = apply_tokeniser_filters(
                all_file_paths, filtering_config
            )
            file_paths_create = filtered_file_paths

            # Generate distribution comparison report
            if filtering_config.get('save_distribution_report', True):
                report_filename = filtering_config.get('distribution_report_path', 'tokeniser_distribution_report.json')
                if output_folder:
                    report_path = os.path.join(output_folder, report_filename)
                elif save_tokenisers_path:
                    report_path = os.path.join(os.path.dirname(save_tokenisers_path), report_filename)
                else:
                    report_path = report_filename

                generate_distribution_report(
                    all_file_paths=all_file_paths,
                    filtered_file_paths=filtered_file_paths,
                    filter_results=filter_results,
                    output_path=report_path,
                    seed=seed_create  # Use same seed as tokeniser creation for consistency
                )
        else:
            print(f"\n[FILTERING] Tokeniser filtering DISABLED - using all households")
            file_paths_create = all_file_paths

        # Apply max_households_create AFTER filtering
        # This ensures you get N filtered households, not N pre-filter households
        if max_households_create is not None:
            available_count = len(file_paths_create)
            if available_count < max_households_create:
                raise ValueError(
                    f"max_households_create ({max_households_create}) exceeds available households "
                    f"after filtering ({available_count}). "
                    f"Either reduce max_households_create, adjust filtering criteria, "
                    f"or set max_households_create to null to use all available households. "
                    f"This check prevents downstream issues with sharding and val/test splits."
                )
            if seed_create is not None:
                random.seed(seed_create)
            file_paths_create = random.sample(file_paths_create, max_households_create)
            print(f"[CREATE] Sampled {max_households_create} households from {available_count} available (after filtering)")
        else:
            print(f"[CREATE] Using all {len(file_paths_create)} available households (after filtering)")

        # Track which PUPRNs were used for tokeniser creation
        tokeniser_puprns = set(os.path.basename(fp).replace('.pkl', '') for fp in file_paths_create)

        # Use memory-efficient incremental approach with per-modality bins
        (elec_boundaries, gas_boundaries, temp_boundaries, solar_boundaries,
         elec_bin_means, gas_bin_means) = build_tokenisers_incremental(
            file_paths=file_paths_create,
            num_bins=num_bins_default,
            num_bins_elec=num_bins_elec,
            num_bins_gas=num_bins_gas,
            num_bins_temp=num_bins_temp,
            num_bins_solar=num_bins_solar,
            value_transform=value_transform,
            binning_strategy=binning_strategy,
            bin_clip_lower=bin_clip_lower,
            bin_clip_upper_percentile=bin_clip_upper_percentile,
            temperature_clip_lower=temperature_clip_lower,
            temperature_clip_upper_percentile=temperature_clip_upper_percentile,
            temperature_clip_upper=temperature_clip_upper,
            solar_clip_lower=solar_clip_lower,
            solar_clip_upper_percentile=solar_clip_upper_percentile
        )

        # Optionally save
        if save_tokenisers_path:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(save_tokenisers_path), exist_ok=True)
            save_tokenisers(elec_boundaries, gas_boundaries, temp_boundaries, solar_boundaries,
                            save_tokenisers_path, elec_bin_means=elec_bin_means, gas_bin_means=gas_bin_means,
                            value_transform=value_transform,
                            binning_strategy=binning_strategy,
                            bin_clip_lower=bin_clip_lower,
                            bin_clip_upper_percentile=bin_clip_upper_percentile,
                            temperature_clip_lower=temperature_clip_lower,
                            temperature_clip_upper_percentile=temperature_clip_upper_percentile,
                            temperature_clip_upper=temperature_clip_upper,
                            solar_clip_lower=solar_clip_lower,
                            solar_clip_upper_percentile=solar_clip_upper_percentile)

            # Save the list of PUPRNs used for tokeniser creation
            puprn_list_path = save_tokenisers_path.replace('.pkl', '_puprns.json')
            with open(puprn_list_path, 'w') as f:
                json.dump(list(tokeniser_puprns), f)
            print(f"[SAVE] Saved {len(tokeniser_puprns)} PUPRNs to {puprn_list_path}")

    # If we still don't have boundaries, and user wants to tokenise => problem
    if do_tokenise and (elec_boundaries is None or gas_boundaries is None or temp_boundaries is None or solar_boundaries is None):
        raise ValueError("Cannot tokenise without tokenisers. "
                         "Either load them or create them first.")

    # 2. tokenise if requested
    if do_tokenise:
        print(f"[tokenise] Starting tokenisation process...")
        print(f"[tokenise] Data folder: {data_folder}")
        print(f"[tokenise] Output folder: {output_folder}")

        # IMPORTANT: Only tokenise households that were used to create the tokenisers
        # This guarantees distribution alignment between tokeniser bins and training data
        if tokeniser_puprns:
            print(f"[tokenise] Using ONLY households from tokeniser creation set ({len(tokeniser_puprns)} households)")
            print(f"[tokenise] This ensures tokenised data distribution matches tokeniser bin distribution")

            # Build file paths from tokeniser PUPRNs
            file_paths_tokenise = []
            missing_files = []
            for puprn in tokeniser_puprns:
                fp = os.path.join(data_folder, f"{puprn}.pkl")
                if os.path.exists(fp):
                    file_paths_tokenise.append(fp)
                else:
                    missing_files.append(puprn)

            if missing_files:
                print(f"[WARNING] {len(missing_files)} PUPRNs from tokeniser creation not found in data_folder")
                print(f"  First 10 missing: {missing_files[:10]}")

            print(f"[tokenise] Found {len(file_paths_tokenise)} files to tokenise (from tokeniser creation set)")
        else:
            # Fallback: no PUPRN list available (e.g., loaded old tokenisers without PUPRN list)
            print(f"[WARNING] No tokeniser PUPRN list available - falling back to sampling all files")
            print(f"[WARNING] This may cause distribution mismatch if different households are used!")
            print(f"[tokenise] Max households: {max_households_tokenise}")

            file_paths_tokenise = get_file_paths(
                folder_path=data_folder,
                max_households=max_households_tokenise,
                seed=seed_tokenise
            )
            print(f"[tokenise] Found {len(file_paths_tokenise)} files to tokenise")

        if len(file_paths_tokenise) == 0:
            print("[WARNING] No files found to tokenise!")
        else:
            tokenise_files(
                file_paths=file_paths_tokenise,
                elec_boundaries=elec_boundaries,
                gas_boundaries=gas_boundaries,
                temp_boundaries=temp_boundaries,
                solar_boundaries=solar_boundaries,
                output_folder=output_folder,
                num_workers=num_workers_tokenise,
                value_transform=value_transform
            )
    end_time = time.time()
    time_elapsed_minutes = (end_time - start_time)/60
    print(f"[DONE] All requested operations completed. Total time elapsed: {time_elapsed_minutes} mins")


###############################################################################
# 5. If Running as a Script
###############################################################################

if __name__ == "__main__":
    main()
