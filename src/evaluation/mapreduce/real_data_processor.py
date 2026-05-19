"""
RealDataProcessor for MapReduce evaluation pipeline.

Processes real household data in parallel, producing RealDataStats
with the same structure as synthetic data for consistent comparison.
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd

from .data_structures import FuelStats, BatchResult, RealDataStats, MissingnessStats
from .month_statistics import MonthStatisticsCalculator


class RealDataProcessor:
    """
    Process real household data in parallel.

    Produces FuelStats in the same format as BatchProcessor,
    enabling consistent comparison methodology.
    """

    # Column name mappings for real data
    ELEC_COL = 'Clean_elec_net_Wh'
    GAS_COL = 'Clean_gas_Wh'
    HALFHOUR_COL = 'Readings_from_midnight_local'

    # Mapping from config conditioning variable names to metadata_catalog_filtered columns
    METADATA_COLUMN_MAP = {
        'building_type': 'building_type_descriptive',
        'age_built': 'age_built_descriptive',
        'num_rooms': 'B5',
        'num_occs': 'C1_new',
        'region': 'Region',
    }

    # Timesteps per week (7 days × 48 half-hours)
    TIMESTEPS_PER_WEEK = 336

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 n_workers: Optional[int] = None):
        """
        Args:
            config: Evaluation configuration
            n_workers: Number of parallel workers (default: CPU count - 1)
        """
        self.config = config or {}
        self.n_workers = n_workers or max(1, os.cpu_count() - 1)
        self.acf_lags = self.config.get('acf_lags', [1, 2, 48, 96, 336, 672, 1008, 1344])
        self.batch_size = self.config.get('real_data_batch_size', 100)

        # Integrated diagnostics config
        self.diag_config = self.config.get('diagnostics_integration', {})
        self.diag_enabled = self.diag_config.get('enabled', False)

        # Load tokeniser bin edges if available (for token analysis)
        self._bin_edges = None
        if self.diag_enabled and self.diag_config.get('token_analysis', {}).get('enabled', True):
            tokeniser_path = self.config.get('tokeniser_file')
            if tokeniser_path:
                self._bin_edges = self._load_tokeniser_bin_edges(tokeniser_path)

    def process(self, data_dir: str, split: str = 'train',
                metadata_path: Optional[str] = None,
                puprn_filter: Optional[List[str]] = None) -> RealDataStats:
        """
        Process all real household files in parallel.

        Args:
            data_dir: Directory containing household pickle files
            split: 'train', 'val', or 'test'
            metadata_path: Optional path to metadata catalog for conditioning
            puprn_filter: Optional list of PUPRNs to include (from shard_metadata splits).
                          If None, all households in the directory are processed.

        Returns:
            RealDataStats with aggregated statistics
        """
        # 1. Discover household files
        household_files = self._discover_households(data_dir)

        # 2. Filter by PUPRN list if provided
        if puprn_filter is not None:
            puprn_set = set(puprn_filter)
            household_files = [
                f for f in household_files
                if f.stem.replace('_tokenised', '') in puprn_set
            ]

        print(f"[REAL] Found {len(household_files)} {split} households" +
              (f" (filtered from {len(puprn_filter)} PUPRNs)" if puprn_filter else ""))

        if not household_files:
            return RealDataStats(
                n_households=0,
                split=split,
                fuel_stats={},
                acf={},
                volatility_stats={},
                profiles={},
                conditional={}
            )

        # 2. Load metadata if available
        metadata = self._load_metadata(metadata_path) if metadata_path else None

        # 3. Group into batches for parallel processing
        batches = self._create_batches(household_files)
        print(f"[REAL] Processing in {len(batches)} batches with {self.n_workers} workers")

        # 4. Process batches in parallel
        batch_results = self._run_parallel(batches, metadata)

        # 5. Aggregate results
        return self._aggregate(batch_results, split)

    def _discover_households(self, data_dir: str) -> List[Path]:
        """Find all household pickle files."""
        data_path = Path(data_dir)

        # Try different patterns for real data files
        patterns = ['*_tokenised.pkl', '*.pkl']

        for pattern in patterns:
            files = sorted(data_path.glob(pattern))
            # Filter out non-household files
            files = [f for f in files if not any(x in f.name.lower()
                     for x in ['metadata', 'catalog', 'summary', 'conditioning'])]
            if files:
                return files

        return []

    def _load_metadata(self, metadata_path: str) -> Optional[pd.DataFrame]:
        """Load metadata catalog for conditioning variables."""
        if not metadata_path or not os.path.exists(metadata_path):
            return None

        try:
            metadata = pd.read_pickle(metadata_path)
            # Set index to puprn column if available (required for PUPRN lookup)
            if 'puprn' in metadata.columns:
                metadata = metadata.set_index('puprn')
            return metadata
        except Exception as e:
            print(f"[REAL] Warning: Failed to load metadata: {e}")
            return None

    def _get_metadata_column(self, var: str) -> str:
        """Get the actual metadata column name for a conditioning variable.

        Args:
            var: Conditioning variable name (e.g., 'building_type')

        Returns:
            The actual column name in metadata_catalog_filtered.pkl
        """
        return self.METADATA_COLUMN_MAP.get(var, var)

    def _create_batches(self, household_files: List[Path]) -> List[List[Path]]:
        """Group household files into batches for parallel processing."""
        return [
            household_files[i:i + self.batch_size]
            for i in range(0, len(household_files), self.batch_size)
        ]

    def _run_parallel(self, batches: List[List[Path]],
                      metadata: Optional[pd.DataFrame]) -> List[BatchResult]:
        """Process household batches in parallel."""
        results = []

        # For small number of batches, process sequentially to avoid overhead
        if len(batches) <= 2 or self.n_workers == 1:
            for i, batch in enumerate(batches):
                try:
                    result = self._process_household_batch(batch, metadata, i)
                    results.append(result)
                    print(f"[REAL]   Completed batch {i+1}/{len(batches)}")
                except Exception as e:
                    print(f"[REAL]   Batch {i+1} failed: {e}")
            return results

        # Parallel processing
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {
                executor.submit(self._process_household_batch, batch, metadata, i): i
                for i, batch in enumerate(batches)
            }

            for future in as_completed(futures):
                batch_idx = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    print(f"[REAL]   Completed batch {batch_idx+1}/{len(batches)}")
                except Exception as e:
                    print(f"[REAL]   Batch {batch_idx+1} failed: {e}")

        return results

    def _process_household_batch(self, household_files: List[Path],
                                  metadata: Optional[pd.DataFrame],
                                  batch_idx: int) -> BatchResult:
        """
        Process a batch of real households.

        Returns BatchResult with same structure as synthetic batches.
        """
        # Collect data from all households
        all_elec_values = []
        all_gas_values = []
        monthly_elec = defaultdict(list)
        monthly_gas = defaultdict(list)

        acf_values = {'electricity': defaultdict(list), 'gas': defaultdict(list)}
        volatility_elec = []
        volatility_gas = []

        profile_contributions = {
            'electricity': {'all': defaultdict(lambda: [0.0, 0]),
                           'weekday': defaultdict(lambda: [0.0, 0]),
                           'weekend': defaultdict(lambda: [0.0, 0])},
            'gas': {'all': defaultdict(lambda: [0.0, 0]),
                   'weekday': defaultdict(lambda: [0.0, 0]),
                   'weekend': defaultdict(lambda: [0.0, 0])}
        }

        # Collect raw electricity values per half-hour for median profiles
        elec_profile_values = {
            'all': defaultdict(list),
            'weekday': defaultdict(list),
            'weekend': defaultdict(list),
        }

        household_month_stats = []
        conditioning_counts = defaultdict(lambda: defaultdict(int))
        n_households = 0
        n_days = 0

        # Conditional distribution tracking: {cond_var: {cond_value: {fuel: [values]}}}
        conditional_values = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        cond_vars = self.config.get('conditioning_vars',
                                     ['building_type', 'num_rooms', 'region', 'age_built', 'num_occs'])

        # Missingness tracking
        missingness_data = {
            'electricity': {'n_total': 0, 'n_missing': 0, 'n_zeros': 0,
                           'n_households_no_data': 0, 'n_households_total': 0,
                           'missing_run_lengths': {}, 'zero_run_lengths': {}},
            'gas': {'n_total': 0, 'n_missing': 0, 'n_zeros': 0,
                   'n_households_no_data': 0, 'n_households_total': 0,
                   'missing_run_lengths': {}, 'zero_run_lengths': {}}
        }

        # Pairwise correlation tracking
        corr_lists: Dict[str, list] = {'elec_gas': [], 'temp_elec': [], 'temp_gas': []}

        # PTG monthly sums for Power-Temperature Gradient analysis
        ptg_monthly: Dict[int, Dict[str, float]] = {}
        for m in range(1, 13):
            ptg_monthly[m] = {'elec_sum': 0.0, 'elec_count': 0, 'gas_sum': 0.0, 'gas_count': 0,
                              'temp_sum': 0.0, 'temp_count': 0}

        # Value samples (reservoir sampling)
        value_samples = {'electricity': [], 'gas': []}
        reservoir_count = {'electricity': 0, 'gas': 0}
        max_samples = 10000
        rng = np.random.default_rng(42)

        # --- Integrated diagnostics tracking ---
        seasonal_household_means: Dict[str, Dict[int, List[float]]] = {
            'electricity': defaultdict(list), 'gas': defaultdict(list)
        }
        monthly_profile_contribs: Dict[str, Dict[int, Dict[int, List]]] = {
            'electricity': defaultdict(lambda: defaultdict(lambda: [0.0, 0, 0.0])),
            'gas': defaultdict(lambda: defaultdict(lambda: [0.0, 0, 0.0]))
        }
        weekly_profile_contribs: Dict[str, Dict[int, Dict[int, List]]] = {
            'electricity': defaultdict(lambda: defaultdict(lambda: [0.0, 0, 0.0])),
            'gas': defaultdict(lambda: defaultdict(lambda: [0.0, 0, 0.0]))
        }
        weekly_values_for_stats: Dict[str, Dict[int, List[float]]] = {
            'electricity': defaultdict(list), 'gas': defaultdict(list)
        }
        token_freq_counts: Dict[str, Optional[List[int]]] = {'electricity': None, 'gas': None}
        monthly_decoded: Dict[str, Dict[int, Dict[str, float]]] = {
            'electricity': {}, 'gas': {}
        }
        weekly_token_counts: Dict[str, Dict[int, List[int]]] = {
            'electricity': {}, 'gas': {}
        }
        num_weeks = self.diag_config.get('weekly_fidelity', {}).get('num_weeks', 4)

        for hh_file in household_files:
            try:
                df = pd.read_pickle(hh_file)
                n_households += 1

                # Extract PUPRN from filename
                puprn = hh_file.stem.replace('_tokenised', '')

                # Process electricity
                if self.ELEC_COL in df.columns:
                    elec_raw = df[self.ELEC_COL].values
                    elec_values = df[self.ELEC_COL].dropna().values
                    all_elec_values.extend(elec_values)

                    # Missingness tracking
                    missingness_data['electricity']['n_households_total'] += 1
                    missingness_data['electricity']['n_total'] += len(elec_raw)
                    is_nan = pd.isna(elec_raw)
                    n_nan = int(np.sum(is_nan))
                    missingness_data['electricity']['n_missing'] += n_nan
                    non_nan_values = elec_raw[~is_nan]
                    missingness_data['electricity']['n_zeros'] += int(np.sum(non_nan_values == 0))
                    if n_nan == len(elec_raw):
                        missingness_data['electricity']['n_households_no_data'] += 1
                    self._update_run_lengths(is_nan, missingness_data['electricity']['missing_run_lengths'])
                    is_zero = np.zeros(len(elec_raw), dtype=bool)
                    is_zero[~is_nan] = (elec_raw[~is_nan] == 0)
                    self._update_run_lengths(is_zero, missingness_data['electricity']['zero_run_lengths'])

                    # Value sampling (reservoir)
                    for val in elec_values:
                        n = reservoir_count['electricity']
                        if n < max_samples:
                            value_samples['electricity'].append(float(val))
                        else:
                            j = rng.integers(0, n + 1)
                            if j < max_samples:
                                value_samples['electricity'][j] = float(val)
                        reservoir_count['electricity'] += 1

                    # Monthly breakdown
                    if 'month' in df.columns:
                        for month in df['month'].unique():
                            month_vals = df[df['month'] == month][self.ELEC_COL].dropna().values
                            monthly_elec[int(month)].extend(month_vals)

                    # ACF (within month if available)
                    self._compute_household_acf(df, self.ELEC_COL, acf_values['electricity'])

                    # Volatility (first differences, respecting month boundaries)
                    self._compute_household_volatility(df, self.ELEC_COL, volatility_elec)

                    # Profile contributions
                    self._compute_household_profiles(df, self.ELEC_COL,
                                                     profile_contributions['electricity'])

                    # Collect per-half-hour values for electricity median profiles
                    if self.HALFHOUR_COL in df.columns:
                        for _, row in df.iterrows():
                            hh = int(row[self.HALFHOUR_COL])
                            val = row[self.ELEC_COL]
                            if pd.isna(val):
                                continue
                            elec_profile_values['all'][hh].append(val)
                            if 'dayofweek' in df.columns:
                                dow = row['dayofweek']
                            elif hasattr(_, 'dayofweek'):
                                dow = _.dayofweek
                            else:
                                continue
                            if dow in [5, 6]:
                                elec_profile_values['weekend'][hh].append(val)
                            else:
                                elec_profile_values['weekday'][hh].append(val)

                # Process gas
                if self.GAS_COL in df.columns:
                    gas_raw = df[self.GAS_COL].values
                    gas_values = df[self.GAS_COL].dropna().values
                    all_gas_values.extend(gas_values)

                    # Missingness tracking
                    missingness_data['gas']['n_households_total'] += 1
                    missingness_data['gas']['n_total'] += len(gas_raw)
                    is_nan = pd.isna(gas_raw)
                    n_nan = int(np.sum(is_nan))
                    missingness_data['gas']['n_missing'] += n_nan
                    non_nan_values = gas_raw[~is_nan]
                    missingness_data['gas']['n_zeros'] += int(np.sum(non_nan_values == 0))
                    if n_nan == len(gas_raw):
                        missingness_data['gas']['n_households_no_data'] += 1
                    self._update_run_lengths(is_nan, missingness_data['gas']['missing_run_lengths'])
                    is_zero = np.zeros(len(gas_raw), dtype=bool)
                    is_zero[~is_nan] = (gas_raw[~is_nan] == 0)
                    self._update_run_lengths(is_zero, missingness_data['gas']['zero_run_lengths'])

                    # Value sampling (reservoir)
                    for val in gas_values:
                        n = reservoir_count['gas']
                        if n < max_samples:
                            value_samples['gas'].append(float(val))
                        else:
                            j = rng.integers(0, n + 1)
                            if j < max_samples:
                                value_samples['gas'][j] = float(val)
                        reservoir_count['gas'] += 1

                    if 'month' in df.columns:
                        for month in df['month'].unique():
                            month_vals = df[df['month'] == month][self.GAS_COL].dropna().values
                            monthly_gas[int(month)].extend(month_vals)

                    self._compute_household_acf(df, self.GAS_COL, acf_values['gas'])
                    self._compute_household_volatility(df, self.GAS_COL, volatility_gas)
                    self._compute_household_profiles(df, self.GAS_COL,
                                                     profile_contributions['gas'])

                # Pairwise correlations (elec-gas, temp-elec, temp-gas)
                # Computed per-month to align with synthetic data (which arrives as month-chunks)
                has_elec = self.ELEC_COL in df.columns
                has_gas = self.GAS_COL in df.columns
                has_temp = 'temp_C' in df.columns
                if 'month' in df.columns:
                    for month in df['month'].unique():
                        month_df = df[df['month'] == month]
                        if has_elec and has_gas:
                            self._add_correlation(month_df[self.ELEC_COL].values, month_df[self.GAS_COL].values, corr_lists['elec_gas'])
                        if has_temp and has_elec:
                            self._add_correlation(month_df['temp_C'].values, month_df[self.ELEC_COL].values, corr_lists['temp_elec'])
                        if has_temp and has_gas:
                            self._add_correlation(month_df['temp_C'].values, month_df[self.GAS_COL].values, corr_lists['temp_gas'])
                else:
                    # Fallback: no month column — compute on full data
                    if has_elec and has_gas:
                        self._add_correlation(df[self.ELEC_COL].values, df[self.GAS_COL].values, corr_lists['elec_gas'])
                    if has_temp and has_elec:
                        self._add_correlation(df['temp_C'].values, df[self.ELEC_COL].values, corr_lists['temp_elec'])
                    if has_temp and has_gas:
                        self._add_correlation(df['temp_C'].values, df[self.GAS_COL].values, corr_lists['temp_gas'])

                # PTG monthly sums (consumption and temperature per month)
                if 'month' in df.columns:
                    for month in df['month'].unique():
                        month_df = df[df['month'] == month]
                        m = int(month)
                        if has_elec:
                            vals = month_df[self.ELEC_COL].dropna().values
                            ptg_monthly[m]['elec_sum'] += vals.sum()
                            ptg_monthly[m]['elec_count'] += len(vals)
                        if has_gas:
                            vals = month_df[self.GAS_COL].dropna().values
                            ptg_monthly[m]['gas_sum'] += vals.sum()
                            ptg_monthly[m]['gas_count'] += len(vals)
                        if has_temp:
                            vals = month_df['temp_C'].dropna().values
                            ptg_monthly[m]['temp_sum'] += vals.sum()
                            ptg_monthly[m]['temp_count'] += len(vals)

                # Compute household-month stats
                stats_before = len(household_month_stats)
                self._compute_real_household_stats(df, puprn, household_month_stats)

                # --- Integrated diagnostics per household ---
                if self.diag_enabled and 'month' in df.columns:
                    self._compute_household_diagnostics(
                        df, has_elec, has_gas,
                        seasonal_household_means, monthly_profile_contribs,
                        weekly_profile_contribs, weekly_values_for_stats,
                        token_freq_counts, monthly_decoded,
                        weekly_token_counts, num_weeks,
                    )

                # Track conditioning variables and values from metadata
                # Also attach to household_month_stats for conditioning breakdown
                hh_conditioning = {}
                if metadata is not None and puprn in metadata.index:
                    for var in cond_vars:
                        # Map conditioning variable name to actual metadata column
                        metadata_col = self._get_metadata_column(var)
                        if metadata_col in metadata.columns:
                            cond_value = metadata.loc[puprn, metadata_col]
                            # Handle numpy types
                            if hasattr(cond_value, 'item'):
                                cond_value = cond_value.item()
                            conditioning_counts[var][cond_value] += 1
                            hh_conditioning[var] = cond_value

                            # Track energy values for this conditioning group
                            if self.ELEC_COL in df.columns:
                                elec_vals = df[self.ELEC_COL].dropna().values
                                conditional_values[var][cond_value]['electricity'].extend(elec_vals)
                            if self.GAS_COL in df.columns:
                                gas_vals = df[self.GAS_COL].dropna().values
                                conditional_values[var][cond_value]['gas'].extend(gas_vals)

                # Attach conditioning to household_month_stats records
                if hh_conditioning:
                    for stat_record in household_month_stats[stats_before:]:
                        stat_record.update(hh_conditioning)

                # Approximate days
                n_days += len(df) // 48

            except Exception as e:
                print(f"[REAL] Warning: Failed to process {hh_file.name}: {e}")

        # Build FuelStats
        fuel_stats = {}

        if all_elec_values:
            monthly_arrays = {m: np.array(v) for m, v in monthly_elec.items()}
            fuel_stats['electricity'] = FuelStats.from_values(
                np.array(all_elec_values),
                n_days=n_days,
                monthly_values=monthly_arrays
            )

        if all_gas_values:
            monthly_arrays = {m: np.array(v) for m, v in monthly_gas.items()}
            fuel_stats['gas'] = FuelStats.from_values(
                np.array(all_gas_values),
                n_days=n_days,
                monthly_values=monthly_arrays
            )

        # Volatility stats + reservoir samples for histogram comparison
        volatility_stats = {}
        volatility_samples = {}
        _rng = np.random.default_rng(42)
        _sample_size = 5000
        for _fuel, _diffs_list in [('electricity', volatility_elec), ('gas', volatility_gas)]:
            if _diffs_list:
                _arr = np.array(_diffs_list)
                volatility_stats[_fuel] = FuelStats.from_values(_arr)
                if len(_arr) <= _sample_size:
                    volatility_samples[_fuel] = _arr.tolist()
                else:
                    _idx = _rng.choice(len(_arr), size=_sample_size, replace=False)
                    volatility_samples[_fuel] = _arr[_idx].tolist()

        # Convert profile contributions to tuples
        profile_result = {}
        for fuel in ['electricity', 'gas']:
            profile_result[fuel] = {}
            for ptype in ['all', 'weekday', 'weekend']:
                profile_result[fuel][ptype] = {
                    hh: (sc[0], sc[1])
                    for hh, sc in profile_contributions[fuel][ptype].items()
                }

        # Compute electricity median profiles from collected values
        profile_med_result = {'electricity': {}}
        for ptype in ['all', 'weekday', 'weekend']:
            profile_med_result['electricity'][ptype] = {}
            for hh, vals in elec_profile_values[ptype].items():
                if vals:
                    profile_med_result['electricity'][ptype][hh] = (
                        float(np.median(vals)), len(vals)
                    )

        # Build MissingnessStats
        missingness_stats = {}
        for fuel in ['electricity', 'gas']:
            md = missingness_data[fuel]
            missingness_stats[fuel] = MissingnessStats(
                n_total=md['n_total'],
                n_missing=md['n_missing'],
                n_zeros=md['n_zeros'],
                n_households_no_data=md['n_households_no_data'],
                n_households_total=md['n_households_total'],
                missing_run_lengths=md['missing_run_lengths'],
                zero_run_lengths=md['zero_run_lengths'],
            )

        # Pairwise correlations
        correlations = {}
        for key, values in corr_lists.items():
            if values:
                correlations[key] = float(np.mean(values))

        # Convert per-batch conditional raw values to FuelStats summaries.
        # Raw values are only held within this batch (small at batch_size ~10),
        # then summarised here so no raw values cross batch boundaries.
        # The reduce phase uses FuelStats.combine() to merge across batches.
        conditional_fuel_stats = {}
        for cond_var, cond_values in conditional_values.items():
            conditional_fuel_stats[cond_var] = {}
            for cond_value, fuel_values in cond_values.items():
                conditional_fuel_stats[cond_var][cond_value] = {}
                for fuel, values in fuel_values.items():
                    if values:
                        conditional_fuel_stats[cond_var][cond_value][fuel] = (
                            FuelStats.from_values(np.array(values))
                        )

        # --- Build diagnostic outputs ---
        diag_seasonal = {}
        diag_monthly_profile = {}
        diag_weekly_profile = {}
        diag_weekly_fuel = {}
        diag_token_freq = {}
        diag_monthly_decoded = {}
        diag_weekly_token = {}

        if self.diag_enabled:
            # Seasonal household means: convert defaultdicts, drop empty
            for fuel in ['electricity', 'gas']:
                fuel_data = dict(seasonal_household_means[fuel])
                if fuel_data:
                    diag_seasonal[fuel] = {m: v for m, v in fuel_data.items() if v}

            # Monthly profile contributions: convert nested defaultdicts to tuples
            for fuel in ['electricity', 'gas']:
                fuel_data = {}
                for month, hh_dict in monthly_profile_contribs[fuel].items():
                    fuel_data[month] = {
                        hh: (e[0], e[1], e[2]) for hh, e in hh_dict.items()
                    }
                if fuel_data:
                    diag_monthly_profile[fuel] = fuel_data

            # Weekly profile contributions: convert to tuples
            for fuel in ['electricity', 'gas']:
                fuel_data = {}
                for week, hh_dict in weekly_profile_contribs[fuel].items():
                    fuel_data[week] = {
                        hh: (e[0], e[1], e[2]) for hh, e in hh_dict.items()
                    }
                if fuel_data:
                    diag_weekly_profile[fuel] = fuel_data

            # Weekly FuelStats from collected values
            for fuel in ['electricity', 'gas']:
                fuel_data = {}
                for week, vals in weekly_values_for_stats[fuel].items():
                    if vals:
                        fuel_data[week] = FuelStats.from_values(np.array(vals))
                if fuel_data:
                    diag_weekly_fuel[fuel] = fuel_data

            # Token frequency, monthly decoded, weekly token
            from scipy import stats as scipy_stats
            for fuel in ['electricity', 'gas']:
                if token_freq_counts[fuel] is not None:
                    diag_token_freq[fuel] = token_freq_counts[fuel]
                # Finalize monthly decoded stats into standard format
                if monthly_decoded[fuel]:
                    finalized = {}
                    for m, entry in monthly_decoded[fuel].items():
                        if entry['n'] > 0:
                            arr = np.array(entry['median_vals']) if entry['median_vals'] else np.array([])
                            finalized[m] = {
                                'mean': entry['mean_sum'] / entry['n'],
                                'median': float(np.median(arr)) if len(arr) > 0 else 0.0,
                                'std': float(np.std(arr)) if len(arr) > 0 else 0.0,
                                'skew': float(scipy_stats.skew(arr)) if len(arr) > 2 else 0.0,
                                'zero_rate': entry['zero_count'] / entry['total_count'] if entry['total_count'] > 0 else 0.0,
                                'n': entry['n'],
                            }
                    if finalized:
                        diag_monthly_decoded[fuel] = finalized
                if weekly_token_counts[fuel]:
                    diag_weekly_token[fuel] = weekly_token_counts[fuel]

        # Convert conditioning_counts to plain dict for BatchResult
        batch_conditioning_n_households = {
            var: dict(cond_vals)
            for var, cond_vals in conditioning_counts.items()
        }

        return BatchResult(
            batch_id=f"real_batch_{batch_idx}",
            n_households=n_households,
            n_months_processed=12,  # Real data typically has all months
            conditioning={},  # Not used for real data (we track conditional_values instead)
            fuel_stats=fuel_stats,
            variance_metrics={},  # Not computed for real data
            acf_values={'electricity': dict(acf_values['electricity']),
                       'gas': dict(acf_values['gas'])},
            volatility_stats=volatility_stats,
            volatility_samples=volatility_samples,
            profile_contributions=profile_result,
            profile_medians=profile_med_result,
            household_month_stats=household_month_stats,
            missingness_stats=missingness_stats,
            correlations=correlations,
            value_samples=value_samples,
            conditional_values=conditional_fuel_stats,
            conditioning_n_households=batch_conditioning_n_households,
            ptg_monthly=ptg_monthly,
            seasonal_household_means=diag_seasonal,
            monthly_profile_contributions=diag_monthly_profile,
            weekly_profile_contributions=diag_weekly_profile,
            weekly_fuel_stats=diag_weekly_fuel,
            token_frequency=diag_token_freq,
            monthly_decoded_stats=diag_monthly_decoded,
            weekly_token_frequency=diag_weekly_token,
        )

    @staticmethod
    def _add_correlation(x: np.ndarray, y: np.ndarray, out: list) -> None:
        """Compute Pearson correlation between x and y, appending to out if valid."""
        valid_mask = ~(pd.isna(x) | pd.isna(y))
        if np.sum(valid_mask) < 10:
            return
        x_valid = x[valid_mask].astype(float)
        y_valid = y[valid_mask].astype(float)
        if np.std(x_valid) == 0 or np.std(y_valid) == 0:
            return
        corr = np.corrcoef(x_valid, y_valid)[0, 1]
        if not np.isnan(corr):
            out.append(corr)

    def _compute_household_acf(self, df: pd.DataFrame, col: str,
                                acf_dict: Dict[int, List[float]]) -> None:
        """Compute ACF for a single household, respecting month boundaries."""
        if 'month' in df.columns:
            for month in df['month'].unique():
                month_data = df[df['month'] == month][col].dropna().values
                if len(month_data) > max(self.acf_lags) + 1:
                    for lag in self.acf_lags:
                        acf = self._compute_single_acf(month_data, lag)
                        if not np.isnan(acf):
                            acf_dict[lag].append(acf)
        else:
            # No month column - compute on full series
            values = df[col].dropna().values
            if len(values) > max(self.acf_lags) + 1:
                for lag in self.acf_lags:
                    acf = self._compute_single_acf(values, lag)
                    if not np.isnan(acf):
                        acf_dict[lag].append(acf)

    def _compute_single_acf(self, values: np.ndarray, lag: int) -> float:
        """Compute autocorrelation at a single lag."""
        if len(values) <= lag:
            return np.nan

        series = values[:-lag]
        lagged = values[lag:]

        series_std = np.std(series)
        lagged_std = np.std(lagged)

        if series_std == 0 or lagged_std == 0:
            return np.nan

        series_mean = np.mean(series)
        lagged_mean = np.mean(lagged)
        covariance = np.mean((series - series_mean) * (lagged - lagged_mean))

        return covariance / (series_std * lagged_std)

    def _compute_household_volatility(self, df: pd.DataFrame, col: str,
                                       volatility_list: List[float]) -> None:
        """Compute first differences, respecting month boundaries."""
        if 'month' in df.columns:
            for month in df['month'].unique():
                month_data = df[df['month'] == month][col].dropna().values
                if len(month_data) > 1:
                    diffs = np.diff(month_data)
                    volatility_list.extend(diffs)
        else:
            values = df[col].dropna().values
            if len(values) > 1:
                diffs = np.diff(values)
                volatility_list.extend(diffs)

    def _compute_household_profiles(self, df: pd.DataFrame, col: str,
                                     contributions: Dict) -> None:
        """Compute profile contributions for a single household."""
        if self.HALFHOUR_COL not in df.columns:
            return

        for idx, row in df.iterrows():
            hh = int(row[self.HALFHOUR_COL])
            val = row[col]

            if pd.isna(val):
                continue

            contributions['all'][hh][0] += val
            contributions['all'][hh][1] += 1

            # Weekday/weekend
            if 'dayofweek' in df.columns:
                dow = row['dayofweek']
            elif hasattr(idx, 'dayofweek'):
                # idx is the timestamp from iterrows(), use its dayofweek directly
                dow = idx.dayofweek
            else:
                continue

            if dow in [5, 6]:
                contributions['weekend'][hh][0] += val
                contributions['weekend'][hh][1] += 1
            else:
                contributions['weekday'][hh][0] += val
                contributions['weekday'][hh][1] += 1

    def _update_run_lengths(self, mask: np.ndarray, run_lengths: Dict[int, int]) -> None:
        """
        Update run length histogram from a boolean mask.

        Args:
            mask: Boolean array where True indicates the condition (missing/zero)
            run_lengths: Dict mapping run_length -> count, updated in place
        """
        if len(mask) == 0:
            return

        # Find runs of True values
        # Pad with False at both ends to detect all transitions
        padded = np.concatenate([[False], mask, [False]])
        diffs = np.diff(padded.astype(int))

        # Starts are where diff == 1, ends are where diff == -1
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]

        for start, end in zip(starts, ends):
            length = end - start
            run_lengths[length] = run_lengths.get(length, 0) + 1

    @staticmethod
    def _load_tokeniser_bin_edges(tokeniser_path: str) -> Optional[Dict[str, Any]]:
        """Load tokeniser bin edges for token-level analysis.

        Supports two tokeniser formats:
        1. Flat dict with '{fuel_prefix}_boundaries' keys (current format)
        2. Nested dict with fuel objects having bin_edges/boundaries attributes
        """
        import pickle
        try:
            with open(tokeniser_path, 'rb') as f:
                tokenisers = pickle.load(f)

            bin_edges = {}
            value_transform = None

            if isinstance(tokenisers, dict):
                # Format 1: flat dict with elec_boundaries, gas_boundaries
                fuel_prefix_map = {
                    'electricity': 'elec_boundaries',
                    'gas': 'gas_boundaries',
                }
                for fuel, key in fuel_prefix_map.items():
                    edges = tokenisers.get(key)
                    if edges is not None:
                        bin_edges[fuel] = np.array(edges)

                value_transform = tokenisers.get('value_transform')

                # Format 2: nested dict with fuel objects
                if not bin_edges:
                    for fuel in ['electricity', 'gas']:
                        tok = tokenisers.get(fuel)
                        if tok is None:
                            continue
                        edges = getattr(tok, 'bin_edges', None)
                        if edges is None:
                            edges = getattr(tok, 'boundaries', None)
                        if edges is not None:
                            bin_edges[fuel] = np.array(edges)

                    sample_tok = tokenisers.get('electricity') or tokenisers.get('gas')
                    if sample_tok:
                        value_transform = getattr(sample_tok, 'value_transform', None)

            return {'bin_edges': bin_edges, 'value_transform': value_transform} if bin_edges else None
        except Exception as e:
            print(f"[REAL] Warning: Failed to load tokeniser from {tokeniser_path}: {e}")
            return None

    def _values_to_tokens(self, values: np.ndarray, fuel: str) -> np.ndarray:
        """Convert Wh values to token indices using bin edges."""
        if self._bin_edges is None or fuel not in self._bin_edges['bin_edges']:
            return np.array([], dtype=int)

        edges = self._bin_edges['bin_edges'][fuel]
        transform = self._bin_edges.get('value_transform')

        vals = values.copy()
        if transform == 'log1p':
            vals = np.log1p(np.maximum(vals, 0))

        tokens = np.digitize(vals, edges) - 1
        tokens = np.clip(tokens, 0, len(edges) - 1)
        return tokens

    def _compute_household_diagnostics(
        self, df: pd.DataFrame, has_elec: bool, has_gas: bool,
        seasonal_means: Dict, monthly_profile: Dict,
        weekly_profile: Dict, weekly_values: Dict,
        token_freq: Dict, monthly_decoded: Dict,
        weekly_token: Dict, num_weeks: int,
    ) -> None:
        """
        Compute all integrated diagnostics for a single household DataFrame.

        Real data has one row per half-hour with month column. We process
        month-by-month to match the synthetic data structure.
        """
        from scipy import stats as scipy_stats

        # Determine half-hour column: real data uses Readings_from_midnight_local (1-48)
        hh_col = self.HALFHOUR_COL if self.HALFHOUR_COL in df.columns else None

        for fuel, col in [('electricity', self.ELEC_COL), ('gas', self.GAS_COL)]:
            if col not in df.columns:
                continue
            is_fuel = (fuel == 'electricity' and has_elec) or (fuel == 'gas' and has_gas)
            if not is_fuel:
                continue

            for month in df['month'].unique():
                month_df = df[df['month'] == month]
                m = int(month)
                vals = month_df[col].dropna().values.astype(float)

                # --- Seasonal household means ---
                if self.diag_config.get('seasonal_analysis', {}).get('enabled', True):
                    if len(vals) > 0:
                        seasonal_means[fuel][m].append(float(np.mean(vals)))

                # --- Monthly profile contributions (sum, count, sum_sq per half-hour) ---
                if self.diag_config.get('daily_patterns', {}).get('enabled', True) and hh_col:
                    half_hours = month_df[hh_col].values.astype(int)
                    fuel_vals = month_df[col].values.astype(float)
                    for hh, val in zip(half_hours, fuel_vals):
                        if not np.isnan(val):
                            entry = monthly_profile[fuel][m][int(hh)]
                            entry[0] += val
                            entry[1] += 1
                            entry[2] += val * val

                # --- Weekly contributions (profile + values for FuelStats) ---
                weekly_cfg = self.diag_config.get('weekly_fidelity', {})
                daily_cfg = self.diag_config.get('daily_patterns', {})
                if weekly_cfg.get('enabled', True) or daily_cfg.get('enabled', True):
                    # Use positional week splitting within each month
                    month_vals_full = month_df[col].values.astype(float)
                    half_hours_full = month_df[hh_col].values.astype(int) if hh_col else None
                    n_rows = len(month_df)

                    for week in range(num_weeks):
                        start = week * self.TIMESTEPS_PER_WEEK
                        end = min(start + self.TIMESTEPS_PER_WEEK, n_rows)
                        if start >= n_rows:
                            break

                        week_vals = month_vals_full[start:end]
                        valid_mask = ~np.isnan(week_vals)
                        valid_vals = week_vals[valid_mask]

                        if len(valid_vals) > 0:
                            weekly_values[fuel][week + 1].extend(valid_vals.tolist())

                        # Profile contributions per half-hour
                        if half_hours_full is not None:
                            week_hh = half_hours_full[start:end]
                            for hh, val in zip(week_hh[valid_mask], valid_vals):
                                entry = weekly_profile[fuel][week + 1][int(hh)]
                                entry[0] += val
                                entry[1] += 1
                                entry[2] += val * val

                # --- Token analysis ---
                if self.diag_config.get('token_analysis', {}).get('enabled', True) and self._bin_edges:
                    if fuel in self._bin_edges['bin_edges']:
                        n_bins = len(self._bin_edges['bin_edges'][fuel])

                        # Initialize frequency counts if not yet done
                        if token_freq[fuel] is None:
                            token_freq[fuel] = [0] * n_bins

                        # Token frequency (overall)
                        if len(vals) > 0:
                            tokens = self._values_to_tokens(vals, fuel)
                            for t in tokens:
                                token_freq[fuel][int(t)] += 1

                        # Monthly decoded stats
                        if len(vals) > 0:
                            n = len(vals)
                            arr = np.array(vals)
                            if m not in monthly_decoded[fuel]:
                                monthly_decoded[fuel][m] = {
                                    'mean_sum': 0.0, 'median_vals': [],
                                    'std_vals': [], 'n': 0,
                                    'zero_count': 0, 'total_count': 0,
                                }
                            entry = monthly_decoded[fuel][m]
                            entry['mean_sum'] += float(np.sum(arr))
                            entry['median_vals'].extend(arr.tolist())
                            entry['n'] += n
                            entry['zero_count'] += int(np.sum(arr == 0))
                            entry['total_count'] += n

                        # Weekly token frequency
                        if weekly_cfg.get('enabled', True) or daily_cfg.get('enabled', True):
                            if fuel not in weekly_token or not weekly_token[fuel]:
                                weekly_token[fuel] = {
                                    w + 1: [0] * n_bins for w in range(num_weeks)
                                }

                            month_vals_full = month_df[col].values.astype(float)
                            n_rows = len(month_df)
                            for week in range(num_weeks):
                                start = week * self.TIMESTEPS_PER_WEEK
                                end = min(start + self.TIMESTEPS_PER_WEEK, n_rows)
                                if start >= n_rows:
                                    break
                                week_vals = month_vals_full[start:end]
                                valid = week_vals[~np.isnan(week_vals)]
                                if len(valid) > 0:
                                    tokens = self._values_to_tokens(valid, fuel)
                                    for t in tokens:
                                        weekly_token[fuel][week + 1][int(t)] += 1

    def _compute_real_household_stats(self, df: pd.DataFrame, puprn: str,
                                       stats_list: List[Dict]) -> None:
        """
        Compute full summary statistics per household-month.

        Uses MonthStatisticsCalculator to compute comprehensive feature set
        matching the synthetic data format.
        """
        calculator = MonthStatisticsCalculator(self.config.get('summary_statistics', {}))

        # Ensure we have half_hour column for timing stats
        # Real data uses Readings_from_midnight_local (1-48), convert to half_hour (0-47)
        df_work = df.copy()
        if self.HALFHOUR_COL in df_work.columns and 'half_hour' not in df_work.columns:
            df_work['half_hour'] = df_work[self.HALFHOUR_COL] - 1

        # Extract temporal columns from DatetimeIndex if not present
        if isinstance(df_work.index, pd.DatetimeIndex):
            if 'month' not in df_work.columns:
                df_work['month'] = df_work.index.month
            if 'dow' not in df_work.columns:
                df_work['dow'] = df_work.index.dayofweek
        elif 'dayofweek' in df_work.columns and 'dow' not in df_work.columns:
            df_work['dow'] = df_work['dayofweek']

        # Process by month if available
        if 'month' not in df_work.columns:
            months = [0]  # Single pseudo-month for full year
            month_dfs = {0: df_work}
        else:
            months = sorted(df_work['month'].unique())
            month_dfs = {int(m): df_work[df_work['month'] == m] for m in months}

        for month in months:
            month_df = month_dfs[month]

            # Electricity stats
            if self.ELEC_COL in month_df.columns:
                elec_stats = calculator.calculate_household_month_stats(
                    df=month_df,
                    household_col=self.ELEC_COL,
                    month=int(month),
                    household_id=puprn,
                    fuel_type='electricity'
                )
                stats_list.append(elec_stats)

            # Gas stats
            if self.GAS_COL in month_df.columns:
                gas_stats = calculator.calculate_household_month_stats(
                    df=month_df,
                    household_col=self.GAS_COL,
                    month=int(month),
                    household_id=puprn,
                    fuel_type='gas'
                )
                stats_list.append(gas_stats)

    def _aggregate(self, batch_results: List[BatchResult], split: str) -> RealDataStats:
        """Aggregate batch results into final real data statistics."""
        if not batch_results:
            return RealDataStats(
                n_households=0,
                split=split,
                fuel_stats={},
                acf={},
                volatility_stats={},
                profiles={},
                conditional={},
                household_month_stats=[]
            )

        n_households = sum(br.n_households for br in batch_results)

        # Aggregate fuel stats
        fuel_stats = {}
        for fuel in ['electricity', 'gas']:
            stats_list = [br.fuel_stats[fuel] for br in batch_results
                         if fuel in br.fuel_stats]
            if stats_list:
                fuel_stats[fuel] = FuelStats.combine(stats_list)

        # Aggregate ACF (average across all values)
        acf = {}
        for fuel in ['electricity', 'gas']:
            acf[fuel] = {}
            for lag in self.acf_lags:
                all_values = []
                for br in batch_results:
                    if fuel in br.acf_values and lag in br.acf_values[fuel]:
                        all_values.extend(br.acf_values[fuel][lag])
                acf[fuel][lag] = float(np.mean(all_values)) if all_values else np.nan

        # Aggregate volatility stats + merge reservoir samples
        volatility_stats = {}
        agg_volatility_samples = {}
        _rng_agg = np.random.default_rng(42)
        _max_samples = 10000
        for fuel in ['electricity', 'gas']:
            stats_list = [br.volatility_stats[fuel] for br in batch_results
                         if fuel in br.volatility_stats]
            if stats_list:
                volatility_stats[fuel] = FuelStats.combine(stats_list)
            all_s = []
            for br in batch_results:
                all_s.extend(br.volatility_samples.get(fuel, []))
            if len(all_s) > _max_samples:
                _idx = _rng_agg.choice(len(all_s), size=_max_samples, replace=False)
                all_s = [all_s[i] for i in _idx]
            if all_s:
                agg_volatility_samples[fuel] = all_s

        # Aggregate profiles
        profiles = {}
        for fuel in ['electricity', 'gas']:
            profiles[fuel] = {}
            for ptype in ['all', 'weekday', 'weekend']:
                combined_sum = defaultdict(float)
                combined_count = defaultdict(int)

                for br in batch_results:
                    if fuel in br.profile_contributions:
                        contribs = br.profile_contributions[fuel].get(ptype, {})
                        for hh, (s, c) in contribs.items():
                            combined_sum[hh] += s
                            combined_count[hh] += c

                # Compute final profile
                profile = []
                for hh in range(1, 49):
                    if combined_count[hh] > 0:
                        profile.append(combined_sum[hh] / combined_count[hh])
                    else:
                        profile.append(np.nan)

                profiles[fuel][ptype] = profile

        # Aggregate electricity median profiles (weighted mean of batch medians)
        profile_medians_agg = {'electricity': {}}
        for ptype in ['all', 'weekday', 'weekend']:
            combined_median_sum = defaultdict(float)
            combined_median_count = defaultdict(int)
            for br in batch_results:
                elec_meds = br.profile_medians.get('electricity', {}).get(ptype, {})
                for hh, (med, n) in elec_meds.items():
                    combined_median_sum[hh] += med * n
                    combined_median_count[hh] += n
            median_profile = []
            for hh in range(1, 49):
                if combined_median_count[hh] > 0:
                    median_profile.append(combined_median_sum[hh] / combined_median_count[hh])
                else:
                    median_profile.append(np.nan)
            profile_medians_agg['electricity'][ptype] = median_profile

        # Aggregate household_month_stats for distance analysis
        household_month_stats = []
        for br in batch_results:
            household_month_stats.extend(br.household_month_stats)

        # Aggregate missingness stats
        missingness_stats = {}
        for fuel in ['electricity', 'gas']:
            stats_list = [br.missingness_stats[fuel] for br in batch_results
                         if fuel in br.missingness_stats]
            if stats_list:
                missingness_stats[fuel] = MissingnessStats.combine(stats_list)

        # Aggregate pairwise correlations across batches
        corr_lists: Dict[str, list] = defaultdict(list)
        for br in batch_results:
            for key, val in br.correlations.items():
                corr_lists[key].append(val)
        correlations = {key: float(np.mean(vals)) for key, vals in corr_lists.items() if vals}

        # Combine value samples (simple concatenation, then subsample if needed)
        value_samples = {'electricity': [], 'gas': []}
        max_samples = 10000
        rng = np.random.default_rng(42)

        for fuel in ['electricity', 'gas']:
            all_samples = []
            for br in batch_results:
                if fuel in br.value_samples:
                    all_samples.extend(br.value_samples[fuel])

            if len(all_samples) > max_samples:
                # Subsample
                indices = rng.choice(len(all_samples), size=max_samples, replace=False)
                value_samples[fuel] = [all_samples[i] for i in indices]
            else:
                value_samples[fuel] = all_samples

        # Aggregate PTG monthly sums across batches
        ptg_monthly: Dict[int, Dict[str, float]] = {}
        for m in range(1, 13):
            ptg_monthly[m] = {'elec_sum': 0.0, 'elec_count': 0, 'gas_sum': 0.0, 'gas_count': 0,
                              'temp_sum': 0.0, 'temp_count': 0}
        for br in batch_results:
            for m, entry in br.ptg_monthly.items():
                for key in ['elec_sum', 'elec_count', 'gas_sum', 'gas_count', 'temp_sum', 'temp_count']:
                    ptg_monthly[m][key] += entry.get(key, 0)

        # Aggregate conditional values into FuelStats
        # Structure: {cond_var: {cond_value: {fuel: FuelStats}}}
        conditional = {}
        cond_vars = self.config.get('conditioning_vars',
                                     ['building_type', 'num_rooms', 'region', 'age_built', 'num_occs'])

        for cond_var in cond_vars:
            # Collect per-batch FuelStats for each conditioning group, then
            # combine across batches using FuelStats.combine(). No raw values
            # are merged — only summary statistics (sum, sum_squared, count,
            # min, max, quantiles).
            merged_stats = defaultdict(lambda: defaultdict(list))

            for br in batch_results:
                if cond_var in br.conditional_values:
                    for cond_value, fuel_stats_dict in br.conditional_values[cond_var].items():
                        for fuel, stats in fuel_stats_dict.items():
                            merged_stats[cond_value][fuel].append(stats)

            if not merged_stats:
                continue

            # Combine per-batch FuelStats into single FuelStats per group
            cond_results = {}
            for cond_value, fuel_stats_lists in merged_stats.items():
                cond_results[cond_value] = {}
                for fuel, stats_list in fuel_stats_lists.items():
                    if stats_list:
                        cond_results[cond_value][fuel] = FuelStats.combine(stats_list)

            if cond_results:
                conditional[cond_var] = cond_results

        # --- Aggregate integrated diagnostics ---
        agg_seasonal = {}
        agg_monthly_profile = {}
        agg_weekly_profile = {}
        agg_weekly_fuel = {}
        agg_token_freq = {}
        agg_monthly_decoded = {}
        agg_weekly_token = {}

        if self.diag_enabled:
            # Seasonal household means: concatenate lists per fuel/month
            for fuel in ['electricity', 'gas']:
                fuel_data: Dict[int, List[float]] = defaultdict(list)
                for br in batch_results:
                    for m, vals in br.seasonal_household_means.get(fuel, {}).items():
                        fuel_data[m].extend(vals)
                if fuel_data:
                    agg_seasonal[fuel] = dict(fuel_data)

            # Monthly profile contributions: sum (sum, count, sum_sq)
            for fuel in ['electricity', 'gas']:
                fuel_data_mp: Dict[int, Dict[int, List]] = defaultdict(lambda: defaultdict(lambda: [0.0, 0, 0.0]))
                for br in batch_results:
                    for m, hh_dict in br.monthly_profile_contributions.get(fuel, {}).items():
                        for hh, (s, c, sq) in hh_dict.items():
                            entry = fuel_data_mp[m][hh]
                            entry[0] += s
                            entry[1] += c
                            entry[2] += sq
                if fuel_data_mp:
                    agg_monthly_profile[fuel] = {
                        m: {hh: (e[0], e[1], e[2]) for hh, e in hh_dict.items()}
                        for m, hh_dict in fuel_data_mp.items()
                    }

            # Weekly profile contributions: sum (sum, count, sum_sq)
            for fuel in ['electricity', 'gas']:
                fuel_data_wp: Dict[int, Dict[int, List]] = defaultdict(lambda: defaultdict(lambda: [0.0, 0, 0.0]))
                for br in batch_results:
                    for week, hh_dict in br.weekly_profile_contributions.get(fuel, {}).items():
                        for hh, (s, c, sq) in hh_dict.items():
                            entry = fuel_data_wp[week][hh]
                            entry[0] += s
                            entry[1] += c
                            entry[2] += sq
                if fuel_data_wp:
                    agg_weekly_profile[fuel] = {
                        w: {hh: (e[0], e[1], e[2]) for hh, e in hh_dict.items()}
                        for w, hh_dict in fuel_data_wp.items()
                    }

            # Weekly FuelStats: combine
            for fuel in ['electricity', 'gas']:
                fuel_data_wf: Dict[int, List] = defaultdict(list)
                for br in batch_results:
                    for week, fs in br.weekly_fuel_stats.get(fuel, {}).items():
                        fuel_data_wf[week].append(fs)
                if fuel_data_wf:
                    agg_weekly_fuel[fuel] = {
                        w: FuelStats.combine(fs_list)
                        for w, fs_list in fuel_data_wf.items() if fs_list
                    }

            # Token frequency: element-wise sum
            for fuel in ['electricity', 'gas']:
                combined = None
                for br in batch_results:
                    counts = br.token_frequency.get(fuel)
                    if counts:
                        if combined is None:
                            combined = list(counts)
                        else:
                            for i in range(len(combined)):
                                combined[i] += counts[i]
                if combined:
                    agg_token_freq[fuel] = combined

            # Monthly decoded stats: combine finalized per-batch stats
            # Each batch has {month: {mean, median, std, skew, zero_rate, n}}
            # We weight-average mean, and keep the largest-N batch's median/std/skew
            # as approximation (exact combination not possible without raw values)
            for fuel in ['electricity', 'gas']:
                merged_months: Dict[int, Dict] = defaultdict(lambda: {
                    'mean_sum': 0.0, 'n': 0,
                    'zero_rate_sum': 0.0, 'zero_rate_n': 0,
                    'median_best': 0.0, 'std_best': 0.0, 'skew_best': 0.0,
                    'best_n': 0,
                })
                for br in batch_results:
                    for m, stats in br.monthly_decoded_stats.get(fuel, {}).items():
                        entry = merged_months[m]
                        n = stats.get('n', 0)
                        if n > 0:
                            entry['mean_sum'] += stats.get('mean', 0) * n
                            entry['n'] += n
                            entry['zero_rate_sum'] += stats.get('zero_rate', 0) * n
                            entry['zero_rate_n'] += n
                            # Keep stats from the batch with most data
                            if n > entry['best_n']:
                                entry['median_best'] = stats.get('median', 0)
                                entry['std_best'] = stats.get('std', 0)
                                entry['skew_best'] = stats.get('skew', 0)
                                entry['best_n'] = n

                if merged_months:
                    fuel_monthly_final = {}
                    for m, entry in merged_months.items():
                        if entry['n'] > 0:
                            fuel_monthly_final[m] = {
                                'mean': entry['mean_sum'] / entry['n'],
                                'median': entry['median_best'],
                                'std': entry['std_best'],
                                'skew': entry['skew_best'],
                                'zero_rate': entry['zero_rate_sum'] / entry['zero_rate_n'] if entry['zero_rate_n'] > 0 else 0.0,
                                'n': entry['n'],
                            }
                    if fuel_monthly_final:
                        agg_monthly_decoded[fuel] = fuel_monthly_final

            # Weekly token frequency: element-wise sum
            for fuel in ['electricity', 'gas']:
                combined_weeks: Dict[int, List[int]] = {}
                for br in batch_results:
                    for week, counts in br.weekly_token_frequency.get(fuel, {}).items():
                        if week not in combined_weeks:
                            combined_weeks[week] = list(counts)
                        else:
                            for i in range(len(combined_weeks[week])):
                                combined_weeks[week][i] += counts[i]
                if combined_weeks:
                    agg_weekly_token[fuel] = combined_weeks

        # Aggregate conditioning_n_households across batches by summing counts
        agg_conditioning_n_households: Dict[str, Dict[Any, int]] = defaultdict(lambda: defaultdict(int))
        for br in batch_results:
            for var, cond_vals in br.conditioning_n_households.items():
                for cond_value, n in cond_vals.items():
                    agg_conditioning_n_households[var][cond_value] += n
        conditioning_n_households = {
            var: dict(cond_vals)
            for var, cond_vals in agg_conditioning_n_households.items()
        }

        return RealDataStats(
            n_households=n_households,
            split=split,
            fuel_stats=fuel_stats,
            acf=acf,
            volatility_stats=volatility_stats,
            volatility_samples=agg_volatility_samples,
            profiles=profiles,
            profile_medians=profile_medians_agg,
            conditional=conditional,
            conditioning_n_households=conditioning_n_households,
            household_month_stats=household_month_stats,
            missingness_stats=missingness_stats,
            correlations=correlations,
            value_samples=value_samples,
            ptg_monthly=ptg_monthly,
            seasonal_household_means=agg_seasonal,
            monthly_profile_contributions=agg_monthly_profile,
            weekly_profile_contributions=agg_weekly_profile,
            weekly_fuel_stats=agg_weekly_fuel,
            token_frequency=agg_token_freq,
            monthly_decoded_stats=agg_monthly_decoded,
            weekly_token_frequency=agg_weekly_token,
        )
