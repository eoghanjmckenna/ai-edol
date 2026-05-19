"""
BatchProcessor for MapReduce evaluation pipeline.

Processes a single synthetic batch (12 month-chunk files) and produces
a BatchResult containing all intermediate computations for aggregation.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import numpy as np
import pandas as pd

from .data_structures import FuelStats, BatchResult, MissingnessStats
from .month_statistics import MonthStatisticsCalculator, compute_all_household_month_stats


class BatchProcessor:
    """
    Process a single batch of month-chunk files.

    Designed to be stateless and parallelizable - each instance
    processes one batch independently.
    """

    # Column name mappings
    ELEC_SUFFIX = '_elec_net_Wh'
    GAS_SUFFIX = '_gas_Wh'

    # Timesteps per week (7 days × 48 half-hours)
    TIMESTEPS_PER_WEEK = 336

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: Evaluation configuration
                - acf_lags: List[int] - lags for ACF computation
                - summary_stat_features: List[str] - features to compute
                - diagnostics_integration: Dict - integrated diagnostics config
                - tokeniser_file: str - path to tokenisers.pkl (for token analysis)
        """
        self.config = config or {}
        self.acf_lags = self.config.get('acf_lags', [1, 2, 48, 96, 336, 672, 1008, 1344])

        # Integrated diagnostics config
        self.diag_config = self.config.get('diagnostics_integration', {})
        self.diag_enabled = self.diag_config.get('enabled', False)

        # Load tokeniser bin edges if available (for token analysis)
        self._bin_edges = None
        if self.diag_enabled and self.diag_config.get('token_analysis', {}).get('enabled', True):
            tokeniser_path = self.config.get('tokeniser_file')
            if tokeniser_path:
                self._bin_edges = self._load_tokeniser_bin_edges(tokeniser_path)

    def process_batch(self, batch_id: str, month_files: List[Path]) -> BatchResult:
        """
        Process all month-chunk files for a single batch.

        Args:
            batch_id: Base batch identifier (e.g., 'batch_0001')
            month_files: List of paths to month-chunk files

        Returns:
            BatchResult with all intermediate computations
        """
        # 1. Load all month data
        month_data = self._load_month_chunks(month_files)

        # 1b. Apply privacy pre-filter if configured
        prefilter_stats = None
        exclusion_lookup = self.config.get('_prefilter_exclusion_lookup')
        if exclusion_lookup and batch_id in exclusion_lookup:
            month_data, prefilter_stats = self._apply_privacy_prefilter(
                month_data, batch_id, exclusion_lookup[batch_id])

        if not month_data:
            return BatchResult(
                batch_id=batch_id,
                n_households=0,
                n_months_processed=0,
                conditioning={},
                fuel_stats={},
                variance_metrics={'severity': 'unknown', 'flags': ['No data loaded']},
                acf_values={},
                volatility_stats={},
                profile_contributions={},
                household_month_stats=[]
            )

        # 2. Extract conditioning (constant for this batch)
        conditioning = self._extract_conditioning(month_data)

        # 3. Count households
        n_households = self._count_households(month_data)

        # 4. Compute fuel statistics (for marginal + conditional distribution)
        fuel_stats = self._compute_fuel_stats(month_data)

        # 5. Compute within-batch variance (per month, then aggregate)
        variance_metrics = self._compute_variance_metrics(month_data)

        # 6. Compute ACF values (per household-month)
        acf_values = self._compute_acf(month_data)

        # 7. Compute volatility statistics (first differences)
        volatility_stats, volatility_samples = self._compute_volatility_stats(month_data)

        # 8. Compute daily profile contributions (+ electricity medians)
        profile_contribs, profile_meds = self._compute_profile_contributions(month_data)

        # 9. Compute summary statistics per household-month (for distance analysis)
        household_month_stats = self._compute_household_month_stats(month_data, batch_id)

        # Attach batch-level conditioning variables to each household_month_stats record
        # so the reduce phase can group by conditioning variable
        for stat_record in household_month_stats:
            stat_record.update(conditioning)

        # 10. Compute missingness statistics per fuel
        missingness_stats = self._compute_missingness_stats(month_data)

        # 11. Compute pairwise correlations (elec-gas, temp-elec, temp-gas)
        correlations = self._compute_correlations(month_data)

        # 12. Collect value samples for distribution comparison
        value_samples = self._compute_value_samples(month_data)

        # 13. Accumulate PTG monthly sums (for Power-Temperature Gradient plot)
        ptg_monthly = self._compute_ptg_monthly(month_data)

        # --- Integrated diagnostics (gated by config) ---
        seasonal_household_means = {}
        monthly_profile_contribs = {}
        weekly_profile_contribs = {}
        weekly_fuel_stats = {}
        token_freq = {}
        monthly_decoded_stats = {}
        weekly_token_freq = {}

        if self.diag_enabled:
            if self.diag_config.get('seasonal_analysis', {}).get('enabled', True):
                seasonal_household_means = self._compute_seasonal_household_means(month_data)

            if self.diag_config.get('daily_patterns', {}).get('enabled', True):
                monthly_profile_contribs = self._compute_monthly_profile_contributions(month_data)

            weekly_cfg = self.diag_config.get('weekly_fidelity', {})
            daily_cfg = self.diag_config.get('daily_patterns', {})
            if weekly_cfg.get('enabled', True) or daily_cfg.get('enabled', True):
                num_weeks = weekly_cfg.get('num_weeks', 4)
                weekly_profile_contribs, weekly_fuel_stats = self._compute_weekly_contributions(
                    month_data, num_weeks=num_weeks)

            if self.diag_config.get('token_analysis', {}).get('enabled', True) and self._bin_edges:
                token_freq, monthly_decoded_stats = self._compute_token_analysis(month_data)
                if weekly_cfg.get('enabled', True) or daily_cfg.get('enabled', True):
                    weekly_token_freq = self._compute_weekly_token_frequency(
                        month_data, num_weeks=num_weeks)

        return BatchResult(
            batch_id=batch_id,
            n_households=n_households,
            n_months_processed=len(month_data),
            conditioning=conditioning,
            fuel_stats=fuel_stats,
            variance_metrics=variance_metrics,
            acf_values=acf_values,
            volatility_stats=volatility_stats,
            volatility_samples=volatility_samples,
            profile_contributions=profile_contribs,
            profile_medians=profile_meds,
            household_month_stats=household_month_stats,
            missingness_stats=missingness_stats,
            correlations=correlations,
            value_samples=value_samples,
            ptg_monthly=ptg_monthly,
            seasonal_household_means=seasonal_household_means,
            monthly_profile_contributions=monthly_profile_contribs,
            weekly_profile_contributions=weekly_profile_contribs,
            weekly_fuel_stats=weekly_fuel_stats,
            token_frequency=token_freq,
            monthly_decoded_stats=monthly_decoded_stats,
            weekly_token_frequency=weekly_token_freq,
            prefilter_stats=prefilter_stats,
        )

    def _load_month_chunks(self, month_files: List[Path]) -> Dict[int, pd.DataFrame]:
        """Load month files into dict keyed by month number."""
        month_data = {}

        # Sort by month number
        def get_month_num(f: Path) -> int:
            name = f.stem
            if '_month_' in name:
                return int(name.split('_month_')[1])
            return 0

        for f in sorted(month_files, key=get_month_num):
            try:
                month_num = get_month_num(f)
                df = pd.read_pickle(f)
                month_data[month_num] = df
            except Exception as e:
                print(f"[BatchProcessor] Warning: Failed to load {f}: {e}")

        return month_data

    def _apply_privacy_prefilter(
        self,
        month_data: Dict[int, pd.DataFrame],
        batch_id: str,
        exclusions: set,
    ) -> Tuple[Dict[int, pd.DataFrame], Dict[str, Any]]:
        """
        Drop flagged household columns from month DataFrames.

        Args:
            month_data: Dict[month_num -> DataFrame] loaded from month-chunk files
            batch_id: Batch identifier (e.g. 'batch_0001')
            exclusions: Set of (hh_suffix, month) tuples to exclude,
                e.g. {('hh003', 1), ('hh007', 3)}

        Returns:
            Tuple of (filtered_month_data, prefilter_stats dict)
        """
        n_excluded = 0
        excluded_per_hh = defaultdict(int)  # hh_suffix -> count of months excluded
        all_hh_suffixes = set()

        for month, df in month_data.items():
            # Identify all household suffixes in this month
            elec_cols = [c for c in df.columns if c.endswith(self.ELEC_SUFFIX)]
            month_hh_suffixes = set()
            for col in elec_cols:
                hh_suffix = col[: -len(self.ELEC_SUFFIX)]
                month_hh_suffixes.add(hh_suffix)
            all_hh_suffixes.update(month_hh_suffixes)

            # Find which households to drop in this month
            cols_to_drop = []
            for hh_suffix in month_hh_suffixes:
                if (hh_suffix, month) in exclusions:
                    cols_to_drop.append(f'{hh_suffix}{self.ELEC_SUFFIX}')
                    cols_to_drop.append(f'{hh_suffix}{self.GAS_SUFFIX}')
                    excluded_per_hh[hh_suffix] += 1
                    n_excluded += 1

            if cols_to_drop:
                existing_cols = [c for c in cols_to_drop if c in df.columns]
                if existing_cols:
                    month_data[month] = df.drop(columns=existing_cols)

        # Remove months that have no household columns left
        empty_months = []
        for month, df in month_data.items():
            if not any(c.endswith(self.ELEC_SUFFIX) for c in df.columns):
                empty_months.append(month)
        for month in empty_months:
            del month_data[month]

        n_total_hh = len(all_hh_suffixes)
        n_fully_excluded = sum(
            1 for hh, cnt in excluded_per_hh.items()
            if cnt >= len(month_data) + len(empty_months)
        )

        stats = {
            'n_household_months_excluded': n_excluded,
            'n_households_fully_excluded': n_fully_excluded,
            'n_households_partially_excluded': len(excluded_per_hh) - n_fully_excluded,
            'batch_empty_after_filter': len(month_data) == 0,
        }

        return month_data, stats

    def _extract_conditioning(self, month_data: Dict[int, pd.DataFrame]) -> Dict[str, Any]:
        """Extract conditioning variables (constant across batch)."""
        df = next(iter(month_data.values()))

        conditioning = {}
        cond_vars = ['building_type', 'num_rooms', 'region', 'age_built', 'num_occs']

        for var in cond_vars:
            if var in df.columns:
                val = df[var].iloc[0]
                # Handle numpy types
                if hasattr(val, 'item'):
                    val = val.item()
                conditioning[var] = val

        return conditioning

    def _count_households(self, month_data: Dict[int, pd.DataFrame]) -> int:
        """Count number of households in batch."""
        df = next(iter(month_data.values()))
        elec_cols = [c for c in df.columns if c.endswith(self.ELEC_SUFFIX)]
        return len(elec_cols)

    def _get_household_columns(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Get electricity and gas column names."""
        elec_cols = sorted([c for c in df.columns if c.endswith(self.ELEC_SUFFIX)])
        gas_cols = sorted([c for c in df.columns if c.endswith(self.GAS_SUFFIX)])
        return elec_cols, gas_cols

    def _compute_fuel_stats(self, month_data: Dict[int, pd.DataFrame]) -> Dict[str, FuelStats]:
        """
        Compute aggregatable statistics for each fuel.

        Collects all values across households and months, computes FuelStats.
        """
        stats = {}

        for fuel, suffix in [('electricity', self.ELEC_SUFFIX), ('gas', self.GAS_SUFFIX)]:
            all_values = []
            monthly_values = defaultdict(list)
            n_days = 0

            for month, df in month_data.items():
                fuel_cols = [c for c in df.columns if c.endswith(suffix)]

                for col in fuel_cols:
                    vals = df[col].dropna().values.astype(float)
                    all_values.extend(vals)
                    monthly_values[month].extend(vals)

                # Approximate days from rows (48 half-hours per day)
                n_days += len(df) // 48

            if all_values:
                # Convert monthly_values to numpy arrays
                monthly_arrays = {m: np.array(v) for m, v in monthly_values.items()}
                stats[fuel] = FuelStats.from_values(
                    np.array(all_values),
                    n_days=n_days,
                    monthly_values=monthly_arrays
                )
            else:
                stats[fuel] = FuelStats(
                    n_values=0, n_days=0, sum_values=0.0, sum_squared=0.0,
                    min_value=np.nan, max_value=np.nan, quantiles={}, monthly_stats={}
                )

        return stats

    def _compute_variance_metrics(self, month_data: Dict[int, pd.DataFrame]) -> Dict[str, Any]:
        """
        Compute within-batch variance per month, then aggregate.

        Uses variance computation functions from variance_analysis module.
        """
        from .variance_analysis import (
            compute_within_batch_correlations,
            compute_within_batch_distances,
            compute_within_batch_peak_diversity,
            flag_potential_overfitting,
        )

        monthly_results = []

        for month, df in month_data.items():
            try:
                corr_metrics = compute_within_batch_correlations(df)
                dist_metrics = compute_within_batch_distances(df)
                peak_metrics = compute_within_batch_peak_diversity(df)

                monthly_results.append({
                    'month': month,
                    **corr_metrics,
                    **dist_metrics,
                    **peak_metrics
                })
            except Exception as e:
                print(f"[BatchProcessor] Warning: Variance computation failed for month {month}: {e}")

        if not monthly_results:
            return {'severity': 'unknown', 'flags': ['No months processed']}

        # Aggregate across months
        aggregated = {}

        # Correlation metrics - use mean/max across months
        if 'mean_pairwise_correlation' in monthly_results[0]:
            values = [m.get('mean_pairwise_correlation', np.nan) for m in monthly_results]
            aggregated['mean_pairwise_correlation'] = np.nanmean(values)

        if 'max_pairwise_correlation' in monthly_results[0]:
            values = [m.get('max_pairwise_correlation', np.nan) for m in monthly_results]
            aggregated['max_pairwise_correlation'] = np.nanmax(values)

        # Distance metrics
        if 'mean_pairwise_distance' in monthly_results[0]:
            values = [m.get('mean_pairwise_distance', np.nan) for m in monthly_results]
            aggregated['mean_pairwise_distance'] = np.nanmean(values)

        if 'min_pairwise_distance' in monthly_results[0]:
            values = [m.get('min_pairwise_distance', np.nan) for m in monthly_results]
            aggregated['min_pairwise_distance'] = np.nanmin(values)

        # Peak diversity
        if 'peak_hour_diversity' in monthly_results[0]:
            values = [m.get('peak_hour_diversity', np.nan) for m in monthly_results]
            aggregated['peak_hour_diversity'] = np.nanmean(values)

        # Flag potential overfitting
        result = flag_potential_overfitting(aggregated)
        result['n_months_analyzed'] = len(monthly_results)

        return result

    def _compute_acf(self, month_data: Dict[int, pd.DataFrame]) -> Dict[str, Dict[int, List[float]]]:
        """
        Compute ACF within each month (no cross-month correlation).

        Returns ACF values per household-month for averaging in reduce phase.
        """
        acf_values = {'electricity': defaultdict(list), 'gas': defaultdict(list)}

        for month, df in month_data.items():
            elec_cols, gas_cols = self._get_household_columns(df)

            # Electricity ACF
            for col in elec_cols:
                values = df[col].dropna().values.astype(float)
                if len(values) > max(self.acf_lags) + 1:
                    for lag in self.acf_lags:
                        acf = self._compute_single_acf(values, lag)
                        if not np.isnan(acf):
                            acf_values['electricity'][lag].append(acf)

            # Gas ACF
            for col in gas_cols:
                values = df[col].dropna().values.astype(float)
                if len(values) > max(self.acf_lags) + 1:
                    for lag in self.acf_lags:
                        acf = self._compute_single_acf(values, lag)
                        if not np.isnan(acf):
                            acf_values['gas'][lag].append(acf)

        # Convert defaultdicts to regular dicts
        return {
            'electricity': dict(acf_values['electricity']),
            'gas': dict(acf_values['gas'])
        }

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

    def _compute_volatility_stats(self, month_data: Dict[int, pd.DataFrame],
                                   sample_size: int = 5000,
                                   ) -> Tuple[Dict[str, 'FuelStats'], Dict[str, List[float]]]:
        """
        Compute statistics on first differences (volatility).

        Computes first differences within each month (no cross-month diffs).

        Returns:
            Tuple of (FuelStats per fuel, reservoir samples of Δx per fuel)
        """
        stats = {}
        samples: Dict[str, List[float]] = {}
        rng = np.random.default_rng(42)

        for fuel, suffix in [('electricity', self.ELEC_SUFFIX), ('gas', self.GAS_SUFFIX)]:
            all_diffs = []
            monthly_diffs = defaultdict(list)

            for month, df in month_data.items():
                fuel_cols = [c for c in df.columns if c.endswith(suffix)]

                for col in fuel_cols:
                    values = df[col].dropna().values.astype(float)
                    if len(values) > 1:
                        diffs = np.diff(values)
                        all_diffs.extend(diffs)
                        monthly_diffs[month].extend(diffs)

            if all_diffs:
                arr = np.array(all_diffs)
                monthly_arrays = {m: np.array(v) for m, v in monthly_diffs.items()}
                stats[fuel] = FuelStats.from_values(arr, monthly_values=monthly_arrays)
                # Reservoir sample for histogram comparison in reduce phase
                if len(arr) <= sample_size:
                    samples[fuel] = arr.tolist()
                else:
                    idx = rng.choice(len(arr), size=sample_size, replace=False)
                    samples[fuel] = arr[idx].tolist()
            else:
                stats[fuel] = FuelStats(
                    n_values=0, n_days=0, sum_values=0.0, sum_squared=0.0,
                    min_value=np.nan, max_value=np.nan, quantiles={}, monthly_stats={}
                )
                samples[fuel] = []

        return stats, samples

    def _compute_profile_contributions(self, month_data: Dict[int, pd.DataFrame]
                                        ) -> Tuple[Dict[str, Dict[str, Dict[int, Tuple[float, int]]]],
                                                   Dict[str, Dict[str, Dict[int, Tuple[float, int]]]]]:
        """
        Compute sum and count per half-hour for profile averaging.

        Returns:
            Tuple of (contributions, medians) where medians are computed for
            electricity only (per half-hour median + count at batch level).
        """
        contributions = {
            'electricity': {'all': defaultdict(lambda: [0.0, 0]),
                           'weekday': defaultdict(lambda: [0.0, 0]),
                           'weekend': defaultdict(lambda: [0.0, 0])},
            'gas': {'all': defaultdict(lambda: [0.0, 0]),
                   'weekday': defaultdict(lambda: [0.0, 0]),
                   'weekend': defaultdict(lambda: [0.0, 0])}
        }

        # Collect raw values per half-hour for electricity median computation
        elec_values = {
            'all': defaultdict(list),
            'weekday': defaultdict(list),
            'weekend': defaultdict(list),
        }

        for month, df in month_data.items():
            # Check for required columns
            hh_col = 'half_hour' if 'half_hour' in df.columns else None
            dow_col = 'dow' if 'dow' in df.columns else None

            if hh_col is None:
                continue

            elec_cols, gas_cols = self._get_household_columns(df)

            for fuel, fuel_cols in [('electricity', elec_cols), ('gas', gas_cols)]:
                for col in fuel_cols:
                    for idx, row in df.iterrows():
                        hh = int(row[hh_col])
                        val = row[col]

                        if pd.isna(val):
                            continue

                        # All profiles
                        contributions[fuel]['all'][hh][0] += val
                        contributions[fuel]['all'][hh][1] += 1

                        # Collect electricity values for median
                        if fuel == 'electricity':
                            elec_values['all'][hh].append(val)

                        # Weekday/weekend split
                        if dow_col and dow_col in df.columns:
                            dow = row[dow_col]
                            if dow in [5, 6]:  # Weekend
                                contributions[fuel]['weekend'][hh][0] += val
                                contributions[fuel]['weekend'][hh][1] += 1
                                if fuel == 'electricity':
                                    elec_values['weekend'][hh].append(val)
                            else:
                                contributions[fuel]['weekday'][hh][0] += val
                                contributions[fuel]['weekday'][hh][1] += 1
                                if fuel == 'electricity':
                                    elec_values['weekday'][hh].append(val)

        # Convert contributions to tuples
        result = {}
        for fuel in ['electricity', 'gas']:
            result[fuel] = {}
            for profile_type in ['all', 'weekday', 'weekend']:
                result[fuel][profile_type] = {
                    hh: (sc[0], sc[1])
                    for hh, sc in contributions[fuel][profile_type].items()
                }

        # Compute batch-level medians for electricity
        medians = {'electricity': {}}
        for profile_type in ['all', 'weekday', 'weekend']:
            medians['electricity'][profile_type] = {}
            for hh, vals in elec_values[profile_type].items():
                if vals:
                    medians['electricity'][profile_type][hh] = (
                        float(np.median(vals)), len(vals)
                    )

        return result, medians

    def _compute_household_month_stats(self, month_data: Dict[int, pd.DataFrame],
                                        batch_id: str) -> List[Dict[str, Any]]:
        """
        Compute full summary statistics per household-month for distance analysis.

        Uses MonthStatisticsCalculator to compute comprehensive feature set:
        - Distributional stats (mean, std, quantiles, skew, kurtosis)
        - Timing stats (peak_hour, time-of-day averages)
        - Temporal stats (first differences, rolling volatility)
        - Load stats (load_factor, weekday_weekend_ratio)
        - Data quality (missing_pct, zeros_pct)

        Returns one dict per household-month-fuel combination.
        """
        # Use the full statistics calculator
        stats_config = self.config.get('summary_statistics', {})
        return compute_all_household_month_stats(month_data, batch_id, stats_config)

    def _compute_missingness_stats(self, month_data: Dict[int, pd.DataFrame]) -> Dict[str, MissingnessStats]:
        """
        Compute missingness statistics for each fuel type.

        Tracks:
        - NaN missingness rate
        - Zero value rate
        - Structural missingness (households with all values missing)
        - Consecutive run lengths of missing/zero values
        """
        stats = {}

        for fuel, suffix in [('electricity', self.ELEC_SUFFIX), ('gas', self.GAS_SUFFIX)]:
            n_total = 0
            n_missing = 0
            n_zeros = 0
            n_households_no_data = 0
            n_households_total = 0
            missing_run_lengths: Dict[int, int] = {}
            zero_run_lengths: Dict[int, int] = {}

            for month, df in month_data.items():
                fuel_cols = [c for c in df.columns if c.endswith(suffix)]
                n_households_total += len(fuel_cols)

                for col in fuel_cols:
                    values = df[col].values
                    n_values = len(values)
                    n_total += n_values

                    # Count NaN values
                    is_nan = pd.isna(values)
                    n_nan = int(np.sum(is_nan))
                    n_missing += n_nan

                    # Count zeros (only among non-NaN values)
                    # Convert to float for numeric comparison
                    non_nan_values = values[~is_nan].astype(float)
                    n_zero = int(np.sum(non_nan_values == 0))
                    n_zeros += n_zero

                    # Structural missingness: all values NaN
                    if n_nan == n_values:
                        n_households_no_data += 1

                    # Compute consecutive run lengths for missing values
                    self._update_run_lengths(is_nan, missing_run_lengths)

                    # Compute consecutive run lengths for zeros
                    is_zero = np.zeros(len(values), dtype=bool)
                    is_zero[~is_nan] = (values[~is_nan] == 0)
                    self._update_run_lengths(is_zero, zero_run_lengths)

            stats[fuel] = MissingnessStats(
                n_total=n_total,
                n_missing=n_missing,
                n_zeros=n_zeros,
                n_households_no_data=n_households_no_data,
                n_households_total=n_households_total,
                missing_run_lengths=missing_run_lengths,
                zero_run_lengths=zero_run_lengths,
            )

        return stats

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

    def _compute_correlations(self, month_data: Dict[int, pd.DataFrame]) -> Dict[str, float]:
        """
        Compute pairwise correlations: elec-gas, temp-elec, temp-gas.

        Computes per-household, per-month correlations and returns the mean
        for each pair. Only includes households where both columns are present.
        Temperature (temp_C) is a shared column across all households.
        """
        corr_lists: Dict[str, list] = {'elec_gas': [], 'temp_elec': [], 'temp_gas': []}

        for month, df in month_data.items():
            elec_cols, gas_cols = self._get_household_columns(df)
            has_temp = 'temp_C' in df.columns

            # Match households by prefix
            for elec_col in elec_cols:
                hh_prefix = elec_col.replace(self.ELEC_SUFFIX, '')
                gas_col = hh_prefix + self.GAS_SUFFIX

                elec_vals = df[elec_col].values
                gas_vals = df[gas_col].values if gas_col in gas_cols else None
                temp_vals = df['temp_C'].values if has_temp else None

                # Elec-gas correlation
                if gas_vals is not None:
                    self._add_correlation(elec_vals, gas_vals, corr_lists['elec_gas'])

                # Temp-elec correlation
                if temp_vals is not None:
                    self._add_correlation(temp_vals, elec_vals, corr_lists['temp_elec'])

                # Temp-gas correlation
                if temp_vals is not None and gas_vals is not None:
                    self._add_correlation(temp_vals, gas_vals, corr_lists['temp_gas'])

        result = {}
        for key, values in corr_lists.items():
            if values:
                result[key] = float(np.mean(values))
        return result

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

    def _compute_value_samples(self, month_data: Dict[int, pd.DataFrame],
                               max_samples: int = 10000) -> Dict[str, List[float]]:
        """
        Collect value samples for distribution comparison using reservoir sampling.

        Samples are used for Wasserstein distance and KS tests in the reduce phase.

        Args:
            month_data: Dict of month -> DataFrame
            max_samples: Maximum samples to collect per fuel

        Returns:
            Dict mapping fuel -> list of sampled values
        """
        samples = {'electricity': [], 'gas': []}
        reservoir_count = {'electricity': 0, 'gas': 0}

        rng = np.random.default_rng(42)  # Reproducible sampling

        for fuel, suffix in [('electricity', self.ELEC_SUFFIX), ('gas', self.GAS_SUFFIX)]:
            # Stream values directly into the reservoir — never accumulate all
            # values in memory. Each value has equal probability (max_samples/N)
            # of being in the final sample.
            for month, df in month_data.items():
                fuel_cols = [c for c in df.columns if c.endswith(suffix)]
                for col in fuel_cols:
                    vals = df[col].dropna().values.astype(float)
                    for val in vals:
                        n = reservoir_count[fuel]
                        if n < max_samples:
                            samples[fuel].append(float(val))
                        else:
                            # Replace with probability max_samples / (n + 1)
                            j = rng.integers(0, n + 1)
                            if j < max_samples:
                                samples[fuel][j] = float(val)
                        reservoir_count[fuel] += 1

        return samples

    def _compute_ptg_monthly(self, month_data: Dict[int, pd.DataFrame]) -> Dict[int, Dict[str, float]]:
        """
        Accumulate per-month consumption and temperature sums for PTG analysis.

        Returns dict of {month: {elec_sum, elec_count, gas_sum, gas_count, temp_sum, temp_count}}.
        """
        ptg_monthly: Dict[int, Dict[str, float]] = {}

        for month, df in month_data.items():
            elec_cols, gas_cols = self._get_household_columns(df)
            entry = {'elec_sum': 0.0, 'elec_count': 0, 'gas_sum': 0.0, 'gas_count': 0,
                     'temp_sum': 0.0, 'temp_count': 0}

            for elec_col in elec_cols:
                vals = df[elec_col].dropna().values
                entry['elec_sum'] += vals.sum()
                entry['elec_count'] += len(vals)

            for gas_col in gas_cols:
                vals = df[gas_col].dropna().values
                entry['gas_sum'] += vals.sum()
                entry['gas_count'] += len(vals)

            if 'temp_C' in df.columns:
                temp_vals = df['temp_C'].dropna().values
                entry['temp_sum'] += temp_vals.sum()
                entry['temp_count'] += len(temp_vals)

            ptg_monthly[int(month)] = entry

        return ptg_monthly

    # --- Integrated Diagnostics: Map Phase Methods ---

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
                # Format 1: flat dict with elec_boundaries, gas_boundaries, etc.
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
            print(f"[BatchProcessor] Warning: Failed to load tokeniser from {tokeniser_path}: {e}")
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

        # np.digitize returns 1-based indices; clip to valid range
        tokens = np.digitize(vals, edges) - 1
        tokens = np.clip(tokens, 0, len(edges) - 1)
        return tokens

    def _compute_seasonal_household_means(
        self, month_data: Dict[int, pd.DataFrame]
    ) -> Dict[str, Dict[int, List[float]]]:
        """
        Compute per-household monthly means for seasonal (autoregressive collapse) analysis.

        Returns {fuel: {month: [household_mean_1, household_mean_2, ...]}}.
        The per-household granularity enables between-household standard errors
        in the reduce phase.
        """
        result: Dict[str, Dict[int, List[float]]] = {}

        for fuel, suffix in [('electricity', self.ELEC_SUFFIX), ('gas', self.GAS_SUFFIX)]:
            fuel_result: Dict[int, List[float]] = {}

            for month, df in month_data.items():
                fuel_cols = [c for c in df.columns if c.endswith(suffix)]
                means = []

                for col in fuel_cols:
                    vals = df[col].dropna().values.astype(float)
                    if len(vals) > 0:
                        means.append(float(np.mean(vals)))

                if means:
                    fuel_result[int(month)] = means

            if fuel_result:
                result[fuel] = fuel_result

        return result

    def _compute_monthly_profile_contributions(
        self, month_data: Dict[int, pd.DataFrame]
    ) -> Dict[str, Dict[int, Dict[int, Tuple[float, int, float]]]]:
        """
        Compute per-month × per-half-hour (sum, count, sum_sq) for heatmap analysis.

        Unlike _compute_profile_contributions which aggregates all months together,
        this keeps months separate for the 12×48 heatmap. Includes sum_of_squares
        for standard error computation in the reduce phase.
        """
        # {fuel: {month: {half_hour: [sum, count, sum_sq]}}}
        contribs: Dict[str, Dict[int, Dict[int, List[float]]]] = {}

        for fuel in ['electricity', 'gas']:
            contribs[fuel] = {}

        for month, df in month_data.items():
            hh_col = 'half_hour' if 'half_hour' in df.columns else None
            if hh_col is None:
                continue

            elec_cols, gas_cols = self._get_household_columns(df)
            half_hours = df[hh_col].values.astype(int)

            for fuel, fuel_cols in [('electricity', elec_cols), ('gas', gas_cols)]:
                month_contribs: Dict[int, List[float]] = defaultdict(lambda: [0.0, 0, 0.0])

                for col in fuel_cols:
                    vals = df[col].values.astype(float)
                    for i, (hh, val) in enumerate(zip(half_hours, vals)):
                        if not np.isnan(val):
                            entry = month_contribs[int(hh)]
                            entry[0] += val
                            entry[1] += 1
                            entry[2] += val * val

                if month_contribs:
                    contribs[fuel][int(month)] = {
                        hh: (e[0], e[1], e[2]) for hh, e in month_contribs.items()
                    }

        return contribs

    def _compute_weekly_contributions(
        self, month_data: Dict[int, pd.DataFrame], num_weeks: int = 4
    ) -> Tuple[Dict[str, Dict[int, Dict[int, Tuple[float, int, float]]]],
               Dict[str, Dict[int, FuelStats]]]:
        """
        Split each month into positional weeks and compute:
        1. Weekly profile contributions (sum, count, sum_sq per half-hour)
        2. Weekly FuelStats for marginal distribution comparison

        Within-month weeks: rows 0-335 = week 1, 336-671 = week 2, etc.
        Week numbering is within-month (resets each month, then combined across months).

        Returns:
            (weekly_profile_contributions, weekly_fuel_stats)
        """
        # {fuel: {week: {half_hour: [sum, count, sum_sq]}}}
        profile_contribs: Dict[str, Dict[int, Dict[int, List[float]]]] = {
            'electricity': defaultdict(lambda: defaultdict(lambda: [0.0, 0, 0.0])),
            'gas': defaultdict(lambda: defaultdict(lambda: [0.0, 0, 0.0]))
        }
        # {fuel: {week: [all_values]}}  — collect for FuelStats
        weekly_values: Dict[str, Dict[int, List[float]]] = {
            'electricity': defaultdict(list),
            'gas': defaultdict(list)
        }

        for month, df in month_data.items():
            hh_col = 'half_hour' if 'half_hour' in df.columns else None
            elec_cols, gas_cols = self._get_household_columns(df)
            n_rows = len(df)

            for fuel, fuel_cols in [('electricity', elec_cols), ('gas', gas_cols)]:
                for col in fuel_cols:
                    vals = df[col].values.astype(float)
                    half_hours = df[hh_col].values.astype(int) if hh_col else None

                    for week in range(num_weeks):
                        start = week * self.TIMESTEPS_PER_WEEK
                        end = min(start + self.TIMESTEPS_PER_WEEK, n_rows)
                        if start >= n_rows:
                            break

                        week_vals = vals[start:end]
                        valid_mask = ~np.isnan(week_vals)
                        valid_vals = week_vals[valid_mask]

                        if len(valid_vals) > 0:
                            weekly_values[fuel][week + 1].extend(valid_vals.tolist())

                        # Profile contributions (if half_hour column available)
                        if half_hours is not None:
                            week_hh = half_hours[start:end]
                            for hh, val in zip(week_hh[valid_mask], valid_vals):
                                entry = profile_contribs[fuel][week + 1][int(hh)]
                                entry[0] += val
                                entry[1] += 1
                                entry[2] += val * val

        # Convert profile_contribs to tuples
        profile_result: Dict[str, Dict[int, Dict[int, Tuple[float, int, float]]]] = {}
        for fuel in ['electricity', 'gas']:
            profile_result[fuel] = {}
            for week, hh_dict in profile_contribs[fuel].items():
                profile_result[fuel][week] = {
                    hh: (e[0], e[1], e[2]) for hh, e in hh_dict.items()
                }

        # Build FuelStats per week
        fuel_stats_result: Dict[str, Dict[int, FuelStats]] = {}
        for fuel in ['electricity', 'gas']:
            fuel_stats_result[fuel] = {}
            for week, vals in weekly_values[fuel].items():
                if vals:
                    fuel_stats_result[fuel][week] = FuelStats.from_values(np.array(vals))

        return profile_result, fuel_stats_result

    def _compute_token_analysis(
        self, month_data: Dict[int, pd.DataFrame]
    ) -> Tuple[Dict[str, List[int]], Dict[str, Dict[int, Dict[str, float]]]]:
        """
        Compute token frequency histograms and monthly decoded-value stats.

        Token frequency: counts per bin index (O(n_bins) memory).
        Monthly decoded stats: mean, median, std, skewness, zero_rate per month.

        Returns:
            (token_frequency, monthly_decoded_stats)
        """
        from scipy import stats as scipy_stats

        token_freq: Dict[str, List[int]] = {}
        monthly_stats: Dict[str, Dict[int, Dict[str, float]]] = {}

        for fuel, suffix in [('electricity', self.ELEC_SUFFIX), ('gas', self.GAS_SUFFIX)]:
            if fuel not in self._bin_edges['bin_edges']:
                continue

            n_bins = len(self._bin_edges['bin_edges'][fuel])
            counts = [0] * n_bins
            fuel_monthly: Dict[int, Dict[str, float]] = {}

            for month, df in month_data.items():
                fuel_cols = [c for c in df.columns if c.endswith(suffix)]
                month_vals = []

                for col in fuel_cols:
                    vals = df[col].dropna().values.astype(float)
                    month_vals.extend(vals.tolist())

                    # Token counting
                    tokens = self._values_to_tokens(vals, fuel)
                    for t in tokens:
                        counts[int(t)] += 1

                # Monthly decoded stats
                if month_vals:
                    arr = np.array(month_vals)
                    n = len(arr)
                    fuel_monthly[int(month)] = {
                        'mean': float(np.mean(arr)),
                        'median': float(np.median(arr)),
                        'std': float(np.std(arr)),
                        'skew': float(scipy_stats.skew(arr)) if n > 2 else 0.0,
                        'zero_rate': float(np.sum(arr == 0) / n),
                        'n': n,
                    }

            token_freq[fuel] = counts
            if fuel_monthly:
                monthly_stats[fuel] = fuel_monthly

        return token_freq, monthly_stats

    def _compute_weekly_token_frequency(
        self, month_data: Dict[int, pd.DataFrame], num_weeks: int = 4
    ) -> Dict[str, Dict[int, List[int]]]:
        """
        Compute token frequency histograms stratified by within-month week.

        Returns {fuel: {week: [counts_per_bin]}}.
        """
        result: Dict[str, Dict[int, List[int]]] = {}

        for fuel, suffix in [('electricity', self.ELEC_SUFFIX), ('gas', self.GAS_SUFFIX)]:
            if fuel not in self._bin_edges['bin_edges']:
                continue

            n_bins = len(self._bin_edges['bin_edges'][fuel])
            # {week: counts}
            weekly_counts: Dict[int, List[int]] = {
                w + 1: [0] * n_bins for w in range(num_weeks)
            }

            for month, df in month_data.items():
                fuel_cols = [c for c in df.columns if c.endswith(suffix)]
                n_rows = len(df)

                for col in fuel_cols:
                    vals = df[col].values.astype(float)
                    for week in range(num_weeks):
                        start = week * self.TIMESTEPS_PER_WEEK
                        end = min(start + self.TIMESTEPS_PER_WEEK, n_rows)
                        if start >= n_rows:
                            break

                        week_vals = vals[start:end]
                        valid = week_vals[~np.isnan(week_vals)]
                        if len(valid) > 0:
                            tokens = self._values_to_tokens(valid, fuel)
                            for t in tokens:
                                weekly_counts[week + 1][int(t)] += 1

            result[fuel] = weekly_counts

        return result
