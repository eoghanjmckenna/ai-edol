"""
ResultsAggregator for MapReduce evaluation pipeline.

Aggregates BatchResults from all synthetic batches and compares
to RealDataStats to produce FinalResults.
"""

from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from .data_structures import FuelStats, BatchResult, RealDataStats, FinalResults, MissingnessStats


class ResultsAggregator:
    """
    Aggregate intermediate results from all batches.

    Implements the reduce phase of the MapReduce pattern.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 real_train_stats: Optional[RealDataStats] = None,
                 real_val_stats: Optional[RealDataStats] = None,
                 real_test_stats: Optional[RealDataStats] = None):
        """
        Args:
            config: Evaluation configuration
            real_train_stats: Pre-computed statistics from real training data
            real_val_stats: Pre-computed statistics from real validation data
            real_test_stats: Pre-computed statistics from real test data
        """
        self.config = config or {}
        self.real_train_stats = real_train_stats
        self.real_val_stats = real_val_stats
        self.real_test_stats = real_test_stats

        # Extract commonly used config
        self.acf_lags = self.config.get('acf_lags', [1, 2, 48, 96, 336, 672, 1008, 1344])
        self.conditioning_vars = self.config.get(
            'conditioning_vars',
            ['building_type', 'num_rooms', 'region', 'age_built', 'num_occs']
        )

    def aggregate(self, batch_results: List[BatchResult]) -> FinalResults:
        """
        Aggregate all batch results into final evaluation.

        Args:
            batch_results: List of BatchResult from map phase

        Returns:
            FinalResults with complete evaluation
        """
        if not batch_results:
            return self._empty_results()

        # Filter out empty batches
        valid_results = [br for br in batch_results if br.n_households > 0]

        if not valid_results:
            return self._empty_results()

        # 1. Compute metadata
        n_batches = len(valid_results)
        n_households = sum(br.n_households for br in valid_results)
        n_months = max(br.n_months_processed for br in valid_results)

        # 2. Aggregate marginal distribution
        marginal_results, n_values = self._aggregate_marginal(valid_results)

        # 3. Aggregate conditional distribution
        conditional_results = self._aggregate_conditional(valid_results)

        # 4. Aggregate temporal patterns
        profile_results = self._aggregate_profiles(valid_results)
        acf_results = self._aggregate_acf(valid_results)
        volatility_results = self._aggregate_volatility(valid_results)

        temporal_patterns = {
            'daily_profiles': profile_results,
            'acf': acf_results,
            'volatility': volatility_results
        }

        # 5. Aggregate within-batch variance
        variance_report = self._aggregate_variance(valid_results)

        # 6. Distance analysis is handled by DistanceVisualizer (with chunked
        #    computation to avoid OOM). See Decision 038.

        # 7. Aggregate missingness analysis
        missingness_results = self._aggregate_missingness(valid_results)

        # 8. Aggregate distribution comparison (Wasserstein/KS)
        distribution_results = self._aggregate_distribution_comparison(valid_results)

        # 9. Aggregate correlation analysis
        correlation_results = self._aggregate_correlation(valid_results)

        # 10. Aggregate PTG analysis
        ptg_results = self._aggregate_ptg(valid_results)

        # --- Integrated diagnostics (gated by config) ---
        diag_config = self.config.get('diagnostics_integration', {})
        seasonal_results = {}
        daily_pattern_results = {}
        hh_month_fidelity_results = {}
        weekly_fidelity_results = {}
        token_results = {}

        if diag_config.get('enabled', False):
            if diag_config.get('seasonal_analysis', {}).get('enabled', True):
                seasonal_results = self._aggregate_seasonal_analysis(valid_results)

            if diag_config.get('daily_patterns', {}).get('enabled', True):
                daily_pattern_results = self._aggregate_monthly_profiles(valid_results)

            if diag_config.get('weekly_fidelity', {}).get('enabled', True):
                weekly_fidelity_results = self._aggregate_weekly_fidelity(valid_results)
                # Add weekly profiles (degradation) into daily_pattern_results
                weekly_profiles = self._aggregate_weekly_profiles(valid_results, diag_config)
                daily_pattern_results['weekly_degradation'] = weekly_profiles

            if diag_config.get('household_month_fidelity', {}).get('enabled', True):
                quintiles = diag_config.get('household_month_fidelity', {}).get('quintiles', [20, 40, 60, 80])
                hh_month_fidelity_results = self._aggregate_household_month_fidelity(
                    valid_results, quintiles=quintiles)

            if diag_config.get('token_analysis', {}).get('enabled', True):
                token_results = self._aggregate_token_analysis(valid_results)

        # 11. Aggregate privacy pre-filter stats (if any batch had filtering)
        prefilter_summary = self._aggregate_prefilter_stats(batch_results)

        # Real household counts for metadata (SDC companion CSVs)
        n_households_train = self.real_train_stats.n_households if self.real_train_stats else 0
        n_households_val = self.real_val_stats.n_households if self.real_val_stats else 0
        n_households_test = self.real_test_stats.n_households if self.real_test_stats else 0

        return FinalResults(
            n_batches=n_batches,
            n_households=n_households,
            n_months=n_months,
            n_values=n_values,
            n_households_train=n_households_train,
            n_households_val=n_households_val,
            n_households_test=n_households_test,
            marginal_distribution=marginal_results,
            conditional_distribution=conditional_results,
            temporal_patterns=temporal_patterns,
            within_batch_variance=variance_report,
            distance_analysis={},
            missingness_analysis=missingness_results,
            distribution_comparison=distribution_results,
            correlation_analysis=correlation_results,
            ptg_analysis=ptg_results,
            seasonal_analysis=seasonal_results,
            daily_pattern_diagnostics=daily_pattern_results,
            household_month_fidelity=hh_month_fidelity_results,
            weekly_fidelity=weekly_fidelity_results,
            token_analysis=token_results,
            prefilter_summary=prefilter_summary,
        )

    def _empty_results(self) -> FinalResults:
        """Return empty FinalResults when no valid batches."""
        return FinalResults(
            n_batches=0,
            n_households=0,
            n_months=0,
            n_values={},
            marginal_distribution={},
            conditional_distribution={},
            temporal_patterns={},
            within_batch_variance={'status': 'no_data'},
            distance_analysis={},
            missingness_analysis={},
            distribution_comparison={},
            correlation_analysis={},
            ptg_analysis={},
        )

    def _aggregate_prefilter_stats(
        self, batch_results: List[BatchResult]
    ) -> Optional[Dict[str, Any]]:
        """Aggregate privacy pre-filter stats across all batches."""
        filtered_batches = [br for br in batch_results if br.prefilter_stats is not None]
        if not filtered_batches:
            return None

        total_hm_excluded = sum(
            br.prefilter_stats['n_household_months_excluded'] for br in filtered_batches)
        total_fully = sum(
            br.prefilter_stats['n_households_fully_excluded'] for br in filtered_batches)
        total_partial = sum(
            br.prefilter_stats['n_households_partially_excluded'] for br in filtered_batches)
        n_emptied = sum(
            1 for br in filtered_batches if br.prefilter_stats['batch_empty_after_filter'])

        # Include CSV-level info from config if available
        pf_config = self.config.get('_prefilter_config', {})

        return {
            'enabled': True,
            'exclusion_csv': pf_config.get('exclusion_csv', 'unknown'),
            'csv_rows': pf_config.get('csv_rows', 0),
            'csv_duplicates': pf_config.get('csv_duplicates', 0),
            'n_household_months_excluded': total_hm_excluded,
            'n_households_fully_excluded': total_fully,
            'n_households_partially_excluded': total_partial,
            'n_batches_affected': len(filtered_batches),
            'n_batches_total': len(batch_results),
            'n_batches_emptied': n_emptied,
        }

    def _aggregate_marginal(self, batch_results: List[BatchResult]
                            ) -> Tuple[Dict[str, Any], Dict[str, int]]:
        """
        Aggregate fuel statistics for marginal distribution comparison.

        Returns:
            Tuple of (marginal_results dict, n_values dict)
        """
        results = {}
        n_values = {}

        for fuel in ['electricity', 'gas']:
            # Collect all FuelStats for this fuel
            stats_list = [
                br.fuel_stats[fuel] for br in batch_results
                if fuel in br.fuel_stats and br.fuel_stats[fuel].n_values > 0
            ]

            if not stats_list:
                results[fuel] = {'status': 'no_data'}
                n_values[fuel] = 0
                continue

            # Combine statistics
            combined = FuelStats.combine(stats_list)
            n_values[fuel] = combined.n_values

            # Extract synthetic statistics
            synth_mean = combined.mean
            synth_std = combined.std
            synth_quantiles = combined.quantiles

            # Get train data statistics for comparison
            train_mean = np.nan
            train_std = np.nan
            train_quantiles = {}

            if self.real_train_stats and fuel in self.real_train_stats.fuel_stats:
                train_stats = self.real_train_stats.fuel_stats[fuel]
                train_mean = train_stats.mean
                train_std = train_stats.std
                train_quantiles = train_stats.quantiles

            # Get val data statistics for comparison
            val_mean = np.nan
            val_std = np.nan
            val_quantiles = {}

            if self.real_val_stats and fuel in self.real_val_stats.fuel_stats:
                val_stats = self.real_val_stats.fuel_stats[fuel]
                val_mean = val_stats.mean
                val_std = val_stats.std
                val_quantiles = val_stats.quantiles

            # Get test data statistics for comparison
            test_mean = np.nan
            test_std = np.nan
            test_quantiles = {}

            if self.real_test_stats and fuel in self.real_test_stats.fuel_stats:
                test_stats = self.real_test_stats.fuel_stats[fuel]
                test_mean = test_stats.mean
                test_std = test_stats.std
                test_quantiles = test_stats.quantiles

            # Compute comparison metrics (synth vs train)
            mean_ratio = synth_mean / train_mean if train_mean and not np.isnan(train_mean) else np.nan
            std_ratio = synth_std / train_std if train_std and not np.isnan(train_std) else np.nan

            # Compute synth vs val metrics
            val_mean_ratio = synth_mean / val_mean if val_mean and not np.isnan(val_mean) else np.nan
            val_std_ratio = synth_std / val_std if val_std and not np.isnan(val_std) else np.nan

            # Compute synth vs test metrics
            test_mean_ratio = synth_mean / test_mean if test_mean and not np.isnan(test_mean) else np.nan
            test_std_ratio = synth_std / test_std if test_std and not np.isnan(test_std) else np.nan

            # Quantile MAE (mean absolute error) vs train
            quantile_mae = np.nan
            if train_quantiles and synth_quantiles:
                errors = []
                for q in synth_quantiles:
                    if q in train_quantiles:
                        synth_q = synth_quantiles[q]
                        train_q = train_quantiles[q]
                        if not np.isnan(synth_q) and not np.isnan(train_q):
                            # Normalize by train quantile to get relative error
                            if train_q != 0:
                                errors.append(abs(synth_q - train_q) / abs(train_q))
                if errors:
                    quantile_mae = np.mean(errors)

            # Quantile MAE vs val
            val_quantile_mae = np.nan
            if val_quantiles and synth_quantiles:
                errors = []
                for q in synth_quantiles:
                    if q in val_quantiles:
                        synth_q = synth_quantiles[q]
                        val_q = val_quantiles[q]
                        if not np.isnan(synth_q) and not np.isnan(val_q):
                            if val_q != 0:
                                errors.append(abs(synth_q - val_q) / abs(val_q))
                if errors:
                    val_quantile_mae = np.mean(errors)

            # Quantile MAE vs test
            test_quantile_mae = np.nan
            if test_quantiles and synth_quantiles:
                errors = []
                for q in synth_quantiles:
                    if q in test_quantiles:
                        synth_q = synth_quantiles[q]
                        test_q = test_quantiles[q]
                        if not np.isnan(synth_q) and not np.isnan(test_q):
                            if test_q != 0:
                                errors.append(abs(synth_q - test_q) / abs(test_q))
                if errors:
                    test_quantile_mae = np.mean(errors)

            # Determine status based on mean ratio (use train as primary comparison)
            if np.isnan(mean_ratio):
                status = 'no_comparison_data'
            elif 0.8 <= mean_ratio <= 1.2:
                status = 'pass'
            elif 0.5 <= mean_ratio <= 2.0:
                status = 'warn'
            else:
                status = 'fail'

            # Collect n_values for SE/CI computation
            synth_n = combined.n_values
            train_n = 0
            val_n = 0
            test_n = 0
            if self.real_train_stats and fuel in self.real_train_stats.fuel_stats:
                train_n = self.real_train_stats.fuel_stats[fuel].n_values
            if self.real_val_stats and fuel in self.real_val_stats.fuel_stats:
                val_n = self.real_val_stats.fuel_stats[fuel].n_values
            if self.real_test_stats and fuel in self.real_test_stats.fuel_stats:
                test_n = self.real_test_stats.fuel_stats[fuel].n_values

            results[fuel] = {
                'synth_mean': synth_mean,
                'synth_std': synth_std,
                'synth_n': synth_n,
                'synth_quantiles': synth_quantiles,
                'synth_min': combined.min_value,
                'synth_max': combined.max_value,
                'train_mean': train_mean,
                'train_std': train_std,
                'train_n': train_n,
                'train_quantiles': train_quantiles,
                'mean_ratio': mean_ratio,
                'std_ratio': std_ratio,
                'quantile_mae': quantile_mae,
                'val_mean': val_mean,
                'val_std': val_std,
                'val_n': val_n,
                'val_quantiles': val_quantiles,
                'val_mean_ratio': val_mean_ratio,
                'val_std_ratio': val_std_ratio,
                'val_quantile_mae': val_quantile_mae,
                'test_mean': test_mean,
                'test_std': test_std,
                'test_n': test_n,
                'test_quantiles': test_quantiles,
                'test_mean_ratio': test_mean_ratio,
                'test_std_ratio': test_std_ratio,
                'test_quantile_mae': test_quantile_mae,
                'status': status
            }

        return results, n_values

    def _aggregate_conditional(self, batch_results: List[BatchResult]) -> Dict[str, Any]:
        """
        Group batches by conditioning variables, aggregate within groups.

        Returns statistics per conditioning variable per value per fuel.
        """
        results = {}

        for cond_var in self.conditioning_vars:
            # Group batches by this conditioning variable
            groups = defaultdict(list)
            for br in batch_results:
                cond_value = br.conditioning.get(cond_var)
                if cond_value is not None:
                    groups[cond_value].append(br)

            if not groups:
                continue

            # Aggregate stats within each group
            cond_results = {}
            for cond_value, group_batches in groups.items():
                cond_results[cond_value] = {}

                for fuel in ['electricity', 'gas']:
                    stats_list = [
                        br.fuel_stats[fuel] for br in group_batches
                        if fuel in br.fuel_stats and br.fuel_stats[fuel].n_values > 0
                    ]

                    if not stats_list:
                        continue

                    combined = FuelStats.combine(stats_list)
                    synth_mean = combined.mean
                    synth_std = combined.std

                    # Get train data conditional mean
                    train_mean = np.nan
                    train_std = np.nan
                    train_n = 0
                    if (self.real_train_stats and
                        cond_var in self.real_train_stats.conditional and
                        cond_value in self.real_train_stats.conditional[cond_var] and
                        fuel in self.real_train_stats.conditional[cond_var][cond_value]):
                        train_fuel_stats = self.real_train_stats.conditional[cond_var][cond_value][fuel]
                        train_mean = train_fuel_stats.mean
                        train_std = train_fuel_stats.std
                        train_n = train_fuel_stats.n_values

                    # Get val data conditional mean
                    val_mean = np.nan
                    val_std = np.nan
                    val_n = 0
                    if (self.real_val_stats and
                        cond_var in self.real_val_stats.conditional and
                        cond_value in self.real_val_stats.conditional[cond_var] and
                        fuel in self.real_val_stats.conditional[cond_var][cond_value]):
                        val_fuel_stats = self.real_val_stats.conditional[cond_var][cond_value][fuel]
                        val_mean = val_fuel_stats.mean
                        val_std = val_fuel_stats.std
                        val_n = val_fuel_stats.n_values

                    # Get test data conditional mean
                    test_mean = np.nan
                    test_std = np.nan
                    test_n = 0
                    if (self.real_test_stats and
                        cond_var in self.real_test_stats.conditional and
                        cond_value in self.real_test_stats.conditional[cond_var] and
                        fuel in self.real_test_stats.conditional[cond_var][cond_value]):
                        test_fuel_stats = self.real_test_stats.conditional[cond_var][cond_value][fuel]
                        test_mean = test_fuel_stats.mean
                        test_std = test_fuel_stats.std
                        test_n = test_fuel_stats.n_values

                    train_ratio = synth_mean / train_mean if train_mean and not np.isnan(train_mean) else np.nan
                    val_ratio = synth_mean / val_mean if val_mean and not np.isnan(val_mean) else np.nan
                    test_ratio = synth_mean / test_mean if test_mean and not np.isnan(test_mean) else np.nan

                    # Household counts per conditioning group (for SDC compliance)
                    n_households_train = 0
                    if (self.real_train_stats and
                            cond_var in self.real_train_stats.conditioning_n_households and
                            cond_value in self.real_train_stats.conditioning_n_households[cond_var]):
                        n_households_train = self.real_train_stats.conditioning_n_households[cond_var][cond_value]

                    n_households_val = 0
                    if (self.real_val_stats and
                            cond_var in self.real_val_stats.conditioning_n_households and
                            cond_value in self.real_val_stats.conditioning_n_households[cond_var]):
                        n_households_val = self.real_val_stats.conditioning_n_households[cond_var][cond_value]

                    n_households_test = 0
                    if (self.real_test_stats and
                            cond_var in self.real_test_stats.conditioning_n_households and
                            cond_value in self.real_test_stats.conditioning_n_households[cond_var]):
                        n_households_test = self.real_test_stats.conditioning_n_households[cond_var][cond_value]

                    cond_results[cond_value][fuel] = {
                        'synth_mean': synth_mean,
                        'synth_std': synth_std,
                        'synth_n': combined.n_values,
                        'train_mean': train_mean,
                        'train_std': train_std,
                        'train_n': train_n,
                        'n_households_train': n_households_train,
                        'train_ratio': train_ratio,
                        'val_mean': val_mean,
                        'val_std': val_std,
                        'val_n': val_n,
                        'n_households_val': n_households_val,
                        'val_ratio': val_ratio,
                        'test_mean': test_mean,
                        'test_std': test_std,
                        'test_n': test_n,
                        'n_households_test': n_households_test,
                        'test_ratio': test_ratio,
                        'n_batches': len(group_batches),
                        'n_values': combined.n_values
                    }

            if cond_results:
                results[cond_var] = cond_results

        return results

    def _aggregate_profiles(self, batch_results: List[BatchResult]) -> Dict[str, Any]:
        """
        Combine profile contributions from all batches.

        Profile contributions are (sum, count) tuples per half-hour.
        Final profile is sum / count for each half-hour.
        """
        results = {}

        for fuel in ['electricity', 'gas']:
            for profile_type in ['all', 'weekday', 'weekend']:
                # Combine sums and counts across all batches
                combined_sum = {}
                combined_count = {}

                for br in batch_results:
                    contribs = br.profile_contributions.get(fuel, {}).get(profile_type, {})
                    for hh, (s, c) in contribs.items():
                        combined_sum[hh] = combined_sum.get(hh, 0.0) + s
                        combined_count[hh] = combined_count.get(hh, 0) + c

                if not combined_sum:
                    continue

                # Compute weighted average profile (48 half-hours: 1-48)
                synth_profile = []
                for hh in range(1, 49):
                    if hh in combined_sum and combined_count.get(hh, 0) > 0:
                        synth_profile.append(combined_sum[hh] / combined_count[hh])
                    else:
                        synth_profile.append(np.nan)

                synth_profile = np.array(synth_profile)

                # Get train data profile for comparison
                train_profile = None
                train_distance = np.nan

                if self.real_train_stats and fuel in self.real_train_stats.profiles:
                    train_profile_dict = self.real_train_stats.profiles.get(fuel, {})
                    if profile_type in train_profile_dict:
                        train_profile = train_profile_dict[profile_type]

                        # Compute distance (RMSE)
                        if train_profile is not None and len(train_profile) == 48:
                            valid_mask = ~np.isnan(synth_profile) & ~np.isnan(train_profile)
                            if np.any(valid_mask):
                                train_distance = np.sqrt(np.mean(
                                    (synth_profile[valid_mask] - np.array(train_profile)[valid_mask]) ** 2
                                ))

                # Get val data profile for comparison
                val_profile = None
                val_distance = np.nan

                if self.real_val_stats and fuel in self.real_val_stats.profiles:
                    val_profile_dict = self.real_val_stats.profiles.get(fuel, {})
                    if profile_type in val_profile_dict:
                        val_profile = val_profile_dict[profile_type]

                        # Compute distance (RMSE)
                        if val_profile is not None and len(val_profile) == 48:
                            valid_mask = ~np.isnan(synth_profile) & ~np.isnan(val_profile)
                            if np.any(valid_mask):
                                val_distance = np.sqrt(np.mean(
                                    (synth_profile[valid_mask] - np.array(val_profile)[valid_mask]) ** 2
                                ))

                # Get test data profile for comparison
                test_profile = None
                test_distance = np.nan

                if self.real_test_stats and fuel in self.real_test_stats.profiles:
                    test_profile_dict = self.real_test_stats.profiles.get(fuel, {})
                    if profile_type in test_profile_dict:
                        test_profile = test_profile_dict[profile_type]

                        # Compute distance (RMSE)
                        if test_profile is not None and len(test_profile) == 48:
                            valid_mask = ~np.isnan(synth_profile) & ~np.isnan(test_profile)
                            if np.any(valid_mask):
                                test_distance = np.sqrt(np.mean(
                                    (synth_profile[valid_mask] - np.array(test_profile)[valid_mask]) ** 2
                                ))

                key = f'{fuel}_{profile_type}'
                result_entry = {
                    'synth': synth_profile.tolist(),
                    'train': list(train_profile) if train_profile is not None else None,
                    'train_distance': float(train_distance) if not np.isnan(train_distance) else None,
                    'val': list(val_profile) if val_profile is not None else None,
                    'val_distance': float(val_distance) if not np.isnan(val_distance) else None,
                    'test': list(test_profile) if test_profile is not None else None,
                    'test_distance': float(test_distance) if not np.isnan(test_distance) else None,
                    'n_observations': sum(combined_count.values())
                }

                # Median profiles for electricity
                if fuel == 'electricity':
                    # Synth median: weighted mean of batch-level medians
                    med_sum = {}
                    med_count = {}
                    for br in batch_results:
                        elec_meds = br.profile_medians.get('electricity', {}).get(profile_type, {})
                        for hh, (med, n) in elec_meds.items():
                            med_sum[hh] = med_sum.get(hh, 0.0) + med * n
                            med_count[hh] = med_count.get(hh, 0) + n
                    synth_median_profile = []
                    for hh in range(1, 49):
                        if med_count.get(hh, 0) > 0:
                            synth_median_profile.append(med_sum[hh] / med_count[hh])
                        else:
                            synth_median_profile.append(np.nan)
                    result_entry['synth_median'] = synth_median_profile

                    # Real data medians
                    for split_name, real_stats in [('train', self.real_train_stats),
                                                   ('val', self.real_val_stats),
                                                   ('test', self.real_test_stats)]:
                        if real_stats and hasattr(real_stats, 'profile_medians'):
                            med_dict = real_stats.profile_medians.get('electricity', {})
                            if profile_type in med_dict:
                                result_entry[f'{split_name}_median'] = med_dict[profile_type]

                results[key] = result_entry

        return results

    def _aggregate_acf(self, batch_results: List[BatchResult]) -> Dict[str, Any]:
        """
        Average ACF values across all batches/months.

        ACF values are collected per household-month, then averaged.
        """
        results = {}

        for fuel in ['electricity', 'gas']:
            results[fuel] = {}

            for lag in self.acf_lags:
                # Collect all ACF values for this lag
                all_values = []
                for br in batch_results:
                    if fuel in br.acf_values and lag in br.acf_values[fuel]:
                        all_values.extend(br.acf_values[fuel][lag])

                synth_acf = np.nanmean(all_values) if all_values else np.nan
                synth_std = np.nanstd(all_values) if all_values else np.nan

                # Get train data ACF for comparison
                train_acf = np.nan
                if self.real_train_stats and fuel in self.real_train_stats.acf:
                    train_acf = self.real_train_stats.acf[fuel].get(lag, np.nan)

                # Get val data ACF for comparison
                val_acf = np.nan
                if self.real_val_stats and fuel in self.real_val_stats.acf:
                    val_acf = self.real_val_stats.acf[fuel].get(lag, np.nan)

                # Get test data ACF for comparison
                test_acf = np.nan
                if self.real_test_stats and fuel in self.real_test_stats.acf:
                    test_acf = self.real_test_stats.acf[fuel].get(lag, np.nan)

                train_diff = abs(synth_acf - train_acf) if (
                    not np.isnan(synth_acf) and not np.isnan(train_acf)
                ) else np.nan

                val_diff = abs(synth_acf - val_acf) if (
                    not np.isnan(synth_acf) and not np.isnan(val_acf)
                ) else np.nan

                test_diff = abs(synth_acf - test_acf) if (
                    not np.isnan(synth_acf) and not np.isnan(test_acf)
                ) else np.nan

                results[fuel][lag] = {
                    'synth': float(synth_acf) if not np.isnan(synth_acf) else None,
                    'synth_std': float(synth_std) if not np.isnan(synth_std) else None,
                    'train': float(train_acf) if not np.isnan(train_acf) else None,
                    'train_diff': float(train_diff) if not np.isnan(train_diff) else None,
                    'val': float(val_acf) if not np.isnan(val_acf) else None,
                    'val_diff': float(val_diff) if not np.isnan(val_diff) else None,
                    'test': float(test_acf) if not np.isnan(test_acf) else None,
                    'test_diff': float(test_diff) if not np.isnan(test_diff) else None,
                    'n_samples': len(all_values)
                }

        return results

    @staticmethod
    def _excess_kurtosis(arr: np.ndarray) -> float:
        """Excess kurtosis (Fisher definition: normal = 0) from an array."""
        n = len(arr)
        if n < 4:
            return np.nan
        std = float(np.std(arr, ddof=1))
        if std == 0:
            return np.nan
        return float(np.mean(((arr - np.mean(arr)) / std) ** 4) - 3)

    @staticmethod
    def _abs_diff_quantiles(arr: np.ndarray) -> Dict[str, float]:
        """Quantiles of |Δx| from an array of signed first differences."""
        abs_arr = np.abs(arr)
        return {
            'p50': float(np.percentile(abs_arr, 50)),
            'p75': float(np.percentile(abs_arr, 75)),
            'p90': float(np.percentile(abs_arr, 90)),
            'p95': float(np.percentile(abs_arr, 95)),
        }

    def _aggregate_volatility(self, batch_results: List[BatchResult]) -> Dict[str, Any]:
        """
        Combine volatility statistics using FuelStats.combine().

        Volatility statistics are computed on first differences.
        """
        results = {}
        _rng = np.random.default_rng(42)
        _max_samples = 10000

        for fuel in ['electricity', 'gas']:
            # Collect all volatility FuelStats for this fuel
            stats_list = [
                br.volatility_stats[fuel] for br in batch_results
                if fuel in br.volatility_stats and br.volatility_stats[fuel].n_values > 0
            ]

            if not stats_list:
                results[fuel] = {'status': 'no_data'}
                continue

            # Combine statistics
            combined = FuelStats.combine(stats_list)

            synth_mean = combined.mean
            synth_std = combined.std
            synth_quantiles = combined.quantiles

            # Get train data volatility for comparison
            train_mean = np.nan
            train_std = np.nan
            train_quantiles = {}

            if self.real_train_stats and fuel in self.real_train_stats.volatility_stats:
                train_stats = self.real_train_stats.volatility_stats[fuel]
                train_mean = train_stats.mean
                train_std = train_stats.std
                train_quantiles = train_stats.quantiles

            # Get val data volatility for comparison
            val_mean = np.nan
            val_std = np.nan
            val_quantiles = {}

            if self.real_val_stats and fuel in self.real_val_stats.volatility_stats:
                val_stats = self.real_val_stats.volatility_stats[fuel]
                val_mean = val_stats.mean
                val_std = val_stats.std
                val_quantiles = val_stats.quantiles

            # Get test data volatility for comparison
            test_mean = np.nan
            test_std = np.nan
            test_quantiles = {}

            if self.real_test_stats and fuel in self.real_test_stats.volatility_stats:
                test_stats = self.real_test_stats.volatility_stats[fuel]
                test_mean = test_stats.mean
                test_std = test_stats.std
                test_quantiles = test_stats.quantiles

            mean_ratio = synth_mean / train_mean if train_mean and not np.isnan(train_mean) else np.nan
            std_ratio = synth_std / train_std if train_std and not np.isnan(train_std) else np.nan
            val_mean_ratio = synth_mean / val_mean if val_mean and not np.isnan(val_mean) else np.nan
            val_std_ratio = synth_std / val_std if val_std and not np.isnan(val_std) else np.nan
            test_mean_ratio = synth_mean / test_mean if test_mean and not np.isnan(test_mean) else np.nan
            test_std_ratio = synth_std / test_std if test_std and not np.isnan(test_std) else np.nan

            # Merge diff samples from batches for histogram + kurtosis + |Δx| quantiles
            def _merge_samples(source_list):
                merged = []
                for s in source_list:
                    merged.extend(s)
                if len(merged) > _max_samples:
                    idx = _rng.choice(len(merged), size=_max_samples, replace=False)
                    merged = [merged[i] for i in idx]
                return np.array(merged) if merged else np.array([])

            synth_arr = _merge_samples([br.volatility_samples.get(fuel, []) for br in batch_results])

            train_arr = np.array([])
            if self.real_train_stats and self.real_train_stats.volatility_samples.get(fuel):
                train_arr = np.array(self.real_train_stats.volatility_samples[fuel])

            test_arr = np.array([])
            if self.real_test_stats and self.real_test_stats.volatility_samples.get(fuel):
                test_arr = np.array(self.real_test_stats.volatility_samples[fuel])

            results[fuel] = {
                'synth_mean': float(synth_mean) if not np.isnan(synth_mean) else None,
                'synth_std': float(synth_std) if not np.isnan(synth_std) else None,
                'synth_quantiles': synth_quantiles,
                'train_mean': float(train_mean) if not np.isnan(train_mean) else None,
                'train_std': float(train_std) if not np.isnan(train_std) else None,
                'train_quantiles': train_quantiles,
                'mean_ratio': float(mean_ratio) if not np.isnan(mean_ratio) else None,
                'std_ratio': float(std_ratio) if not np.isnan(std_ratio) else None,
                'val_mean': float(val_mean) if not np.isnan(val_mean) else None,
                'val_std': float(val_std) if not np.isnan(val_std) else None,
                'val_quantiles': val_quantiles,
                'val_mean_ratio': float(val_mean_ratio) if not np.isnan(val_mean_ratio) else None,
                'val_std_ratio': float(val_std_ratio) if not np.isnan(val_std_ratio) else None,
                'test_mean': float(test_mean) if not np.isnan(test_mean) else None,
                'test_std': float(test_std) if not np.isnan(test_std) else None,
                'test_quantiles': test_quantiles,
                'test_mean_ratio': float(test_mean_ratio) if not np.isnan(test_mean_ratio) else None,
                'test_std_ratio': float(test_std_ratio) if not np.isnan(test_std_ratio) else None,
                'n_values': combined.n_values,
                # Diff samples for histogram
                'synth_diff_samples': synth_arr.tolist() if len(synth_arr) else [],
                'train_diff_samples': train_arr.tolist() if len(train_arr) else [],
                'test_diff_samples': test_arr.tolist() if len(test_arr) else [],
                # Kurtosis of Δx (excess, Fisher: normal = 0)
                'synth_kurtosis': self._excess_kurtosis(synth_arr) if len(synth_arr) > 3 else None,
                'train_kurtosis': self._excess_kurtosis(train_arr) if len(train_arr) > 3 else None,
                'test_kurtosis': self._excess_kurtosis(test_arr) if len(test_arr) > 3 else None,
                # |Δx| quantiles for spike magnitude comparison
                'synth_abs_quantiles': self._abs_diff_quantiles(synth_arr) if len(synth_arr) else {},
                'train_abs_quantiles': self._abs_diff_quantiles(train_arr) if len(train_arr) else {},
                'test_abs_quantiles': self._abs_diff_quantiles(test_arr) if len(test_arr) else {},
            }

        return results

    def _aggregate_variance(self, batch_results: List[BatchResult]) -> Dict[str, Any]:
        """
        Collect within-batch variance metrics, compute overall statistics.

        Each batch has pre-computed variance metrics with a severity level.
        """
        severity_counts = {'none': 0, 'low': 0, 'medium': 0, 'high': 0, 'unknown': 0}
        flagged_batches = []

        correlation_values = []
        distance_values = []
        peak_diversity_values = []

        for br in batch_results:
            vm = br.variance_metrics
            severity = vm.get('severity', 'unknown')
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

            # Collect flagged batches
            if severity not in ['none', 'unknown']:
                flagged_batches.append({
                    'batch_id': br.batch_id,
                    'severity': severity,
                    'flags': vm.get('flags', []),
                    'mean_correlation': vm.get('mean_pairwise_correlation'),
                    'min_distance': vm.get('min_pairwise_distance')
                })

            # Collect aggregate metrics
            if 'mean_pairwise_correlation' in vm:
                val = vm['mean_pairwise_correlation']
                if not np.isnan(val):
                    correlation_values.append(val)

            if 'mean_pairwise_distance' in vm:
                val = vm['mean_pairwise_distance']
                if not np.isnan(val):
                    distance_values.append(val)

            if 'peak_hour_diversity' in vm:
                val = vm['peak_hour_diversity']
                if not np.isnan(val):
                    peak_diversity_values.append(val)

        # Determine overall status
        if severity_counts.get('high', 0) > 0:
            overall_status = 'fail'
        elif severity_counts.get('medium', 0) > len(batch_results) * 0.1:
            overall_status = 'warn'
        else:
            overall_status = 'pass'

        return {
            'n_batches_analyzed': len(batch_results),
            'severity_counts': severity_counts,
            'flagged_batches': flagged_batches[:20],  # Limit to 20 for readability
            'overall_status': overall_status,
            'aggregate_metrics': {
                'mean_correlation': float(np.mean(correlation_values)) if correlation_values else None,
                'mean_distance': float(np.mean(distance_values)) if distance_values else None,
                'mean_peak_diversity': float(np.mean(peak_diversity_values)) if peak_diversity_values else None
            }
        }

    def _aggregate_missingness(self, batch_results: List[BatchResult]) -> Dict[str, Any]:
        """
        Aggregate missingness statistics across all batches.

        Compares synthetic missingness patterns to real data missingness.
        """
        results = {}

        for fuel in ['electricity', 'gas']:
            # Collect all MissingnessStats for this fuel
            stats_list = [
                br.missingness_stats[fuel] for br in batch_results
                if fuel in br.missingness_stats
            ]

            if not stats_list:
                results[fuel] = {'status': 'no_data'}
                continue

            # Combine statistics
            combined = MissingnessStats.combine(stats_list)

            fuel_results = {
                'synth_missing_pct': combined.missing_pct,
                'synth_zeros_pct': combined.zeros_pct,
                'synth_structural_pct': combined.structural_missing_pct,
                'synth_n_total': combined.n_total,
                'synth_n_missing': combined.n_missing,
                'synth_n_zeros': combined.n_zeros,
                'synth_n_households_no_data': combined.n_households_no_data,
                'synth_n_households_total': combined.n_households_total,
            }

            # Add run length statistics (summarize the histogram)
            if combined.missing_run_lengths:
                run_lengths = list(combined.missing_run_lengths.keys())
                fuel_results['synth_missing_run_stats'] = {
                    'max_run_length': max(run_lengths),
                    'mean_run_length': float(np.average(
                        list(combined.missing_run_lengths.keys()),
                        weights=list(combined.missing_run_lengths.values())
                    )),
                    'n_runs': sum(combined.missing_run_lengths.values()),
                }

            if combined.zero_run_lengths:
                run_lengths = list(combined.zero_run_lengths.keys())
                fuel_results['synth_zero_run_stats'] = {
                    'max_run_length': max(run_lengths),
                    'mean_run_length': float(np.average(
                        list(combined.zero_run_lengths.keys()),
                        weights=list(combined.zero_run_lengths.values())
                    )),
                    'n_runs': sum(combined.zero_run_lengths.values()),
                }

            # Compare to train data if available
            if self.real_train_stats and fuel in self.real_train_stats.missingness_stats:
                train_stats = self.real_train_stats.missingness_stats[fuel]
                fuel_results['train_missing_pct'] = train_stats.missing_pct
                fuel_results['train_zeros_pct'] = train_stats.zeros_pct
                fuel_results['train_structural_pct'] = train_stats.structural_missing_pct

                # Compute differences
                fuel_results['missing_pct_diff'] = combined.missing_pct - train_stats.missing_pct
                fuel_results['zeros_pct_diff'] = combined.zeros_pct - train_stats.zeros_pct
                fuel_results['structural_pct_diff'] = (
                    combined.structural_missing_pct - train_stats.structural_missing_pct
                )

                # Add train run length stats
                if train_stats.missing_run_lengths:
                    fuel_results['train_missing_run_stats'] = {
                        'max_run_length': max(train_stats.missing_run_lengths.keys()),
                        'mean_run_length': float(np.average(
                            list(train_stats.missing_run_lengths.keys()),
                            weights=list(train_stats.missing_run_lengths.values())
                        )),
                        'n_runs': sum(train_stats.missing_run_lengths.values()),
                    }

                if train_stats.zero_run_lengths:
                    fuel_results['train_zero_run_stats'] = {
                        'max_run_length': max(train_stats.zero_run_lengths.keys()),
                        'mean_run_length': float(np.average(
                            list(train_stats.zero_run_lengths.keys()),
                            weights=list(train_stats.zero_run_lengths.values())
                        )),
                        'n_runs': sum(train_stats.zero_run_lengths.values()),
                    }

            # Compare to val data if available
            if self.real_val_stats and fuel in self.real_val_stats.missingness_stats:
                val_stats = self.real_val_stats.missingness_stats[fuel]
                fuel_results['val_missing_pct'] = val_stats.missing_pct
                fuel_results['val_zeros_pct'] = val_stats.zeros_pct
                fuel_results['val_structural_pct'] = val_stats.structural_missing_pct

            # Compare to test data if available
            if self.real_test_stats and fuel in self.real_test_stats.missingness_stats:
                test_stats = self.real_test_stats.missingness_stats[fuel]
                fuel_results['test_missing_pct'] = test_stats.missing_pct
                fuel_results['test_zeros_pct'] = test_stats.zeros_pct
                fuel_results['test_structural_pct'] = test_stats.structural_missing_pct

            # Determine status
            if 'train_missing_pct' not in fuel_results:
                fuel_results['status'] = 'no_comparison_data'
            elif abs(fuel_results.get('missing_pct_diff', 0)) < 5.0:
                fuel_results['status'] = 'pass'
            elif abs(fuel_results.get('missing_pct_diff', 0)) < 15.0:
                fuel_results['status'] = 'warn'
            else:
                fuel_results['status'] = 'fail'

            results[fuel] = fuel_results

        return results

    def _aggregate_distribution_comparison(self, batch_results: List[BatchResult]) -> Dict[str, Any]:
        """
        Compute Wasserstein distance and KS test between synthetic and real distributions.

        Uses value samples collected via reservoir sampling in map phase.
        """
        from scipy.stats import wasserstein_distance, ks_2samp

        results = {}

        for fuel in ['electricity', 'gas']:
            # Collect all value samples for this fuel
            all_samples = []
            for br in batch_results:
                if fuel in br.value_samples:
                    all_samples.extend(br.value_samples[fuel])

            if not all_samples:
                results[fuel] = {'status': 'no_data'}
                continue

            synth_samples = np.array(all_samples)
            fuel_results = {'n_synth_samples': len(synth_samples)}

            # Compare to train data if available
            if self.real_train_stats and fuel in self.real_train_stats.value_samples:
                train_samples = np.array(self.real_train_stats.value_samples[fuel])
                if len(train_samples) > 0:
                    # Wasserstein distance (normalized by train std)
                    train_std = np.std(train_samples)
                    if train_std > 0:
                        wd = wasserstein_distance(synth_samples, train_samples)
                        wd_normalized = wd / train_std
                    else:
                        wd = wasserstein_distance(synth_samples, train_samples)
                        wd_normalized = np.nan

                    # KS test
                    ks_stat, ks_pvalue = ks_2samp(synth_samples, train_samples)

                    fuel_results['synth_train'] = {
                        'wasserstein': float(wd),
                        'wasserstein_normalized': float(wd_normalized) if not np.isnan(wd_normalized) else None,
                        'ks_statistic': float(ks_stat),
                        'ks_pvalue': float(ks_pvalue),
                        'n_train_samples': len(train_samples),
                    }

            # Compare to val data if available
            if self.real_val_stats and fuel in self.real_val_stats.value_samples:
                val_samples = np.array(self.real_val_stats.value_samples[fuel])
                if len(val_samples) > 0:
                    val_std = np.std(val_samples)
                    if val_std > 0:
                        wd = wasserstein_distance(synth_samples, val_samples)
                        wd_normalized = wd / val_std
                    else:
                        wd = wasserstein_distance(synth_samples, val_samples)
                        wd_normalized = np.nan

                    ks_stat, ks_pvalue = ks_2samp(synth_samples, val_samples)

                    fuel_results['synth_val'] = {
                        'wasserstein': float(wd),
                        'wasserstein_normalized': float(wd_normalized) if not np.isnan(wd_normalized) else None,
                        'ks_statistic': float(ks_stat),
                        'ks_pvalue': float(ks_pvalue),
                        'n_val_samples': len(val_samples),
                    }

            # Compare to test data if available
            if self.real_test_stats and fuel in self.real_test_stats.value_samples:
                test_samples = np.array(self.real_test_stats.value_samples[fuel])
                if len(test_samples) > 0:
                    test_std = np.std(test_samples)
                    if test_std > 0:
                        wd = wasserstein_distance(synth_samples, test_samples)
                        wd_normalized = wd / test_std
                    else:
                        wd = wasserstein_distance(synth_samples, test_samples)
                        wd_normalized = np.nan

                    ks_stat, ks_pvalue = ks_2samp(synth_samples, test_samples)

                    fuel_results['synth_test'] = {
                        'wasserstein': float(wd),
                        'wasserstein_normalized': float(wd_normalized) if not np.isnan(wd_normalized) else None,
                        'ks_statistic': float(ks_stat),
                        'ks_pvalue': float(ks_pvalue),
                        'n_test_samples': len(test_samples),
                    }

            # Compare train to test (baseline) if both available
            if (self.real_train_stats and fuel in self.real_train_stats.value_samples and
                self.real_test_stats and fuel in self.real_test_stats.value_samples):
                train_samples = np.array(self.real_train_stats.value_samples[fuel])
                test_samples = np.array(self.real_test_stats.value_samples[fuel])
                if len(train_samples) > 0 and len(test_samples) > 0:
                    train_std = np.std(train_samples)
                    if train_std > 0:
                        wd = wasserstein_distance(train_samples, test_samples)
                        wd_normalized = wd / train_std
                    else:
                        wd = wasserstein_distance(train_samples, test_samples)
                        wd_normalized = np.nan

                    ks_stat, ks_pvalue = ks_2samp(train_samples, test_samples)

                    fuel_results['train_test'] = {
                        'wasserstein': float(wd),
                        'wasserstein_normalized': float(wd_normalized) if not np.isnan(wd_normalized) else None,
                        'ks_statistic': float(ks_stat),
                        'ks_pvalue': float(ks_pvalue),
                    }

            # Determine status based on KS test p-value
            # (If synth-train KS p-value > 0.05, distributions are not significantly different)
            if 'synth_train' not in fuel_results:
                fuel_results['status'] = 'no_comparison_data'
            elif fuel_results['synth_train']['ks_pvalue'] > 0.05:
                fuel_results['status'] = 'pass'
            elif fuel_results['synth_train']['ks_pvalue'] > 0.01:
                fuel_results['status'] = 'warn'
            else:
                fuel_results['status'] = 'fail'

            results[fuel] = fuel_results

        return results

    def _aggregate_correlation(self, batch_results: List[BatchResult]) -> Dict[str, Any]:
        """
        Aggregate pairwise correlation analysis.

        Compares elec-gas, temp-elec, and temp-gas correlations between
        synthetic and real data.

        Keys use {metric}_{split} naming for easy per-metric comparison:
            elec_gas_corr_synth, elec_gas_corr_train, elec_gas_corr_test, etc.
        """
        # Collect per-pair correlations from synthetic batches
        synth_corr_lists: Dict[str, list] = defaultdict(list)
        for br in batch_results:
            for key, val in br.correlations.items():
                synth_corr_lists[key].append(val)

        if not synth_corr_lists:
            return {'status': 'no_data'}

        results: Dict[str, Any] = {}
        pair_keys = ['elec_gas', 'temp_elec', 'temp_gas']
        worst_status = 'pass'

        for pair_key in pair_keys:
            synth_vals = synth_corr_lists.get(pair_key, [])
            if not synth_vals:
                continue

            synth_mean = float(np.mean(synth_vals))
            results[f'{pair_key}_corr_synth'] = synth_mean
            results[f'{pair_key}_corr_std_synth'] = float(np.std(synth_vals))
            results[f'{pair_key}_n_batch_correlations'] = len(synth_vals)

            # Compare to train data
            if self.real_train_stats and pair_key in self.real_train_stats.correlations:
                train_corr = self.real_train_stats.correlations[pair_key]
                results[f'{pair_key}_corr_train'] = train_corr
                diff = synth_mean - train_corr
                results[f'{pair_key}_diff_synth_train'] = diff

                if abs(diff) >= 0.2:
                    worst_status = 'fail'
                elif abs(diff) >= 0.1 and worst_status != 'fail':
                    worst_status = 'warn'

            # Compare to val data
            if self.real_val_stats and pair_key in self.real_val_stats.correlations:
                val_corr = self.real_val_stats.correlations[pair_key]
                results[f'{pair_key}_corr_val'] = val_corr
                results[f'{pair_key}_diff_synth_val'] = synth_mean - val_corr

            # Compare to test data
            if self.real_test_stats and pair_key in self.real_test_stats.correlations:
                test_corr = self.real_test_stats.correlations[pair_key]
                results[f'{pair_key}_corr_test'] = test_corr
                results[f'{pair_key}_diff_synth_test'] = synth_mean - test_corr

        # Determine overall status
        has_train = any(k.endswith('_corr_train') for k in results)
        if not has_train:
            results['status'] = 'no_comparison_data'
        else:
            results['status'] = worst_status

        return results

    def _aggregate_ptg(self, batch_results: List[BatchResult]) -> Dict[str, Any]:
        """
        Aggregate PTG (Power-Temperature Gradient) monthly sums across batches.

        For each split (synth, train, val, test), computes monthly mean
        consumption and monthly mean temperature for electricity and gas.
        """
        # Sum ptg_monthly across all synthetic batches
        synth_ptg: Dict[int, Dict[str, float]] = {}
        for m in range(1, 13):
            synth_ptg[m] = {'elec_sum': 0.0, 'elec_count': 0, 'gas_sum': 0.0, 'gas_count': 0,
                            'temp_sum': 0.0, 'temp_count': 0}

        for br in batch_results:
            for m, entry in br.ptg_monthly.items():
                for key in ['elec_sum', 'elec_count', 'gas_sum', 'gas_count', 'temp_sum', 'temp_count']:
                    synth_ptg[m][key] += entry.get(key, 0)

        def _ptg_to_means(ptg: Dict[int, Dict[str, float]]) -> Dict[str, Any]:
            """Convert ptg sums to monthly means per fuel."""
            result = {}
            for fuel_key, sum_key, count_key in [
                ('electricity', 'elec_sum', 'elec_count'),
                ('gas', 'gas_sum', 'gas_count'),
            ]:
                months = []
                mean_consumption = []
                mean_temp = []
                for m in sorted(ptg.keys()):
                    entry = ptg[m]
                    if entry[count_key] > 0 and entry['temp_count'] > 0:
                        months.append(m)
                        mean_consumption.append(entry[sum_key] / entry[count_key])
                        mean_temp.append(entry['temp_sum'] / entry['temp_count'])
                result[fuel_key] = {
                    'months': months,
                    'mean_consumption': mean_consumption,
                    'mean_temp': mean_temp,
                }
            return result

        results = {'synth': _ptg_to_means(synth_ptg)}

        # Real data splits
        for split_name, real_stats in [
            ('train', self.real_train_stats),
            ('val', self.real_val_stats),
            ('test', self.real_test_stats),
        ]:
            if real_stats and real_stats.ptg_monthly:
                results[split_name] = _ptg_to_means(real_stats.ptg_monthly)

        return results

    # --- Integrated Diagnostics: Reduce Phase Methods ---

    @staticmethod
    def _stats_with_ci(values: list) -> Dict[str, float]:
        """Compute mean, median, std, n, SE, and 95% CI from a list of values."""
        arr = np.array(values, dtype=float)
        arr = arr[~np.isnan(arr)]
        n = len(arr)
        if n == 0:
            return {'mean': np.nan, 'median': np.nan, 'std': np.nan,
                    'n': 0, 'se': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan}
        mean = float(np.mean(arr))
        median = float(np.median(arr))
        std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
        se = std / np.sqrt(n) if n > 0 else np.nan
        return {
            'mean': mean,
            'median': median,
            'std': std,
            'n': n,
            'se': float(se),
            'ci_lower': float(mean - 1.96 * se) if not np.isnan(se) else np.nan,
            'ci_upper': float(mean + 1.96 * se) if not np.isnan(se) else np.nan,
        }

    def _aggregate_seasonal_analysis(self, batch_results: List[BatchResult]) -> Dict[str, Any]:
        """
        Aggregate per-household monthly means across batches for seasonal analysis.

        Computes per-month statistics (mean, median, SE, CI) and collapse ratio.
        Compares synthetic vs real data splits.
        """
        results = {}

        for fuel in ['electricity', 'gas']:
            # Collect all household means per month across all batches
            monthly_means: Dict[int, List[float]] = defaultdict(list)
            for br in batch_results:
                fuel_data = br.seasonal_household_means.get(fuel, {})
                for month, means in fuel_data.items():
                    monthly_means[month].extend(means)

            if not monthly_means:
                continue

            # Compute per-month stats
            synth_monthly = {}
            for month in sorted(monthly_means.keys()):
                synth_monthly[month] = self._stats_with_ci(monthly_means[month])

            # Collapse ratio: month 12 / month 1
            collapse_ratio = np.nan
            if 12 in synth_monthly and 1 in synth_monthly:
                m1 = synth_monthly[1]['mean']
                m12 = synth_monthly[12]['mean']
                if m1 != 0 and not np.isnan(m1):
                    collapse_ratio = float(m12 / m1)

            fuel_result = {
                'synth': synth_monthly,
                'collapse_ratio': collapse_ratio,
            }

            # Real data comparison
            for split_name, real_stats in [
                ('train', self.real_train_stats),
                ('test', self.real_test_stats),
            ]:
                if real_stats and real_stats.seasonal_household_means.get(fuel):
                    real_monthly = {}
                    for month, means in real_stats.seasonal_household_means[fuel].items():
                        real_monthly[month] = self._stats_with_ci(means)
                    fuel_result[split_name] = real_monthly

            results[fuel] = fuel_result

        return results

    def _aggregate_monthly_profiles(self, batch_results: List[BatchResult]) -> Dict[str, Any]:
        """
        Aggregate monthly × half-hour profile contributions for heatmap analysis.

        Combines (sum, count, sum_sq) across batches to produce per-cell
        mean, SE, and 95% CI for the 12×48 heatmap.
        """
        results = {}

        for fuel in ['electricity', 'gas']:
            # Combine contributions: {month: {half_hour: [sum, count, sum_sq]}}
            combined: Dict[int, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(lambda: [0.0, 0, 0.0]))

            for br in batch_results:
                fuel_data = br.monthly_profile_contributions.get(fuel, {})
                for month, hh_dict in fuel_data.items():
                    for hh, (s, n, s2) in hh_dict.items():
                        entry = combined[month][hh]
                        entry[0] += s
                        entry[1] += n
                        entry[2] += s2

            if not combined:
                continue

            # Compute per-cell statistics
            heatmap_mean = {}
            heatmap_se = {}
            heatmap_n = {}
            for month in sorted(combined.keys()):
                heatmap_mean[month] = {}
                heatmap_se[month] = {}
                heatmap_n[month] = {}
                for hh in sorted(combined[month].keys()):
                    s, n, s2 = combined[month][hh]
                    if n > 0:
                        mean = s / n
                        variance = (s2 / n) - mean ** 2
                        variance = max(variance, 0.0)  # Guard against float rounding
                        se = np.sqrt(variance / n) if n > 1 else 0.0
                        heatmap_mean[month][hh] = float(mean)
                        heatmap_se[month][hh] = float(se)
                        heatmap_n[month][hh] = n

            fuel_result = {
                'synth': {
                    'mean': heatmap_mean,
                    'se': heatmap_se,
                    'n': heatmap_n,
                },
            }

            # Real data comparison
            for split_name, real_stats in [('train', self.real_train_stats), ('test', self.real_test_stats)]:
                if real_stats and real_stats.monthly_profile_contributions.get(fuel):
                    real_combined: Dict[int, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(lambda: [0.0, 0, 0.0]))
                    for month, hh_dict in real_stats.monthly_profile_contributions[fuel].items():
                        for hh, (s, n, s2) in hh_dict.items():
                            entry = real_combined[month][hh]
                            entry[0] += s
                            entry[1] += n
                            entry[2] += s2

                    real_mean = {}
                    real_se = {}
                    real_n = {}
                    for month in sorted(real_combined.keys()):
                        real_mean[month] = {}
                        real_se[month] = {}
                        real_n[month] = {}
                        for hh in sorted(real_combined[month].keys()):
                            s, n, s2 = real_combined[month][hh]
                            if n > 0:
                                mean = s / n
                                variance = max((s2 / n) - mean ** 2, 0.0)
                                se = np.sqrt(variance / n) if n > 1 else 0.0
                                real_mean[month][hh] = float(mean)
                                real_se[month][hh] = float(se)
                                real_n[month][hh] = n

                    fuel_result[split_name] = {'mean': real_mean, 'se': real_se, 'n': real_n}

            results[fuel] = fuel_result

        return results

    def _aggregate_weekly_profiles(self, batch_results: List[BatchResult],
                                   diag_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggregate weekly profile contributions and compute degradation metrics.

        Computes CV ratio, profile correlation, peak-to-trough ratio per week.
        """
        results = {}

        for fuel in ['electricity', 'gas']:
            # Combine: {week: {half_hour: [sum, count, sum_sq]}}
            combined: Dict[int, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(lambda: [0.0, 0, 0.0]))

            for br in batch_results:
                fuel_data = br.weekly_profile_contributions.get(fuel, {})
                for week, hh_dict in fuel_data.items():
                    for hh, (s, n, s2) in hh_dict.items():
                        entry = combined[week][hh]
                        entry[0] += s
                        entry[1] += n
                        entry[2] += s2

            if not combined:
                continue

            # Compute per-week profile arrays (48 half-hours)
            weekly_profiles = {}
            for week in sorted(combined.keys()):
                profile = np.full(48, np.nan)
                for hh in range(48):
                    if hh in combined[week]:
                        s, n, s2 = combined[week][hh]
                        if n > 0:
                            profile[hh] = s / n
                weekly_profiles[week] = profile

            # Degradation metrics
            metrics = []
            ref_profile = weekly_profiles.get(1)  # Week 1 as reference
            for week in sorted(weekly_profiles.keys()):
                profile = weekly_profiles[week]
                valid = ~np.isnan(profile)
                if not np.any(valid):
                    continue

                p = profile[valid]
                cv = float(np.std(p) / np.mean(p)) if np.mean(p) != 0 else 0.0
                pt_ratio = float(np.max(p) / np.min(p)) if np.min(p) > 0 else np.nan

                corr = np.nan
                if ref_profile is not None:
                    ref_valid = ~np.isnan(ref_profile)
                    both_valid = valid & ref_valid
                    if np.sum(both_valid) > 2:
                        r = np.corrcoef(profile[both_valid], ref_profile[both_valid])
                        corr = float(r[0, 1]) if not np.isnan(r[0, 1]) else np.nan

                metrics.append({
                    'week': week,
                    'cv': cv,
                    'peak_to_trough': pt_ratio,
                    'correlation_to_week1': corr,
                    'mean': float(np.nanmean(profile)),
                    'n_valid_halfhours': int(np.sum(valid)),
                })

            fuel_result = {
                'synth': {
                    'weekly_profiles': {w: p.tolist() for w, p in weekly_profiles.items()},
                    'degradation_metrics': metrics,
                },
                'context_length': diag_config.get('daily_patterns', {}).get('context_length'),
            }

            # Compute same metrics for real data (train/test)
            for split_name, real_stats in [('train', self.real_train_stats), ('test', self.real_test_stats)]:
                if not real_stats:
                    continue
                real_combined: Dict[int, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(lambda: [0.0, 0, 0.0]))
                real_fuel_data = real_stats.weekly_profile_contributions.get(fuel, {})
                for week, hh_dict in real_fuel_data.items():
                    for hh, (s, n, s2) in hh_dict.items():
                        entry = real_combined[week][hh]
                        entry[0] += s
                        entry[1] += n
                        entry[2] += s2

                if not real_combined:
                    continue

                real_weekly_profiles = {}
                for week in sorted(real_combined.keys()):
                    profile = np.full(48, np.nan)
                    for hh in range(48):
                        if hh in real_combined[week]:
                            s, n, s2 = real_combined[week][hh]
                            if n > 0:
                                profile[hh] = s / n
                    real_weekly_profiles[week] = profile

                real_metrics = []
                real_ref = real_weekly_profiles.get(1)
                for week in sorted(real_weekly_profiles.keys()):
                    profile = real_weekly_profiles[week]
                    valid = ~np.isnan(profile)
                    if not np.any(valid):
                        continue
                    p = profile[valid]
                    cv = float(np.std(p) / np.mean(p)) if np.mean(p) != 0 else 0.0
                    pt_ratio = float(np.max(p) / np.min(p)) if np.min(p) > 0 else np.nan
                    corr = np.nan
                    if real_ref is not None:
                        ref_valid = ~np.isnan(real_ref)
                        both_valid = valid & ref_valid
                        if np.sum(both_valid) > 2:
                            r = np.corrcoef(profile[both_valid], real_ref[both_valid])
                            corr = float(r[0, 1]) if not np.isnan(r[0, 1]) else np.nan
                    real_metrics.append({
                        'week': week, 'cv': cv, 'peak_to_trough': pt_ratio,
                        'correlation_to_week1': corr, 'mean': float(np.nanmean(profile)),
                        'n_valid_halfhours': int(np.sum(valid)),
                    })

                fuel_result[split_name] = {
                    'weekly_profiles': {w: p.tolist() for w, p in real_weekly_profiles.items()},
                    'degradation_metrics': real_metrics,
                }

            results[fuel] = fuel_result

        return results

    def _aggregate_weekly_fidelity(self, batch_results: List[BatchResult]) -> Dict[str, Any]:
        """
        Aggregate weekly FuelStats for within-month fidelity comparison.

        Combines weekly FuelStats across batches, compares synth vs real,
        computes week-over-week drift.
        """
        results = {}

        for fuel in ['electricity', 'gas']:
            # Collect FuelStats per week across batches
            weekly_stats_lists: Dict[int, List[FuelStats]] = defaultdict(list)
            for br in batch_results:
                fuel_data = br.weekly_fuel_stats.get(fuel, {})
                for week, fs in fuel_data.items():
                    weekly_stats_lists[week].append(fs)

            if not weekly_stats_lists:
                continue

            # Combine per week
            synth_weekly = {}
            for week in sorted(weekly_stats_lists.keys()):
                combined = FuelStats.combine(weekly_stats_lists[week])
                synth_weekly[week] = {
                    'mean': combined.mean,
                    'std': combined.std,
                    'n': combined.n_values,
                    'se': float(combined.std / np.sqrt(combined.n_values))
                          if combined.n_values > 0 and not np.isnan(combined.std) else np.nan,
                    'quantiles': combined.quantiles,
                }

            # Week-over-week drift (ratio to week 1)
            week1_mean = synth_weekly.get(1, {}).get('mean', np.nan)
            for week, stats in synth_weekly.items():
                if week1_mean and not np.isnan(week1_mean) and week1_mean != 0:
                    stats['ratio_to_week1'] = float(stats['mean'] / week1_mean)
                else:
                    stats['ratio_to_week1'] = np.nan

            fuel_result = {'synth': synth_weekly}

            # Real data comparison
            for split_name, real_stats in [('train', self.real_train_stats), ('test', self.real_test_stats)]:
                if real_stats and real_stats.weekly_fuel_stats.get(fuel):
                    real_weekly = {}
                    for week, fs in real_stats.weekly_fuel_stats[fuel].items():
                        real_weekly[week] = {
                            'mean': fs.mean,
                            'std': fs.std,
                            'n': fs.n_values,
                        }
                    fuel_result[split_name] = real_weekly

            results[fuel] = fuel_result

        return results

    def _aggregate_household_month_fidelity(
        self, batch_results: List[BatchResult], quintiles: List[int] = None
    ) -> Dict[str, Any]:
        """
        Aggregate household-month-mean level fidelity from existing household_month_stats.

        Reuses the household_month_stats already computed for distance analysis.
        Computes quintile comparison and conditioning-group breakdown with CIs.
        """
        if quintiles is None:
            quintiles = [20, 40, 60, 80]

        results = {}

        # Collect all household-month stats from synthetic batches
        synth_stats = []
        for br in batch_results:
            synth_stats.extend(br.household_month_stats)

        if not synth_stats:
            return results

        # Extract means per fuel, filtering by fuel_type field
        for fuel in ['electricity', 'gas']:
            synth_means = []
            synth_hh_ids: set = set()
            for s in synth_stats:
                # Filter: only include stats matching this fuel type
                stat_fuel = s.get('fuel_type', '')
                if stat_fuel and stat_fuel != fuel:
                    continue
                val = s.get('mean')
                if val is not None and not np.isnan(val):
                    synth_means.append(float(val))
                    synth_hh_ids.add(s.get('household_id'))

            if not synth_means:
                continue

            # Quintile comparison
            synth_arr = np.array(synth_means)
            synth_quintiles = {f'Q{q}': float(np.percentile(synth_arr, q)) for q in quintiles}
            synth_quintiles['mean'] = self._stats_with_ci(synth_means)
            synth_quintiles['n_households'] = len(synth_hh_ids)

            fuel_result = {'synth': synth_quintiles}

            # Real data comparison
            for split_name, real_stats in [('train', self.real_train_stats), ('val', self.real_val_stats), ('test', self.real_test_stats)]:
                if real_stats and real_stats.household_month_stats:
                    real_means = []
                    real_hh_ids: set = set()
                    for s in real_stats.household_month_stats:
                        stat_fuel = s.get('fuel_type', '')
                        if stat_fuel and stat_fuel != fuel:
                            continue
                        val = s.get('mean')
                        if val is not None and not np.isnan(val):
                            real_means.append(float(val))
                            real_hh_ids.add(s.get('household_id'))
                    if real_means:
                        real_arr = np.array(real_means)
                        real_quintiles = {f'Q{q}': float(np.percentile(real_arr, q)) for q in quintiles}
                        real_quintiles['mean'] = self._stats_with_ci(real_means)
                        real_quintiles['n_households'] = len(real_hh_ids)
                        fuel_result[split_name] = real_quintiles

            # Conditioning-group breakdown
            conditioning_results = {}
            for cond_var in self.conditioning_vars:
                groups: Dict[Any, List[float]] = defaultdict(list)
                group_hh_ids: Dict[Any, set] = defaultdict(set)
                for s in synth_stats:
                    stat_fuel = s.get('fuel_type', '')
                    if stat_fuel and stat_fuel != fuel:
                        continue
                    cond_val = s.get(cond_var)
                    val = s.get('mean')
                    if cond_val is not None and val is not None and not np.isnan(val):
                        groups[cond_val].append(float(val))
                        group_hh_ids[cond_val].add(s.get('household_id'))

                if groups:
                    group_stats = {}
                    for cond_val, vals in sorted(groups.items(), key=lambda x: str(x[0])):
                        stats = self._stats_with_ci(vals)
                        stats['n_households'] = len(group_hh_ids[cond_val])
                        group_stats[str(cond_val)] = stats
                    conditioning_results[cond_var] = {'synth': group_stats}

                    # Real data conditioning for train / val / test
                    for split_name, split_stats in [
                        ('train', self.real_train_stats),
                        ('val',   self.real_val_stats),
                        ('test',  self.real_test_stats),
                    ]:
                        if not (split_stats and split_stats.household_month_stats):
                            continue
                        real_groups: Dict[Any, List[float]] = defaultdict(list)
                        real_group_hh_ids: Dict[Any, set] = defaultdict(set)
                        for s in split_stats.household_month_stats:
                            stat_fuel = s.get('fuel_type', '')
                            if stat_fuel and stat_fuel != fuel:
                                continue
                            cond_val = s.get(cond_var)
                            val = s.get('mean')
                            if cond_val is not None and val is not None and not np.isnan(val):
                                real_groups[cond_val].append(float(val))
                                real_group_hh_ids[cond_val].add(s.get('household_id'))
                        if real_groups:
                            real_group_stats = {}
                            for cond_val, vals in sorted(real_groups.items(), key=lambda x: str(x[0])):
                                stats = self._stats_with_ci(vals)
                                stats['n_households'] = len(real_group_hh_ids[cond_val])
                                real_group_stats[str(cond_val)] = stats
                            conditioning_results[cond_var][split_name] = real_group_stats

            fuel_result['conditioning'] = conditioning_results
            results[fuel] = fuel_result

        return results

    def _aggregate_token_analysis(self, batch_results: List[BatchResult]) -> Dict[str, Any]:
        """
        Aggregate token frequency, distribution, and weekly cascade analyses.

        Covers diagnostics 4 (token distribution), 5 (token frequency),
        and 7 (weekly token cascade).
        """
        results = {}

        for fuel in ['electricity', 'gas']:
            fuel_result = {}

            # --- Token Frequency (diagnostic 5) ---
            freq_lists = [br.token_frequency.get(fuel) for br in batch_results
                          if br.token_frequency.get(fuel)]
            if freq_lists:
                n_bins = len(freq_lists[0])
                total_counts = [0] * n_bins
                for counts in freq_lists:
                    for i, c in enumerate(counts):
                        if i < n_bins:
                            total_counts[i] += c

                total = sum(total_counts)
                frequencies = [c / total if total > 0 else 0.0 for c in total_counts]
                uniform = 1.0 / n_bins if n_bins > 0 else 0.0
                deviation = [f - uniform for f in frequencies]

                # CDF
                cdf = np.cumsum(frequencies).tolist()

                # Shannon entropy
                entropy = 0.0
                for f in frequencies:
                    if f > 0:
                        entropy -= f * np.log2(f)

                fuel_result['frequency'] = {
                    'counts': total_counts,
                    'frequencies': frequencies,
                    'deviation_from_uniform': deviation,
                    'cdf': cdf,
                    'entropy': float(entropy),
                    'max_entropy': float(np.log2(n_bins)) if n_bins > 0 else 0.0,
                    'n_tokens_total': total,
                    'n_bins': n_bins,
                }

                # Real data comparison
                for split_name, real_stats in [('train', self.real_train_stats), ('test', self.real_test_stats)]:
                    if real_stats and real_stats.token_frequency.get(fuel):
                        real_counts = real_stats.token_frequency[fuel]
                        real_total = sum(real_counts)
                        real_freq = [c / real_total if real_total > 0 else 0.0 for c in real_counts]
                        real_cdf = np.cumsum(real_freq).tolist()
                        fuel_result['frequency'][f'{split_name}_counts'] = real_counts
                        fuel_result['frequency'][f'{split_name}_n_tokens_total'] = real_total
                        fuel_result['frequency'][f'{split_name}_frequencies'] = real_freq
                        fuel_result['frequency'][f'{split_name}_cdf'] = real_cdf
                        # Max CDF gap
                        if len(cdf) == len(real_cdf):
                            max_gap = float(max(abs(a - b) for a, b in zip(cdf, real_cdf)))
                            fuel_result['frequency'][f'{split_name}_max_cdf_gap'] = max_gap

            # --- Monthly Decoded Stats (diagnostic 4) ---
            monthly_combined: Dict[int, Dict[str, float]] = defaultdict(lambda: {
                'sum_mean': 0.0, 'sum_median': 0.0, 'sum_n': 0,
                'sum_zero_n': 0.0, 'total_n': 0
            })
            for br in batch_results:
                fuel_monthly = br.monthly_decoded_stats.get(fuel, {})
                for month, stats in fuel_monthly.items():
                    entry = monthly_combined[month]
                    n = stats.get('n', 0)
                    entry['sum_mean'] += stats.get('mean', 0) * n
                    entry['sum_median'] += stats.get('median', 0) * n
                    entry['sum_zero_n'] += stats.get('zero_rate', 0) * n
                    entry['total_n'] += n
                    entry['sum_n'] += 1

            if monthly_combined:
                monthly_stats = {}
                for month in sorted(monthly_combined.keys()):
                    e = monthly_combined[month]
                    n = e['total_n']
                    if n > 0:
                        monthly_stats[month] = {
                            'mean': float(e['sum_mean'] / n),
                            'median': float(e['sum_median'] / n),
                            'zero_rate': float(e['sum_zero_n'] / n),
                            'n': n,
                        }
                fuel_result['monthly_decoded'] = monthly_stats

            # --- Weekly Token Cascade (diagnostic 7) ---
            weekly_freq_lists: Dict[int, List[List[int]]] = defaultdict(list)
            for br in batch_results:
                fuel_weekly = br.weekly_token_frequency.get(fuel, {})
                for week, counts in fuel_weekly.items():
                    weekly_freq_lists[week].append(counts)

            if weekly_freq_lists:
                weekly_cascade = {}
                for week in sorted(weekly_freq_lists.keys()):
                    counts_lists = weekly_freq_lists[week]
                    if not counts_lists:
                        continue
                    n_bins = len(counts_lists[0])
                    total_counts = [0] * n_bins
                    for counts in counts_lists:
                        for i, c in enumerate(counts):
                            if i < n_bins:
                                total_counts[i] += c

                    total = sum(total_counts)
                    freqs = [c / total if total > 0 else 0.0 for c in total_counts]

                    # Cascade metrics
                    freq_arr = np.array(freqs)
                    freq_std = float(np.std(freq_arr))
                    entropy = 0.0
                    for f in freqs:
                        if f > 0:
                            entropy -= f * np.log2(f)

                    # Third-mass decomposition
                    third = n_bins // 3
                    lower_mass = float(sum(freqs[:third]))
                    middle_mass = float(sum(freqs[third:2*third]))
                    upper_mass = float(sum(freqs[2*third:]))

                    weekly_cascade[week] = {
                        'counts': total_counts,
                        'frequencies': freqs,
                        'freq_std': freq_std,
                        'entropy': float(entropy),
                        'lower_third_mass': lower_mass,
                        'middle_third_mass': middle_mass,
                        'upper_third_mass': upper_mass,
                        'n_tokens_total': total,
                    }

                # Two-mechanism decomposition (if we have real data baseline)
                real_freq = fuel_result.get('frequency', {}).get('train_frequencies')
                if real_freq and 1 in weekly_cascade and len(weekly_cascade) > 1:
                    last_week = max(weekly_cascade.keys())
                    week1_freqs = weekly_cascade[1]['frequencies']
                    weekN_freqs = weekly_cascade[last_week]['frequencies']

                    mechanism1 = [w1 - r for w1, r in zip(week1_freqs, real_freq)]
                    mechanism2 = [wn - w1 for wn, w1 in zip(weekN_freqs, week1_freqs)]

                    weekly_cascade['decomposition'] = {
                        'mechanism1_learned_bias': mechanism1,
                        'mechanism2_cascade_drift': mechanism2,
                        'mechanism1_magnitude': float(np.sum(np.abs(mechanism1))),
                        'mechanism2_magnitude': float(np.sum(np.abs(mechanism2))),
                    }

                fuel_result['weekly_cascade'] = weekly_cascade

            if fuel_result:
                results[fuel] = fuel_result

        # --- Value samples for distribution histograms ---
        # Collect synth + real value samples (already reservoir-sampled, manageable size)
        for fuel in ['electricity', 'gas']:
            if fuel not in results:
                results[fuel] = {}
            synth_samples = []
            for br in batch_results:
                if fuel in br.value_samples:
                    synth_samples.extend(br.value_samples[fuel])
            if synth_samples:
                results[fuel]['value_samples_synth'] = synth_samples
            if self.real_train_stats and fuel in self.real_train_stats.value_samples:
                results[fuel]['value_samples_real'] = list(self.real_train_stats.value_samples[fuel])

        # --- Bin edges for bin width analysis ---
        tokeniser_path = self.config.get('tokeniser_file')
        if tokeniser_path:
            try:
                from .batch_processor import BatchProcessor
                bin_info = BatchProcessor._load_tokeniser_bin_edges(tokeniser_path)
                if bin_info:
                    value_transform = bin_info.get('value_transform')
                    for fuel, edges in bin_info.get('bin_edges', {}).items():
                        if fuel not in results:
                            results[fuel] = {}
                        # Convert edges to Wh space for bin width display
                        edges_wh = np.expm1(edges) if value_transform == 'log1p' else edges
                        bin_widths = np.diff(edges_wh).tolist()
                        results[fuel]['bin_edges_wh'] = edges_wh.tolist()
                        results[fuel]['bin_widths_wh'] = bin_widths
                        results[fuel]['value_transform'] = value_transform
            except Exception as e:
                print(f"[ResultsAggregator] Warning: Failed to load tokeniser for bin edges: {e}")

        return results
