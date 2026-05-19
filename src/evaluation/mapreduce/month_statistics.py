"""
Month-stratified summary statistics for MapReduce evaluation pipeline.

Computes summary statistics adapted for month-chunk data format, where each
household-month is an independent sample. Removes features that require
cross-month data (e.g., seasonality ratio).

Based on: src/evaluation/summary_statistics.py
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import warnings


class MonthStatisticsCalculator:
    """
    Calculate summary statistics for household-month data.

    Adapted from SummaryStatisticsCalculator for month-chunk format:
    - All statistics computed within single month boundaries
    - Removed seasonality_ratio (requires cross-month data)
    - Added month identifier to output
    """

    # Column name mappings for synthetic data
    ELEC_SUFFIX = '_elec_net_Wh'
    GAS_SUFFIX = '_gas_Wh'

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize calculator.

        Args:
            config: Configuration dictionary with:
                - quantiles: List of quantile values (default: [0.1, 0.25, 0.5, 0.75, 0.9])
                - rolling_window_days: Window for volatility (default: 1)
                - time_bins: Time-of-day bins (default: night/morning/afternoon/evening)
        """
        self.config = config or {}
        self.quantiles = self.config.get('quantiles', [0.1, 0.25, 0.5, 0.75, 0.9])
        self.rolling_window_days = self.config.get('rolling_window_days', 1)
        self.rolling_window = self.rolling_window_days * 48  # Half-hours

        self.time_bins = self.config.get('time_bins', {
            'night': [0, 6],
            'morning': [6, 12],
            'afternoon': [12, 18],
            'evening': [18, 24]
        })

    def calculate_household_month_stats(
        self,
        df: pd.DataFrame,
        household_col: str,
        month: int,
        household_id: str,
        fuel_type: str = 'electricity'
    ) -> Dict[str, Any]:
        """
        Calculate all summary statistics for one household-month.

        Args:
            df: Month DataFrame with household columns
            household_col: Column name for this household's fuel data
            month: Month number (1-12)
            household_id: Unique identifier for this household
            fuel_type: 'electricity' or 'gas'

        Returns:
            Dictionary with all statistics and metadata
        """
        series = df[household_col].dropna()

        stats = {
            'household_id': household_id,
            'month': month,
            'fuel_type': fuel_type,
            'n_values': len(series),
        }

        # Skip if no valid data
        if len(series) == 0:
            stats['has_valid_data'] = False
            return stats

        stats['has_valid_data'] = True

        # Distributional statistics
        stats.update(self._calc_distributional_stats(series))

        # Timing statistics (need half_hour column)
        if 'half_hour' in df.columns:
            stats.update(self._calc_timing_stats(df, household_col))

        # Temporal dynamics (first differences, volatility)
        stats.update(self._calc_temporal_stats(series))

        # Load characteristics
        stats.update(self._calc_load_stats(df, household_col))

        # Data quality
        stats.update(self._calc_data_quality_stats(df[household_col]))

        return stats

    def _calc_distributional_stats(self, series: pd.Series) -> Dict[str, float]:
        """Calculate distributional statistics."""
        if len(series) == 0:
            return {
                'mean': np.nan, 'std': np.nan, 'max': np.nan, 'min': np.nan,
                'skew': np.nan, 'kurtosis': np.nan
            }

        stats = {
            'mean': float(series.mean()),
            'std': float(series.std()),
            'max': float(series.max()),
            'min': float(series.min()),
            'skew': float(series.skew()) if len(series) > 2 else np.nan,
            'kurtosis': float(series.kurtosis()) if len(series) > 3 else np.nan,
        }

        # Add quantiles
        for q in self.quantiles:
            q_name = f"q{int(q * 100)}"
            stats[q_name] = float(series.quantile(q))

        return stats

    def _calc_timing_stats(self, df: pd.DataFrame, column: str) -> Dict[str, float]:
        """Calculate timing-related statistics."""
        stats = {}

        # Average consumption by half-hour
        if 'half_hour' in df.columns:
            period_avg = df.groupby('half_hour')[column].mean()

            if len(period_avg) > 0 and not period_avg.isna().all():
                # Peak hour (convert half-hour 0-47 to hour 0-23)
                peak_hh = period_avg.idxmax()
                stats['peak_hour'] = int(peak_hh // 2) if not pd.isna(peak_hh) else np.nan

                # Time-of-day averages
                for bin_name, (start_hour, end_hour) in self.time_bins.items():
                    start_hh = start_hour * 2
                    end_hh = end_hour * 2
                    bin_periods = period_avg.loc[
                        (period_avg.index >= start_hh) & (period_avg.index < end_hh)
                    ]
                    stats[f'{bin_name}_avg'] = float(bin_periods.mean()) if len(bin_periods) > 0 else np.nan
            else:
                stats['peak_hour'] = np.nan
                for bin_name in self.time_bins:
                    stats[f'{bin_name}_avg'] = np.nan
        else:
            stats['peak_hour'] = np.nan
            for bin_name in self.time_bins:
                stats[f'{bin_name}_avg'] = np.nan

        return stats

    def _calc_temporal_stats(self, series: pd.Series) -> Dict[str, float]:
        """Calculate temporal dynamics statistics."""
        stats = {}

        # First differences
        if len(series) > 1:
            first_diff = series.diff().dropna()
            stats['first_diff_mean'] = float(first_diff.mean())
            stats['first_diff_std'] = float(first_diff.std())
        else:
            stats['first_diff_mean'] = np.nan
            stats['first_diff_std'] = np.nan

        # Rolling volatility
        if len(series) >= self.rolling_window:
            rolling_vol = series.rolling(window=self.rolling_window).std().dropna()
            stats['rolling_volatility_mean'] = float(rolling_vol.mean()) if len(rolling_vol) > 0 else np.nan
            stats['rolling_volatility_std'] = float(rolling_vol.std()) if len(rolling_vol) > 0 else np.nan
        else:
            stats['rolling_volatility_mean'] = np.nan
            stats['rolling_volatility_std'] = np.nan

        return stats

    def _calc_load_stats(self, df: pd.DataFrame, column: str) -> Dict[str, float]:
        """Calculate load characteristics."""
        stats = {}

        series = df[column].dropna()
        if len(series) == 0:
            return {'load_factor': np.nan, 'weekday_weekend_ratio': np.nan}

        # Load factor: average / peak
        avg_load = series.mean()
        peak_load = series.max()
        stats['load_factor'] = float(avg_load / peak_load) if peak_load > 0 else np.nan

        # Weekday/weekend ratio
        if 'dow' in df.columns:
            valid_df = df[df[column].notna()]
            weekday_vals = valid_df[valid_df['dow'] < 5][column]
            weekend_vals = valid_df[valid_df['dow'] >= 5][column]

            weekday_avg = weekday_vals.mean() if len(weekday_vals) > 0 else np.nan
            weekend_avg = weekend_vals.mean() if len(weekend_vals) > 0 else np.nan

            if not np.isnan(weekend_avg) and weekend_avg > 0:
                stats['weekday_weekend_ratio'] = float(weekday_avg / weekend_avg)
            else:
                stats['weekday_weekend_ratio'] = np.nan
        else:
            stats['weekday_weekend_ratio'] = np.nan

        # Note: seasonality_ratio removed - requires cross-month data

        return stats

    def _calc_data_quality_stats(self, series: pd.Series) -> Dict[str, float]:
        """Calculate data quality indicators."""
        total = len(series)
        if total == 0:
            return {'missing_pct': 100.0, 'zeros_pct': 0.0}

        missing = series.isna().sum()
        zeros = (series == 0).sum()

        return {
            'missing_pct': float(missing / total * 100),
            'zeros_pct': float(zeros / total * 100),
        }

    def get_feature_names(self) -> List[str]:
        """Get ordered list of feature names for vector creation."""
        features = [
            # Distributional
            'mean', 'std', 'max', 'min', 'skew', 'kurtosis',
        ]

        # Add quantile names
        for q in self.quantiles:
            features.append(f"q{int(q * 100)}")

        # Timing
        features.append('peak_hour')
        for bin_name in ['night', 'morning', 'afternoon', 'evening']:
            features.append(f'{bin_name}_avg')

        # Temporal
        features.extend([
            'first_diff_mean', 'first_diff_std',
            'rolling_volatility_mean', 'rolling_volatility_std'
        ])

        # Load
        features.extend(['load_factor', 'weekday_weekend_ratio'])

        # Data quality
        features.extend(['missing_pct', 'zeros_pct'])

        return features

    def stats_to_vector(self, stats: Dict[str, Any]) -> np.ndarray:
        """
        Convert statistics dictionary to feature vector.

        Args:
            stats: Dictionary from calculate_household_month_stats()

        Returns:
            Numpy array with features in consistent order
        """
        feature_names = self.get_feature_names()
        vector = []

        for name in feature_names:
            val = stats.get(name, np.nan)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                vector.append(np.nan)
            else:
                vector.append(float(val))

        return np.array(vector)


def compute_all_household_month_stats(
    month_data: Dict[int, pd.DataFrame],
    batch_id: str,
    config: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Compute full summary statistics for all household-months in a batch.

    Args:
        month_data: Dict mapping month number to DataFrame
        batch_id: Batch identifier
        config: Optional configuration

    Returns:
        List of statistics dictionaries, one per household-month-fuel
    """
    calculator = MonthStatisticsCalculator(config)
    all_stats = []

    elec_suffix = MonthStatisticsCalculator.ELEC_SUFFIX
    gas_suffix = MonthStatisticsCalculator.GAS_SUFFIX

    for month, df in month_data.items():
        # Find household columns
        elec_cols = sorted([c for c in df.columns if c.endswith(elec_suffix)])
        gas_cols = sorted([c for c in df.columns if c.endswith(gas_suffix)])

        for elec_col in elec_cols:
            hh_prefix = elec_col.replace(elec_suffix, '')
            household_id = f"{batch_id}_{hh_prefix}"

            # Electricity stats
            elec_stats = calculator.calculate_household_month_stats(
                df=df,
                household_col=elec_col,
                month=month,
                household_id=household_id,
                fuel_type='electricity'
            )
            all_stats.append(elec_stats)

            # Gas stats (if corresponding column exists)
            gas_col = hh_prefix + gas_suffix
            if gas_col in df.columns:
                gas_stats = calculator.calculate_household_month_stats(
                    df=df,
                    household_col=gas_col,
                    month=month,
                    household_id=household_id,
                    fuel_type='gas'
                )
                all_stats.append(gas_stats)

    return all_stats
