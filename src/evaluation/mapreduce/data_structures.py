"""
Data structures for MapReduce evaluation pipeline.

These dataclasses define the intermediate and final results for the
parallel evaluation pipeline. They are designed for efficient aggregation
using weighted formulas (Chan's parallel algorithm for variance, etc.).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import numpy as np


@dataclass
class MissingnessStats:
    """
    Statistics about missing values and zeros for one fuel type.

    Tracks both NaN missingness and zero values, which are important
    for gas data where zeros may indicate no connection or no usage.
    """
    # Total values (including NaN and zeros)
    n_total: int = 0

    # Missing (NaN) statistics
    n_missing: int = 0

    # Zero statistics
    n_zeros: int = 0

    # Structural missingness (households with ALL values missing for this fuel)
    n_households_no_data: int = 0
    n_households_total: int = 0

    # Consecutive run statistics (for temporal patterns)
    # Stored as histogram: run_length -> count
    missing_run_lengths: Dict[int, int] = field(default_factory=dict)
    zero_run_lengths: Dict[int, int] = field(default_factory=dict)

    @property
    def missing_pct(self) -> float:
        """Percentage of values that are NaN."""
        return (self.n_missing / self.n_total * 100) if self.n_total > 0 else 0.0

    @property
    def zeros_pct(self) -> float:
        """Percentage of values that are zero."""
        return (self.n_zeros / self.n_total * 100) if self.n_total > 0 else 0.0

    @property
    def structural_missing_pct(self) -> float:
        """Percentage of households with no data for this fuel."""
        return (self.n_households_no_data / self.n_households_total * 100) if self.n_households_total > 0 else 0.0

    @classmethod
    def combine(cls, stats_list: List['MissingnessStats']) -> 'MissingnessStats':
        """Combine multiple MissingnessStats."""
        if not stats_list:
            return cls()

        combined = cls(
            n_total=sum(s.n_total for s in stats_list),
            n_missing=sum(s.n_missing for s in stats_list),
            n_zeros=sum(s.n_zeros for s in stats_list),
            n_households_no_data=sum(s.n_households_no_data for s in stats_list),
            n_households_total=sum(s.n_households_total for s in stats_list),
        )

        # Combine run length histograms
        for s in stats_list:
            for length, count in s.missing_run_lengths.items():
                combined.missing_run_lengths[length] = combined.missing_run_lengths.get(length, 0) + count
            for length, count in s.zero_run_lengths.items():
                combined.zero_run_lengths[length] = combined.zero_run_lengths.get(length, 0) + count

        return combined


@dataclass
class FuelStats:
    """
    Aggregatable summary statistics for one fuel type.

    Designed to be combined across batches using weighted formulas:
    - Mean: sum_values / n_values
    - Variance: Chan's parallel algorithm using sum_squared
    - Quantiles: Weighted average (approximation)

    All statistics are computed respecting month boundaries.
    """
    n_values: int                    # Total number of non-NaN values
    n_days: int                      # Total days (for weighting)
    sum_values: float                # Sum for weighted mean
    sum_squared: float               # Sum of squares for variance
    min_value: float
    max_value: float

    # Quantiles (approximate - will be averaged across batches)
    quantiles: Dict[str, float] = field(default_factory=dict)
    # {'p5': x, 'p10': x, 'p25': x, 'p50': x, 'p75': x, 'p90': x, 'p95': x}

    # Per-month sums and counts (for seasonal analysis)
    monthly_stats: Dict[int, Tuple[float, int]] = field(default_factory=dict)
    # month -> (sum, count)

    @property
    def mean(self) -> float:
        """Compute mean from sum and count."""
        if self.n_values == 0:
            return np.nan
        return self.sum_values / self.n_values

    @property
    def variance(self) -> float:
        """Compute variance from sum of squares."""
        if self.n_values == 0:
            return np.nan
        mean = self.mean
        return (self.sum_squared / self.n_values) - mean ** 2

    @property
    def std(self) -> float:
        """Compute standard deviation."""
        var = self.variance
        return np.sqrt(var) if not np.isnan(var) and var >= 0 else np.nan

    @classmethod
    def from_values(cls, values: np.ndarray, n_days: int = 0,
                    monthly_values: Optional[Dict[int, np.ndarray]] = None) -> 'FuelStats':
        """
        Create FuelStats from an array of values.

        Args:
            values: Array of consumption values
            n_days: Number of days represented
            monthly_values: Optional dict of month -> values for monthly stats
        """
        if len(values) == 0:
            return cls(
                n_values=0,
                n_days=n_days,
                sum_values=0.0,
                sum_squared=0.0,
                min_value=np.nan,
                max_value=np.nan,
                quantiles={},
                monthly_stats={}
            )

        # Remove NaN values
        values = values[~np.isnan(values)]
        if len(values) == 0:
            return cls(
                n_values=0,
                n_days=n_days,
                sum_values=0.0,
                sum_squared=0.0,
                min_value=np.nan,
                max_value=np.nan,
                quantiles={},
                monthly_stats={}
            )

        # Compute quantiles
        quantiles = {
            'p5': float(np.percentile(values, 5)),
            'p10': float(np.percentile(values, 10)),
            'p25': float(np.percentile(values, 25)),
            'p50': float(np.percentile(values, 50)),
            'p75': float(np.percentile(values, 75)),
            'p90': float(np.percentile(values, 90)),
            'p95': float(np.percentile(values, 95)),
        }

        # Compute monthly stats if provided
        monthly_stats = {}
        if monthly_values:
            for month, mv in monthly_values.items():
                mv = mv[~np.isnan(mv)]
                if len(mv) > 0:
                    monthly_stats[month] = (float(np.sum(mv)), len(mv))

        return cls(
            n_values=len(values),
            n_days=n_days,
            sum_values=float(np.sum(values)),
            sum_squared=float(np.sum(values ** 2)),
            min_value=float(np.min(values)),
            max_value=float(np.max(values)),
            quantiles=quantiles,
            monthly_stats=monthly_stats
        )

    @classmethod
    def combine(cls, stats_list: List['FuelStats']) -> 'FuelStats':
        """
        Combine multiple FuelStats using weighted formulas.

        Uses Chan's parallel algorithm for variance.
        """
        if not stats_list:
            return cls(
                n_values=0, n_days=0, sum_values=0.0, sum_squared=0.0,
                min_value=np.nan, max_value=np.nan, quantiles={}, monthly_stats={}
            )

        # Filter out empty stats
        stats_list = [s for s in stats_list if s.n_values > 0]
        if not stats_list:
            return cls(
                n_values=0, n_days=0, sum_values=0.0, sum_squared=0.0,
                min_value=np.nan, max_value=np.nan, quantiles={}, monthly_stats={}
            )

        n_total = sum(s.n_values for s in stats_list)
        n_days_total = sum(s.n_days for s in stats_list)
        sum_total = sum(s.sum_values for s in stats_list)
        sum_sq_total = sum(s.sum_squared for s in stats_list)
        min_total = min(s.min_value for s in stats_list)
        max_total = max(s.max_value for s in stats_list)

        # Weighted average of quantiles (approximation)
        quantiles = {}
        for q in ['p5', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95']:
            weighted_sum = sum(s.quantiles.get(q, 0) * s.n_values for s in stats_list)
            quantiles[q] = weighted_sum / n_total if n_total > 0 else np.nan

        # Combine monthly stats
        monthly_stats = {}
        all_months = set()
        for s in stats_list:
            all_months.update(s.monthly_stats.keys())

        for month in all_months:
            month_sum = 0.0
            month_count = 0
            for s in stats_list:
                if month in s.monthly_stats:
                    ms, mc = s.monthly_stats[month]
                    month_sum += ms
                    month_count += mc
            if month_count > 0:
                monthly_stats[month] = (month_sum, month_count)

        return cls(
            n_values=n_total,
            n_days=n_days_total,
            sum_values=sum_total,
            sum_squared=sum_sq_total,
            min_value=min_total,
            max_value=max_total,
            quantiles=quantiles,
            monthly_stats=monthly_stats
        )


@dataclass
class BatchResult:
    """
    Intermediate results from processing a single synthetic batch.

    Contains all computed statistics from 12 month-chunk files,
    ready for aggregation in the reduce phase.
    """
    batch_id: str
    n_households: int
    n_months_processed: int

    # Conditioning variables for this batch (CONSTANT across all households)
    conditioning: Dict[str, Any] = field(default_factory=dict)

    # Fuel statistics (for marginal + conditional distribution)
    fuel_stats: Dict[str, FuelStats] = field(default_factory=dict)
    # {'electricity': FuelStats(...), 'gas': FuelStats(...)}

    # Within-batch variance metrics (per-month, aggregated for this batch)
    variance_metrics: Dict[str, Any] = field(default_factory=dict)
    # {
    #     'mean_correlation': float,
    #     'max_correlation': float,
    #     'mean_distance': float,
    #     'min_distance': float,
    #     'peak_diversity': float,
    #     'severity': str,
    #     'flags': List[str],
    # }

    # ACF values per household-month (to be averaged in reduce)
    acf_values: Dict[str, Dict[int, List[float]]] = field(default_factory=dict)
    # {'electricity': {lag: [values]}, 'gas': {...}}

    # Volatility statistics (first differences)
    volatility_stats: Dict[str, FuelStats] = field(default_factory=dict)
    # {'electricity': FuelStats(...), 'gas': FuelStats(...)}

    # Daily profile contributions (sum and count for weighted average)
    profile_contributions: Dict[str, Dict[str, Dict[int, Tuple[float, int]]]] = field(default_factory=dict)
    # {'electricity': {'all': {halfhour: (sum, count)}, 'weekday': {...}, 'weekend': {...}}}

    # Daily profile medians per half-hour (batch-level, for cross-batch aggregation)
    profile_medians: Dict[str, Dict[str, Dict[int, Tuple[float, int]]]] = field(default_factory=dict)
    # {'electricity': {'all': {halfhour: (median, count)}, ...}}

    # Summary statistics per household-month (for distance analysis)
    household_month_stats: List[Dict[str, Any]] = field(default_factory=list)
    # [{'household_id': str, 'month': int, 'elec_mean': float, ...}]

    # Missingness statistics per fuel
    missingness_stats: Dict[str, MissingnessStats] = field(default_factory=dict)
    # {'electricity': MissingnessStats(...), 'gas': MissingnessStats(...)}

    # Pairwise correlations: {'elec_gas': float, 'temp_elec': float, 'temp_gas': float}
    correlations: Dict[str, float] = field(default_factory=dict)

    # Value samples for distribution comparison (reservoir sampling)
    value_samples: Dict[str, List[float]] = field(default_factory=dict)
    # {'electricity': [sampled values], 'gas': [sampled values]}

    # Conditional FuelStats for real data aggregation (only used by RealDataProcessor).
    # Each batch computes FuelStats per conditioning group; the reduce phase
    # combines them via FuelStats.combine() without needing raw values.
    conditional_values: Dict[str, Dict[Any, Dict[str, 'FuelStats']]] = field(default_factory=dict)
    # {cond_var: {cond_value: {fuel: FuelStats}}}

    # Household counts per conditioning group (only used by RealDataProcessor).
    # {cond_var: {cond_value: n_households}}
    conditioning_n_households: Dict[str, Dict[Any, int]] = field(default_factory=dict)

    # PTG monthly sums for Power-Temperature Gradient analysis
    ptg_monthly: Dict[int, Dict[str, float]] = field(default_factory=dict)
    # {month: {'elec_sum': float, 'elec_count': int, 'gas_sum': float, 'gas_count': int, 'temp_sum': float, 'temp_count': int}}

    # --- Integrated Diagnostics Fields ---

    # Seasonal analysis: per-household monthly means for between-household SE
    # {fuel: {month: [household_mean_1, household_mean_2, ...]}}
    seasonal_household_means: Dict[str, Dict[int, List[float]]] = field(default_factory=dict)

    # Monthly half-hour profiles with sum_sq for SE computation
    # {fuel: {month: {half_hour: (sum, count, sum_sq)}}}
    monthly_profile_contributions: Dict[str, Dict[int, Dict[int, Tuple[float, int, float]]]] = field(default_factory=dict)

    # Weekly (within-month) half-hour profiles with sum_sq
    # {fuel: {week: {half_hour: (sum, count, sum_sq)}}}
    weekly_profile_contributions: Dict[str, Dict[int, Dict[int, Tuple[float, int, float]]]] = field(default_factory=dict)

    # Weekly (within-month) marginal distribution stats
    # {fuel: {week: FuelStats}}
    weekly_fuel_stats: Dict[str, Dict[int, 'FuelStats']] = field(default_factory=dict)

    # Reservoir samples of first differences for volatility histogram comparison
    # {fuel: [diff_1, diff_2, ...]}
    volatility_samples: Dict[str, List[float]] = field(default_factory=dict)

    # Token frequency histograms (requires tokeniser)
    # {fuel: list of int counts per bin}
    token_frequency: Dict[str, List[int]] = field(default_factory=dict)

    # Monthly decoded-value stats for token distribution analysis
    # {fuel: {month: {'mean': float, 'median': float, 'std': float, 'skew': float, 'zero_rate': float, 'n': int}}}
    monthly_decoded_stats: Dict[str, Dict[int, Dict[str, float]]] = field(default_factory=dict)

    # Weekly token frequency (requires tokeniser)
    # {fuel: {week: list of int counts per bin}}
    weekly_token_frequency: Dict[str, Dict[int, List[int]]] = field(default_factory=dict)

    # Privacy pre-filter stats (None if pre-filter not applied)
    prefilter_stats: Optional[Dict[str, Any]] = None
    # {
    #     'n_household_months_excluded': int,
    #     'n_households_fully_excluded': int,
    #     'n_households_partially_excluded': int,
    #     'batch_empty_after_filter': bool,
    # }


@dataclass
class RealDataStats:
    """
    Aggregated statistics from real household data.

    Produced by RealDataProcessor, used by ResultsAggregator for comparison.
    """
    n_households: int
    split: str  # 'train' or 'test'

    # Per-fuel aggregated statistics
    fuel_stats: Dict[str, FuelStats] = field(default_factory=dict)

    # ACF values (averaged across households)
    acf: Dict[str, Dict[int, float]] = field(default_factory=dict)
    # {fuel: {lag: value}}

    # Volatility statistics
    volatility_stats: Dict[str, FuelStats] = field(default_factory=dict)

    # Reservoir samples of first differences for volatility histogram
    # {fuel: [diff_1, diff_2, ...]}
    volatility_samples: Dict[str, List[float]] = field(default_factory=dict)

    # Daily profiles (means)
    profiles: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)
    # {fuel: {'all': [48 values], 'weekday': [...], 'weekend': [...]}}

    # Daily profiles (medians, electricity only)
    profile_medians: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)
    # {'electricity': {'all': [48 values], 'weekday': [...], 'weekend': [...]}}

    # Conditional statistics (grouped by conditioning variable)
    conditional: Dict[str, Dict[Any, Dict[str, FuelStats]]] = field(default_factory=dict)
    # {cond_var: {cond_value: {fuel: FuelStats}}}

    # Household counts per conditioning group (for SDC compliance).
    # {cond_var: {cond_value: n_households}}
    conditioning_n_households: Dict[str, Dict[Any, int]] = field(default_factory=dict)

    # Summary statistics per household-month (for distance analysis)
    household_month_stats: List[Dict[str, Any]] = field(default_factory=list)
    # Same format as BatchResult.household_month_stats

    # Missingness statistics per fuel
    missingness_stats: Dict[str, MissingnessStats] = field(default_factory=dict)

    # Pairwise correlations: {'elec_gas': float, 'temp_elec': float, 'temp_gas': float}
    correlations: Dict[str, float] = field(default_factory=dict)

    # Value samples for distribution comparison
    value_samples: Dict[str, List[float]] = field(default_factory=dict)

    # PTG monthly sums for Power-Temperature Gradient analysis
    ptg_monthly: Dict[int, Dict[str, float]] = field(default_factory=dict)
    # {month: {'elec_sum': float, 'elec_count': int, 'gas_sum': float, 'gas_count': int, 'temp_sum': float, 'temp_count': int}}

    # --- Integrated Diagnostics Fields (mirror of BatchResult fields) ---

    seasonal_household_means: Dict[str, Dict[int, List[float]]] = field(default_factory=dict)
    monthly_profile_contributions: Dict[str, Dict[int, Dict[int, Tuple[float, int, float]]]] = field(default_factory=dict)
    weekly_profile_contributions: Dict[str, Dict[int, Dict[int, Tuple[float, int, float]]]] = field(default_factory=dict)
    weekly_fuel_stats: Dict[str, Dict[int, 'FuelStats']] = field(default_factory=dict)
    token_frequency: Dict[str, List[int]] = field(default_factory=dict)
    monthly_decoded_stats: Dict[str, Dict[int, Dict[str, float]]] = field(default_factory=dict)
    weekly_token_frequency: Dict[str, Dict[int, List[int]]] = field(default_factory=dict)


@dataclass
class FinalResults:
    """
    Final aggregated evaluation results.

    Produced by ResultsAggregator after combining all BatchResults
    and comparing to RealDataStats.
    """
    # Metadata
    n_batches: int
    n_households: int
    n_months: int
    n_values: Dict[str, int] = field(default_factory=dict)

    # Category A: Marginal Distribution
    marginal_distribution: Dict[str, Any] = field(default_factory=dict)
    # {
    #     'electricity': {
    #         'synth_mean': float, 'synth_std': float, 'synth_quantiles': {...},
    #         'train_mean': float, 'train_std': float, 'train_quantiles': {...},
    #         'mean_ratio': float, 'std_ratio': float, 'quantile_mae': float,
    #         'status': str
    #     },
    #     'gas': {...}
    # }

    # Category B: Conditional Distribution
    conditional_distribution: Dict[str, Any] = field(default_factory=dict)
    # {cond_var: {cond_value: {fuel: {'synth_mean': x, 'train_mean': x, 'ratio': x}}}}

    # Category C: Temporal Patterns
    temporal_patterns: Dict[str, Any] = field(default_factory=dict)
    # {
    #     'daily_profiles': {...},
    #     'acf': {...},
    #     'volatility': {...}
    # }

    # Category D: Within-batch variance
    within_batch_variance: Dict[str, Any] = field(default_factory=dict)
    # {
    #     'n_batches_analyzed': int,
    #     'severity_counts': {...},
    #     'flagged_batches': [...],
    #     'overall_status': str
    # }

    # Distance Analysis
    distance_analysis: Dict[str, Any] = field(default_factory=dict)

    # Category E: Missingness Analysis
    missingness_analysis: Dict[str, Any] = field(default_factory=dict)
    # {
    #     'electricity': {
    #         'synth_missing_pct': float, 'train_missing_pct': float, 'test_missing_pct': float,
    #         'synth_zeros_pct': float, 'train_zeros_pct': float, 'test_zeros_pct': float,
    #         'synth_structural_pct': float, 'train_structural_pct': float,
    #         'run_length_comparison': {...}
    #     },
    #     'gas': {...}
    # }

    # Distribution Comparison (Wasserstein, KS)
    distribution_comparison: Dict[str, Any] = field(default_factory=dict)
    # {
    #     'electricity': {
    #         'synth_train': {'wasserstein': float, 'ks_statistic': float, 'ks_pvalue': float},
    #         'synth_test': {...},
    #         'train_test': {...}
    #     },
    #     'gas': {...}
    # }

    # Correlation Structure
    correlation_analysis: Dict[str, Any] = field(default_factory=dict)

    # PTG analysis: monthly mean consumption vs temperature per split
    ptg_analysis: Dict[str, Any] = field(default_factory=dict)
    # {
    #     'synth_elec_gas_corr': float,
    #     'train_elec_gas_corr': float,
    #     'test_elec_gas_corr': float,
    #     'correlation_difference': float
    # }

    # --- Integrated Diagnostics Results ---
    seasonal_analysis: Dict[str, Any] = field(default_factory=dict)
    daily_pattern_diagnostics: Dict[str, Any] = field(default_factory=dict)
    household_month_fidelity: Dict[str, Any] = field(default_factory=dict)
    weekly_fidelity: Dict[str, Any] = field(default_factory=dict)
    token_analysis: Dict[str, Any] = field(default_factory=dict)

    # Household counts per split (for SDC companion CSVs).
    # Populated by ResultsAggregator from real_*_stats.n_households.
    n_households_train: int = 0
    n_households_val: int = 0
    n_households_test: int = 0

    # Privacy pre-filter summary (None if not applied)
    prefilter_summary: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'metadata': {
                'n_batches': self.n_batches,
                'n_households': self.n_households,
                'n_months': self.n_months,
                'n_values': self.n_values,
                'n_households_train': self.n_households_train,
                'n_households_val': self.n_households_val,
                'n_households_test': self.n_households_test,
                'prefilter_summary': self.prefilter_summary,
            },
            'marginal_distribution': self.marginal_distribution,
            'conditional_distribution': self.conditional_distribution,
            'temporal_patterns': self.temporal_patterns,
            'within_batch_variance': self.within_batch_variance,
            'distance_analysis': self.distance_analysis,
            'missingness_analysis': self.missingness_analysis,
            'distribution_comparison': self.distribution_comparison,
            'correlation_analysis': self.correlation_analysis,
            'ptg_analysis': self.ptg_analysis,
            'seasonal_analysis': self.seasonal_analysis,
            'daily_pattern_diagnostics': self.daily_pattern_diagnostics,
            'household_month_fidelity': self.household_month_fidelity,
            'weekly_fidelity': self.weekly_fidelity,
            'token_analysis': self.token_analysis,
        }
