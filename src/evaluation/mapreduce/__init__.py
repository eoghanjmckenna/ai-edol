"""
MapReduce evaluation pipeline for month-chunk synthetic data.

This module implements a parallelizable evaluation pipeline using the MapReduce
pattern. Synthetic batches and real households are processed in parallel (map phase),
and results are aggregated for final comparison (reduce phase).

Components:
- BatchProcessor: Processes one synthetic batch (12 month-chunks)
- RealDataProcessor: Processes real household data in parallel
- ResultsAggregator: Aggregates batch results and compares to real data
- ParallelEvaluator: Orchestrates the full pipeline

See: specs/evaluation-mapreduce-architecture.md for detailed specification.
"""

from .data_structures import FuelStats, BatchResult, RealDataStats, FinalResults, MissingnessStats
from .batch_processor import BatchProcessor
from .real_data_processor import RealDataProcessor
from .results_aggregator import ResultsAggregator
from .parallel_evaluator import ParallelEvaluator, run_evaluation
from .month_statistics import MonthStatisticsCalculator, compute_all_household_month_stats
from .distance_visualizer import DistanceVisualizer
from .results_visualizer import ResultsVisualizer, visualize_results
from .variance_analysis import (
    compute_within_batch_correlations,
    compute_within_batch_distances,
    compute_within_batch_peak_diversity,
    flag_potential_overfitting,
    generate_within_batch_variance_report,
    plot_within_batch_variance,
    OVERFITTING_THRESHOLDS,
)

__all__ = [
    'FuelStats',
    'BatchResult',
    'RealDataStats',
    'FinalResults',
    'MissingnessStats',
    'BatchProcessor',
    'RealDataProcessor',
    'ResultsAggregator',
    'ParallelEvaluator',
    'run_evaluation',
    'MonthStatisticsCalculator',
    'compute_all_household_month_stats',
    'DistanceVisualizer',
    'ResultsVisualizer',
    'visualize_results',
    'compute_within_batch_correlations',
    'compute_within_batch_distances',
    'compute_within_batch_peak_diversity',
    'flag_potential_overfitting',
    'generate_within_batch_variance_report',
    'plot_within_batch_variance',
    'OVERFITTING_THRESHOLDS',
]
