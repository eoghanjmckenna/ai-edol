"""
Evaluation pipeline entry point.

This module provides a simple interface to run the MapReduce evaluation pipeline.
All evaluation functionality is now implemented in src/evaluation/mapreduce/.

NOTE: As of Phase 4 refactoring (2026-02), legacy evaluation code has been removed:
- Training visualization (now done directly in training loop)
- Privacy cost analysis (masked training deprecated)
- Summary statistics calculator (replaced by mapreduce/month_statistics.py)
- Distance analyzer (replaced by mapreduce/distance_visualizer.py)
- Fidelity module (integrated into mapreduce)
- Within-batch variance (moved to mapreduce/variance_analysis.py)

Use run_evaluation() or ParallelEvaluator directly for synthetic data evaluation.
"""
import os
from typing import Dict, Any, Optional

from .mapreduce import ParallelEvaluator, run_evaluation
from .diagnostics import run_all_diagnostics


def evaluate_synthetic_data(
    synthetic_dir: str,
    output_dir: str,
    real_train_dir: Optional[str] = None,
    real_test_dir: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    n_workers: Optional[int] = None,
    run_visualizations: bool = True,
    logger=None,
) -> Dict[str, Any]:
    """
    Run full evaluation pipeline on synthetic data.

    This is the main entry point for evaluating synthetic month-chunk data.
    It uses the MapReduce architecture for scalable parallel processing.

    Args:
        synthetic_dir: Path to directory containing batch_XXXX_month_MM.pkl files
        output_dir: Path to save evaluation results
        real_train_dir: Optional path to real training data for comparison
        real_test_dir: Optional path to real test data for comparison
        config: Optional configuration dictionary
        n_workers: Number of parallel workers (None = auto-detect)
        run_visualizations: Whether to generate visualization plots
        logger: Optional structured logger

    Returns:
        Dictionary containing evaluation results (FinalResults.to_dict())

    Example:
        >>> results = evaluate_synthetic_data(
        ...     synthetic_dir='runs/run_001/generated_data',
        ...     output_dir='runs/run_001/evaluation',
        ...     real_train_dir='data/real_households/train',
        ... )
        >>> print(results['marginal_distribution']['electricity']['status'])
    """
    config = config or {}

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Run MapReduce evaluation
    results = run_evaluation(
        synthetic_dir=synthetic_dir,
        output_dir=output_dir,
        real_train_dir=real_train_dir,
        real_test_dir=real_test_dir,
        config=config,
        n_workers=n_workers,
        run_visualizations=run_visualizations,
    )

    if logger:
        logger.info(f"Evaluation complete: {results.n_batches} batches processed")
        logger.log_custom_metric('evaluation_n_batches', results.n_batches)
        logger.log_custom_metric('evaluation_n_households', results.n_households)

    return results.to_dict()


def run_diagnostics(
    synthetic_dir: str,
    output_dir: str,
    real_dir: Optional[str] = None,
    metadata_path: Optional[str] = None,
    tokeniser_path: Optional[str] = None,
    sample_size: int = 50,
    context_length: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run quick diagnostic scripts on synthetic data.

    These diagnostics provide rapid insights into potential issues:
    1. Autoregressive Collapse: Detects value degradation across months
    2. Daily Patterns: Analyzes half-hourly profile fidelity
    3. Full Year Fidelity: Checks marginal distributions and conditioning
    4. Token Distribution: Analyzes token/value distributions for median bias

    Args:
        synthetic_dir: Path to synthetic data directory
        output_dir: Path to save diagnostic outputs
        real_dir: Optional path to real data for comparison
        metadata_path: Optional path to metadata catalog
        tokeniser_path: Optional path to tokeniser file
        sample_size: Number of households to sample
        context_length: Model context length (for degradation analysis)

    Returns:
        Dictionary with diagnostic results per category
    """
    os.makedirs(output_dir, exist_ok=True)

    return run_all_diagnostics(
        synthetic_dir=synthetic_dir,
        real_dir=real_dir,
        output_dir=output_dir,
        metadata_path=metadata_path,
        tokeniser_path=tokeniser_path,
        sample_size=sample_size,
        context_length=context_length,
    )


# Re-export key classes for backward compatibility
__all__ = [
    'evaluate_synthetic_data',
    'run_diagnostics',
    'ParallelEvaluator',
    'run_evaluation',
]
