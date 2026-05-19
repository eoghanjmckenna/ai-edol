"""
Diagnostic scripts for rapid fidelity assessment.

These diagnostics run before the formal fidelity evaluation pipeline
to provide quick insights into potential issues:

1. Autoregressive Collapse: Detects if values degrade across months
2. Daily Patterns: Analyzes half-hourly profile fidelity and weekly degradation
3. Household-Month Fidelity: Checks marginal distributions and conditioning effects
4. Token Distribution: Analyzes token/value distributions for median bias

Each diagnostic can be run standalone or via run_all_diagnostics().
"""

from .diagnose_autoregressive_collapse import main as run_collapse_diagnostic
from .diagnose_daily_patterns import main as run_daily_patterns_diagnostic
from .diagnose_full_year_fidelity import main as run_full_year_diagnostic
from .diagnose_token_distribution import main as run_token_distribution_diagnostic


def run_all_diagnostics(
    synthetic_dir: str,
    real_dir: str = None,
    output_dir: str = None,
    metadata_path: str = None,
    tokeniser_path: str = None,
    sample_size: int = 50,
    context_length: int = None,
    train_puprns: list = None,
) -> dict:
    """
    Run all diagnostic scripts sequentially.

    Args:
        synthetic_dir: Path to generated_data directory containing batch files
        real_dir: Path to tokenised_data directory for comparison (optional)
        output_dir: Directory to save diagnostic outputs (default: {synthetic_dir}/../diagnosis)
        metadata_path: Path to metadata_catalog_filtered.pkl for conditioning analysis
        tokeniser_path: Path to tokenisers.pkl for token distribution analysis
        sample_size: Number of real households to sample for comparison
        context_length: Model context length in timesteps (for degradation marking)
        train_puprns: List of training PUPRNs to filter real data (None = no filter)

    Returns:
        Dictionary with results from each diagnostic
    """
    import os

    # Default output directory
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(synthetic_dir), 'diagnosis')

    os.makedirs(output_dir, exist_ok=True)

    results = {}

    # 1. Autoregressive Collapse Detection
    print("\n" + "="*80)
    print("DIAGNOSTIC 1/4: AUTOREGRESSIVE COLLAPSE DETECTION")
    print("="*80)
    try:
        collapse_result = run_collapse_diagnostic(
            synthetic_dir=synthetic_dir,
            real_dir=real_dir,
            output_dir=output_dir,
            sample_size=sample_size,
            train_puprns=train_puprns,
        )
        results['collapse'] = collapse_result or {'status': 'completed'}
    except Exception as e:
        print(f"[WARNING] Collapse diagnostic failed: {e}")
        results['collapse'] = {'status': 'failed', 'error': str(e)}

    # 2. Daily Patterns Analysis
    print("\n" + "="*80)
    print("DIAGNOSTIC 2/4: DAILY PATTERN ANALYSIS")
    print("="*80)
    try:
        daily_result = run_daily_patterns_diagnostic(
            synthetic_dir=synthetic_dir,
            real_dir=real_dir,
            output_dir=output_dir,
            sample_size=sample_size,
            context_length=context_length,
            train_puprns=train_puprns,
        )
        results['daily_patterns'] = daily_result or {'status': 'completed'}
    except Exception as e:
        print(f"[WARNING] Daily patterns diagnostic failed: {e}")
        results['daily_patterns'] = {'status': 'failed', 'error': str(e)}

    # 3. Household-Month Fidelity
    print("\n" + "="*80)
    print("DIAGNOSTIC 3/4: HOUSEHOLD-MONTH FIDELITY ANALYSIS")
    print("="*80)
    try:
        full_year_result = run_full_year_diagnostic(
            synthetic_dir=synthetic_dir,
            real_dir=real_dir,
            output_dir=output_dir,
            metadata_path=metadata_path,
            sample_size=sample_size,
            train_puprns=train_puprns,
        )
        results['full_year'] = full_year_result or {'status': 'completed'}
    except Exception as e:
        print(f"[WARNING] Full year fidelity diagnostic failed: {e}")
        results['full_year'] = {'status': 'failed', 'error': str(e)}

    # 4. Token Distribution Analysis
    print("\n" + "="*80)
    print("DIAGNOSTIC 4/4: TOKEN DISTRIBUTION ANALYSIS")
    print("="*80)
    try:
        token_result = run_token_distribution_diagnostic(
            synthetic_dir=synthetic_dir,
            real_dir=real_dir,
            output_dir=output_dir,
            tokeniser_path=tokeniser_path,
            sample_size=sample_size,
            train_puprns=train_puprns,
        )
        results['token_distribution'] = token_result or {'status': 'completed'}
    except Exception as e:
        print(f"[WARNING] Token distribution diagnostic failed: {e}")
        results['token_distribution'] = {'status': 'failed', 'error': str(e)}

    print("\n" + "="*80)
    print("ALL DIAGNOSTICS COMPLETE")
    print(f"Results saved to: {output_dir}")
    print("="*80)

    return results
