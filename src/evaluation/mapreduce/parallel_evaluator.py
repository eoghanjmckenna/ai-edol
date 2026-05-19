"""
ParallelEvaluator for MapReduce evaluation pipeline.

Orchestrates parallel batch processing (map phase) and
results aggregation (reduce phase).
"""

import os
import json
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

from .data_structures import BatchResult, RealDataStats, FinalResults
from .batch_processor import BatchProcessor
from .real_data_processor import RealDataProcessor
from .results_aggregator import ResultsAggregator
from .distance_visualizer import DistanceVisualizer


def _process_batch_worker(args: tuple) -> BatchResult:
    """
    Worker function for parallel batch processing.

    Must be a top-level function (not a method) for pickling in ProcessPoolExecutor.
    """
    batch_id, month_files, config = args
    processor = BatchProcessor(config)
    return processor.process_batch(batch_id, month_files)


class ParallelEvaluator:
    """
    Orchestrate parallel evaluation using MapReduce pattern.

    Manages worker pool for map phase and aggregation for reduce phase.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 n_workers: Optional[int] = None):
        """
        Args:
            config: Evaluation configuration
            n_workers: Number of parallel workers (default: CPU count - 1)
        """
        self.config = config or {}
        self.n_workers = n_workers or max(1, (os.cpu_count() or 2) - 1)
        self.aggregator = None

    def evaluate(self, synthetic_dir: str,
                 real_train_dir: Optional[str] = None,
                 real_test_dir: Optional[str] = None,
                 output_dir: Optional[str] = None,
                 run_visualizations: bool = True,
                 progress_callback: Optional[callable] = None,
                 train_puprns: Optional[List[str]] = None,
                 val_puprns: Optional[List[str]] = None,
                 test_puprns: Optional[List[str]] = None,
                 train_sample: Optional[int] = None,
                 val_sample: Optional[int] = None,
                 test_sample: Optional[int] = None,
                 synth_batch_limit: Optional[int] = None,
                 metadata_catalog_path: Optional[str] = None) -> FinalResults:
        """
        Run full evaluation pipeline.

        Args:
            synthetic_dir: Directory containing batch_*_month_*.pkl files
            real_train_dir: Directory with real training data (or combined data folder)
            real_test_dir: Directory with real test data (or same as train_dir if using PUPRN filters)
            output_dir: Directory for output files (distance analysis, plots, etc.)
            run_visualizations: Whether to run distance analysis visualizations
            progress_callback: Optional callback(stage, current, total) for progress
            train_puprns: List of PUPRNs to use for training split (from shard_metadata)
            val_puprns: List of PUPRNs to use for validation split (from shard_metadata)
            test_puprns: List of PUPRNs to use for test split (from shard_metadata)
            train_sample: Max number of training households to process (None = all)
            val_sample: Max number of validation households to process (None = all)
            test_sample: Max number of test households to process (None = all)
            synth_batch_limit: Max number of synthetic batches to process (None = all)
            metadata_catalog_path: Path to metadata_catalog_filtered.pkl for real data
                conditioning variable lookup

        Returns:
            FinalResults with complete evaluation
        """
        import random

        # 1. Discover synthetic batches
        batches = self._discover_batches(synthetic_dir)
        print(f"[EVAL] Found {len(batches)} synthetic batches to process")

        # Apply synthetic batch limit if specified
        if synth_batch_limit and synth_batch_limit < len(batches):
            batch_ids = sorted(batches.keys())[:synth_batch_limit]
            batches = {bid: batches[bid] for bid in batch_ids}
            print(f"[EVAL] Limited to {len(batches)} batches (synth_batch_limit={synth_batch_limit})")

        if not batches:
            print("[EVAL] No synthetic batches found!")
            return FinalResults(
                n_batches=0, n_households=0, n_months=0, n_values={},
                marginal_distribution={}, conditional_distribution={},
                temporal_patterns={}, within_batch_variance={'status': 'no_data'},
                distance_analysis={}
            )

        # 2. Process real data in parallel (if provided)
        real_train_stats = None
        real_test_stats = None
        real_val_stats = None

        # Apply sampling to PUPRN lists if specified
        def sample_puprns(puprn_list: Optional[List[str]], sample_size: Optional[int]) -> Optional[List[str]]:
            if puprn_list is None:
                return None
            if sample_size is None or sample_size >= len(puprn_list):
                return puprn_list
            random.seed(42)  # Reproducible sampling
            return random.sample(puprn_list, sample_size)

        sampled_train_puprns = sample_puprns(train_puprns, train_sample)
        sampled_val_puprns = sample_puprns(val_puprns, val_sample)
        sampled_test_puprns = sample_puprns(test_puprns, test_sample)

        if real_train_dir and (sampled_train_puprns is None or sampled_train_puprns):
            print(f"[EVAL] Processing real training data from {real_train_dir}...")
            real_processor = RealDataProcessor(self.config, n_workers=self.n_workers)
            real_train_stats = real_processor.process(
                real_train_dir, split='train', puprn_filter=sampled_train_puprns,
                metadata_path=metadata_catalog_path
            )
            print(f"[EVAL]   Processed {real_train_stats.n_households} training households")

        # Process validation data if we have val_puprns (uses same directory as train)
        if real_train_dir and sampled_val_puprns:
            print(f"[EVAL] Processing real validation data...")
            real_processor = RealDataProcessor(self.config, n_workers=self.n_workers)
            real_val_stats = real_processor.process(
                real_train_dir, split='val', puprn_filter=sampled_val_puprns,
                metadata_path=metadata_catalog_path
            )
            print(f"[EVAL]   Processed {real_val_stats.n_households} validation households")

        if real_test_dir and (sampled_test_puprns is None or sampled_test_puprns):
            print(f"[EVAL] Processing real test data...")
            real_processor = RealDataProcessor(self.config, n_workers=self.n_workers)
            real_test_stats = real_processor.process(
                real_test_dir, split='test', puprn_filter=sampled_test_puprns,
                metadata_path=metadata_catalog_path
            )
            print(f"[EVAL]   Processed {real_test_stats.n_households} test households")

        # 3. Initialize aggregator with real data stats
        self.aggregator = ResultsAggregator(
            config=self.config,
            real_train_stats=real_train_stats,
            real_val_stats=real_val_stats,
            real_test_stats=real_test_stats
        )

        # 4. Map phase: process synthetic batches in parallel
        print(f"[EVAL] Processing {len(batches)} synthetic batches with {self.n_workers} workers...")
        batch_results = self._run_map_phase(batches, progress_callback)
        print(f"[EVAL]   Completed {len(batch_results)} batches")

        # 5. Reduce phase: aggregate results
        print(f"[EVAL] Aggregating results...")
        final_results = self.aggregator.aggregate(batch_results)

        print(f"[EVAL] Evaluation complete:")
        print(f"[EVAL]   - {final_results.n_batches} batches")
        print(f"[EVAL]   - {final_results.n_households} synthetic households")
        print(f"[EVAL]   - {final_results.n_values.get('electricity', 0):,} electricity values")
        print(f"[EVAL]   - {final_results.n_values.get('gas', 0):,} gas values")

        # 6. Run distance analysis visualizations (if enabled)
        if run_visualizations and output_dir:
            print(f"[EVAL] Running distance analysis visualizations...")
            self._run_distance_visualization(
                batch_results=batch_results,
                real_train_stats=real_train_stats,
                real_test_stats=real_test_stats,
                output_dir=output_dir
            )

        # 7. Write run_metadata.json (provenance record)
        if output_dir:
            self._write_run_metadata(output_dir)

        return final_results

    def _write_run_metadata(self, output_dir: str) -> None:
        """
        Write run_metadata.json to output_dir for provenance tracking.

        Contains enough information to reconstruct the evaluation setup
        without having to inspect the full results file.
        """
        tokeniser_file = self.config.get('tokeniser_file', 'unknown')
        tokeniser_version = 'unknown'

        # Try to read tokeniser version from the tokeniser file
        if tokeniser_file and tokeniser_file != 'unknown':
            try:
                import pickle
                tok_path = Path(tokeniser_file)
                if tok_path.exists():
                    with open(tok_path, 'rb') as fh:
                        tok_data = pickle.load(fh)
                    # Tokeniser is typically a dict with a metadata key
                    if isinstance(tok_data, dict):
                        meta = tok_data.get('metadata', tok_data.get('meta', {}))
                        if isinstance(meta, dict):
                            tokeniser_version = str(
                                meta.get('version', meta.get('tokeniser_version', 'unknown'))
                            )
            except Exception:
                pass  # Non-critical; fall back to 'unknown'

        metadata = {
            'experiment_config': self.config.get('experiment_config', 'unknown'),
            'epoch': self.config.get('epoch', 'unknown'),
            'tokeniser_file': tokeniser_file,
            'tokeniser_version': tokeniser_version,
            'evaluation_pipeline_version': '1.0.0',
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'n_workers': self.n_workers,
        }

        # Include pre-filter config if present
        pf_config = self.config.get('_prefilter_config')
        if pf_config:
            metadata['privacy_prefilter'] = pf_config

        out_path = Path(output_dir) / 'run_metadata.json'
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as fh:
            json.dump(metadata, fh, indent=2)
        print(f"[EVAL] run_metadata.json written to {out_path}")

    def _run_distance_visualization(
        self,
        batch_results: List[BatchResult],
        real_train_stats: Optional[RealDataStats],
        real_test_stats: Optional[RealDataStats],
        output_dir: str
    ) -> None:
        """
        Run comprehensive distance analysis and generate visualizations.

        Args:
            batch_results: List of BatchResult from map phase
            real_train_stats: Statistics from training data (if available)
            real_test_stats: Statistics from test data (if available)
            output_dir: Directory for output files
        """
        # Collect all household_month_stats from synthetic batches
        synth_stats = []
        for br in batch_results:
            synth_stats.extend(br.household_month_stats)

        if not synth_stats:
            print("[EVAL]   No synthetic stats available for distance analysis")
            return

        # Get real data stats
        train_stats = None
        test_stats = None

        if real_train_stats and hasattr(real_train_stats, 'household_month_stats'):
            train_stats = real_train_stats.household_month_stats

        if real_test_stats and hasattr(real_test_stats, 'household_month_stats'):
            test_stats = real_test_stats.household_month_stats

        # Create output directory
        distance_output = Path(output_dir) / 'distance_analysis'

        # Initialize and run visualizer
        visualizer = DistanceVisualizer(
            synth_stats=synth_stats,
            train_stats=train_stats,
            test_stats=test_stats,
            output_folder=str(distance_output),
            config=self.config
        )

        try:
            visualizer.run_full_analysis()
            print(f"[EVAL]   Distance analysis complete. Outputs in: {distance_output}")
        except Exception as e:
            print(f"[EVAL]   Distance visualization failed: {e}")
            import traceback
            traceback.print_exc()

    def _discover_batches(self, synthetic_dir: str) -> Dict[str, List[Path]]:
        """
        Discover all batches and their month files.

        Returns:
            Dict mapping batch_id to list of month file paths
        """
        synthetic_path = Path(synthetic_dir)
        if not synthetic_path.exists():
            print(f"[EVAL] Warning: Synthetic directory does not exist: {synthetic_dir}")
            return {}

        batches = defaultdict(list)

        # Find month-chunk format files: batch_XXXX_month_MM.pkl
        for f in synthetic_path.glob('batch_*_month_*.pkl'):
            # Extract base batch ID
            name = f.stem
            if '_month_' in name:
                batch_id = name.split('_month_')[0]
                batches[batch_id].append(f)

        # Also handle old single-file format (batch_XXXX.pkl)
        for f in synthetic_path.glob('batch_*.pkl'):
            if '_month_' not in f.name:
                batch_id = f.stem
                if batch_id not in batches:  # Don't override month-chunk batches
                    batches[batch_id].append(f)

        # Sort files within each batch by month number
        def get_month_num(path: Path) -> int:
            name = path.stem
            if '_month_' in name:
                try:
                    return int(name.split('_month_')[1])
                except (ValueError, IndexError):
                    return 0
            return 0

        for batch_id in batches:
            batches[batch_id] = sorted(batches[batch_id], key=get_month_num)

        return dict(batches)

    def _run_map_phase(self, batches: Dict[str, List[Path]],
                       progress_callback: Optional[callable] = None) -> List[BatchResult]:
        """
        Process all batches in parallel.

        Uses ProcessPoolExecutor for true parallelism.
        """
        batch_results = []
        total = len(batches)
        completed = 0

        # Prepare work items
        work_items = [
            (batch_id, month_files, self.config)
            for batch_id, month_files in batches.items()
        ]

        if self.n_workers == 1:
            # Sequential execution (for debugging)
            for batch_id, month_files, config in work_items:
                try:
                    result = _process_batch_worker((batch_id, month_files, config))
                    batch_results.append(result)
                    completed += 1
                    if progress_callback:
                        progress_callback('map', completed, total)
                    print(f"[EVAL]   Completed: {batch_id} ({completed}/{total})")
                except Exception as e:
                    print(f"[EVAL]   Failed: {batch_id} - {e}")
        else:
            # Parallel execution
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                # Submit all batch processing tasks
                futures = {
                    executor.submit(_process_batch_worker, work_item): work_item[0]
                    for work_item in work_items
                }

                # Collect results as they complete
                for future in as_completed(futures):
                    batch_id = futures[future]
                    try:
                        result = future.result()
                        batch_results.append(result)
                        completed += 1
                        if progress_callback:
                            progress_callback('map', completed, total)
                        print(f"[EVAL]   Completed: {batch_id} ({completed}/{total})")
                    except Exception as e:
                        print(f"[EVAL]   Failed: {batch_id} - {e}")

        return batch_results

    def evaluate_synthetic_only(self, synthetic_dir: str,
                                output_dir: Optional[str] = None,
                                run_visualizations: bool = True,
                                progress_callback: Optional[callable] = None) -> FinalResults:
        """
        Run evaluation without real data comparison.

        Useful for quick checks of synthetic data quality.
        """
        return self.evaluate(
            synthetic_dir=synthetic_dir,
            real_train_dir=None,
            real_test_dir=None,
            output_dir=output_dir,
            run_visualizations=run_visualizations,
            progress_callback=progress_callback
        )

    def get_batch_count(self, synthetic_dir: str) -> int:
        """Get number of batches without running full evaluation."""
        return len(self._discover_batches(synthetic_dir))


def run_evaluation(synthetic_dir: str,
                   real_train_dir: Optional[str] = None,
                   real_test_dir: Optional[str] = None,
                   config: Optional[Dict[str, Any]] = None,
                   n_workers: Optional[int] = None,
                   output_path: Optional[str] = None,
                   output_dir: Optional[str] = None,
                   run_visualizations: bool = True) -> FinalResults:
    """
    Convenience function to run full evaluation pipeline.

    Args:
        synthetic_dir: Directory containing synthetic batch files
        real_train_dir: Optional directory with real training data
        real_test_dir: Optional directory with real test data
        config: Optional evaluation configuration
        n_workers: Optional number of workers (default: auto)
        output_path: Optional path to save results as JSON
        output_dir: Optional directory for distance analysis outputs
        run_visualizations: Whether to run distance analysis (default: True)

    Returns:
        FinalResults with complete evaluation
    """
    import json

    evaluator = ParallelEvaluator(config=config, n_workers=n_workers)
    results = evaluator.evaluate(
        synthetic_dir=synthetic_dir,
        real_train_dir=real_train_dir,
        real_test_dir=real_test_dir,
        output_dir=output_dir,
        run_visualizations=run_visualizations
    )

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results.to_dict(), f, indent=2, default=str)
        print(f"[EVAL] Results saved to {output_path}")

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run MapReduce evaluation pipeline')
    parser.add_argument('synthetic_dir', help='Directory containing synthetic batch files')
    parser.add_argument('--real-train', help='Directory with real training data')
    parser.add_argument('--real-test', help='Directory with real test data')
    parser.add_argument('--output', '-o', help='Output path for results JSON')
    parser.add_argument('--output-dir', '-d', help='Directory for distance analysis outputs')
    parser.add_argument('--workers', '-w', type=int, help='Number of parallel workers')
    parser.add_argument('--no-visualizations', action='store_true',
                        help='Skip distance analysis visualizations')

    args = parser.parse_args()

    results = run_evaluation(
        synthetic_dir=args.synthetic_dir,
        real_train_dir=args.real_train,
        real_test_dir=args.real_test,
        n_workers=args.workers,
        output_path=args.output,
        output_dir=args.output_dir,
        run_visualizations=not args.no_visualizations
    )

    # Print summary
    print("\n=== EVALUATION SUMMARY ===")
    print(f"Batches: {results.n_batches}")
    print(f"Households: {results.n_households}")

    for fuel in ['electricity', 'gas']:
        if fuel in results.marginal_distribution:
            md = results.marginal_distribution[fuel]
            print(f"\n{fuel.upper()}:")
            print(f"  Synth mean: {md.get('synth_mean', 'N/A'):.2f}")
            print(f"  Train mean: {md.get('train_mean', 'N/A')}")
            print(f"  Mean ratio: {md.get('mean_ratio', 'N/A')}")
            print(f"  Status: {md.get('status', 'N/A')}")

    wbv = results.within_batch_variance
    print(f"\nWithin-batch variance:")
    print(f"  Status: {wbv.get('overall_status', 'N/A')}")
    print(f"  Severity counts: {wbv.get('severity_counts', {})}")
