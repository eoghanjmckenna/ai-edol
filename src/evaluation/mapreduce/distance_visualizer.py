"""
DistanceVisualizer for MapReduce evaluation pipeline.

Provides comprehensive distance analysis and visualization based on
household_month_stats from MapReduce outputs. Replicates all functionality
from the original DistanceAnalyzer but adapted for month-stratified data.

Key outputs:
- Distance matrices (synth-synth, synth-train, synth-test, etc.)
- Summary statistics CSV
- Per-household statistics CSV
- Privacy leakage analysis with statistical tests
- Outlier detection
- PCA projection plot
- Distance distribution plots
- Paired differences plots
- Human-readable analysis report
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import ttest_rel, percentileofscore
import matplotlib.pyplot as plt
import seaborn as sns

from .data_structures import RealDataStats, FinalResults
from .month_statistics import MonthStatisticsCalculator


@dataclass
class CrossGroupStats:
    """Pre-computed statistics from a cross-group distance matrix.

    Replaces the full (n_synth × n_other) matrix to avoid OOM at production
    scale. Computed via chunked cdist — see _compute_cross_group_chunked().

    All statistics are exact except percentiles, which use a reservoir sample.
    """
    n_synth_rows: int           # number of synthetic household-months
    n_other_cols: int           # number of train or test household-months
    synth_row_means: np.ndarray  # (n_synth_rows,) mean dist per synth row
    synth_row_mins: np.ndarray   # (n_synth_rows,) min dist per synth row
    other_col_means: np.ndarray  # (n_other_cols,) mean dist per other col
    other_col_mins: np.ndarray   # (n_other_cols,) min dist per other col
    global_mean: float          # exact mean of all distances (Chan's algorithm)
    global_std: float           # exact std of all distances (Chan's algorithm)
    global_min: float           # exact global minimum distance
    global_max: float           # exact global maximum distance
    global_n: int               # total number of distance values
    reservoir_sample: np.ndarray  # (~100K) random sample for approximate percentiles


class DistanceVisualizer:
    """
    Comprehensive distance analysis and visualization for MapReduce outputs.

    Takes household_month_stats from synthetic batches and real data,
    constructs feature matrices, and generates all distance analysis outputs.
    """

    def __init__(
        self,
        synth_stats: List[Dict[str, Any]],
        train_stats: Optional[List[Dict[str, Any]]] = None,
        test_stats: Optional[List[Dict[str, Any]]] = None,
        output_folder: str = 'distance_analysis',
        config: Optional[Dict[str, Any]] = None,
        logger=None
    ):
        """
        Initialize DistanceVisualizer.

        Args:
            synth_stats: List of household_month_stats from synthetic batches
            train_stats: Optional list of household_month_stats from training data
            test_stats: Optional list of household_month_stats from test data
            output_folder: Directory for outputs
            config: Configuration dictionary
            logger: Optional logger instance
        """
        self.config = config or {}
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.logger = logger

        # Distance analysis configuration (nested under 'distance_analysis' key
        # when passed from ParallelEvaluator, or at top level for direct use)
        distance_config = self.config.get('distance_analysis', self.config)
        self.metric = distance_config.get('metric', 'euclidean')
        self.normalize_features = distance_config.get('normalize_features', True)
        self.outlier_threshold_percentile = distance_config.get('outlier_threshold_percentile', 95)

        # Max households for within-group distance matrices (train_train, test_test,
        # synth_synth). These scale as N² and can OOM at production scale.
        # Cross-group matrices (train_synth, test_synth) are NOT affected — they
        # always use full populations for comprehensive privacy analysis.
        # None = no limit.
        self.within_group_max_households = distance_config.get('within_group_max_households', None)

        # Cross-group matrices are computed in chunks to avoid materializing the
        # full N×M matrix. See Decision 038 for rationale.
        # Memory per chunk: chunk_size × n_other_rows × 8 bytes
        # e.g. 1000 × 100,000 × 8 = ~800 MB per chunk
        # None = compute full matrix at once (may OOM at production scale)
        self.cross_group_chunk_size = distance_config.get('cross_group_chunk_size', 1000)

        # Initialize calculator for feature vectors
        self.calculator = MonthStatisticsCalculator(self.config.get('summary_statistics', {}))
        self.feature_names = self.calculator.get_feature_names()

        # Build feature matrices and metadata
        self._log("Building feature matrices from household_month_stats...")
        self.feature_matrix, self.metadata = self._build_feature_matrix(
            synth_stats, train_stats, test_stats
        )

        # Preprocess features
        self.feature_matrix_processed = self._preprocess_features()

        # Storage for computed results
        self.distance_matrices = {}       # Within-group: full square matrices
        self.cross_group_stats = {}       # Cross-group: pre-computed stats (not full matrices)
        self.distance_summaries = {}

        self._log(f"Initialized with {len(self.metadata)} household-months, "
                  f"{self.feature_matrix.shape[1]} features")
        self._log(f"Data types: {self.metadata['data_type'].value_counts().to_dict()}")

    def _log(self, message: str, level: str = 'info'):
        """Log a message."""
        if self.logger:
            getattr(self.logger, level)(message)
        else:
            print(f"[DistanceVisualizer] {message}")

    def _build_feature_matrix(
        self,
        synth_stats: List[Dict[str, Any]],
        train_stats: Optional[List[Dict[str, Any]]],
        test_stats: Optional[List[Dict[str, Any]]]
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """Build feature matrix and metadata from stats lists."""
        all_stats = []
        data_types = []

        # Add synthetic stats
        for s in synth_stats:
            all_stats.append(s)
            data_types.append('synthetic')

        # Add train stats
        if train_stats:
            for s in train_stats:
                all_stats.append(s)
                data_types.append('train')

        # Add test stats
        if test_stats:
            for s in test_stats:
                all_stats.append(s)
                data_types.append('test')

        # Build feature vectors
        vectors = []
        metadata_rows = []

        for i, (stats, dtype) in enumerate(zip(all_stats, data_types)):
            vector = self.calculator.stats_to_vector(stats)
            vectors.append(vector)

            metadata_rows.append({
                'household_id': stats.get('household_id', f'unknown_{i}'),
                'data_type': dtype,
                'month': stats.get('month', 0),
                'fuel_type': stats.get('fuel_type', 'unknown'),
            })

        feature_matrix = np.array(vectors)
        metadata = pd.DataFrame(metadata_rows)

        return feature_matrix, metadata

    def _preprocess_features(self) -> np.ndarray:
        """Preprocess features (impute NaN, optionally normalize)."""
        features = self.feature_matrix.copy()

        # Impute NaN with column means
        col_means = np.nanmean(features, axis=0)
        nan_mask = np.isnan(features)
        for col_idx in range(features.shape[1]):
            col_nans = nan_mask[:, col_idx]
            if np.any(col_nans):
                fill_val = col_means[col_idx] if not np.isnan(col_means[col_idx]) else 0.0
                features[col_nans, col_idx] = fill_val

        if self.normalize_features:
            # Z-score normalization
            mean = np.mean(features, axis=0)
            std = np.std(features, axis=0)
            std[std == 0] = 1.0  # Avoid division by zero
            features = (features - mean) / std
            self._log("Features normalized using z-score standardization")

        return features

    def _sample_within_group_features(self, features: np.ndarray, group_name: str) -> np.ndarray:
        """
        Subsample features for within-group distance matrices if they exceed
        the configured household limit.

        Within-group matrices (e.g. train_train) scale as N² and can OOM at
        production scale. This samples by household count (rows / 12 months)
        to keep memory bounded, while cross-group matrices always use full
        populations for privacy analysis.

        Args:
            features: Feature matrix for a single data type (N_rows × N_features).
                      Rows are household-month combinations.
            group_name: Name for logging (e.g. 'train', 'test', 'synthetic')

        Returns:
            Subsampled feature matrix, or the original if no limit applies.
        """
        if self.within_group_max_households is None:
            return features

        # Each household contributes up to 12 monthly rows
        # Estimate household count conservatively (some may have fewer months)
        n_rows = len(features)
        max_rows = self.within_group_max_households * 12

        if n_rows <= max_rows:
            return features

        # Reproducible random subsample of rows
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(n_rows, size=max_rows, replace=False)
        sample_idx.sort()  # Preserve original ordering

        self._log(
            f"Sampled {group_name} within-group features: {n_rows} → {max_rows} rows "
            f"(within_group_max_households={self.within_group_max_households})"
        )
        return features[sample_idx]

    def _compute_cross_group_chunked(
        self,
        synth_features: np.ndarray,
        other_features: np.ndarray,
        matrix_name: str,
        chunk_size: int = 1000,
        reservoir_size: int = 100_000,
    ) -> CrossGroupStats:
        """
        Compute cross-group distance statistics without materializing the full matrix.

        Instead of storing the full N_synth × N_other matrix (which can exceed 10 GB),
        processes cdist in chunks of synthetic rows. For each chunk, extracts per-row
        statistics, accumulates per-column statistics, updates global running statistics
        (Chan's parallel algorithm for exact mean/variance), and collects a random
        sample for approximate percentile estimation. The chunk is freed after each
        iteration.

        See Decision 038 for full rationale and accuracy analysis.

        Args:
            synth_features: Feature matrix for synthetic data (n_synth × n_features)
            other_features: Feature matrix for train or test data (n_other × n_features)
            matrix_name: Name for logging (e.g. 'train_synth')
            chunk_size: Number of synthetic rows per chunk. Memory per chunk is
                        chunk_size × n_other × 8 bytes.
            reservoir_size: Number of distance values to sample for percentile estimation.

        Returns:
            CrossGroupStats with all derived statistics.
        """
        n_synth = synth_features.shape[0]
        n_other = other_features.shape[0]
        total_distances = n_synth * n_other

        # Pre-allocate per-row arrays (exact)
        synth_row_means = np.empty(n_synth)
        synth_row_mins = np.empty(n_synth)

        # Per-column accumulators (exact, accumulated across chunks)
        other_col_sum = np.zeros(n_other)
        other_col_min = np.full(n_other, np.inf)

        # Chan's parallel algorithm accumulators for global stats (exact)
        global_n = 0
        global_mean = 0.0
        global_m2 = 0.0  # sum of squared deviations from mean
        global_min = np.inf
        global_max = -np.inf

        # Proportional sampling for percentile estimation
        rng = np.random.RandomState(42)
        reservoir_chunks = []

        for chunk_start in range(0, n_synth, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_synth)
            chunk_synth = synth_features[chunk_start:chunk_end]

            # Compute chunk distance matrix: (chunk_rows, n_other)
            chunk_dists = cdist(chunk_synth, other_features, metric=self.metric)

            # Per-synth-row stats (exact)
            synth_row_means[chunk_start:chunk_end] = np.mean(chunk_dists, axis=1)
            synth_row_mins[chunk_start:chunk_end] = np.min(chunk_dists, axis=1)

            # Per-other-col accumulation (exact)
            other_col_sum += np.sum(chunk_dists, axis=0)
            np.minimum(other_col_min, np.min(chunk_dists, axis=0), out=other_col_min)

            # Global stats via Chan's parallel algorithm (exact mean/std)
            chunk_flat = chunk_dists.ravel()
            chunk_n = len(chunk_flat)
            chunk_mean = np.mean(chunk_flat)
            chunk_m2 = np.var(chunk_flat) * chunk_n

            if global_n == 0:
                global_n = chunk_n
                global_mean = chunk_mean
                global_m2 = chunk_m2
            else:
                delta = chunk_mean - global_mean
                new_n = global_n + chunk_n
                global_mean = (global_n * global_mean + chunk_n * chunk_mean) / new_n
                global_m2 += chunk_m2 + delta ** 2 * global_n * chunk_n / new_n
                global_n = new_n

            global_min = min(global_min, float(np.min(chunk_flat)))
            global_max = max(global_max, float(np.max(chunk_flat)))

            # Proportional random sampling for percentile estimation
            # Take a proportional share of reservoir_size from this chunk
            samples_target = max(1, int(reservoir_size * chunk_n / total_distances))
            sample_count = min(samples_target, chunk_n)
            sample_idx = rng.choice(chunk_n, size=sample_count, replace=False)
            reservoir_chunks.append(chunk_flat[sample_idx])

            self._log(f"  {matrix_name} chunk [{chunk_start}:{chunk_end}] / {n_synth}")

            # chunk_dists freed when it goes out of scope

        # Finalize per-col means
        other_col_means = other_col_sum / n_synth

        # Finalize global std
        global_std = float(np.sqrt(global_m2 / global_n)) if global_n > 0 else 0.0

        # Combine reservoir samples; downsample if over target
        reservoir = np.concatenate(reservoir_chunks) if reservoir_chunks else np.array([])
        if len(reservoir) > reservoir_size:
            idx = rng.choice(len(reservoir), size=reservoir_size, replace=False)
            reservoir = reservoir[idx]

        return CrossGroupStats(
            n_synth_rows=n_synth,
            n_other_cols=n_other,
            synth_row_means=synth_row_means,
            synth_row_mins=synth_row_mins,
            other_col_means=other_col_means,
            other_col_mins=other_col_min,
            global_mean=float(global_mean),
            global_std=global_std,
            global_min=float(global_min),
            global_max=float(global_max),
            global_n=global_n,
            reservoir_sample=reservoir,
        )

    def calculate_all_distances(self):
        """
        Calculate all pairwise distance matrices.

        Computes:
        - synth_synth: Within synthetic distances (sampled if over limit)
        - train_train: Within training distances (sampled if over limit)
        - test_test: Within test distances (sampled if over limit)
        - train_synth: Synthetic to training distances (FULL - required for privacy)
        - test_synth: Synthetic to test distances (FULL - required for privacy)

        Within-group matrices use within_group_max_households to cap memory.
        Cross-group matrices always use full populations for privacy analysis.
        """
        self._log("Calculating pairwise distance matrices...")

        # Get indices for each data type
        synth_idx = self.metadata['data_type'] == 'synthetic'
        train_idx = self.metadata['data_type'] == 'train'
        test_idx = self.metadata['data_type'] == 'test'

        # Full feature sets — used for cross-group (privacy) matrices
        synth_features = self.feature_matrix_processed[synth_idx]
        train_features = self.feature_matrix_processed[train_idx]
        test_features = self.feature_matrix_processed[test_idx]

        # Subsampled feature sets — used only for within-group (fidelity) matrices
        # These scale as N² and can OOM at production scale (e.g. 48,000² = 18+ GB)
        synth_features_sampled = self._sample_within_group_features(synth_features, 'synthetic')
        train_features_sampled = self._sample_within_group_features(train_features, 'train')
        test_features_sampled = self._sample_within_group_features(test_features, 'test')

        # Within-group distances (use sampled features for memory safety)
        if len(synth_features_sampled) >= 2:
            self.distance_matrices['synth_synth'] = squareform(
                pdist(synth_features_sampled, metric=self.metric)
            )
            np.fill_diagonal(self.distance_matrices['synth_synth'], np.nan)
            self._log(f"Calculated synth_synth: {self.distance_matrices['synth_synth'].shape}")

        if len(train_features_sampled) >= 2:
            self.distance_matrices['train_train'] = squareform(
                pdist(train_features_sampled, metric=self.metric)
            )
            np.fill_diagonal(self.distance_matrices['train_train'], np.nan)
            self._log(f"Calculated train_train: {self.distance_matrices['train_train'].shape}")

        if len(test_features_sampled) >= 2:
            self.distance_matrices['test_test'] = squareform(
                pdist(test_features_sampled, metric=self.metric)
            )
            np.fill_diagonal(self.distance_matrices['test_test'], np.nan)
            self._log(f"Calculated test_test: {self.distance_matrices['test_test'].shape}")

        # Cross-group distances (FULL populations — critical for privacy analysis)
        # Computed in chunks to avoid materializing the full N×M matrix.
        # Results stored as CrossGroupStats, not full matrices. See Decision 038.
        chunk_size = self.cross_group_chunk_size or max(len(synth_features), 1)

        if len(synth_features) > 0 and len(train_features) > 0:
            self._log(f"Computing train_synth chunked: "
                      f"({len(synth_features)}, {len(train_features)}), "
                      f"chunk_size={chunk_size}")
            self.cross_group_stats['train_synth'] = self._compute_cross_group_chunked(
                synth_features, train_features, 'train_synth',
                chunk_size=chunk_size,
            )
            cg = self.cross_group_stats['train_synth']
            self._log(f"Calculated train_synth: ({cg.n_synth_rows}, {cg.n_other_cols})")

        if len(synth_features) > 0 and len(test_features) > 0:
            self._log(f"Computing test_synth chunked: "
                      f"({len(synth_features)}, {len(test_features)}), "
                      f"chunk_size={chunk_size}")
            self.cross_group_stats['test_synth'] = self._compute_cross_group_chunked(
                synth_features, test_features, 'test_synth',
                chunk_size=chunk_size,
            )
            cg = self.cross_group_stats['test_synth']
            self._log(f"Calculated test_synth: ({cg.n_synth_rows}, {cg.n_other_cols})")

    def generate_summary_statistics(self) -> pd.DataFrame:
        """Generate summary statistics for each distance matrix."""
        self._log("Generating distance summary statistics...")

        summary_rows = []

        # Within-group matrices (stored as full matrices)
        for matrix_name, distances in self.distance_matrices.items():
            dist_flat = distances.flatten()
            dist_flat = dist_flat[~np.isnan(dist_flat)]

            if len(dist_flat) == 0:
                continue

            summary = {
                'matrix_type': matrix_name,
                'n_comparisons': len(dist_flat),
                'mean': np.mean(dist_flat),
                'std': np.std(dist_flat),
                'min': np.min(dist_flat),
                'q25': np.percentile(dist_flat, 25),
                'median': np.percentile(dist_flat, 50),
                'q75': np.percentile(dist_flat, 75),
                'max': np.max(dist_flat),
                'iqr': np.percentile(dist_flat, 75) - np.percentile(dist_flat, 25)
            }
            summary_rows.append(summary)

        # Cross-group matrices (stored as pre-computed stats, not full matrices)
        for matrix_name, cg in self.cross_group_stats.items():
            reservoir = np.sort(cg.reservoir_sample)
            if len(reservoir) == 0:
                continue

            q25 = float(np.percentile(reservoir, 25))
            q75 = float(np.percentile(reservoir, 75))
            summary = {
                'matrix_type': matrix_name,
                'n_comparisons': cg.global_n,
                'mean': cg.global_mean,
                'std': cg.global_std,
                'min': cg.global_min,
                'q25': q25,
                'median': float(np.percentile(reservoir, 50)),
                'q75': q75,
                'max': cg.global_max,
                'iqr': q75 - q25,
            }
            summary_rows.append(summary)

        summary_df = pd.DataFrame(summary_rows)

        # Save to CSV
        output_path = self.output_folder / 'distance_summary_statistics.csv'
        summary_df.to_csv(output_path, index=False, float_format='%.4f')
        self._log(f"Summary statistics saved to {output_path}")

        self.distance_summaries['summary_statistics'] = summary_df
        return summary_df

    def _is_within_group_sampled(self, matrix_name: str, group_size: int) -> bool:
        """Check if a within-group matrix was built from a subsample."""
        if matrix_name not in self.distance_matrices:
            return False
        return self.distance_matrices[matrix_name].shape[0] < group_size

    def generate_per_household_statistics(self) -> pd.DataFrame:
        """
        Calculate per-household distance statistics.

        Cross-group distances (mean_dist_to_train, mean_dist_to_test) are
        always available for all households since those matrices use full
        populations.

        Within-group distances (e.g. mean_dist_to_synth for synthetic
        households) are skipped when the within-group matrix was sampled,
        because the random subsample breaks the row-to-household index
        correspondence. The aggregate within-group statistics from
        generate_summary_statistics() remain valid.
        """
        self._log("Generating per-household distance statistics...")

        household_stats = []

        synth_idx = np.where(self.metadata['data_type'] == 'synthetic')[0]
        train_idx = np.where(self.metadata['data_type'] == 'train')[0]
        test_idx = np.where(self.metadata['data_type'] == 'test')[0]

        # Check which within-group matrices were sampled (row indices no longer
        # correspond to metadata order, so per-household lookups are invalid)
        synth_sampled = self._is_within_group_sampled('synth_synth', len(synth_idx))
        train_sampled = self._is_within_group_sampled('train_train', len(train_idx))
        test_sampled = self._is_within_group_sampled('test_test', len(test_idx))

        if synth_sampled:
            self._log("synth_synth was sampled — skipping per-household within-group stats for synthetic")
        if train_sampled:
            self._log("train_train was sampled — skipping per-household within-group stats for train")
        if test_sampled:
            self._log("test_test was sampled — skipping per-household within-group stats for test")

        # Process synthetic households
        if len(synth_idx) > 0:
            for i, global_idx in enumerate(synth_idx):
                household_id = self.metadata.iloc[global_idx]['household_id']
                month = self.metadata.iloc[global_idx]['month']
                fuel = self.metadata.iloc[global_idx]['fuel_type']

                stats = {
                    'household_id': household_id,
                    'month': month,
                    'fuel_type': fuel,
                    'data_type': 'synthetic'
                }

                # Distances to train (cross-group — pre-computed from chunked cdist)
                if 'train_synth' in self.cross_group_stats and len(train_idx) > 0:
                    cg = self.cross_group_stats['train_synth']
                    stats['mean_dist_to_train'] = float(cg.synth_row_means[i])
                    stats['min_dist_to_train'] = float(cg.synth_row_mins[i])

                # Distances to test (cross-group — pre-computed from chunked cdist)
                if 'test_synth' in self.cross_group_stats and len(test_idx) > 0:
                    cg = self.cross_group_stats['test_synth']
                    stats['mean_dist_to_test'] = float(cg.synth_row_means[i])
                    stats['min_dist_to_test'] = float(cg.synth_row_mins[i])

                # Distances to other synthetic (within-group — only when not sampled)
                if not synth_sampled and 'synth_synth' in self.distance_matrices:
                    distances = self.distance_matrices['synth_synth'][i, :]
                    valid = distances[~np.isnan(distances)]
                    if len(valid) > 0:
                        stats['mean_dist_to_synth'] = np.mean(valid)
                        stats['min_dist_to_synth'] = np.min(valid)

                household_stats.append(stats)

        # Process train households
        if len(train_idx) > 0:
            for i, global_idx in enumerate(train_idx):
                household_id = self.metadata.iloc[global_idx]['household_id']
                month = self.metadata.iloc[global_idx]['month']
                fuel = self.metadata.iloc[global_idx]['fuel_type']

                stats = {
                    'household_id': household_id,
                    'month': month,
                    'fuel_type': fuel,
                    'data_type': 'train'
                }

                # Within-group distances (only when not sampled)
                if not train_sampled and 'train_train' in self.distance_matrices:
                    distances = self.distance_matrices['train_train'][i, :]
                    valid = distances[~np.isnan(distances)]
                    if len(valid) > 0:
                        stats['mean_dist_to_train'] = np.mean(valid)
                        stats['min_dist_to_train'] = np.min(valid)

                # Cross-group distances (pre-computed column stats from chunked cdist)
                if 'train_synth' in self.cross_group_stats and len(synth_idx) > 0:
                    cg = self.cross_group_stats['train_synth']
                    stats['mean_dist_to_synth'] = float(cg.other_col_means[i])
                    stats['min_dist_to_synth'] = float(cg.other_col_mins[i])

                household_stats.append(stats)

        # Process test households
        if len(test_idx) > 0:
            for i, global_idx in enumerate(test_idx):
                household_id = self.metadata.iloc[global_idx]['household_id']
                month = self.metadata.iloc[global_idx]['month']
                fuel = self.metadata.iloc[global_idx]['fuel_type']

                stats = {
                    'household_id': household_id,
                    'month': month,
                    'fuel_type': fuel,
                    'data_type': 'test'
                }

                # Within-group distances (only when not sampled)
                if not test_sampled and 'test_test' in self.distance_matrices:
                    distances = self.distance_matrices['test_test'][i, :]
                    valid = distances[~np.isnan(distances)]
                    if len(valid) > 0:
                        stats['mean_dist_to_test'] = np.mean(valid)
                        stats['min_dist_to_test'] = np.min(valid)

                # Cross-group distances (pre-computed column stats from chunked cdist)
                if 'test_synth' in self.cross_group_stats and len(synth_idx) > 0:
                    cg = self.cross_group_stats['test_synth']
                    stats['mean_dist_to_synth'] = float(cg.other_col_means[i])
                    stats['min_dist_to_synth'] = float(cg.other_col_mins[i])

                household_stats.append(stats)

        stats_df = pd.DataFrame(household_stats)

        # Save to CSV
        output_path = self.output_folder / 'per_household_statistics.csv'
        stats_df.to_csv(output_path, index=False, float_format='%.4f')
        self._log(f"Per-household statistics saved to {output_path}")

        return stats_df

    def detect_privacy_leakage(self) -> pd.DataFrame:
        """
        Detect potential privacy leakage by comparing synthetic-to-train
        vs synthetic-to-test distances.
        """
        self._log("Detecting privacy leakage...")

        if 'train_synth' not in self.cross_group_stats or 'test_synth' not in self.cross_group_stats:
            self._log("Missing train_synth or test_synth stats. Skipping privacy detection.", 'warning')
            return pd.DataFrame()

        synth_idx = np.where(self.metadata['data_type'] == 'synthetic')[0]
        train_cg = self.cross_group_stats['train_synth']
        test_cg = self.cross_group_stats['test_synth']

        privacy_records = []
        all_mean_dists_train = train_cg.synth_row_means.tolist()
        all_mean_dists_test = test_cg.synth_row_means.tolist()

        for i, global_idx in enumerate(synth_idx):
            household_id = self.metadata.iloc[global_idx]['household_id']
            month = self.metadata.iloc[global_idx]['month']
            fuel = self.metadata.iloc[global_idx]['fuel_type']

            min_distance_train = float(train_cg.synth_row_mins[i])
            mean_distance_train = float(train_cg.synth_row_means[i])
            min_distance_test = float(test_cg.synth_row_mins[i])
            mean_distance_test = float(test_cg.synth_row_means[i])

            mean_distance_diff = mean_distance_train - mean_distance_test
            is_closer_to_train = mean_distance_diff < 0

            privacy_records.append({
                'household_id': household_id,
                'month': month,
                'fuel_type': fuel,
                'min_distance_to_train': min_distance_train,
                'mean_distance_to_train': mean_distance_train,
                'min_distance_to_test': min_distance_test,
                'mean_distance_to_test': mean_distance_test,
                'mean_distance_diff': mean_distance_diff,
                'is_closer_to_train': is_closer_to_train,
            })

        privacy_df = pd.DataFrame(privacy_records)

        # Statistical test
        t_stat, p_value = ttest_rel(all_mean_dists_train, all_mean_dists_test)

        mean_diff = np.mean(np.array(all_mean_dists_train) - np.array(all_mean_dists_test))
        std_diff = np.std(np.array(all_mean_dists_train) - np.array(all_mean_dists_test), ddof=1)
        cohens_d = mean_diff / std_diff if std_diff > 0 else 0

        # Flag high-risk households
        diff_threshold = np.percentile(privacy_df['mean_distance_diff'], 25)
        privacy_df['is_high_privacy_risk'] = privacy_df['mean_distance_diff'] < diff_threshold

        # Log results
        self._log(f"Privacy leakage statistical test (paired t-test):")
        self._log(f"  Mean distance to train: {np.mean(all_mean_dists_train):.4f}")
        self._log(f"  Mean distance to test: {np.mean(all_mean_dists_test):.4f}")
        self._log(f"  Mean difference (train - test): {mean_diff:.4f}")
        self._log(f"  t-statistic: {t_stat:.4f}, p-value: {p_value:.4e}")
        self._log(f"  Cohen's d (effect size): {cohens_d:.4f}")

        if p_value < 0.05:
            if mean_diff < 0:
                self._log("  ⚠ SIGNIFICANT: Synthetic closer to TRAIN than TEST (memorization detected)", 'warning')
            else:
                self._log("  ✓ Synthetic farther from train than test (good!)")
        else:
            self._log("  ✓ No significant difference between train and test distances")

        # Save results
        output_path = self.output_folder / 'privacy_leakage_analysis.csv'
        privacy_df.to_csv(output_path, index=False, float_format='%.4f')
        self._log(f"Privacy leakage analysis saved to {output_path}")

        # Save statistical test results
        stats_results = {
            'test_type': 'paired_t_test',
            'n_synthetic': len(synth_idx),
            'n_train': train_cg.n_other_cols,
            'n_test': test_cg.n_other_cols,
            'mean_dist_to_train': np.mean(all_mean_dists_train),
            'mean_dist_to_test': np.mean(all_mean_dists_test),
            'mean_difference': mean_diff,
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'significant': p_value < 0.05,
            'memorization_detected': (p_value < 0.05) and (mean_diff < 0)
        }
        stats_df = pd.DataFrame([stats_results])
        stats_output = self.output_folder / 'privacy_statistical_test.csv'
        stats_df.to_csv(stats_output, index=False, float_format='%.6f')
        self._log(f"Statistical test results saved to {stats_output}")

        return privacy_df

    def detect_outliers(self) -> pd.DataFrame:
        """
        Detect outlier households based on within-group distance patterns.

        When a within-group matrix was sampled, that group is skipped because
        the random subsample breaks the row-to-household index correspondence
        (row i in the sampled matrix does not correspond to the i-th household
        in metadata). The aggregate distance statistics from
        generate_summary_statistics() remain valid for sampled matrices.
        """
        self._log("Detecting outlier households...")

        outlier_records = []

        # Detect outliers in each group using within-group distance matrices
        for group_name, matrix_name in [
            ('train', 'train_train'),
            ('test', 'test_test'),
            ('synthetic', 'synth_synth')
        ]:
            if matrix_name not in self.distance_matrices:
                continue

            group_idx = np.where(self.metadata['data_type'] == group_name)[0]
            distances = self.distance_matrices[matrix_name]

            # Skip groups where within-group matrix was sampled — row indices
            # no longer correspond to metadata order
            if distances.shape[0] < len(group_idx):
                self._log(f"Skipping outlier detection for {group_name}: within-group "
                          f"matrix was sampled ({distances.shape[0]} of {len(group_idx)} rows)")
                continue

            all_distances = distances.flatten()
            all_distances = all_distances[~np.isnan(all_distances)]

            if len(all_distances) == 0:
                continue

            threshold = np.percentile(all_distances, self.outlier_threshold_percentile)

            # Pre-sort for O(log n) percentile lookup instead of O(n) per call
            sorted_distances = np.sort(all_distances)
            n_distances = len(sorted_distances)

            for i, global_idx in enumerate(group_idx):
                household_id = self.metadata.iloc[global_idx]['household_id']
                month = self.metadata.iloc[global_idx]['month']
                fuel = self.metadata.iloc[global_idx]['fuel_type']

                row_distances = distances[i, :]
                valid = row_distances[~np.isnan(row_distances)]

                if len(valid) > 0:
                    mean_dist = np.mean(valid)
                    is_outlier = mean_dist > threshold

                    # Use binary search for O(log n) percentile calculation
                    percentile = 100.0 * np.searchsorted(sorted_distances, mean_dist) / n_distances

                    outlier_records.append({
                        'household_id': household_id,
                        'month': month,
                        'fuel_type': fuel,
                        'data_type': group_name,
                        'mean_distance': mean_dist,
                        'outlier_threshold': threshold,
                        'is_outlier': is_outlier,
                        'percentile': percentile
                    })

        outlier_df = pd.DataFrame(outlier_records)

        output_path = self.output_folder / 'outlier_detection.csv'
        outlier_df.to_csv(output_path, index=False, float_format='%.4f')
        self._log(f"Outlier detection results saved to {output_path}")

        return outlier_df

    def plot_pca_projection(self):
        """Create 2D PCA projection of the feature space."""
        self._log("Generating PCA projection visualization...")

        from sklearn.decomposition import PCA

        # Apply PCA
        pca = PCA(n_components=2, random_state=42)
        features_2d = pca.fit_transform(self.feature_matrix_processed)
        explained_var = pca.explained_variance_ratio_

        # Save projection data
        pca_df = pd.DataFrame({
            'pc1': features_2d[:, 0],
            'pc2': features_2d[:, 1],
            'data_type': self.metadata['data_type'].values,
            'household_id': self.metadata['household_id'].values,
            'month': self.metadata['month'].values,
        })
        pca_df.to_csv(self.output_folder / 'pca_projection.csv', index=False)
        self._log(f"PCA projection data saved to {self.output_folder / 'pca_projection.csv'}")

        # Save loadings and explained variance
        loadings_df = pd.DataFrame(
            pca.components_.T,
            columns=['PC1', 'PC2'],
            index=self.feature_names
        )
        loadings_df.index.name = 'feature'
        loadings_df.loc['_explained_variance_ratio'] = explained_var
        loadings_df.to_csv(self.output_folder / 'pca_loadings.csv')
        self._log(f"PCA loadings saved to {self.output_folder / 'pca_loadings.csv'}")

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 9))

        # Define colors for each data type
        type_config = {
            'train': {'color': '#e74c3c', 'label': 'Training', 'marker': 'o', 'size': 80, 'alpha': 0.6},
            'test': {'color': '#3498db', 'label': 'Test', 'marker': 's', 'size': 80, 'alpha': 0.6},
            'synthetic': {'color': '#27ae60', 'label': 'Synthetic', 'marker': 'D', 'size': 100, 'alpha': 0.8}
        }

        for data_type, config in type_config.items():
            mask = self.metadata['data_type'] == data_type
            if mask.sum() == 0:
                continue

            ax.scatter(features_2d[mask, 0], features_2d[mask, 1],
                       c=config['color'], label=config['label'],
                       marker=config['marker'], s=config['size'],
                       alpha=config['alpha'], edgecolors='black', linewidth=0.5)

        ax.set_xlabel(f'PC1 ({explained_var[0]:.1%} variance)', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'PC2 ({explained_var[1]:.1%} variance)', fontsize=12, fontweight='bold')
        ax.set_title('PCA Projection: Household-Month Feature Space (Month-Stratified)',
                     fontsize=14, fontweight='bold', pad=15)

        ax.legend(loc='best', fontsize=11, frameon=True, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')

        # Interpretation guide
        interpretation_text = (
            'Privacy Assessment:\n'
            '  ✓ Synthetic separate from train → Good privacy\n'
            '  ⚠ Synthetic overlapping train → Memorization risk\n'
            f'  Total variance explained: {sum(explained_var):.1%}'
        )
        ax.text(0.02, 0.98, interpretation_text,
                transform=ax.transAxes, fontsize=9, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.8', facecolor='white', alpha=0.8))

        plt.tight_layout()

        output_path = self.output_folder / 'pca_projection.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        self._log(f"PCA projection saved to {output_path}")
        self._log(f"PC1 explains {explained_var[0]:.1%}, PC2 explains {explained_var[1]:.1%}")

    def plot_distance_distributions(self):
        """Create violin plots comparing distance distributions."""
        self._log("Generating distance distribution plots...")

        MAX_SAMPLES = 50000  # Limit samples for violin plots to avoid OOM

        # Privacy-focused plot: synth-to-train vs synth-to-test
        if 'train_synth' in self.cross_group_stats and 'test_synth' in self.cross_group_stats:
            fig, ax = plt.subplots(figsize=(10, 6))

            train_cg = self.cross_group_stats['train_synth']
            test_cg = self.cross_group_stats['test_synth']

            # Use exact global means for annotation
            train_mean = train_cg.global_mean
            test_mean = test_cg.global_mean

            # Use reservoir samples for violin plot (already bounded in size)
            train_dists = train_cg.reservoir_sample
            test_dists = test_cg.reservoir_sample

            # Subsample if reservoir is larger than MAX_SAMPLES
            if len(train_dists) > MAX_SAMPLES:
                train_dists = np.random.choice(train_dists, MAX_SAMPLES, replace=False)
            if len(test_dists) > MAX_SAMPLES:
                test_dists = np.random.choice(test_dists, MAX_SAMPLES, replace=False)

            # Use numpy arrays for type labels - much more memory efficient
            data = pd.DataFrame({
                'Distance': np.concatenate([train_dists, test_dists]),
                'Type': np.concatenate([
                    np.full(len(train_dists), 'Synth-to-Train'),
                    np.full(len(test_dists), 'Synth-to-Test')
                ])
            })
            data.to_csv(self.output_folder / 'distance_distributions_privacy.csv', index=False)
            self._log(f"Privacy distribution data saved to {self.output_folder / 'distance_distributions_privacy.csv'}")

            sns.violinplot(x='Type', y='Distance', data=data, ax=ax, palette=['#e74c3c', '#3498db'])
            ax.set_title('Privacy Analysis: Synthetic Distances to Train vs Test',
                         fontsize=14, fontweight='bold')
            ax.set_ylabel('Distance', fontsize=12)
            ax.set_xlabel('')

            # Add annotation (using pre-computed full means)
            annotation = f'Mean to Train: {train_mean:.2f}\nMean to Test: {test_mean:.2f}'
            if train_mean < test_mean:
                annotation += '\n⚠ Closer to train (memorization risk)'
            else:
                annotation += '\n✓ Farther from train (good)'

            ax.text(0.98, 0.98, annotation, transform=ax.transAxes,
                    fontsize=10, va='top', ha='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            plt.tight_layout()
            output_path = self.output_folder / 'distance_distributions_privacy.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            self._log(f"Privacy distribution plot saved to {output_path}")

        # Within-group distances comparison
        # Use numpy arrays directly instead of list-of-dicts to avoid OOM
        MAX_SAMPLES_PER_GROUP = 50000  # Limit samples for violin plot
        distance_arrays = []
        type_arrays = []
        color_map = {
            'Train-Train': '#e74c3c',
            'Test-Test': '#3498db',
            'Synth-Synth': '#27ae60'
        }
        present_types = []
        for matrix_name, label in [
            ('train_train', 'Train-Train'),
            ('test_test', 'Test-Test'),
            ('synth_synth', 'Synth-Synth')
        ]:
            if matrix_name in self.distance_matrices:
                dists = self.distance_matrices[matrix_name].flatten()
                dists = dists[~np.isnan(dists)]
                # Sample if too many distances to avoid memory issues
                if len(dists) > MAX_SAMPLES_PER_GROUP:
                    dists = np.random.choice(dists, MAX_SAMPLES_PER_GROUP, replace=False)
                distance_arrays.append(dists)
                type_arrays.append(np.full(len(dists), label))
                present_types.append(label)

        if distance_arrays:
            fig, ax = plt.subplots(figsize=(10, 6))
            # Concatenate numpy arrays directly - much more memory efficient
            data = pd.DataFrame({
                'Distance': np.concatenate(distance_arrays),
                'Type': np.concatenate(type_arrays)
            })
            data.to_csv(self.output_folder / 'distance_distributions_within_group.csv', index=False)
            self._log(f"Within-group distribution data saved to {self.output_folder / 'distance_distributions_within_group.csv'}")

            # Build palette dynamically based on present types
            palette = [color_map[t] for t in present_types]
            sns.violinplot(x='Type', y='Distance', hue='Type', data=data, ax=ax,
                           palette=palette, legend=False)
            ax.set_title('Within-Group Distance Distributions', fontsize=14, fontweight='bold')
            ax.set_ylabel('Distance', fontsize=12)
            ax.set_xlabel('')

            plt.tight_layout()
            output_path = self.output_folder / 'distance_distributions_within_group.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            self._log(f"Within-group distribution plot saved to {output_path}")

    def plot_paired_differences(self):
        """Create scatter plot of paired differences (train vs test distances)."""
        self._log("Generating paired differences plot...")

        if 'train_synth' not in self.cross_group_stats or 'test_synth' not in self.cross_group_stats:
            self._log("Missing cross-group stats for paired differences plot", 'warning')
            return

        synth_idx = np.where(self.metadata['data_type'] == 'synthetic')[0]

        # Pre-computed mean distances for each synthetic household-month
        mean_to_train = self.cross_group_stats['train_synth'].synth_row_means
        mean_to_test = self.cross_group_stats['test_synth'].synth_row_means

        fig, ax = plt.subplots(figsize=(10, 10))

        ax.scatter(mean_to_train, mean_to_test, alpha=0.6, s=50, c='#27ae60', edgecolors='black')

        # Add diagonal line (equal distances)
        min_val = min(mean_to_train.min(), mean_to_test.min())
        max_val = max(mean_to_train.max(), mean_to_test.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Equal distance')

        ax.set_xlabel('Mean Distance to Train', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean Distance to Test', fontsize=12, fontweight='bold')
        ax.set_title('Paired Distances: Each Synthetic Household-Month',
                     fontsize=14, fontweight='bold')

        # Count points above/below diagonal
        above = np.sum(mean_to_test > mean_to_train)
        below = np.sum(mean_to_test < mean_to_train)
        total = len(mean_to_train)

        annotation = (f'Above diagonal (closer to train): {above} ({above/total*100:.1f}%)\n'
                      f'Below diagonal (farther from train): {below} ({below/total*100:.1f}%)')
        ax.text(0.02, 0.98, annotation, transform=ax.transAxes,
                fontsize=10, va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = self.output_folder / 'paired_differences.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        self._log(f"Paired differences plot saved to {output_path}")

    def generate_analysis_report(self):
        """Generate human-readable text report."""
        self._log("Generating comprehensive analysis report...")

        lines = []
        lines.append("=" * 80)
        lines.append("DISTANCE ANALYSIS REPORT (Month-Stratified)")
        lines.append("=" * 80)
        lines.append("")

        # Dataset overview
        lines.append("DATASET OVERVIEW")
        lines.append("-" * 80)
        type_counts = self.metadata['data_type'].value_counts()
        for data_type, count in type_counts.items():
            lines.append(f"  {data_type.capitalize()}: {count} household-months")
        lines.append(f"  Features per household-month: {self.feature_matrix.shape[1]}")
        lines.append(f"  Distance metric: {self.metric}")
        lines.append(f"  Feature normalization: {'Yes' if self.normalize_features else 'No'}")
        lines.append("")

        # Distance distributions
        if 'summary_statistics' in self.distance_summaries:
            lines.append("DISTANCE DISTRIBUTIONS")
            lines.append("-" * 80)
            for _, row in self.distance_summaries['summary_statistics'].iterrows():
                lines.append(f"\n  {row['matrix_type'].upper()}")
                lines.append(f"    Comparisons: {int(row['n_comparisons'])}")
                lines.append(f"    Mean: {row['mean']:.4f} ± {row['std']:.4f}")
                lines.append(f"    Range: [{row['min']:.4f}, {row['max']:.4f}]")
                lines.append(f"    Quartiles: Q1={row['q25']:.4f}, Median={row['median']:.4f}, Q3={row['q75']:.4f}")
            lines.append("")

        # Privacy leakage
        privacy_path = self.output_folder / 'privacy_leakage_analysis.csv'
        stats_path = self.output_folder / 'privacy_statistical_test.csv'

        if privacy_path.exists():
            privacy_df = pd.read_csv(privacy_path)
            lines.append("PRIVACY LEAKAGE ANALYSIS")
            lines.append("-" * 80)

            if stats_path.exists():
                stats_df = pd.read_csv(stats_path)
                stats = stats_df.iloc[0]

                lines.append("  POPULATION-LEVEL STATISTICAL TEST (Paired t-test)")
                lines.append(f"    Synthetic household-months: {int(stats['n_synthetic'])}")
                lines.append(f"    Train household-months: {int(stats['n_train'])}")
                lines.append(f"    Test household-months: {int(stats['n_test'])}")
                lines.append("")
                lines.append(f"    Mean distance to train: {stats['mean_dist_to_train']:.4f}")
                lines.append(f"    Mean distance to test:  {stats['mean_dist_to_test']:.4f}")
                lines.append(f"    Mean difference: {stats['mean_difference']:.4f}")
                lines.append("")
                lines.append(f"    t-statistic: {stats['t_statistic']:.4f}")
                lines.append(f"    p-value: {stats['p_value']:.6f}")
                lines.append(f"    Cohen's d: {stats['cohens_d']:.4f}")
                lines.append("")

                if stats['memorization_detected']:
                    lines.append("    ⚠ SIGNIFICANT MEMORIZATION DETECTED")
                    lines.append("      → Synthetic data closer to TRAIN than TEST")
                elif stats['significant']:
                    lines.append("    ✓ Synthetic data farther from train (good!)")
                else:
                    lines.append("    ✓ No significant difference")
                lines.append("")

            high_risk = privacy_df['is_high_privacy_risk'].sum()
            closer_to_train = privacy_df['is_closer_to_train'].sum()
            total = len(privacy_df)

            lines.append("  INDIVIDUAL HOUSEHOLD PRIVACY RISK")
            lines.append(f"    Total synthetic: {total}")
            lines.append(f"    Closer to train: {closer_to_train} ({closer_to_train/total*100:.1f}%)")
            lines.append(f"    High privacy risk: {high_risk} ({high_risk/total*100:.1f}%)")
            lines.append("")

        # Outlier findings
        outlier_path = self.output_folder / 'outlier_detection.csv'
        if outlier_path.exists():
            try:
                outlier_df = pd.read_csv(outlier_path)
                if len(outlier_df) > 0:
                    lines.append("OUTLIER DETECTION")
                    lines.append("-" * 80)
                    for data_type in outlier_df['data_type'].unique():
                        type_df = outlier_df[outlier_df['data_type'] == data_type]
                        outlier_count = type_df['is_outlier'].sum()
                        lines.append(f"  {data_type.capitalize()}: {outlier_count}/{len(type_df)} "
                                     f"({outlier_count/len(type_df)*100:.1f}%)")
                    lines.append("")
            except Exception:
                pass

        lines.append("=" * 80)
        lines.append("END OF REPORT")
        lines.append("=" * 80)

        report_text = "\n".join(lines)
        output_path = self.output_folder / 'analysis_report.txt'
        with open(output_path, 'w') as f:
            f.write(report_text)

        self._log(f"Analysis report saved to {output_path}")
        print("\n" + report_text)

    def _save_feature_artifacts(self):
        """Save feature matrix, feature names, and metadata to disk."""
        matrix_path = self.output_folder / 'feature_matrix.npy'
        np.save(matrix_path, self.feature_matrix)
        self._log(f"Feature matrix saved to {matrix_path} (shape: {self.feature_matrix.shape})")

        names_path = self.output_folder / 'feature_names.txt'
        with open(names_path, 'w') as f:
            for name in self.feature_names:
                f.write(f"{name}\n")
        self._log(f"Feature names saved to {names_path} ({len(self.feature_names)} features)")

        metadata_path = self.output_folder / 'metadata.csv'
        self.metadata.to_csv(metadata_path, index=False)
        self._log(f"Metadata saved to {metadata_path} ({len(self.metadata)} rows)")

    def run_full_analysis(self):
        """Execute complete distance analysis pipeline."""
        self._log("Running full distance analysis pipeline...")

        # 1. Calculate distances
        self.calculate_all_distances()

        # 2. Generate statistics
        self.generate_summary_statistics()
        self.generate_per_household_statistics()

        # 3. Privacy and outlier analysis
        self.detect_privacy_leakage()
        self.detect_outliers()

        # 4. Visualizations
        self.plot_pca_projection()
        self.plot_distance_distributions()
        self.plot_paired_differences()

        # 5. Report
        self.generate_analysis_report()

        # 6. Save feature artifacts
        self._save_feature_artifacts()

        self._log("Full analysis complete!")
