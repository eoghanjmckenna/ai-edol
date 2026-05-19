"""
Data Sharding Module for AI-EDOL Privacy-Preserving Synthetic Data Generation.

This module implements a comprehensive data sharding pipeline with k-anonymity privacy
guarantees for training transformer models on sensitive household energy data.

Key Features:
    - Comprehensive filtering pipeline (missingness, PV exclusion, variable completeness, k-anonymity)
    - Configurable conditioning variable strategy (masked OR unmasked)
    - Train/validation/test split assignment with k-anonymity filtering
    - Multi-shard training data generation with augmentation
    - K10 aggregated synthetic data generation for privacy benchmarking

Conditioning Variable Approaches:
    1. Unmasked (simplified): Uses filtered conditioning variables directly (k≥3 anonymity on input)
       - Configured via: use_mask_tokens=false in sharding.data_filtering
       - Training uses unmasked shards exclusively
       - Val/test include unmasked versions for secondary validation metrics

    2. Masked (legacy): Progressive k-anonymity masking with curriculum learning (k≥10 anonymity)
       - Configured via: use_mask_tokens=true in sharding.data_filtering
       - Training uses either constrained OR unconstrained masked shards
       - Val/test include masked + unmasked versions

Filtering Pipeline:
    1. Household-level missingness calculation (electricity and gas)
    2. PV household exclusion (exclude exporters)
    3. Variable completeness check (all required conditioning variables present)
    4. K-anonymity filtering (k≥3 for input filtering, k≥10 for masking-based privacy)

    Produces filtered metadata (metadata_catalog_filtered, metadata_tokenised_filtered) used
    for all downstream shard creation.

K10 Aggregation:
    Generates privacy baseline synthetic data by aggregating time series from 10 real
    households that share conditioning variable values. Serves as privacy benchmark for
    Membership Inference Attack (MIA) evaluation. If transformer synthetic data provides
    at least equivalent privacy protection as k10 aggregates, both can be publicly released.

    The k10 aggregation is integrated into create_masked_conditioning_variables() and
    executes inline during masked conditioning table generation, ensuring the same k≥10
    groups are used for both privacy masking and synthetic data generation.

Configuration:
    See configs/sharding.yaml for full configuration options including:
    - data_filtering: Filtering thresholds and conditioning variable selection
    - k_anonymity: Privacy masking settings (min_group_size, augmentation_factor)
    - k10_aggregation: K10 synthetic data generation settings

Author: AI-EDOL Team
License: See LICENSE file
"""

import os
import pandas as pd
import numpy as np
import torch
import pickle
import json
import time
import multiprocessing as mp
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

# Import the existing preprocessing function
from preprocessing.preprocessing import process_multi_household_data_with_conditioning
from preprocessing.data_loader import _process_household_data


def _calculate_missingness_worker(args: tuple) -> dict:
    """
    Worker function for parallel missingness and mean consumption calculation.

    Calculates both data quality metrics (missingness) and mean consumption
    values used for stratified sampling.

    Args:
        args: Tuple of (file_path, puprn)

    Returns:
        dict containing:
            - puprn: Household identifier
            - elec_missingness: Fraction of NaN values in electricity column (0.0-1.0)
            - gas_missingness: Fraction of NaN values in gas column (0.0-1.0)
            - elec_mean: Mean electricity consumption (Wh per half-hour), excluding NaNs
            - gas_mean: Mean gas consumption (Wh per half-hour), excluding NaNs
            - error: Error message string if processing failed, else None
    """
    file_path, puprn = args
    try:
        df = pd.read_pickle(file_path)

        elec_col = 'Clean_elec_net_Wh'
        gas_col = 'Clean_gas_Wh'

        elec_missing = df[elec_col].isna().mean() if elec_col in df.columns else 1.0
        gas_missing = df[gas_col].isna().mean() if gas_col in df.columns else 1.0

        elec_mean = df.loc[~df[elec_col].isna(), elec_col].mean()
        gas_mean = df.loc[~df[gas_col].isna(), gas_col].mean()

        return {
            'puprn': puprn,
            'elec_missingness': elec_missing,
            'gas_missingness': gas_missing,
            'elec_mean': elec_mean,
            'gas_mean': gas_mean,
            'error': None
        }
    except Exception as e:
        return {
            'puprn': puprn,
            'elec_missingness': None,
            'gas_missingness': None,
            'elec_mean': None,
            'gas_mean': None,
            'error': str(e)
        }


class DataSharder:
    def __init__(self, config: Dict[str, Any], run_path: str, logger: Optional[Any] = None):
        """
        Initialize data sharder with configuration, run path, and optional logger.

        Args:
            config: Configuration dictionary containing sharding, training, and k10_aggregation settings
            run_path: Path to experiment run directory for saving outputs
            logger: Optional logger instance for structured logging

        Attributes:
            k10_aggregate_count: Counter for k10 synthetic households generated
            k10_metadata: List storing metadata for each k10 aggregate (source PUPRNs, conditioning vars, etc.)
        """
        self.config = config
        self.run_path = run_path
        self.logger = logger
        self.metadata_catalog = None

        # Load k-anonymity configuration
        self._load_k_anonymity_config()

        # Set random seed for reproducible sharding
        self._set_random_seed()

        # Initialize k10 aggregation tracking
        self.k10_aggregate_count: int = 0
        self.k10_metadata: List[Dict[str, Any]] = []
    
    def _set_random_seed(self) -> None:
        """Set random seed for reproducible sharding pipeline."""
        # Get random seed from sharding config, fall back to training config, then default
        sharding_config = self.config.get('sharding', {})
        training_config = self.config.get('training', {})
        
        random_seed = sharding_config.get('random_seed') or training_config.get('random_seed', 42)
        
        # Set seeds for all random number generators
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)
        
        self.random_seed = random_seed
        self._log(f"Set random seed to {random_seed} for reproducible sharding")

    def _load_k_anonymity_config(self) -> None:
        """Load k-anonymity configuration and set instance variables."""
        k_config = self.config.get('sharding', {}).get('k_anonymity', {})
        self.k_anonymity = k_config.get('min_group_size', 10)  # Default: k=10

        # Warn if using reduced k-anonymity (testing mode)
        if self.k_anonymity < 10:
            warning_msg = (
                f"⚠️  WARNING: k-anonymity set to k={self.k_anonymity} (< 10). "
                "This is ONLY suitable for testing! "
                "DO NOT use for production/TRE deployment."
            )
            self._log(warning_msg, level='warning')

    def _should_create_constrained_shards(self) -> bool:
        """Check if constrained shards should be created."""
        k_anonymity_config = self.config.get('sharding', {}).get('k_anonymity', {})
        return k_anonymity_config.get('create_constrained_shards', False)  
    
    def _log(self, message: str, level: str = 'info') -> None:
        """Log message using logger if available, otherwise print with [SHARDING] prefix."""
        if self.logger:
            if level == 'info':
                self.logger.info(message)
            elif level == 'warning':
                self.logger.warning(message)
            elif level == 'error':
                self.logger.error(message)
            elif level == 'debug':
                self.logger.debug(message)
        else:
            print(f"[SHARDING] {message}", flush=True)
        
    def load_household_metadata(self) -> pd.DataFrame:
        """
        Step 1: Load all household metadata (PUPRN, file paths, conditioning variables).

        Returns:
            DataFrame with columns: puprn, file_path, B5, B5_err, C1_new
        """
        self._log("Step 1: Loading household metadata...")

        # Get tokenised data directory from config
        training_config = self.config.get('training', {})
        tokenised_dir = training_config['data']['path_to_module_1_data']
        
        if not tokenised_dir:
            raise ValueError("tokenised_data_dir not specified in sharding config")
        
        if not os.path.exists(tokenised_dir):
            raise FileNotFoundError(f"Tokenised data directory not found: {tokenised_dir}")
        
        # Get all tokenised household files
        puprn_files = [f for f in os.listdir(tokenised_dir) if f.endswith('.pkl')]
        
        if not puprn_files:
            raise ValueError(f"No .pkl files found in tokenised data directory: {tokenised_dir}")
        
        self._log(f"Found {len(puprn_files)} tokenised household files")
        
        # Create basic file metadata
        # Extract PUPRN from filename format: {puprn}_tokenised.pkl
        puprns = []
        for f in puprn_files:
            # Split on underscore and extract the PUPRN part
            filename_parts = f.split('_')
            if len(filename_parts) >= 2 and filename_parts[1] == 'tokenised.pkl':
                puprn = filename_parts[0]
                puprns.append(puprn)
            else:
                self._log(f"Unexpected filename format: {f}", level='warning')
        
        file_data = {
            'puprn': puprns,
            'file_path': [os.path.join(tokenised_dir, f"{puprn}_tokenised.pkl") for puprn in puprns]
        }
        
        # Load SERL survey data for conditioning variables
        serl_survey_path = training_config.get('data', {}).get('path_to_serl_survey_data')
        
        if not serl_survey_path:
            raise ValueError("path_to_serl_survey_data not specified in training config")
        
        if not os.path.exists(serl_survey_path):
            raise FileNotFoundError(f"SERL survey data file not found: {serl_survey_path}")
        
        self._log(f"Loading SERL survey data from: {serl_survey_path}")
        serl_survey = pd.read_csv(serl_survey_path)
        
        # load EPC data
        epc_data_path = training_config.get('data',{}).get('path_to_epc_data')
        if not epc_data_path:
            raise ValueError("path_to_epc_data not specified in training config")
        if not os.path.exists(epc_data_path):
            raise FileNotFoundError(f"EPC data file not found: {epc_data_path}")
        self._log(f"Loading EPC data from: {epc_data_path}")
        epc_data = pd.read_csv(epc_data_path)

        # load participant summary data
        participant_summary_path = training_config.get('data',{}).get('path_to_participant_summary_data')
        if not participant_summary_path:
            raise ValueError("path_to_participant_summary_data not specified in training config")
        if not os.path.exists(participant_summary_path):
            raise FileNotFoundError(f"Participant summary data file not found: {participant_summary_path}")
        self._log(f"Loading participant summary data from: {participant_summary_path}")
        participant_summary = pd.read_csv(participant_summary_path)

        # load follow-up survey data
        followup_survey_path = training_config.get('data',{}).get('path_to_followup_survey_data')
        if not followup_survey_path:
            raise ValueError("path_to_followup_survey_data not specified in training config")
        if not os.path.exists(followup_survey_path):
            raise FileNotFoundError(f"Follow-up survey data file not found: {followup_survey_path}")
        self._log(f"Loading follow-up survey data from: {followup_survey_path}")
        followup_survey = pd.read_csv(followup_survey_path, encoding='latin-1')
        
        # load list of exporter puprns
        exporter_puprns_path = training_config.get('data',{}).get('path_to_list_of_exporter_puprns')
        if not exporter_puprns_path:
            raise ValueError("path_to_list_of_exporter_puprns not specified in training config")
        if not os.path.exists(exporter_puprns_path):
            raise FileNotFoundError(f"List of exporter PUPRNs file not found: {exporter_puprns_path}")
        self._log(f"Loading list of exporter PUPRNs from: {exporter_puprns_path}")
        exporter_puprns_df = pd.read_csv(exporter_puprns_path)
        exporter_puprns_df.columns = ['PUPRN']  # Ensure column name is PUPRN
        exporter_puprns_df['is_exporter'] = True  # Mark these PUPRNs as exporters

        # Create metadata DataFrame
        metadata_df = pd.DataFrame(file_data)

        # Merge with SERL survey data to get conditioning variables
        self.metadata_catalog = metadata_df.merge(
            serl_survey[['PUPRN'] + training_config.get('serl_survey_context', [])],
            left_on='puprn',
            right_on='PUPRN',
            how='inner', # only keep households with survey data and energy data
        ).drop('PUPRN', axis=1)

        # merge with epc data
        self.metadata_catalog = self.metadata_catalog.merge(
            epc_data[['PUPRN'] + training_config.get('epc_context', [])],
            left_on='puprn',
            right_on='PUPRN',
            how='left', # don't discard from sample if puprn missing from epc data
        ).drop('PUPRN',axis=1)

        # merge with participant summary data
        participant_summary_cols = ['PUPRN'] + training_config.get('participant_summary_context', [])
        participant_summary_subset = participant_summary[participant_summary_cols]

        self.metadata_catalog = self.metadata_catalog.merge(
            participant_summary_subset,
            left_on='puprn',
            right_on='PUPRN',
            how='inner',
        ).drop('PUPRN',axis=1)

        # merge with follow-up survey
        self.metadata_catalog = self.metadata_catalog.merge(
            followup_survey[['PUPRN'] + training_config.get('followup_survey_context', [])],
            left_on='puprn',
            right_on='PUPRN',
            how='left', # don't discard if puprn not in follow-up survey
            suffixes=('', '_fu')
        ).drop('PUPRN',axis=1)

        # merge with list of exporters
        self.metadata_catalog = self.metadata_catalog.merge(
            exporter_puprns_df,
            left_on='puprn',
            right_on='PUPRN',
            how='left' # don't discard if puprn not in exporter list
        ).drop('PUPRN',axis=1)
        self.metadata_catalog['is_exporter'] = self.metadata_catalog['is_exporter'].fillna(False)

        # Validate and filter conditioning contexts
        self._validate_and_filter_contexts()
        
        
        # Report metadata statistics
        total_households = len(self.metadata_catalog)
        
        # Initialize all households as training by default
        self.metadata_catalog['split'] = 'train'
        
        self._log("Metadata catalog created:")
        self._log(f"  Total households: {total_households}")
        
        # Show conditioning variable distributions
        # Note: Only D4_fu has _fu suffix (due to column collision); other follow-up vars have no collision
        context_vars = ["B5", "B5_err", "C1_new",'A301','A302','A303','A304','A305','A306','A307','A308','A309','A310','A3_Other','A3_sum','A3_err','A301_edit','B1','B4','B9','C5','D4','A1_corr_C', 'A1_err', 'B3_1_yes', 'B3_4_yes', 'C1', 'D4_fu', 'D5']
        for var in context_vars:
            if var in self.metadata_catalog.columns:
                self._log(f"  {var} distribution: {dict(self.metadata_catalog[var].value_counts().sort_index())}")
        
        return self.metadata_catalog
    
    def household_missingness(self):
        """
        Calculate household-level time series missingness for electricity and gas.

        This method computes the proportion of missing data (NaN values) in electricity
        and gas time series for each household, enabling data quality filtering based on
        missingness thresholds. Results are stored in metadata_catalog for downstream
        filtering operations.

        Data Flow:
            1. Loads tokenised household pickle files from training data directory
            2. Calculates missingness rate for Clean_elec_net_Wh and Clean_gas_Wh columns
            3. Updates metadata_catalog with elec_missingness and gas_missingness columns
            4. Validates that PUPRNs in files match metadata_catalog entries

        Configuration Requirements:
            - config['training']['data']['path_to_module_1_data']: Path to tokenised household files
            - Expected filename format: {PUPRN}_tokenised.pkl

        Side Effects:
            - Adds 'elec_missingness' column to self.metadata_catalog (float, 0.0-1.0)
            - Adds 'gas_missingness' column to self.metadata_catalog (float, 0.0-1.0)
            - Initializes columns with np.nan before calculation

        Error Handling:
            - Raises ValueError if tokenised_data_dir not specified in config
            - Raises FileNotFoundError if tokenised_data_dir doesn't exist
            - Raises ValueError if no .pkl files found in directory
            - Warns and skips files with unexpected filename format
            - Warns and skips PUPRNs not found in metadata_catalog
            - Warns and continues on individual household processing errors

        Output Columns:
            - elec_missingness: Proportion of NaN values in electricity time series (0.0 = complete, 1.0 = all missing)
            - gas_missingness: Proportion of NaN values in gas time series (0.0 = complete, 1.0 = all missing)
            - Returns 1.0 if column doesn't exist in household dataframe

        Usage:
            Called early in sharding pipeline before filtering to enable missingness-based
            household exclusion (e.g., config['sharding']['data_filtering']['max_missingness_rate']).

        Notes:
            - Uses multiprocessing for parallel file reading (up to 8 workers)
            - For 13k households, expect ~30-60 seconds runtime (vs several minutes sequential)
            - Missingness filtering should occur after this step in the pipeline

        Raises:
            ValueError: If tokenised_data_dir not configured or no .pkl files found
            FileNotFoundError: If tokenised_data_dir doesn't exist
        """
        self._log("Step 1.1: Calculate household-level missingness...")

        # Get tokenised data directory from config
        training_config = self.config.get('training', {})
        tokenised_dir = training_config['data']['path_to_module_1_data']
        
        if not tokenised_dir:
            raise ValueError("tokenised_data_dir not specified in sharding config")
        
        if not os.path.exists(tokenised_dir):
            raise FileNotFoundError(f"Tokenised data directory not found: {tokenised_dir}")
        
        # Get all tokenised household files
        puprn_files = [f for f in os.listdir(tokenised_dir) if f.endswith('.pkl')]
        
        if not puprn_files:
            raise ValueError(f"No .pkl files found in tokenised data directory: {tokenised_dir}")
        
        self._log(f"Found {len(puprn_files)} tokenised household files")

        # Build list of (file_path, puprn) tuples for parallel processing
        worker_args = []
        skipped_files = []
        for puprn_file in puprn_files:
            # Split on underscore and extract the PUPRN part
            filename_parts = puprn_file.split('_')
            if len(filename_parts) >= 2 and filename_parts[1] == 'tokenised.pkl':
                puprn = filename_parts[0]
                file_path = os.path.join(tokenised_dir, puprn_file)
                worker_args.append((file_path, puprn))
            else:
                skipped_files.append(puprn_file)

        if skipped_files:
            self._log(f"Skipped {len(skipped_files)} files with unexpected filename format", level='warning')

        # Calculate missingness in parallel
        # Determine number of workers: use config value if provided, otherwise auto-detect
        # Hard limit of 8 workers (diminishing returns for I/O-bound work)
        available_cores = mp.cpu_count()
        data_filtering_config = self.config.get('sharding', {}).get('data_filtering', {})
        configured_workers = data_filtering_config.get('num_workers')
        if configured_workers is not None:
            num_workers = max(1, min(configured_workers, 8))
        else:
            num_workers = max(1, min(available_cores - 1, 8))
        self._log(f"\nCalculating time series missingness for {len(worker_args)} households...")
        self._log(f"  Using {num_workers}/{available_cores} CPU cores for parallel processing")

        start_time = time.time()
        with mp.Pool(processes=num_workers) as pool:
            results = pool.map(_calculate_missingness_worker, worker_args)
        elapsed = time.time() - start_time
        self._log(f"  Completed in {elapsed:.1f}s ({len(worker_args)/elapsed:.0f} files/sec)")

        # Initialize columns and batch-update metadata_catalog with results
        self.metadata_catalog['elec_missingness'] = np.nan
        self.metadata_catalog['gas_missingness'] = np.nan
        self.metadata_catalog['elec_mean'] = np.nan
        self.metadata_catalog['gas_mean'] = np.nan

        # Create a lookup dict for fast updates
        puprn_to_idx = {puprn: idx for idx, puprn in enumerate(self.metadata_catalog['puprn'])}

        processed_count = 0
        error_count = 0
        not_found_count = 0

        for result in results:
            puprn = result['puprn']
            if result['error']:
                self._log(f"Warning: Could not process {puprn}: {result['error']}", level='warning')
                error_count += 1
                continue

            if puprn not in puprn_to_idx:
                not_found_count += 1
                continue

            idx = puprn_to_idx[puprn]
            self.metadata_catalog.at[idx, 'elec_missingness'] = result['elec_missingness']
            self.metadata_catalog.at[idx, 'gas_missingness'] = result['gas_missingness']
            self.metadata_catalog.at[idx, 'elec_mean'] = result['elec_mean']
            self.metadata_catalog.at[idx, 'gas_mean'] = result['gas_mean']
            processed_count += 1

        self._log(f"\n  Missingness calculation complete:")
        self._log(f"    Households processed: {processed_count}")
        if error_count > 0:
            self._log(f"    Errors: {error_count}", level='warning')
        if not_found_count > 0:
            self._log(f"    PUPRNs not in metadata_catalog: {not_found_count}", level='warning')  
        
    def _validate_and_filter_contexts(self) -> None:
        """
        Validate and filter households with invalid conditioning contexts.
       
        """
        if self.metadata_catalog is None:
            raise ValueError("metadata_catalog not initialized")
        
        initial_count = len(self.metadata_catalog)

        # Core conditioning context validation (from assess_context_availability)
        # Note: A1_err doesn't have _fu suffix because there's no column collision during merge
        # Fill NaN values with False (treat missing follow-up survey data as "no error")
        valid_mask = (
            (~self.metadata_catalog['B5_err']) &
            (~self.metadata_catalog['A3_err']) &
            (~self.metadata_catalog['A1_err'].fillna(False))
        )
        
        # Apply filter
        self.metadata_catalog = self.metadata_catalog[valid_mask].reset_index(drop=True)
        filtered_count = len(self.metadata_catalog)
        invalid_count = initial_count - filtered_count
        
        # Report validation results
        if invalid_count > 0:
            self._log(f"Filtered out {invalid_count} households with invalid conditioning contexts", level='warning')
            self._log(f"Retained {filtered_count} households with valid conditioning contexts")
        
        if filtered_count == 0:
            raise ValueError("No households with valid conditioning contexts found! Check your data quality.")
        
        # recode negative values as nans
        # Note: Only D4_fu has _fu suffix (due to column collision); other follow-up vars have no collision
        # Only check numeric columns for negative values
        context_vars = ["B5", "B5_err", "C1_new",'A301','A302','A303','A304','A305','A306','A307','A308','A309','A310','A3_Other','A3_sum','A3_err','A301_edit','B1','B4','B9','C5','D4','A1_corr_C', 'A1_err', 'B3_1_yes', 'B3_4_yes', 'C1', 'D4_fu', 'D5']
        for var in context_vars:
            if var in self.metadata_catalog.columns:
                # Skip non-numeric columns
                if not pd.api.types.is_numeric_dtype(self.metadata_catalog[var]):
                    continue
                negative_mask = self.metadata_catalog[var] < 0
                num_negatives = negative_mask.sum()
                if num_negatives > 0:
                    self._log(f"Found {num_negatives} negative values in {var}, recoding as NaN", level='warning')
                    self.metadata_catalog.loc[negative_mask, var] = np.nan

    def rationalise_heating_types(self) -> None:
        '''
        Create simplified single-category heating type (no combinations).
        Priority: None > Gas > Electric > Other
        '''

        # Priority-based categorization (first match wins)
        conditions = [
            self.metadata_catalog['A301'] == 1,  # No central heating
            self.metadata_catalog['A302'] == 1,  # Gas
            (self.metadata_catalog['A303'] == 1) |
            (self.metadata_catalog['A304'] == 1) |
            (self.metadata_catalog['A305'] == 1),  # Electric
            (self.metadata_catalog['A306'] == 1) |
            (self.metadata_catalog['A307'] == 1) |
            (self.metadata_catalog['A308'] == 1) |
            (self.metadata_catalog['A309'] == 1) |
            (self.metadata_catalog['A310'] == 1),  # Other
        ]

        categories = [
            'No central heating',
            'Gas central heating',
            'Electric heating',
            'Other heating'
        ]

        # Use np.select for efficient categorization
        # Use 'Unknown' string instead of np.nan to avoid dtype mismatch
        self.metadata_catalog['heating_type'] = np.select(
            conditions,
            categories,
            default='Unknown'
        )

        # Log distribution
        self._log("Heating type distribution:")
        for cat, count in self.metadata_catalog['heating_type'].value_counts().items():
            self._log(f"  {cat}: {count}")

    def build_floor_area_tokeniser(self,num_bins=10) -> None:
        '''
        totalFloorArea is a continuous variable, which if naively tokenised could lead to vocabulary bloat.
        Instead bin into num_bins quantiles.
        '''
        from collections import Counter
        # Round values and filter out NaN (households without EPC data)
        self.metadata_catalog['total_floor_area_rounded'] = self.metadata_catalog['totalFloorArea'].round()
        valid_floor_areas = self.metadata_catalog['total_floor_area_rounded'].dropna()
        floor_area_counter = Counter(valid_floor_areas)
        from utils.universal_tokeniser_naive import build_boundaries_from_counter
        floor_area_boundaries = build_boundaries_from_counter(floor_area_counter, num_bins=num_bins)
        return floor_area_boundaries

    def tokenise_conditioning_variables(self) -> None:
        """
        Tokenize conditioning variables based on config-specified variable list.

        Processes metadata_catalog to create tokenized representations of conditioning
        variables for model training. Only variables specified in
        config['sharding']['data_filtering']['conditioning_variables'] are included.

        Operations:
            1. Filters variables based on config list
            2. Merges small categories for sparse variables (region, building_type, num_rooms, num_occs)
            3. Creates encoder/decoder mappings (vtoi/itov) for categorical/ordinal variables
            4. Applies quantile binning for continuous variables (total_floor_area)
            5. Validates encoding/decoding with assertions

        Side Effects:
            - Creates self.conditioning_variable_tokenisers (dict of tokenizers with vtoi/itov)
            - Creates self.metadata_tokenised (DataFrame with tokenized values)
            - Modifies metadata_catalog columns for category merging (e.g., 'NORTH EAST'→'NORTH')
            - Saves metadata_tokenised.pkl and conditioning_variable_tokenisers.pkl to disk

        Available Variables (selectable via config):
            region, imd, heating_type, num_rooms, total_floor_area, epc_rating,
            household_income_band, managing_financially, num_occs, building_type,
            age_built, tenure, has_pv, has_ev
        """
        data_filtering_config = self.config.get('sharding', {}).get('data_filtering')
        conditioning_vars_filter = data_filtering_config.get('conditioning_variables')

        self.conditioning_variable_tokenisers = {}
        # a dict of dicts holding original variable source name, encoder, decoder
        # region
        if 'region' in conditioning_vars_filter:
            self.conditioning_variable_tokenisers['region'] = {'original_variable_name': 'Region',
                                                            'source': 'participant_summary',
                                                            'type': 'categorical'}
            # Merge NORTH EAST and NORTH WEST
            mask = self.metadata_catalog['Region'].isin(['NORTH EAST', 'NORTH WEST'])
            self.metadata_catalog.loc[mask,'Region'] = 'NORTH'
        # imd
        if 'imd' in conditioning_vars_filter:
            self.conditioning_variable_tokenisers['imd'] = {'original_variable_name': 'IMD_quintile',
                                                            'source': 'participant_summary',
                                                            'type':'ordinal'}
        # heating type
        if 'heating_type' in conditioning_vars_filter:
            self.rationalise_heating_types()
            self.conditioning_variable_tokenisers['heating_type'] = {'original_variable_name': 'heating_type',
                                                            'source': 'serl_survey',
                                                            'type':'categorical'}
        # num_rooms
        if 'num_rooms' in conditioning_vars_filter:
            self.conditioning_variable_tokenisers['num_rooms'] = {'original_variable_name': 'B5',
                                                            'source': 'serl_survey',
                                                            'type':'categorical'}
            # merge categories
            mask_1_to_3 = self.metadata_catalog['B5'] <= 3
            mask_9plus = self.metadata_catalog['B5'] >= 9
            self.metadata_catalog.loc[mask_1_to_3, 'B5'] = '1-3'
            self.metadata_catalog.loc[mask_9plus, 'B5'] = '9+'
            # Convert entire column to string for consistent dtype
            self.metadata_catalog['B5'] = self.metadata_catalog['B5'].astype(str)

        # total_floor_area is continuous so build a quantile binned tokeniser
        if 'total_floor_area' in conditioning_vars_filter:
            num_bins = self.config.get('training', {}).get('num_bins', 10)
            floor_area_boundaries = self.build_floor_area_tokeniser(num_bins=num_bins)
            self.conditioning_variable_tokenisers['total_floor_area'] = {'original_variable_name': 'totalFloorArea',
                                                            'source': 'epc',
                                                            'type':'continuous'}
        # epc rating
        if 'epc_rating' in conditioning_vars_filter:
            self.conditioning_variable_tokenisers['epc_rating'] = {'original_variable_name': 'currentEnergyRating',
                                                            'source': 'epc',
                                                            'type':'categorical'}
        # household_income_band
        # Note: C1 has no _fu suffix because it doesn't collide with C1_new from SERL survey
        if 'household_income_band' in conditioning_vars_filter:
            self.conditioning_variable_tokenisers['household_income_band'] = {'original_variable_name': 'C1',
                                                            'source': 'follow_up_survey',
                                                            'type':'categorical'}
        # num_occs
        if 'num_occs' in conditioning_vars_filter:
            self.conditioning_variable_tokenisers['num_occs'] = {'original_variable_name': 'C1_new',
                                                            'source': 'serl_survey',
                                                            'type':'categorical'}
            # merge categories
            mask = self.metadata_catalog['C1_new'] >= 5
            self.metadata_catalog.loc[mask, 'C1_new'] = '5+'
            # Convert entire column to string for consistent dtype
            self.metadata_catalog['C1_new'] = self.metadata_catalog['C1_new'].astype(str)

        # building_type - Map B1 codes to descriptive categories
        if 'building_type' in conditioning_vars_filter:
            conditions = [self.metadata_catalog['B1'] == 1,
                        self.metadata_catalog['B1'] == 2,
                        self.metadata_catalog['B1'] == 3,
                        self.metadata_catalog['B1'] == 4,
                        self.metadata_catalog['B1'] >= 5,
                        ]
            categories = ['Detached',
                        'Semi-detached',
                        'Terraced',
                        'Flat/tenement',
                        'Other', # merge 'Converted/shared house', and 'Commercial building'
                        ]
            self.metadata_catalog['building_type_descriptive'] = np.select(conditions,
                                                                            categories,
                                                                            default='Unknown')
            self.conditioning_variable_tokenisers['building_type'] = {'original_variable_name': 'building_type_descriptive',
                                                            'source': 'serl_survey',
                                                            'type':'categorical'}

        # age_built - Map B9 codes to descriptive categories
        if 'age_built' in conditioning_vars_filter:
            conditions = [self.metadata_catalog['B9'] == 1,
                        self.metadata_catalog['B9'] == 2,
                        self.metadata_catalog['B9'] == 3,
                        self.metadata_catalog['B9'] == 4,
                        self.metadata_catalog['B9'] == 5,
                        self.metadata_catalog['B9'] == 6,
                        self.metadata_catalog['B9'] == 7]
            categories = ['Before 1900',
                        '1900-1929',
                        '1930-1949',
                        '1950-1975',
                        '1976-1990',
                        '1991-2002',
                        '2003 onwards']
            self.metadata_catalog['age_built_descriptive'] = np.select(conditions,
                                                                    categories,
                                                                    default='Unknown')
            self.conditioning_variable_tokenisers['age_built'] = {'original_variable_name': 'age_built_descriptive',
                                                            'source': 'serl_survey',
                                                            'type':'categorical'}

        # tenure - Map B4 codes to descriptive categories
        if 'tenure' in conditioning_vars_filter:
            conditions = [self.metadata_catalog['B4'] == 1,
                        self.metadata_catalog['B4'] == 2,
                        self.metadata_catalog['B4'] == 3,
                        self.metadata_catalog['B4'] == 4,
                        self.metadata_catalog['B4'] == 5]
            categories = ['Owner occupier',
                        'Shared ownership',
                        'Private rented',
                        'Social rented',
                        'Rent free']
            self.metadata_catalog['tenure_descriptive'] = np.select(conditions,
                                                                    categories,
                                                                    default='Unknown')
            self.conditioning_variable_tokenisers['tenure'] = {'original_variable_name': 'tenure_descriptive',
                                                            'source': 'serl_survey',
                                                            'type':'categorical'}

        # managing_financially - Map D4 codes to descriptive categories
        if 'managing_financially' in conditioning_vars_filter:
            conditions = [self.metadata_catalog['D4'] == 1,
                        self.metadata_catalog['D4'] == 2,
                        self.metadata_catalog['D4'] == 3,
                        self.metadata_catalog['D4'] == 4,
                        self.metadata_catalog['D4'] == 5]
            categories = ['Living comfortably',
                        'Doing alright',
                        'Just about getting by',
                        'Finding it quite difficult',
                        'Finding it very difficult']
            self.metadata_catalog['managing_financially_descriptive'] = np.select(conditions,
                                                                                categories,
                                                                                default='Unknown')
            self.conditioning_variable_tokenisers['managing_financially'] = {'original_variable_name': 'managing_financially_descriptive',
                                                            'source': 'serl_survey',
                                                            'type':'categorical'}

        # has_pv
        if 'has_pv' in conditioning_vars_filter:
            self.conditioning_variable_tokenisers['has_pv'] = {'original_variable_name': 'is_exporter',
                                                            'source': 'smart_meter_data',
                                                            'type':'boolean'}

        # has_ev - Map C5 codes to descriptive categories
        if 'has_ev' in conditioning_vars_filter:
            conditions = [self.metadata_catalog['C5'] == -1, # Don't know
                        self.metadata_catalog['C5'] == 1, # has ev
                        self.metadata_catalog['C5'] == 2] # no ev
            categories = ["Don't know",
                        'Has EV',
                        'No EV']
            # Use 'Unknown' string instead of np.nan to avoid dtype mismatch
            self.metadata_catalog['has_ev_descriptive'] = np.select(conditions,
                                                                    categories,
                                                                    default='Unknown')
            self.conditioning_variable_tokenisers['has_ev'] = {'original_variable_name': 'has_ev_descriptive',
                                                            'source': 'serl_survey',
                                                            'type':'categorical'}
        
        # Check if masking is enabled
        use_masked_tokens = self.config.get('sharding', {}).get('data_filtering', {}).get('use_masked_tokens', False)

        dict_metadata_tokenised = {}
        dict_metadata_tokenised['puprn'] = self.metadata_catalog['puprn'].to_list()
        # now create the encoder and decoders
        for key, value in self.conditioning_variable_tokenisers.items():
            original_variable_name = value['original_variable_name']
            if value['type'] in ['categorical', 'ordinal', 'cardinal']:
                # Get unique values and sort for deterministic tokenization
                unique_vals = self.metadata_catalog[original_variable_name].unique()
                sorted_unique_vals = np.sort(unique_vals[~pd.isna(unique_vals)])

                if use_masked_tokens:
                    # Masking enabled: include mask token in vocabulary
                    mask_token = len(sorted_unique_vals)
                    itov = {mask_token: np.nan}  # index to value
                    vtoi = {}   # value to index (don't add NaN as key!)
                    for i, val in enumerate(sorted_unique_vals):
                        itov[i] = val
                        vtoi[val] = i
                    value['mask_token'] = mask_token
                else:
                    # Masking disabled: no mask token in vocabulary
                    itov = {}
                    vtoi = {}
                    for i, val in enumerate(sorted_unique_vals):
                        itov[i] = val
                        vtoi[val] = i
                    value['mask_token'] = None  # No mask token

                value['itov'] = itov
                value['vtoi'] = vtoi
                value['vocab_size'] = len(itov)

                # Store original dtype for proper decoding
                original = self.metadata_catalog[original_variable_name]
                value['original_dtype'] = str(original.dtype)

                # encode conditioning variable
                if use_masked_tokens:
                    dict_metadata_tokenised[key] = self.metadata_catalog[original_variable_name].map(vtoi).fillna(value['mask_token']).astype(int)
                else:
                    # Without masking, NaN should not exist (filtered out by require_all_conditioning_vars)
                    encoded = self.metadata_catalog[original_variable_name].map(vtoi)
                    if encoded.isna().any():
                        raise ValueError(f"NaN values found in {key} during unmasked tokenization. "
                                       f"This should not happen with require_all_conditioning_vars=True")
                    dict_metadata_tokenised[key] = encoded.astype(int)
                # test
                if original.dtype == 'int64':
                    # if original is int64, then need to convert decoded to astype int for comparison to be valid
                    decoded = dict_metadata_tokenised[key].map(itov).astype(int)
                else:
                    decoded = dict_metadata_tokenised[key].map(itov)
                assert decoded.equals(original), \
                    f"Mismatch in {key}: decoded values don't match original"
                self._log(f"Tokenized {key}: {len(sorted_unique_vals)} unique values, vocab_size={len(itov)}")
            elif value['type'] == 'continuous':
                # quantile binning for continuous variables
                if key == 'total_floor_area':
                    from utils.universal_tokeniser_naive import TokenVocabulary
                    value['boundaries'] = floor_area_boundaries

                    # Create TokenVocabulary for consistent token calculations
                    floor_area_vocab = TokenVocabulary(boundaries=list(floor_area_boundaries), zero_aware=False)

                    if use_masked_tokens:
                        # Masking enabled: include mask token in vocabulary
                        mask_token = floor_area_vocab.missing_token
                        vocab_size = floor_area_vocab.vocab_size

                        # encode using vocab
                        dict_metadata_tokenised[key] = self.metadata_catalog[original_variable_name].\
                            apply(lambda x: floor_area_vocab.encode(x))
                        value['mask_token'] = mask_token
                    else:
                        # Masking disabled: no mask token, only bins (num_bins, not vocab_size)
                        vocab_size = floor_area_vocab.num_bins

                        # encode (missing values will encode to missing_token, which we'll check for)
                        dict_metadata_tokenised[key] = self.metadata_catalog[original_variable_name].\
                            apply(lambda x: floor_area_vocab.encode(x))

                        # Verify no NaN values were encoded (they would get missing_token)
                        if (dict_metadata_tokenised[key] == floor_area_vocab.missing_token).any():
                            raise ValueError(f"NaN values found in {key} during unmasked tokenization. "
                                           f"This should not happen with require_all_conditioning_vars=True")

                        value['mask_token'] = None  # No mask token

                    # no assertion as quantile boundary tokenisation returns mid-point of bins i.e. inherently lossy, not expecting equivalence
                    value['itov'] = None # uses quantile boundaries not map
                    value['vtoi'] = None
                    value['vocab_size'] = vocab_size
                    value['num_bins'] = num_bins
                    self._log(f"Tokenized {key}: {len(floor_area_boundaries)} bins, vocab_size={vocab_size}")
                else:
                    self._log(f'Warning: unknown continuous variable {key}',level='warning')
                    continue
            elif value['type'] == 'boolean':
                if use_masked_tokens:
                    # Masking enabled: include mask token in vocabulary
                    mask_token = 2
                    itov = {0: False,
                            1: True,
                            mask_token: np.nan}  # index to value
                    vtoi = {
                        False: 0,
                        True: 1,
                    }
                    value['mask_token'] = mask_token
                else:
                    # Masking disabled: no mask token in vocabulary
                    itov = {0: False,
                            1: True}
                    vtoi = {
                        False: 0,
                        True: 1,
                    }
                    value['mask_token'] = None  # No mask token

                value['itov'] = itov
                value['vtoi'] = vtoi
                value['vocab_size'] = len(itov)

                # Store original dtype for proper decoding
                original = self.metadata_catalog[original_variable_name]
                value['original_dtype'] = str(original.dtype)

                # encode conditioning variable
                if use_masked_tokens:
                    dict_metadata_tokenised[key] = self.metadata_catalog[original_variable_name].map(vtoi).fillna(value['mask_token']).astype(int)
                else:
                    # Without masking, NaN should not exist (filtered out by require_all_conditioning_vars)
                    encoded = self.metadata_catalog[original_variable_name].map(vtoi)
                    if encoded.isna().any():
                        raise ValueError(f"NaN values found in {key} during unmasked tokenization. "
                                       f"This should not happen with require_all_conditioning_vars=True")
                    dict_metadata_tokenised[key] = encoded.astype(int)
                # test
                if original.dtype == 'int64':
                    # if original is int64, then need to convert decoded to astype int for comparison to be valid
                    decoded = dict_metadata_tokenised[key].map(itov).astype(int)
                elif original.dtype == 'bool':
                    decoded = dict_metadata_tokenised[key].map(itov).astype(bool)
                else:
                    decoded = dict_metadata_tokenised[key].map(itov)
                assert decoded.equals(original), \
                    f"Mismatch in {key}: decoded values don't match original"
                self._log(f"Tokenized {key}: vocab_size={len(itov)}")
            else:
                raise ValueError(f"Unknown variable type: {value['type']}")
        
        self.metadata_tokenised = pd.DataFrame(dict_metadata_tokenised)
        # save the conditioning_variable_tokeniser to disk
        shard_output_dir = os.path.join(self.run_path, self.config['sharding']['shard_output_dir'])
        os.makedirs(shard_output_dir, exist_ok=True)

        self._log(f"Saving tokenised conditioning variables and tokenisers to: {shard_output_dir}")

        self.metadata_tokenised.to_pickle(os.path.join(shard_output_dir, 'metadata_tokenised.pkl'))
        with open (os.path.join(shard_output_dir, 'conditioning_variable_tokenisers.pkl'),'wb') as f:
            pickle.dump(self.conditioning_variable_tokenisers, f)
        self._log("Successfully saved tokenised metadata and tokenizers")
    
    def determine_k_anonymity(self):
        self._log("=" * 70)
        self._log("K ANONYMITY ANALYSIS")
        self._log("=" * 70)

        results = {}
        data_filtering_config = self.config.get('sharding', {}).get('data_filtering')
        conditioning_vars_filter = data_filtering_config.get('conditioning_variables')

        k = data_filtering_config['k_anonymity']

        # IMPORTANT: Exclude region and has_pv from k-anonymity calculation
        # These are used for filtering/metadata only, NOT as training features
        training_vars = [v for v in conditioning_vars_filter if v not in ['region', 'has_pv']]

        self._log(f"Conditioning variables (all): {conditioning_vars_filter}")
        self._log(f"Training variables (for k-anonymity): {training_vars}")

        # Map conditioning variable names to original column names
        orig_col_names = []
        for var in training_vars:
            if var in self.conditioning_variable_tokenisers:
                orig_col = self.conditioning_variable_tokenisers[var]['original_variable_name']
                orig_col_names.append(orig_col)
            else:
                # Variable not in tokenisers (shouldn't happen)
                self._log(f"WARNING: Variable '{var}' not found in conditioning_variable_tokenisers", level='warning')
                orig_col_names.append(var)  # Use as-is

        self._log(f"Original column names (for k-anonymity): {orig_col_names}")

        # Check all variables exist
        missing_vars = [v for v in orig_col_names if v not in self.metadata_catalog.columns]
        if missing_vars:
            self._log(f"WARNING: Missing variables: {missing_vars}", level='warning')
            results['error'] = f'Missing variables: {missing_vars}'

        # Filter to rows with all variables present
        valid_df = self.metadata_catalog.dropna(subset=orig_col_names).copy()
        self._log(f"Households in metadata catalog: {len(self.metadata_catalog)}")
        self._log(f"Households with all variables present: {len(valid_df)}")

        if len(valid_df) == 0:
            results['error'] = 'No valid households'
            self._log("WARNING: No valid households!", level='warning')

        # Group by variable combination (using original column names)
        grouped = valid_df.groupby(orig_col_names).size().reset_index(name='count')

        # Analyze group sizes
        total_groups = len(grouped)
        k_groups = grouped[grouped['count'] >= k]
        k_group_count = len(k_groups)

        # Households in k-satisfying groups (use original column names for merge)
        households_in_k_group = valid_df.merge(
            k_groups[orig_col_names],
            on=orig_col_names,
            how='inner'
        )
        k_household_count = len(households_in_k_group)

        # Calculate percentages
        pct_groups_k = 100 * k_group_count / total_groups if total_groups > 0 else 0
        pct_households_k = 100 * k_household_count / len(valid_df) if len(valid_df) > 0 else 0

        # update metadata catalog with k anonymity results (use original column names for merge)
        self.metadata_with_k_groups = valid_df.merge(
            grouped,
            on=orig_col_names,
            how='inner'
        )
        # save summary results
        results['summary'] = {
            'variables': training_vars,  # Only training variables used for k-anonymity
            'total_households_with_vars': len(valid_df),
            'total_unique_groups': total_groups,
            f'k{k}_satisfying_groups': k_group_count,
            f'pct_groups_k{k}': pct_groups_k,
            f'households_in_k{k}_groups': k_household_count,
            f'pct_households_in_k{k}': pct_households_k,
            f'households_lost_to_k{k}': len(valid_df) - k_household_count,
            'group_size_distribution': {
                'min': int(grouped['count'].min()),
                'max': int(grouped['count'].max()),
                'mean': float(grouped['count'].mean()),
                'median': float(grouped['count'].median()),
            }
        }

        self._log(f"Total unique groups: {total_groups}")
        self._log(f"Groups satisfying k>={k}: {k_group_count} ({pct_groups_k:.1f}%)")
        self._log(f"Households in k{k} groups: {k_household_count} ({pct_households_k:.1f}%)")
        self._log(f"Households LOST to k{k} filter: {len(valid_df) - k_household_count}")
        self._log(f"Group size: min={grouped['count'].min()}, max={grouped['count'].max()}, "
            f"mean={grouped['count'].mean():.1f}, median={grouped['count'].median():.0f}")

        shard_output_dir = os.path.join(self.run_path, self.config['sharding']['shard_output_dir'])
        os.makedirs(shard_output_dir, exist_ok=True)

        results_path = os.path.join(shard_output_dir, 'k_anonymity_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        self._log(f"Saved k anonymity results: {results_path}")
        
        # save metadata catalogue with k anonymity results
        save_path = os.path.join(shard_output_dir, 'metadata_with_k_groups.pkl')
        self.metadata_with_k_groups.to_pickle(save_path)
        self._log(f"Saved metadata with k anonymity results: {save_path}")

    def apply_filters_and_analyze(self) -> pd.DataFrame:
        """
        Apply successive filters with logging and analyze category distributions.

        Filtering stages:
            1. Exclude PV households (is_exporter=False)
            2. Missingness < threshold for both gas and electricity
            3. All required conditioning variables present
            4. K-anonymity >= threshold

        After filtering, analyzes category distributions for conditioning variables.

        Side Effects:
            - Saves metadata_catalog_unfiltered.pkl (before filtering)
            - Saves metadata_catalog_filtered.pkl (after filtering)
            - Saves filtering_analysis.json (stage statistics + category distributions)
            - Logs detailed console output for each stage

        Returns:
            pd.DataFrame: Filtered metadata_catalog ready for shard creation
        """
        self._log("=" * 70)
        self._log("FILTERING ANALYSIS")
        self._log("=" * 70)

        data_filtering_config = self.config.get('sharding', {}).get('data_filtering', {})
        conditioning_vars = data_filtering_config.get('conditioning_variables', [])
        max_missingness = data_filtering_config.get('max_missingness_rate', 0.20)
        k = data_filtering_config.get('k_anonymity', 10)

        shard_output_dir = os.path.join(self.run_path, self.config['sharding']['shard_output_dir'])
        os.makedirs(shard_output_dir, exist_ok=True)

        # Save unfiltered version (has missingness, tokenization, k-anonymity)
        unfiltered_path = os.path.join(shard_output_dir, 'metadata_catalog_unfiltered.pkl')
        self.metadata_catalog.to_pickle(unfiltered_path)
        self._log(f"Saved unfiltered metadata: {unfiltered_path}")

        # Initialize filtering results
        results = {
            'stages': [],
            'initial_count': len(self.metadata_catalog),
        }

        current_df = self.metadata_catalog.copy()
        initial_count = len(current_df)

        # Stage 0: Initial count
        results['stages'].append({
            'stage': 0,
            'name': 'Initial (unfiltered)',
            'count': len(current_df),
            'removed': 0,
            'pct_of_initial': 100.0,
        })
        self._log(f"\nStage 0: Initial count = {len(current_df)} households")

        # Stage 1: Filter PV households
        if data_filtering_config.get('exclude_pv_households', True):
            before = len(current_df)
            current_df = current_df[~current_df['is_exporter']].copy()
            removed = before - len(current_df)
            pct = 100 * len(current_df) / initial_count
            results['stages'].append({
                'stage': 1,
                'name': 'Exclude PV households (is_exporter=False)',
                'count': len(current_df),
                'removed': removed,
                'pct_of_initial': pct,
            })
            self._log(f"\nStage 1: Exclude PV households")
            self._log(f"  Removed: {removed} ({100*removed/before:.1f}%)")
            self._log(f"  Remaining: {len(current_df)} ({pct:.1f}% of initial)")

        # Stage 2: Filter for missingness threshold
        before = len(current_df)
        if 'elec_missingness' in current_df.columns and 'gas_missingness' in current_df.columns:
            low_missing = (
                (current_df['elec_missingness'] < max_missingness) &
                (current_df['gas_missingness'] < max_missingness)
            )
            current_df = current_df[low_missing].copy()
            removed = before - len(current_df)
            pct = 100 * len(current_df) / initial_count
            results['stages'].append({
                'stage': 2,
                'name': f'Missingness < {max_missingness*100:.0f}% (gas and elec)',
                'count': len(current_df),
                'removed': removed,
                'pct_of_initial': pct,
            })
            self._log(f"\nStage 2: Missingness < {max_missingness*100:.0f}%")
            self._log(f"  Removed: {removed} ({100*removed/before:.1f}% of previous)")
            self._log(f"  Remaining: {len(current_df)} ({pct:.1f}% of initial)")
        else:
            self._log(f"\nStage 2: Missingness filter SKIPPED (columns not found)")

        # Stage 3: Require all conditioning variables present
        if data_filtering_config.get('require_all_conditioning_vars', True):
            before = len(current_df)
            # Exclude 'region' and 'has_pv' from required variables (used for filtering/inference only)
            required_vars = [v for v in conditioning_vars if v not in ['region', 'has_pv']]
            for var in required_vars:
                # Get original column name from tokenisers
                if hasattr(self, 'conditioning_variable_tokenisers') and var in self.conditioning_variable_tokenisers:
                    orig_col = self.conditioning_variable_tokenisers[var]['original_variable_name']
                    if orig_col in current_df.columns:
                        current_df = current_df[current_df[orig_col].notna()].copy()
            removed = before - len(current_df)
            pct = 100 * len(current_df) / initial_count
            results['stages'].append({
                'stage': 3,
                'name': f'All required variables present ({", ".join(required_vars)})',
                'count': len(current_df),
                'removed': removed,
                'pct_of_initial': pct,
            })
            self._log(f"\nStage 3: All required variables present ({', '.join(required_vars)})")
            self._log(f"  Removed: {removed} ({100*removed/before:.1f}% of previous)")
            self._log(f"  Remaining: {len(current_df)} ({pct:.1f}% of initial)")

        # Stage 4: K-anonymity filter
        before = len(current_df)
        # Merge with k-anonymity results
        if hasattr(self, 'metadata_with_k_groups') and 'count' in self.metadata_with_k_groups.columns:
            # Keep only households that satisfy k-anonymity
            k_satisfied = self.metadata_with_k_groups[self.metadata_with_k_groups['count'] >= k]
            current_df = current_df[current_df['puprn'].isin(k_satisfied['puprn'])].copy()
            # Merge count column for reference
            current_df = current_df.merge(
                k_satisfied[['puprn', 'count']].rename(columns={'count': 'k_group_size'}),
                on='puprn',
                how='left'
            )
            removed = before - len(current_df)
            pct = 100 * len(current_df) / initial_count
            results['stages'].append({
                'stage': 4,
                'name': f'K-anonymity >= {k}',
                'count': len(current_df),
                'removed': removed,
                'pct_of_initial': pct,
            })
            self._log(f"\nStage 4: K-anonymity >= {k}")
            self._log(f"  Removed: {removed} ({100*removed/before:.1f}% of previous)")
            self._log(f"  Remaining: {len(current_df)} ({pct:.1f}% of initial)")
        else:
            self._log(f"\nStage 4: K-anonymity filter SKIPPED (metadata_with_k_groups not available)")

        results['after_all_filters'] = len(current_df)

        # Analyze category distributions
        self._log("=" * 70)
        self._log("CATEGORY DISTRIBUTION ANALYSIS")
        self._log("=" * 70)
        self._log("(After all filters, showing merged categories)")

        category_results = {}
        # Exclude region and has_pv from distribution analysis
        analysis_vars = [v for v in conditioning_vars if v not in ['region', 'has_pv']]

        for var in analysis_vars:
            if var not in self.conditioning_variable_tokenisers:
                continue

            orig_col = self.conditioning_variable_tokenisers[var]['original_variable_name']
            if orig_col not in current_df.columns:
                self._log(f"\n--- {var}: NOT FOUND IN DATA ---")
                continue

            self._log(f"\n--- {var} ---")
            self._log(f"  Source: {orig_col}")

            # Get value counts
            counts = current_df[orig_col].value_counts(dropna=False).sort_index()
            total = len(current_df)
            valid_total = current_df[orig_col].notna().sum()

            self._log(f"  Total households: {total}")
            self._log(f"  Valid (non-NaN): {valid_total} ({100*valid_total/total:.1f}%)")
            self._log(f"  Unique values: {current_df[orig_col].nunique()}")
            self._log(f"\n  Category breakdown:")

            category_data = []
            for value, count in counts.items():
                pct = 100 * count / total
                flag = " ⚠️ SMALL" if pct < 5 and not pd.isna(value) else ""

                if pd.isna(value):
                    label = "Missing/NaN"
                else:
                    label = str(value)

                self._log(f"    {label:>15}: {count:>5} ({pct:>5.1f}%){flag}")

                category_data.append({
                    'value': label,
                    'count': int(count),
                    'pct': float(pct),
                })

            category_results[var] = {
                'original_column': orig_col,
                'total': total,
                'valid': int(valid_total),
                'unique_values': int(current_df[orig_col].nunique()),
                'categories': category_data,
            }

        # Save filtered metadata
        filtered_path = os.path.join(shard_output_dir, 'metadata_catalog_filtered.pkl')
        current_df.to_pickle(filtered_path)
        self._log(f"Saved filtered metadata: {filtered_path}")

        # Filter metadata_tokenised to match filtered PUPRNs
        # This creates the tokenised version used for downstream sharding/masking
        filtered_puprns = set(current_df['puprn'].tolist())
        initial_tokenised_count = len(self.metadata_tokenised)
        self.metadata_tokenised_filtered = self.metadata_tokenised[
            self.metadata_tokenised['puprn'].isin(filtered_puprns)
        ].copy()
        removed_count = initial_tokenised_count - len(self.metadata_tokenised_filtered)

        # Save filtered tokenised metadata (used for training)
        tokenised_filtered_path = os.path.join(shard_output_dir, 'metadata_tokenised_filtered.pkl')
        self.metadata_tokenised_filtered.to_pickle(tokenised_filtered_path)
        self._log(f"Filtered metadata_tokenised: {initial_tokenised_count} → {len(self.metadata_tokenised_filtered)} households (removed {removed_count})")
        self._log(f"Saved filtered tokenised metadata: {tokenised_filtered_path}")

        # Save analysis results
        analysis_results = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'conditioning_variables': conditioning_vars,
                'max_missingness_rate': max_missingness,
                'k_anonymity': k,
                'exclude_pv_households': data_filtering_config.get('exclude_pv_households', True),
            },
            'filtering_stages': results['stages'],
            'initial_count': results['initial_count'],
            'final_count': results['after_all_filters'],
            'category_distributions': category_results,
        }

        analysis_path = os.path.join(shard_output_dir, 'filtering_analysis.json')
        with open(analysis_path, 'w') as f:
            json.dump(analysis_results, f, indent=2)
        self._log(f"Saved filtering analysis: {analysis_path}")

        self._log("=" * 70)
        self._log("FILTERING COMPLETE")
        self._log(f"  Initial: {initial_count} households")
        self._log(f"  Final: {len(current_df)} households ({100*len(current_df)/initial_count:.1f}%)")
        self._log("=" * 70)

        return current_df

    def assign_splits(self) -> None:
        """
        Assign households to train/val/test splits to prevent data leakage.
        This must be called after apply_filters_and_analyze().

        Uses metadata_catalog_filtered (after all filters have been applied).
        """
        self._log("Assigning households to train/val/test splits...")

        if not hasattr(self, 'metadata_catalog_filtered') or self.metadata_catalog_filtered is None:
            raise ValueError("Must call apply_filters_and_analyze() first")

        # Get configuration
        sharding_config = self.config.get('sharding', {})
        val_households = sharding_config.get('val_households', 200)
        test_households = sharding_config.get('test_households', 200)

        total_households = len(self.metadata_catalog_filtered)

        # Validate we have enough households
        if val_households + test_households >= total_households:
            raise ValueError(f"Cannot create {val_households} val + {test_households} test from only {total_households} total households")

        # Initialize 'split' column to 'train' for all filtered households
        self.metadata_catalog_filtered['split'] = 'train'

        # Step 1: Sample validation households (without replacement)
        val_sample = self.metadata_catalog_filtered.sample(n=val_households, random_state=self.random_seed)
        val_indices = val_sample.index
        self.metadata_catalog_filtered.loc[val_indices, 'split'] = 'val'

        # Step 2: Sample test households from remaining (without replacement)
        remaining_after_val = self.metadata_catalog_filtered[self.metadata_catalog_filtered['split'] == 'train']
        test_sample = remaining_after_val.sample(n=test_households, random_state=self.random_seed + 1)  # Different seed
        test_indices = test_sample.index
        self.metadata_catalog_filtered.loc[test_indices, 'split'] = 'test'

        # Step 3: Remaining households stay as 'train'
        train_households = (self.metadata_catalog_filtered['split'] == 'train').sum()

        self._log("Split assignment completed:")
        self._log(f"  Validation: {val_households} households")
        self._log(f"  Test: {test_households} households")
        self._log(f"  Training pool: {train_households} households")
        self._log(f"  Total: {val_households + test_households + train_households} households")

        # Save split assignments for debugging/analysis
        split_counts = self.metadata_catalog_filtered['split'].value_counts()
        self._log(f"  Split verification: {dict(split_counts)}")
    
    def create_masked_conditioning_variables(self, augmentation_factor:int = 1,
                                             seed:int = 42) -> None:
        """
        Generate privacy-preserving masked conditioning variable table with progressive unmasking.

        Implements progressive k-anonymity masking: for each household, sequentially unmasks
        conditioning variables while checking that k≥10 households share the same values.
        Creates multiple masked rows per household with varying specificity levels.

        **K10 Aggregation Integration:**
        If enabled in config (sharding.k10_aggregation.enabled=true), this method also generates
        k10 aggregated synthetic data as a privacy benchmark. For each valid k≥10 conditioning
        variable combination:
        - Filters to TRAINING SPLIT households only (prevents data leakage)
        - Randomly selects 10 households from the training group
        - Loads their original (non-tokenized) time series data
        - Aggregates via mean (treating NaN gas as 0)
        - Adds decoded conditioning variables
        - Saves as synthetic household for MIA evaluation

        Args:
            augmentation_factor: Number of random variable orderings per household (default: 1)
                Higher values create more diverse masked rows per household
            seed: Random seed for reproducible variable shuffling and k10 sampling (default: 42)

        Returns:
            pd.DataFrame: Masked conditioning table with columns:
                - puprn: Household identifier
                - var_count: Number of unmasked variables (2-14)
                - sample_size_n: k-anonymity group size (always ≥10)
                - augmentation_n: Augmentation iteration (0 to augmentation_factor-1)
                - 14 conditioning variable columns (unmasked values or mask_token)

        Side Effects:
            - Saves masked_conditioning_table.pkl to shard_output_dir
            - If k10_aggregation enabled:
                - Generates k10 synthetic households to k10_aggregation.output_dir
                - Saves k10_aggregation_metadata.pkl (source PUPRNs, conditioning vars, filepaths)
                - Saves k10_aggregation_stats.json (count, avg group size, load failures)
                - Increments self.k10_aggregate_count
                - Populates self.k10_metadata list

        Configuration (sharding.k10_aggregation):
            - enabled: Generate k10 aggregates (default: false)
            - max_synthetic_households: Cap on number of k10 aggregates
            - aggregation_method: "mean" or "median" (default: "mean")
            - random_seed: For reproducible household sampling
            - path_to_data_to_aggregate: Path to original (non-tokenized) household data
              (relative paths resolved against run_path; absolute paths used as-is)
            - output_dir: Where to save k10 aggregates (relative to run_path)

        Privacy Guarantees:
            - All masked rows satisfy k≥10 anonymity
            - K10 aggregates use only groups with k≥10
            - Progressive unmasking stops when next variable would violate k<10

        Example:
            >>> sharder.create_masked_conditioning_variables(augmentation_factor=3, seed=42)
            # Creates ~10-15 masked rows per household with varying var_count (2-14)
            # If k10_aggregation enabled, also generates synthetic households
        """
        # Use filtered tokenised metadata (only households that passed all filters)
        global_table = self.metadata_tokenised_filtered

        # Get training split PUPRNs to prevent data leakage (for k10 aggregation)
        train_puprns_set = set(self.get_puprns_for_split('train'))

        # create list to store results
        list_of_masked_conditioning_variables = []
        
        # we need to know the order of the columns in the global_array
        global_table_list_columns = global_table.columns.to_numpy()
        columns_to_shuffle = global_table_list_columns[~np.isin(global_table_list_columns,['puprn','region','imd'])]
        # get flag for whether to do k10 aggregation as part of creating masked conditioning variable
        k10_config = self.config.get('sharding',{}).get('k10_aggregation',{})
        k10_enabled = k10_config.get('enabled', False)
        # prespecify folder to get original data
        if k10_enabled:
            # Resolve relative to run_path (absolute paths pass through unchanged)
            original_data_folder = os.path.join(self.run_path, k10_config.get('path_to_data_to_aggregate', ''))
            # create k10 aggregated synthetic data output folder
            k10_output_folder = os.path.join(self.run_path, k10_config.get('output_dir', ''))
            os.makedirs(k10_output_folder, exist_ok=True)
        
        # iterate through the table, as many times as we want the data to be augmented
        counter = 0
        for augmentation_n in range(augmentation_factor):
            # rename metadata_tokenised for ease of reference
            for row in global_table.itertuples():
                counter += 1
                # start random seed
                rng = np.random.RandomState(seed + counter)
                # shuffle column order, keeping region and imd at the beginning
                shuffled_column_names = ['region','imd'] + list(rng.permutation(columns_to_shuffle))
                # iterate through columns, add conditions to match for in global, and count result, unmasking until N drops below 10, else assign mask_token
                # start with a True condition (include all), then add with AND logic
                mask = pd.Series([True] * len(global_table))
                # keep track of how many conditioning variables we are able to record each time
                var_count = 0
                for this_column_name in shuffled_column_names:
                    var_count += 1
                    this_row_dict = {}
                    # create a new condition
                    new_condition = global_table[this_column_name] == getattr(row, this_column_name)
                    # update the mask to include the new condition
                    mask = mask & new_condition
                    # count how many other rows with this condition
                    count_with_condition = len(global_table[mask])
                    if count_with_condition < self.k_anonymity:
                        # then this is a disclosive combination, so exit this loop and proceed to next row (puprn)
                        break
                    # otherwise proceed to register
                    this_row_dict['puprn'] = getattr(row, 'puprn')
                    this_row_dict['sample_size_n'] = count_with_condition
                    this_row_dict['var_count'] = var_count
                    this_row_dict['augmentation_n'] = augmentation_n
                    for col in shuffled_column_names[:var_count]:
                        this_row_dict[col] = getattr(row, col)
                    list_of_masked_conditioning_variables.append(this_row_dict)

                    # ========================================
                    # K10 AGGREGATION: Generate Privacy Baseline Synthetic Data
                    # ========================================
                    # For each valid k≥10 conditioning variable combination, create a synthetic
                    # household by aggregating time series from 10 randomly selected real households.
                    # This provides a privacy benchmark for MIA evaluation: if transformer synthetic
                    # data is at least as privacy-preserving as k10 aggregates, both can be released.

                    if k10_enabled and self.k10_aggregate_count < k10_config.get('max_synthetic_households', 0):
                        
                        # 1. Filter to TRAINING SPLIT households matching conditioning variables
                        matching_puprns = global_table.loc[mask, 'puprn'].tolist()
                        list_puprn_with_condition = np.array([p for p in matching_puprns if p in train_puprns_set])

                        # Check if we have at least 10 TRAINING households to sample from
                        if len(list_puprn_with_condition) < 10:
                            continue  # Skip k10 aggregation for this combination (not enough training households)

                        # Reproducible random selection
                        k10_rng = np.random.RandomState(k10_config['random_seed'] + self.k10_aggregate_count)
                        selected_puprn = k10_rng.choice(a=list_puprn_with_condition, size=10, replace=False)

                        # 2. Load puprn original time series data with error handling
                        list_of_household_df = []
                        load_failures = []

                        for this_puprn in selected_puprn:
                            try:
                                with open(os.path.join(original_data_folder, f'{this_puprn}.pkl'), 'rb') as f:
                                    df = pickle.load(f)
                                list_of_household_df.append(df)
                            except Exception as e:
                                load_failures.append(this_puprn)
                                self._log(f"Failed to load {this_puprn}: {str(e)}", level='warning')

                        # Require at least 10 successful loads
                        if len(list_of_household_df) < 10:
                            self._log(f"Skipping k10 aggregate - only {len(list_of_household_df)}/10 households loaded successfully", level='warning')
                            continue

                        # 3. Perform aggregation
                        aggregated_df = self._generate_k10_aggregate(list_of_household_df)

                        # 4. Add decoded conditioning variables to aggregated data
                        aggregated_df = self._add_decoded_conditioning_vars(
                            aggregated_df,
                            {col: getattr(row, col) for col in shuffled_column_names[:var_count]}
                        )

                        # 5. Save aggregated synthetic data
                        synthetic_id = f'k10_synthetic_{self.k10_aggregate_count}'
                        output_path = os.path.join(k10_output_folder, f'{synthetic_id}.pkl')
                        aggregated_df.to_pickle(output_path)

                        # 6. Record k10 metadata with enhanced tracking
                        self.k10_metadata.append({
                            'synthetic_id': synthetic_id,
                            'source_puprns': selected_puprn.tolist(),
                            'conditioning_vars': {col: getattr(row, col) for col in shuffled_column_names[:var_count]},
                            'filepath': output_path,
                            'group_size': count_with_condition,  # Total households (all splits)
                            'group_size_train': len(list_puprn_with_condition),  # Training split only
                            'load_failures': load_failures,
                            'var_count': var_count,
                            'augmentation_n': augmentation_n,
                            'sample_size_n': count_with_condition,
                            'split': 'train'  # Mark which split was used
                        })

                        # 7. Iterate k10 aggregation count
                        self.k10_aggregate_count += 1

        masked_conditioning_table = pd.DataFrame(list_of_masked_conditioning_variables)

        # finally, for each conditioning variable column fill NaN with relevant mask_token
        for col in masked_conditioning_table.drop(columns=['puprn',
                                                                'sample_size_n',
                                                                'var_count',
                                                                'augmentation_n']).columns:
            # get mask_token
            mask_token = self.conditioning_variable_tokenisers[col]['mask_token']
            # fill na with mask_token (fix for pandas 3.0 compatibility)
            masked_conditioning_table[col] = masked_conditioning_table[col].fillna(value=mask_token)

        # Save masked conditioning table (only if sharding config exists - not during inference)
        if 'sharding' in self.config and 'shard_output_dir' in self.config['sharding']:
            shard_output_dir = os.path.join(self.run_path, self.config['sharding']['shard_output_dir'])
            if os.path.exists(shard_output_dir):
                masked_table_path = os.path.join(shard_output_dir, 'masked_conditioning_table.pkl')
                masked_conditioning_table.to_pickle(masked_table_path)
                self._log(f"Saved masked conditioning table: {masked_table_path}")
            else:
                self._log("Skipping save of masked conditioning table (shard directory does not exist - inference mode)")

        # save k10 metadata and statistics
        if k10_enabled and self.k10_metadata:
            # Save metadata DataFrame
            k10_metadata_df = pd.DataFrame(self.k10_metadata)
            metadata_path = os.path.join(self.run_path,
                                         k10_config.get('output_dir', ''),
                                         'k10_aggregation_metadata.pkl')
            k10_metadata_df.to_pickle(metadata_path)
            self._log(f"Saved k10 metadata: {metadata_path}")

            # Save statistics JSON
            stats_path = os.path.join(self.run_path,
                                      k10_config.get('output_dir', ''),
                                      'k10_aggregation_stats.json')
            with open(stats_path, 'w') as f:
                json.dump({
                    'total_k10_aggregates': self.k10_aggregate_count,
                    'generation_timestamp': datetime.now().isoformat(),
                    'config': k10_config,
                    'avg_group_size': float(k10_metadata_df['group_size'].mean()) if len(k10_metadata_df) > 0 else 0,
                    'total_load_failures': sum(len(m.get('load_failures', [])) for m in self.k10_metadata)
                }, f, indent=2)
            self._log(f"Saved k10 statistics: {stats_path}")

        # Log k-anonymity privacy masking statistics for transparency
        self._log("=" * 60)
        self._log(f"k={self.k_anonymity} Privacy Masking Statistics")
        self._log("=" * 60)
        self._log(f"  Total unique households: {len(global_table)}")
        self._log(f"  Total masked rows generated: {len(masked_conditioning_table)}")
        self._log(f"  Rows per household (avg): {len(masked_conditioning_table) / len(global_table):.1f}")
        self._log(f"  Augmentation factor: {augmentation_factor}")
        self._log("")
        self._log("Variable Unmasking Distribution:")
        var_count_dist = masked_conditioning_table['var_count'].value_counts().sort_index()
        for var_count, count in var_count_dist.items():
            pct = (count / len(masked_conditioning_table)) * 100
            self._log(f"  {var_count} variables unmasked: {count} rows ({pct:.1f}%)")
        self._log("")
        self._log("k-Anonymity Verification:")
        min_sample_size = masked_conditioning_table['sample_size_n'].min()
        max_sample_size = masked_conditioning_table['sample_size_n'].max()
        avg_sample_size = masked_conditioning_table['sample_size_n'].mean()
        self._log(f"  Min sample size (k): {min_sample_size}")
        self._log(f"  Max sample size (k): {max_sample_size}")
        self._log(f"  Avg sample size (k): {avg_sample_size:.1f}")

        # Verify k-anonymity privacy criterion is satisfied
        assert min_sample_size >= self.k_anonymity, f"k={self.k_anonymity} privacy violated! Minimum sample size is {min_sample_size}"
        self._log(f"  ✓ k={self.k_anonymity} privacy criterion SATISFIED (min k = {min_sample_size} ≥ {self.k_anonymity})")
        self._log("=" * 60)

        # Log k10 aggregation statistics if enabled
        if k10_enabled and self.k10_aggregate_count > 0:
            self._log("")
            self._log("=" * 60)
            self._log("k=10 Aggregation Synthetic Data Statistics")
            self._log("=" * 60)
            self._log(f"  Total k10 aggregates generated: {self.k10_aggregate_count}")
            self._log(f"  Split used: TRAINING ONLY")
            self._log(f"  Output directory: {k10_output_folder}")
            self._log(f"  Aggregation method: {k10_config.get('aggregation_method', 'mean')}")

            # Calculate and log failure statistics
            if self.k10_metadata:
                total_load_failures = sum(len(m.get('load_failures', [])) for m in self.k10_metadata)
                if total_load_failures > 0:
                    self._log(f"  Warning: {total_load_failures} household load failures across all aggregates")

                avg_group_size = sum(m.get('group_size', 0) for m in self.k10_metadata) / len(self.k10_metadata)
                self._log(f"  Average k-anonymity group size: {avg_group_size:.1f}")

            self._log("=" * 60)

        return masked_conditioning_table

    def _generate_k10_aggregate(self, list_of_household_df: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Aggregate multiple household time series into one synthetic household.

        Args:
            list_of_household_df: List of DataFrames with original household data

        Returns:
            DataFrame with aggregated time series and calendar features
        """
        # Validate index consistency
        reference_index = list_of_household_df[0].index
        reference_length = len(reference_index)
        
        for i, df in enumerate(list_of_household_df[1:], start=1):
            if len(df) != reference_length:
                raise ValueError(
                    f"Household {i} has {len(df)} rows, expected {reference_length} (reference household 0)"
                )
            if not df.index.equals(reference_index):
                raise ValueError(
                    f"Household {i} has different index values than reference household 0"
                )
            
        # Extract key columns
        elec_data = [df['Clean_elec_net_Wh'].fillna(0) for df in list_of_household_df]
        gas_data = [df['Clean_gas_Wh'].fillna(0) for df in list_of_household_df]  # Treat NaN as 0
        temp_data = [df['temp_C'].fillna(0) for df in list_of_household_df]
        solar_data = [df['surface_solar_radiation_downwards'].fillna(0) for df in list_of_household_df]

        # Stack and aggregate
        aggregation_method = self.config.get('sharding',{}).get('k10_aggregation',{}).get('aggregation_method', 'mean')
        if aggregation_method == 'mean':
            agg_elec = pd.concat(elec_data, axis=1).mean(axis=1)
            agg_gas = pd.concat(gas_data, axis=1).mean(axis=1)
            agg_temp = pd.concat(temp_data, axis=1).mean(axis=1)
            agg_solar = pd.concat(solar_data, axis=1).mean(axis=1)
        elif aggregation_method == 'median':
            agg_elec = pd.concat(elec_data, axis=1).median(axis=1)
            agg_gas = pd.concat(gas_data, axis=1).median(axis=1)
            agg_temp = pd.concat(temp_data, axis=1).median(axis=1)
            agg_solar = pd.concat(solar_data, axis=1).median(axis=1)
        else:
            raise ValueError(f'Unknown k10 aggregation method {aggregation_method}')

        # Create result DataFrame (use structure from first household)
        result_df = list_of_household_df[0][[
            'Read_date_time_local',
            'Read_date_effective_local',
            'Readings_from_midnight_local'
        ]].copy()

        # Add aggregated time series
        result_df['Clean_elec_net_Wh'] = agg_elec.values
        result_df['Clean_gas_Wh'] = agg_gas.values
        result_df['temp_C'] = agg_temp.values
        result_df['surface_solar_radiation_downwards'] = agg_solar.values

        # Add calendar features (use index for dow and month as requested)
        # Extract datetime index for calendar features (convert only if needed)
        result_df['hh'] = result_df['Readings_from_midnight_local']
        if isinstance(result_df.index, pd.DatetimeIndex):
            dt_index = result_df.index 
        else:
            dt_index = pd.to_datetime(result_df.index)
        
        result_df['dow'] = dt_index.dayofweek
        result_df['month'] = dt_index.month

        return result_df

    def _add_decoded_conditioning_vars(
        self,
        df: pd.DataFrame,
        conditioning_vars: Dict[str, int]
    ) -> pd.DataFrame:
        """
        Add decoded conditioning variables to aggregated k10 synthetic data.

        Args:
            df: DataFrame with aggregated time series
            conditioning_vars: Dict mapping variable names to token values

        Returns:
            DataFrame with decoded conditioning variables added as columns
        """
        for var_name, token_value in conditioning_vars.items():
            var_config = self.conditioning_variable_tokenisers[var_name]
            decoded_value = self._decode_conditioning_token(token_value, var_config)
            df[var_name] = decoded_value

        return df

    def _decode_conditioning_token(self, token: int, var_config: Dict) -> Any:
        """
        Decode a single conditioning variable token to human-readable value.

        Args:
            token: Token value to decode
            var_config: Variable configuration from conditioning_variable_tokenisers

        Returns:
            Decoded human-readable value
        """
        data_type = var_config.get('type', '')

        if data_type in ['categorical', 'ordinal', 'cardinal', 'boolean']:
            # Use index-to-value mapping
            itov = var_config.get('itov', {})
            token = int(token) # ensure token is int for dict lookup
            return itov.get(token, np.nan)

        elif data_type == 'continuous':
            # For continuous variables, return bin midpoint
            boundaries = var_config.get('boundaries', [])
            mask_token = var_config.get('mask_token', None)

            # Check if masked (only if mask_token is defined)
            if mask_token is not None and token == mask_token:
                return np.nan

            # Check bounds
            if token < 0 or token >= len(boundaries):
                return np.nan

            # Return bin midpoint
            if token == 0:
                return boundaries[0]
            elif token == len(boundaries) - 1:
                return boundaries[-1]
            else:
                return (boundaries[token] + boundaries[token + 1]) / 2

        return np.nan

    def _get_k_anonymity_config(self) -> Dict[str, int]:
        """
        Get k-anonymity masking configuration parameters from config.

        Returns:
            Dict with keys: augmentation_factor, max_var_count_constrained
        """
        k_config = self.config.get('sharding', {}).get('k_anonymity', {})
        return {
            'augmentation_factor': k_config.get('augmentation_factor', 1),
            'max_var_count_constrained': k_config.get('max_var_count_constrained', 6)
        }

    def get_puprns_for_split(self, split: str) -> List[str]:
        """
        Get list of PUPRNs assigned to a specific split.

        Args:
            split: 'train', 'val', or 'test'

        Returns:
            List of PUPRNs for the specified split
        """
        if not hasattr(self, 'metadata_catalog_filtered') or self.metadata_catalog_filtered is None:
            raise ValueError("Must call apply_filters_and_analyze() and assign_splits() first")

        if split not in ['train', 'val', 'test']:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")

        split_data = self.metadata_catalog_filtered[self.metadata_catalog_filtered['split'] == split]
        return split_data['puprn'].tolist()
    
    def get_conditioning_contexts(self, split: Optional[str] = None) -> List[Tuple[int, int]]:
        """
        DEPRECATED: legacy implementation using only two conditioning variables. see get_constrained_conditioning_context() and similar
        Extract conditioning variable pairs (B5, C1_new) from metadata catalog.
        
        Args:
            split: Optional split filter ('train', 'val', 'test'). If None, returns all contexts.
            
        Returns:
            List of (B5, C1_new) tuples for inference generation
        """
        if self.metadata_catalog is None:
            raise ValueError("Must call load_household_metadata() first")
        
        if split is not None:
            if split not in ['train', 'val', 'test']:
                raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")
            data = self.metadata_catalog[self.metadata_catalog['split'] == split]
        else:
            data = self.metadata_catalog
        
        # Extract (B5, C1_new) pairs, handling NaN values
        conditioning_contexts = []
        for _, row in data.iterrows():
            b5 = int(row['B5']) if pd.notna(row['B5']) else 0
            c1_new = int(row['C1_new']) if pd.notna(row['C1_new']) else 0
            conditioning_contexts.append((b5, c1_new))
        
        return conditioning_contexts
    
    def get_constrained_conditioning_contexts(
        self,
        split: str,
        max_var_count: int,
        augmentation_factor: int
    ) -> pd.DataFrame:
        """
        Get balanced constrained masked conditioning contexts for a split.

        Returns DataFrame with exactly (num_households × augmentation_factor) rows,
        where each household contributes augmentation_factor rows with maximum
        var_count <= max_var_count.
        """
        puprns = self.get_puprns_for_split(split)
        selected_rows = []

        for puprn in puprns:
            for aug_n in range(augmentation_factor):
                candidates = self.masked_conditioning_table[
                    (self.masked_conditioning_table['puprn'] == puprn) &
                    (self.masked_conditioning_table['augmentation_n'] == aug_n) &
                    (self.masked_conditioning_table['var_count'] <= max_var_count)
                ]

                if len(candidates) == 0:
                    raise ValueError(f"No constrained rows for {puprn} aug_n={aug_n}")

                # Select max var_count row
                max_vc = candidates['var_count'].max()
                selected = candidates[candidates['var_count'] == max_vc].iloc[0]
                selected_rows.append(selected)

        return pd.DataFrame(selected_rows)
    
    def get_unconstrained_conditioning_contexts(
        self,
        split: str,
        augmentation_factor: int
    ) -> pd.DataFrame:
        """
        Get balanced unconstrained conditioning contexts for a split.

        Returns DataFrame with exactly (num_households × augmentation_factor) rows,
        where each household contributes augmentation_factor rows with maximum var_count.
        """
        puprns = self.get_puprns_for_split(split)
        selected_rows = []

        for puprn in puprns:
            for aug_n in range(augmentation_factor):
                candidates = self.masked_conditioning_table[
                    (self.masked_conditioning_table['puprn'] == puprn) &
                    (self.masked_conditioning_table['augmentation_n'] == aug_n)
                ]

                # Select max var_count row (most unmasked)
                max_vc = candidates['var_count'].max()
                selected = candidates[candidates['var_count'] == max_vc].iloc[0]
                selected_rows.append(selected)

        return pd.DataFrame(selected_rows)
    
    def get_unmasked_conditioning_contexts(self, split: str) -> pd.DataFrame:
        """
        Get unmasked conditioning contexts for a split.

        Returns DataFrame with exactly num_households rows (one per household).
        """
        puprns = self.get_puprns_for_split(split)

        # Extract from filtered metadata_tokenised (only households that passed filters)
        return self.metadata_tokenised_filtered[self.metadata_tokenised_filtered['puprn'].isin(puprns)]

    def _get_conditioning_contexts_for_puprns(  
            self, 
            puprn_list: List[str], 
            constrained: bool,
            max_var_count: Optional[int] = None,
            augmentation_factor: int = 1
        ) -> pd.DataFrame:
      """Get conditioning contexts for specific list of puprns. Note this gets masked conditioning contexts"""
      selected_rows = []
      for puprn in puprn_list:
          for aug_n in range(augmentation_factor):
              if constrained:
                  candidates = self.masked_conditioning_table[
                      (self.masked_conditioning_table['puprn'] == puprn) &
                      (self.masked_conditioning_table['augmentation_n'] == aug_n) &
                      (self.masked_conditioning_table['var_count'] <= max_var_count)
                  ]
              else:
                  candidates = self.masked_conditioning_table[
                      (self.masked_conditioning_table['puprn'] == puprn) &
                      (self.masked_conditioning_table['augmentation_n'] == aug_n)
                  ]

              max_vc = candidates['var_count'].max()
              selected = candidates[candidates['var_count'] == max_vc].iloc[0]
              selected_rows.append(selected)

      return pd.DataFrame(selected_rows)
    
    def load_household_data_for_puprns(self, puprn_list: List[str]) -> List[pd.DataFrame]:
        """
        Load actual household data for a list of PUPRNs.
        
        Args:
            puprn_list: List of household PUPRNs to load
            
        Returns:
            List of household DataFrames
        """
        list_of_mod_1_data = []
        
        serl_survey_data = pd.read_csv(self.config['training']['data']['path_to_serl_survey_data'])
        
        for puprn in puprn_list:
            file_path = os.path.join(
                self.config['training']['data']['path_to_module_1_data'], 
                f"{puprn}_tokenised.pkl"
            )
            
            if not os.path.exists(file_path):
                self._log(f"File not found for PUPRN {puprn}: {file_path}", level='warning')
                continue
                
            try:
                with open(file_path, 'rb') as f:
                    household_data = pickle.load(f)
                mask = (serl_survey_data['PUPRN'] == puprn)
                my_serl_survey_data = serl_survey_data.loc[mask, self.config['training']['serl_survey_context']]
                num_rooms = my_serl_survey_data.loc[:, 'B5'].values[0]
                num_occs = my_serl_survey_data.loc[:, 'C1_new'].values[0]
                processed_data = _process_household_data(household_data, num_rooms, num_occs)
                list_of_mod_1_data.append(processed_data)
            except Exception as e:
                self._log(f"Failed to load data for PUPRN {puprn}: {str(e)}", level='error')
                continue
        
        self._log(f"Loaded data for {len(list_of_mod_1_data)}/{len(puprn_list)} households")
        return list_of_mod_1_data
    
    def _load_single_household_energy(self, puprn:str) -> pd.DataFrame:
        file_path = os.path.join(
            self.config['training']['data']['path_to_module_1_data'], 
            f"{puprn}_tokenised.pkl"
        )
        
        if not os.path.exists(file_path):
            self._log(f"File not found for PUPRN {puprn}: {file_path}", level='warning')
            return pd.DataFrame()
            
        try:
            with open(file_path, 'rb') as f:
                household_data = pickle.load(f)
        except Exception as e:
            self._log(f"Failed to load data for PUPRN {puprn}: {str(e)}", level='error')
            return pd.DataFrame()
    
        return household_data
    
    def load_household_data_with_conditioning(
        self,
        conditioning_contexts: pd.DataFrame
    ) -> List[Tuple[pd.DataFrame, pd.Series]]:
        """
        Load household energy data paired with conditioning contexts.

        Args:
            conditioning_contexts: DataFrame with columns [puprn, <conditioning_vars>, ...]
                For masked approach: includes all 14 variables + var_count, sample_size_n
                For unmasked approach: includes only configured conditioning_variables

        Returns:
            List of (energy_data, conditioning_row) tuples
        """
        household_data_pairs = []

        # Get configured conditioning variables from config
        data_filtering_config = self.config.get('sharding', {}).get('data_filtering', {})
        configured_vars = data_filtering_config.get('conditioning_variables', [])

        # For masked approach, use all 14 variables (legacy behavior)
        # For unmasked approach, use only configured variables that exist in dataframe
        use_masked_tokens = data_filtering_config.get('use_masked_tokens', False)

        if use_masked_tokens:
            # Masked approach: hardcoded 14-variable list
            conditioning_var_cols = ['region', 'imd', 'heating_type', 'num_rooms',
                                    'total_floor_area', 'epc_rating', 'household_income_band',
                                    'managing_financially', 'num_occs', 'building_type',
                                    'age_built', 'tenure', 'has_pv', 'has_ev']
        else:
            # Unmasked approach: use configured variables EXCEPT region and has_pv
            # (region and has_pv are for filtering/metadata only, not training features)
            training_vars = [v for v in configured_vars if v not in ['region', 'has_pv']]
            conditioning_var_cols = [v for v in training_vars if v in conditioning_contexts.columns]

        for idx, row in conditioning_contexts.iterrows():
            puprn = row['puprn']

            # Load energy sequence
            energy_data = self._load_single_household_energy(puprn)

            # Extract conditioning variables
            conditioning_vars = row[conditioning_var_cols]

            household_data_pairs.append((energy_data, conditioning_vars))

        return household_data_pairs
    
    def create_shard(self,
                     conditioning_context: pd.DataFrame,
                     shard_name: str,
                     vocab_size_elec: int,
                     vocab_size_gas: int) -> Dict[str, torch.Tensor]:
        """
        Create a training-ready shard from a list of PUPRNs.

        Args:
            conditioning_context: dataframe containing one augmentation_n rows per household, containing conditioning variables (masked constrained, unconstrained masked, unmasked)
            shard_name: Name for the shard (for logging)
            vocab_size_elec: Electricity vocabulary size
            vocab_size_gas: Gas vocabulary size

        Returns:
            Dictionary of tensors ready for training
        """
        self._log(f"Creating {shard_name} with {len(conditioning_context)} conditioning contexts...")
        
        # Load household data
        household_data_pairs = self.load_household_data_with_conditioning(conditioning_context)
        
        if not household_data_pairs:
            raise ValueError(f"No household data loaded for {shard_name}")
        
        # Use existing preprocessing function to create tensors
        tensor_data = process_multi_household_data_with_conditioning(
            household_data_pairs,
            self.config['training'],
            vocab_size_elec,
            vocab_size_gas,
        )
        
        self._log(f"{shard_name} created successfully:")
        for tensor_name, tensor in tensor_data.items():
            if hasattr(tensor, 'shape'):
                self._log(f"  {tensor_name}: {list(tensor.shape)}")
        
        return tensor_data
    
    def create_validation_shard(self,
                                vocab_size_elec: int,
                                vocab_size_gas: int,
                                masked: bool = True,
                                constrained: bool = False) -> Tuple[List[str], Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        Create validation shard using pre-assigned splits with specified masking strategy.

        This method supports three conditioning context modes:
        1. Masked constrained (masked=True, constrained=True): k-anonymity masking with curriculum learning
        2. Masked unconstrained (masked=True, constrained=False): k-anonymity masking, all variables unmasked
        3. Unmasked (masked=False): No masking, uses filtered conditioning variables directly

        Args:
            vocab_size_elec: Vocabulary size for electricity tokens
            vocab_size_gas: Vocabulary size for gas tokens
            masked: If True, use k-anonymity masked conditioning contexts; if False, use unmasked
            constrained: If True and masked=True, use constrained masking (curriculum learning)

        Returns:
            Tuple of (selected PUPRNs, tensor data dict with X_base and y_base)

        Raises:
            ValueError: If no households assigned to validation split
        """
        print("[SHARDING] Step 2: Creating validation shard...")
        
        # Get validation PUPRNs from split assignments
        val_puprns = self.get_puprns_for_split('val')
        
        if not val_puprns:
            raise ValueError("No households assigned to validation split. Call assign_splits() first.")
        
        self._log(f"Using {len(val_puprns)} households for validation")
        
        # get conditioning variables
        k_anonymity_config = self._get_k_anonymity_config()
        if masked:
            if constrained:
                conditioning_contexts = self.get_constrained_conditioning_contexts(
                    split='val',
                    augmentation_factor=k_anonymity_config['augmentation_factor'],
                    max_var_count=k_anonymity_config['max_var_count_constrained']
                )
            else:
                conditioning_contexts = self.get_unconstrained_conditioning_contexts(
                    split='val',
                    augmentation_factor=k_anonymity_config['augmentation_factor'],
                )
        else:
            conditioning_contexts = self.get_unmasked_conditioning_contexts(
                split = 'val'
            )
        # Create validation shard using existing preprocessing
        val_tensor_data = self.create_shard(
            conditioning_contexts,
            "validation shard",
            vocab_size_elec,
            vocab_size_gas,
        )
        
        return val_puprns, val_tensor_data
    
    def create_test_shard(self,
                          vocab_size_elec: int,
                          vocab_size_gas: int,
                          masked: bool = True,
                          constrained: bool = False) -> Tuple[List[str], Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        Create test shard using pre-assigned splits with specified masking strategy.

        This method supports three conditioning context modes:
        1. Masked constrained (masked=True, constrained=True): k-anonymity masking with curriculum learning
        2. Masked unconstrained (masked=True, constrained=False): k-anonymity masking, all variables unmasked
        3. Unmasked (masked=False): No masking, uses filtered conditioning variables directly

        Args:
            vocab_size_elec: Vocabulary size for electricity tokens
            vocab_size_gas: Vocabulary size for gas tokens
            masked: If True, use k-anonymity masked conditioning contexts; if False, use unmasked
            constrained: If True and masked=True, use constrained masking (curriculum learning)

        Returns:
            Tuple of (selected PUPRNs, tensor data dict with X_base and y_base)

        Raises:
            ValueError: If no households assigned to test split
        """
        print("[SHARDING] Step 3: Creating test shard...")
        
        # Get test PUPRNs from split assignments  
        test_puprns = self.get_puprns_for_split('test')
        
        if not test_puprns:
            raise ValueError("No households assigned to test split. Call assign_splits() first.")
        
        self._log(f"Using {len(test_puprns)} households for test")
        # get conditioning variables
        k_anonymity_config = self._get_k_anonymity_config()
        if masked:
            if constrained:
                conditioning_contexts = self.get_constrained_conditioning_contexts(
                    split='test',
                    augmentation_factor=k_anonymity_config['augmentation_factor'],
                    max_var_count=k_anonymity_config['max_var_count_constrained']
                )
            else:
                conditioning_contexts = self.get_unconstrained_conditioning_contexts(
                    split='test',
                    augmentation_factor=k_anonymity_config['augmentation_factor'],
                )
        else:
            conditioning_contexts = self.get_unmasked_conditioning_contexts(
                split = 'test'
            )
        # Create test shard using existing preprocessing
        test_tensor_data = self.create_shard(
            conditioning_contexts,
            "test shard",
            vocab_size_elec,
            vocab_size_gas,
        )
        
        return test_puprns, test_tensor_data
    
    def stratify_on_energy_use(self) -> None:
        """
        Assign households to consumption tiers (low/medium/high) based on total energy use.

        Calculates total mean energy consumption (electricity + gas) for each household
        and assigns them to tercile-based categories. Used by create_training_shards()
        to distribute tiers evenly across shards.

        What this DOES:
            - Ensures each training shard has ~equal representation of low/medium/high
              tercile households
            - Provides slightly more consistent per-epoch gradients when using multiple
              shards (each shard sees balanced consumption levels)

        What this does NOT do:
            - Does NOT address rare/extreme consumption patterns. Terciles by definition
              create three equal-sized groups (33% each), so there's nothing "rare" to
              balance. True outliers (e.g., top/bottom 5%) remain underrepresented.
            - Has NO effect when num_shards=1 (all households in same shard anyway)
            - Marginal benefit with many epochs (random assignment averages out)

        Future consideration:
            To help the model learn rare consumption patterns, oversampling of true
            outliers (not terciles) would be needed. However, this carries memorisation
            and privacy leakage risks that require careful evaluation.

        Adds columns to metadata_catalog_filtered:
            - total_energy_mean: Sum of elec_mean and gas_mean (Wh per half-hour)
            - energy_use_category: 'low', 'medium', or 'high' based on terciles

        Note:
            Requires elec_mean and gas_mean columns to be populated first
            (done during household_missingness calculation).

            The tier is used for training balance only and is NOT exposed as a
            conditioning variable at generation time.
        """
        self.metadata_catalog_filtered['total_energy_mean'] = (
            self.metadata_catalog_filtered['elec_mean'] + self.metadata_catalog_filtered['gas_mean']
        )
        terciles = self.metadata_catalog_filtered['total_energy_mean'].quantile([0.33, 0.66]).values

        def categorize_energy_use(value):
            if value <= terciles[0]:
                return 'low'
            elif value <= terciles[1]:
                return 'medium'
            else:
                return 'high'

        self.metadata_catalog_filtered['energy_use_category'] = (
            self.metadata_catalog_filtered['total_energy_mean'].apply(categorize_energy_use)
        )


    def create_training_shards(self, vocab_size_elec: int, vocab_size_gas: int) -> List[Dict[str, Any]]:
        """
        Create training shards from training pool with configurable masking strategy.

        Supports two mutually exclusive modes based on config.sharding.data_filtering.use_mask_tokens:
        1. Masked shards: Uses k-anonymity masking (constrained OR unconstrained based on create_constrained_shards)
        2. Unmasked shards: Uses filtered conditioning variables directly (no masking)

        When use_mask_tokens=False, this method creates unmasked training shards by:
        - Loading all training split unmasked conditioning contexts upfront (efficient)
        - For each shard, filtering contexts to the sampled PUPRNs
        - Creating shards with full conditioning variable information

        Args:
            vocab_size_elec: Vocabulary size for electricity tokens
            vocab_size_gas: Vocabulary size for gas tokens

        Returns:
            List of metadata dicts for each shard containing:
                - shard_index: Shard number
                - puprns: List of household PUPRNs in this shard
                - constrained_file: Path to constrained shard file (None if not created)
                - unconstrained_file: Path to unconstrained shard file (None if not created)
                - unmasked_file: Path to unmasked shard file (None if not created)
                - household_count: Number of households

        Raises:
            ValueError: If no households assigned to training split or sampling strategy is infeasible
        """
        print("[SHARDING] Step 4: Creating training shards (masked constrained/unconstrained)...")
        
        # Get training pool (excludes val and test households)
        training_pool_puprns = self.get_puprns_for_split('train')
        
        if not training_pool_puprns:
            raise ValueError("No households assigned to training split. Call assign_splits() first.")
        
        # assign puprn to low, medium and high terciles based on total average energy use
        self.stratify_on_energy_use()
        
        # assign pool puprns to low, medium, high pools using efficient dict lookup
        category_lookup = self.metadata_catalog_filtered.set_index('puprn')['energy_use_category'].to_dict()
        training_pool_puprns_low = []
        training_pool_puprns_medium = []
        training_pool_puprns_high = []
        for puprn in training_pool_puprns:
            category = category_lookup.get(puprn)
            if category == 'low':
                training_pool_puprns_low.append(puprn)
            elif category == 'medium':
                training_pool_puprns_medium.append(puprn)
            elif category == 'high':
                training_pool_puprns_high.append(puprn)
            # Skip households with missing category (shouldn't happen but defensive)

        self._log(f"  Stratified pools: low={len(training_pool_puprns_low)}, medium={len(training_pool_puprns_medium)}, high={len(training_pool_puprns_high)}")

        # Get configuration
        sharding_config = self.config.get('sharding', {})
        num_shards = sharding_config.get('num_shards', 1)
        households_per_shard_config = sharding_config.get('households_per_shard', 500)
        sampling_strategy = sharding_config.get('sampling_strategy', 'without_replacement')

        # Calculate tier sizes for auto mode and validation
        tier_sizes = {
            'low': len(training_pool_puprns_low),
            'medium': len(training_pool_puprns_medium),
            'high': len(training_pool_puprns_high)
        }
        min_tier_size = min(tier_sizes.values())
        min_tier_name = min(tier_sizes, key=tier_sizes.get)

        # Handle auto-calculation of households_per_shard
        if households_per_shard_config == 'auto':
            # Calculate maximum achievable households_per_shard based on smallest tier
            # Each shard needs households_per_shard // 3 from each tier
            # So: min_tier_size >= num_shards * (households_per_shard // 3)
            # Therefore: households_per_shard = (min_tier_size // num_shards) * 3
            max_per_tier_per_shard = min_tier_size // num_shards
            households_per_shard = max_per_tier_per_shard * 3
            self._log(f"Auto-calculated households_per_shard: {households_per_shard} "
                     f"(limited by '{min_tier_name}' tier with {min_tier_size} households, "
                     f"{max_per_tier_per_shard} per tier × 3 tiers)")
        else:
            households_per_shard = households_per_shard_config

        # Get k10 masking configuration
        k_anonymity_config = self._get_k_anonymity_config()
        augmentation_factor = k_anonymity_config['augmentation_factor']
        max_var_count = k_anonymity_config['max_var_count_constrained']

        self._log(f"Training pool: {len(training_pool_puprns)} households")
        self._log(f"Creating {num_shards} shards with {households_per_shard} households each")
        self._log(f"Sampling strategy: {sampling_strategy}")
        
        # Validate sampling feasibility
        if sampling_strategy == 'without_replacement':
            total_needed = num_shards * households_per_shard
            if total_needed > len(training_pool_puprns):
                raise ValueError(f"Cannot sample {total_needed} households without replacement from pool of {len(training_pool_puprns)}")

        # Validate stratified sampling feasibility - check each tier has enough households
        # (This validation is mainly for when households_per_shard is manually specified;
        # when 'auto' is used, this should always pass)
        n_per_tier_per_shard = households_per_shard // 3
        if sampling_strategy == 'without_replacement':
            total_needed_per_tier = num_shards * n_per_tier_per_shard
            for tier_name, tier_size in tier_sizes.items():
                if tier_size < total_needed_per_tier:
                    raise ValueError(
                        f"Insufficient households in '{tier_name}' consumption tier: "
                        f"need {total_needed_per_tier} for {num_shards} shards × {n_per_tier_per_shard} per tier, "
                        f"but only {tier_size} available. "
                        f"Consider using households_per_shard: 'auto', reducing num_shards, or use 'with_replacement' sampling."
                    )

        # check if should create masked or constrained shards
        create_masked_shards = self.config.get('sharding', {}).get('data_filtering',{}).get('use_mask_tokens',False)
        create_constrained = self._should_create_constrained_shards()
        self._log(f"Creating masked shards: {create_masked_shards}")
        self._log(f"Creating constrained shards: {create_constrained}")
    
        training_shard_metadata = []

        # Use seeded random generator for reproducible sampling
        rng = np.random.RandomState(self.random_seed + 2)  # Different from val/test seeds
        
        # load unmasked conditioning contexts
        unmasked_conditioning_contexts_for_all_train = self.get_unmasked_conditioning_contexts('train')
        
        for shard_idx in range(num_shards):
            self._log(f"Creating training shard {shard_idx + 1}/{num_shards}...")
            
            # Sample households for this shard
            shard_puprns = []
            n_low = households_per_shard//3
            n_high= n_low
            n_medium = households_per_shard - n_low - n_high
            if sampling_strategy == 'with_replacement':
                shard_puprns_low = rng.choice(
                # Can sample same households across different shards
                    training_pool_puprns_low, 
                    size=n_low, 
                    replace=True
                ).tolist()
                shard_puprns_medium = rng.choice(
                    training_pool_puprns_medium, 
                    size=n_medium, 
                    replace=True
                ).tolist()
                shard_puprns_high = rng.choice(
                    training_pool_puprns_high, 
                    size=n_high, 
                    replace=True
                ).tolist()
                shard_puprns = shard_puprns_low + shard_puprns_medium + shard_puprns_high
            else:  # without_replacement
                # Remove sampled households from pool
                if len(training_pool_puprns) < households_per_shard:
                    raise ValueError(f"Not enough households remaining for shard {shard_idx}")
                
                shard_puprns_low = rng.choice(
                    training_pool_puprns_low, 
                    size=n_low, 
                    replace=False
                ).tolist()
                shard_puprns_medium = rng.choice(
                    training_pool_puprns_medium, 
                    size=n_medium, 
                    replace=False
                ).tolist()
                shard_puprns_high = rng.choice(
                    training_pool_puprns_high, 
                    size=n_high, 
                    replace=False
                ).tolist()
                shard_puprns = shard_puprns_low + shard_puprns_medium + shard_puprns_high
                
                # Remove from pools for next shard
                training_pool_puprns_low = [puprn for puprn in training_pool_puprns_low if puprn not in shard_puprns]
                training_pool_puprns_medium = [puprn for puprn in training_pool_puprns_medium if puprn not in shard_puprns]
                training_pool_puprns_high = [puprn for puprn in training_pool_puprns_high if puprn not in shard_puprns]
            
            constrained_filename = None
            unconstrained_filename = None
            unmasked_filename = None
            if create_masked_shards:
                if create_constrained:
                    # Get CONSTRAINED conditioning contexts for these households
                    constrained_contexts = self._get_conditioning_contexts_for_puprns(
                        shard_puprns,
                        constrained=True,
                        max_var_count=max_var_count,
                        augmentation_factor=augmentation_factor
                    )
                    # Create constrained shard
                    constrained_shard_data = self.create_shard(
                        constrained_contexts,
                        f"training constrained shard {shard_idx}",
                        vocab_size_elec,
                        vocab_size_gas,
                    )
                    # immediately save shard to avoid OOM
                    constrained_filename = self._save_single_training_shard(
                        shard_idx, shard_puprns, constrained_shard_data, 'constrained'
                        )
                else:
                    # Get UNCONSTRAINED conditioning contexts for same households
                    unconstrained_contexts = self._get_conditioning_contexts_for_puprns(
                        shard_puprns,
                        constrained=False,
                        augmentation_factor=augmentation_factor
                    )

                    # Create unconstrained shard
                    unconstrained_shard_data = self.create_shard(
                        unconstrained_contexts,
                        f"training unconstrained shard {shard_idx}",
                        vocab_size_elec,
                        vocab_size_gas,
                    )
                    # immediately save shard to avoid OOM
                    unconstrained_filename = self._save_single_training_shard(
                        shard_idx, shard_puprns, unconstrained_shard_data, 'unconstrained'
                        )
            else:
                # create unmasked shard
                # get unmasked conditioning contexts for shard_puprns
                unmasked_contexts_for_shard = unmasked_conditioning_contexts_for_all_train.loc[unmasked_conditioning_contexts_for_all_train['puprn'].isin(shard_puprns)]

                # create unmasked shard
                unmasked_shard_data = self.create_shard(
                    conditioning_context=unmasked_contexts_for_shard,
                    shard_name=f"training unmasked shard {shard_idx}",
                    vocab_size_elec=vocab_size_elec,
                    vocab_size_gas=vocab_size_gas,
                )
                # immediately save shard to avoid OOM
                unmasked_filename = self._save_single_training_shard(
                        shard_idx, shard_puprns, unmasked_shard_data, 'unmasked'
                        )

            # Store only lightweight metadata
            training_shard_metadata.append({
                'shard_index': shard_idx,
                'puprns': shard_puprns,
                'constrained_file': constrained_filename, # None if not created
                'unconstrained_file': unconstrained_filename, # None if not created
                'unmasked_file': unmasked_filename, # None if not created
                'household_count': len(shard_puprns)
            })
        self._log(f"Created {len(training_shard_metadata)} training shards successfully (constrained/unconstrained)")
        return training_shard_metadata
    
    def _save_single_training_shard(self, shard_idx: int, shard_puprns: List[str],
                            shard_tensor_data: Dict, masking_type: str) -> str:
        """Save a single training shard immediately after creation."""
        shard_output_dir = os.path.join(self.run_path, self.config['sharding']['shard_output_dir'])
        k_anonymity_config = self._get_k_anonymity_config()

        filename = f'train_{masking_type}_{shard_idx:03d}.pt'
        shard_path = os.path.join(shard_output_dir, filename)

        torch.save({
            'X_train': shard_tensor_data['X_base'],
            'y_train': shard_tensor_data['y_base'],
            'metadata': {
                'shard_type': f'training_{masking_type}',
                'masking_type': masking_type,
                'max_var_count': k_anonymity_config.get('max_var_count_constrained') if masking_type == 'constrained' else None,
                'augmentation_factor': k_anonymity_config['augmentation_factor'],
                'shard_index': shard_idx,
                'household_count': len(shard_puprns),
                'sequence_count': shard_tensor_data['X_base'].shape[0],
                'puprn_list': shard_puprns
            }
        }, shard_path)

        self._log(f"Saved: {shard_path}")
        return filename

    def generate_k10_aggregates_unmasked(self) -> None:
        """
        Generate k10 aggregated synthetic data for unmasked conditioning approach.

        This method creates privacy baseline synthetic data by aggregating time series from
        10 real households that share conditioning variable values. Works with human-readable
        metadata_catalog_filtered (no tokenization/decoding needed).

        Key differences from masked approach:
        - Uses metadata_catalog_filtered (human-readable conditioning vars)
        - No progressive masking - uses all conditioning filter variables
        - Samples with/without replacement from k>=10 groups
        - Output excludes source PUPRNs for privacy preservation
        - Includes sample_size_n for SDC compliance checking

        Process:
        1. Get training split PUPRNs only (prevents data leakage)
        2. Group by conditioning filter variables, find k>=10 groups
        3. For each group (up to max_synthetic_households):
           - Randomly select 10 households
           - Load original time series data
           - Aggregate via mean/median
           - Add conditioning variables and k-anonymity group size
           - Save WITHOUT source PUPRNs
        4. Save metadata and statistics

        Configuration (sharding.k10_aggregation):
            - enabled: Generate k-aggregates (default: false)
            - k_value: K-anonymity level for aggregation (default: 10)
                Sets both minimum group size and number of households to aggregate
            - max_synthetic_households: Cap on number of k-aggregates
            - aggregation_method: "mean" or "median" (default: "mean")
            - random_seed: For reproducible household sampling
            - path_to_data_to_aggregate: Path to original (non-tokenized) household data
              (relative paths resolved against run_path; absolute paths used as-is)
            - output_dir: Where to save k-aggregates (relative to run_path)

        Side Effects:
            - Generates k-aggregated synthetic households to k10_aggregation.output_dir
            - Saves k10_aggregation_metadata.pkl (conditioning vars, group sizes, filepaths)
            - Saves k10_aggregation_stats.json (count, avg group size, load failures)
            - Increments self.k10_aggregate_count
            - Populates self.k10_metadata list

        Privacy Guarantees:
            - Only uses k>=config.k_value groups (default k>=10)
            - Only samples config.k_value households per aggregate (default 10)
            - Only samples from training split
            - Output files exclude source PUPRNs
            - Includes sample_size_n for SDC verification
        """
        k10_config = self.config.get('sharding', {}).get('k10_aggregation', {})
        k10_enabled = k10_config.get('enabled', False)

        if not k10_enabled:
            self._log("K-aggregation disabled, skipping")
            return

        # Get k-anonymity level for aggregation
        k_value = k10_config.get('k_value', 10)

        self._log("=" * 60)
        self._log(f"K{k_value} AGGREGATION (Unmasked Approach)")
        self._log("=" * 60)

        # Get configuration
        max_synthetic = k10_config.get('max_synthetic_households', 0)
        aggregation_method = k10_config.get('aggregation_method', 'mean')
        k10_seed = k10_config.get('random_seed', 42)

        # Get conditioning filter variables (exclude region and has_pv)
        data_filtering_config = self.config.get('sharding', {}).get('data_filtering', {})
        conditioning_vars = data_filtering_config.get('conditioning_variables', [])
        conditioning_vars_filter = [v for v in conditioning_vars if v not in ['region', 'has_pv']]

        self._log(f"K-anonymity level: k={k_value}")
        self._log(f"Conditioning variables for grouping: {conditioning_vars_filter}")
        self._log(f"Max synthetic households: {max_synthetic}")
        self._log(f"Aggregation method: {aggregation_method}")

        # Get training split households only (prevent data leakage)
        train_households = self.metadata_catalog_filtered[
            self.metadata_catalog_filtered['split'] == 'train'
        ].copy()

        self._log(f"Training households available: {len(train_households)}")

        # Get original column names for conditioning variables
        conditioning_cols_original = []
        for var in conditioning_vars_filter:
            if var in self.conditioning_variable_tokenisers:
                orig_col = self.conditioning_variable_tokenisers[var]['original_variable_name']
                conditioning_cols_original.append(orig_col)
            else:
                self._log(f"Warning: {var} not in tokeniser config, skipping", level='warning')

        if not conditioning_cols_original:
            self._log("No valid conditioning variables found, skipping k10 aggregation", level='warning')
            return

        self._log(f"Original column names: {conditioning_cols_original}")

        # Group by conditioning variables and find k>=k_value groups
        grouped = train_households.groupby(conditioning_cols_original, dropna=False).size().reset_index(name='group_size')
        k_groups = grouped[grouped['group_size'] >= k_value]

        self._log(f"Found {len(k_groups)} k>={k_value} groups in training split")

        if len(k_groups) == 0:
            self._log(f"No k>={k_value} groups found, skipping k-aggregation", level='warning')
            return

        # Setup output directory
        # Resolve relative to run_path (absolute paths pass through unchanged)
        original_data_folder = os.path.join(self.run_path, k10_config.get('path_to_data_to_aggregate', ''))

        k10_output_folder = os.path.join(self.run_path, k10_config.get('output_dir', ''))
        os.makedirs(k10_output_folder, exist_ok=True)

        self._log(f"Original data folder: {original_data_folder}")
        self._log(f"Output folder: {k10_output_folder}")

        # Generate k-aggregates
        k10_rng = np.random.RandomState(k10_seed)

        for group_idx, group_row in k_groups.iterrows():
            if self.k10_aggregate_count >= max_synthetic:
                self._log(f"Reached maximum synthetic households ({max_synthetic}), stopping")
                break

            # Get conditioning variable values for this group
            group_conditions = {col: group_row[col] for col in conditioning_cols_original}
            group_size = group_row['group_size']

            # Find all training households matching this group
            mask = pd.Series([True] * len(train_households))
            for col, value in group_conditions.items():
                if pd.isna(value):
                    mask = mask & train_households[col].isna()
                else:
                    mask = mask & (train_households[col] == value)

            matching_puprns = train_households.loc[mask, 'puprn'].tolist()

            if len(matching_puprns) < k_value:
                self._log(f"Group {group_idx}: Only {len(matching_puprns)} households, skipping", level='warning')
                continue

            # Randomly select k_value households
            selected_puprns = k10_rng.choice(matching_puprns, size=k_value, replace=False)

            # Load household data
            list_of_household_df = []
            load_failures = []

            for puprn in selected_puprns:
                try:
                    with open(os.path.join(original_data_folder, f'{puprn}.pkl'), 'rb') as f:
                        df = pickle.load(f)
                    list_of_household_df.append(df)
                except Exception as e:
                    load_failures.append(puprn)
                    self._log(f"Failed to load {puprn}: {str(e)}", level='warning')

            # Require at least k_value successful loads
            if len(list_of_household_df) < k_value:
                self._log(f"Group {group_idx}: Only {len(list_of_household_df)}/{k_value} loaded successfully, skipping", level='warning')
                continue

            # Perform aggregation
            aggregated_df = self._generate_k10_aggregate(list_of_household_df)

            # Add conditioning variables (already human-readable from metadata_catalog_filtered)
            for var_name in conditioning_vars_filter:
                orig_col = self.conditioning_variable_tokenisers[var_name]['original_variable_name']
                aggregated_df[var_name] = group_conditions[orig_col]

            # Add k-anonymity group size for SDC compliance checking
            aggregated_df['sample_size_n'] = group_size

            # Save WITHOUT source PUPRNs (privacy preservation)
            synthetic_id = f'k10_synthetic_{self.k10_aggregate_count}'
            output_path = os.path.join(k10_output_folder, f'{synthetic_id}.pkl')
            aggregated_df.to_pickle(output_path)

            # Record metadata (keep source PUPRNs in metadata for transparency, not in output)
            self.k10_metadata.append({
                'synthetic_id': synthetic_id,
                'source_puprns': selected_puprns.tolist(),  # Metadata only, not in output
                'conditioning_vars': {var: group_conditions[self.conditioning_variable_tokenisers[var]['original_variable_name']]
                                     for var in conditioning_vars_filter},
                'filepath': output_path,
                'group_size': group_size,
                'load_failures': load_failures,
                'split': 'train'
            })

            self.k10_aggregate_count += 1

            if (self.k10_aggregate_count % 10 == 0) or (self.k10_aggregate_count == max_synthetic):
                self._log(f"Generated {self.k10_aggregate_count}/{max_synthetic} k10 aggregates...")

        # Save metadata and statistics
        if self.k10_metadata:
            # Save metadata DataFrame
            k10_metadata_df = pd.DataFrame(self.k10_metadata)
            metadata_path = os.path.join(k10_output_folder, 'k10_aggregation_metadata.pkl')
            k10_metadata_df.to_pickle(metadata_path)
            self._log(f"Saved k10 metadata: {metadata_path}")

            # Save statistics JSON
            stats_path = os.path.join(k10_output_folder, 'k10_aggregation_stats.json')
            with open(stats_path, 'w') as f:
                json.dump({
                    'k_value': k_value,
                    'total_k_aggregates': self.k10_aggregate_count,
                    'generation_timestamp': datetime.now().isoformat(),
                    'config': k10_config,
                    'conditioning_variables': conditioning_vars_filter,
                    'avg_group_size': float(k10_metadata_df['group_size'].mean()),
                    'total_load_failures': sum(len(m.get('load_failures', [])) for m in self.k10_metadata),
                    'approach': 'unmasked'
                }, f, indent=2)
            self._log(f"Saved k-aggregation statistics: {stats_path}")

        self._log("=" * 60)
        self._log(f"K{k_value} aggregation completed: {self.k10_aggregate_count} synthetic households generated")
        self._log("=" * 60)

    def shard_data(self):
        """
        Execute complete sharding pipeline with filtering, k-anonymity, and configurable masking.

        Pipeline stages:
        1. Load household metadata and calculate missingness
        2. Tokenize conditioning variables with category merging
        3. Determine k-anonymity group sizes
        4. Apply filters (PV, missingness, required variables, k-anonymity)
        5. Assign train/val/test splits (prevents data leakage)
        6. Create validation shards (masked + unmasked for secondary validation metrics)
        7. Create test shards (masked + unmasked for secondary evaluation metrics)
        8. Create training shards (EITHER masked OR unmasked, mutually exclusive)
        9. Save all shards to disk with metadata
        10. Validate shard integrity

        Shard Creation Strategy:
        - When use_mask_tokens=True: Creates masked shards for training, masked+unmasked for val/test
        - When use_mask_tokens=False: Creates unmasked shards only (simplified conditioning approach)
        - Val/test always include unmasked versions to support secondary validation metrics

        Raises:
            ValueError: If shard validation fails
        """
        self._log("Starting complete sharding pipeline...")
        
        # Step 1: Load household metadata
        self._log("Step 1: Loading household metadata...")
        self.load_household_metadata()

        # save the metadata_catalog to disk
        shard_output_dir = os.path.join(self.run_path, self.config['sharding']['shard_output_dir'])
        os.makedirs(shard_output_dir, exist_ok=True)

        self._log(f"Saving raw metadata_catalog: {shard_output_dir}")

        self.metadata_catalog.to_pickle(os.path.join(shard_output_dir, 'metadata_catalog_raw.pkl'))
        
        # Step 1.1: calculate household-level missingness and add info to metadata_catalog
        self.household_missingness()

        # Step 1.2: tokenise metadata catalog
        self.tokenise_conditioning_variables()

        # Step 1.3: determine k-anonymity
        self.determine_k_anonymity()

        # Step 1.4: apply filters, log exclusion counts, analyse remaining category distributions, and produce final metadata set to use going forward for training
        self._log("Step 1.4: Applying filters and analyzing distributions...")
        self.metadata_catalog_filtered = self.apply_filters_and_analyze()

        # Assign train/val/test splits to prevent data leakage (on filtered data)
        self._log("Assigning train/val/test splits...")
        self.assign_splits()

        # create table of masked conditioning variables, if masking is being used
        create_masked_shards = self.config.get('sharding', {}).get('data_filtering',{}).get('use_mask_tokens',False)
        if create_masked_shards:
            # Masked approach: k10 aggregation integrated in create_masked_conditioning_variables()
            seed = self.random_seed
            augmentation_factor = self.config.get('sharding',{}).get('k_anonymity',{}).get('augmentation_factor',1)
            masked_conditioning_table = self.create_masked_conditioning_variables(augmentation_factor=augmentation_factor,
                                                                                seed=seed)
            self.masked_conditioning_table = masked_conditioning_table
        else:
            # Unmasked approach: k10 aggregation as separate step
            self.generate_k10_aggregates_unmasked()
        
        # Load tokenizers to get vocabulary sizes using TokenVocabulary
        self._log("Loading tokenizers for vocabulary sizes...")
        from utils.universal_tokeniser_naive import load_tokenisers_as_vocab
        training_config = self.config.get('training', {})
        tokeniser_file = training_config.get('data', {}).get('tokeniser_file')

        if not tokeniser_file or not os.path.exists(tokeniser_file):
            raise FileNotFoundError(f"Tokeniser file not found: {tokeniser_file}")

        vocab = load_tokenisers_as_vocab(tokeniser_file)
        vocab_size_elec = vocab['electricity'].vocab_size
        vocab_size_gas = vocab['gas'].vocab_size
        vocab_size_temp = vocab['temperature'].vocab_size
        vocab_size_solar = vocab['solar'].vocab_size

        self._log(f"Vocabulary sizes - Elec: {vocab_size_elec}, Gas: {vocab_size_gas}, Temp: {vocab_size_temp}, Solar: {vocab_size_solar}")

        # Check if should create constrained shards    
        create_constrained = self._should_create_constrained_shards()
    
        # Step 2: Create validation shard
        print("[SHARDING] Step 2: Creating validation shards (masked constrained/unconstrained, unmasked)...")
        val_tensor_data_masked_constrained = None
        val_tensor_data_masked_unconstrained = None
        if create_masked_shards:
            if create_constrained:
                val_puprns, val_tensor_data_masked_constrained = self.create_validation_shard(
                    vocab_size_elec,
                    vocab_size_gas,
                    masked=True,
                    constrained=True,
                )
            else:
                val_puprns, val_tensor_data_masked_unconstrained = self.create_validation_shard(
                    vocab_size_elec,
                    vocab_size_gas,
                    masked=True,
                    constrained=False,
                )

        val_puprns, val_tensor_data_unmasked = self.create_validation_shard(
            vocab_size_elec,
            vocab_size_gas,
            masked=False,
            constrained=False,
        )
        
        # Step 3: Create test shard
        print("[SHARDING] Step 3: Creating test shards (masked constrained/unconstrained, unmasked)...")
        
        test_tensor_data_masked_constrained = None
        test_tensor_data_masked_unconstrained = None
        if create_masked_shards:
            if create_constrained:
                test_puprns, test_tensor_data_masked_constrained = self.create_test_shard(
                    vocab_size_elec,
                    vocab_size_gas,
                    masked=True,
                    constrained=True,
                )
            else:
                test_puprns, test_tensor_data_masked_unconstrained = self.create_test_shard(
                    vocab_size_elec,
                    vocab_size_gas,
                    masked=True,
                    constrained=False,
                )

        test_puprns, test_tensor_data_unmasked = self.create_test_shard(
            vocab_size_elec,
            vocab_size_gas,
            masked=False,
            constrained=False,
        )
        
        # Step 4: Create training shards
        print("[SHARDING] Step 4: Creating training shards...")
        training_shard_metadata = self.create_training_shards(vocab_size_elec, vocab_size_gas)
        
        # Step 5: Save all shards to disk
        print("[SHARDING] Step 5: Saving shards to disk...")
        self.save_shards(
            {
                'constrained': val_tensor_data_masked_constrained,
                'unconstrained': val_tensor_data_masked_unconstrained,
                'unmasked': val_tensor_data_unmasked
            },
            {
                'constrained': test_tensor_data_masked_constrained,
                'unconstrained': test_tensor_data_masked_unconstrained,
                'unmasked': test_tensor_data_unmasked
            },
            training_shard_metadata
        )
        # Step 6: Generate and save metadata and statistics
        print("[SHARDING] Step 6: Generating shard metadata and statistics...")
        self.save_shard_metadata(val_puprns, test_puprns, training_shard_metadata)

        # Step 7: Validate all created shards
        print("[SHARDING] Step 7: Validating shards...")
        validation_passed = self.validate_shards()
        if not validation_passed:
            raise ValueError("Shard validation failed! See logs for details.")

        print("[SHARDING] Sharding pipeline completed successfully!")
        
    def save_shards(self, val_tensor_data_dict, test_tensor_data_dict, training_shard_metadata):
        """
        Save validation and test shards as .pt files with new naming convention.

        Handles None values gracefully - only saves shards that were actually created.
        Training shards are saved immediately during creation to avoid OOM issues.

        Args:
            val_tensor_data_dict: Dict with keys 'constrained', 'unconstrained', 'unmasked' (values can be None)
            test_tensor_data_dict: Dict with keys 'constrained', 'unconstrained', 'unmasked' (values can be None)
            training_shard_metadata: List of metadata dicts for training shards (already saved during creation)

        Note:
            - When use_mask_tokens=False, only unmasked shards are created (constrained/unconstrained are None)
            - When use_mask_tokens=True, creates either constrained OR unconstrained (one will be None)
            - Val/test always include unmasked versions for secondary validation metrics
        """
        # Create shards directory
        shard_output_dir = os.path.join(self.run_path, self.config['sharding']['shard_output_dir'])
        self.config['sharding']['shard_output_dir'] = shard_output_dir  # Update config with absolute path
        os.makedirs(shard_output_dir, exist_ok=True)

        self._log(f"Saving shards to: {shard_output_dir}")

        # Get k10 masking config for metadata
        k_anonymity_config = self._get_k_anonymity_config()

        # Save validation shards (3 types: constrained, unconstrained, unmasked)
        self._log("Saving validation shards...")
        shard_count = 0
        val_puprns_count = len(self.get_puprns_for_split('val'))

        if val_tensor_data_dict['constrained'] is not None:
            # Constrained validation
            val_constrained_path = os.path.join(shard_output_dir, 'val_constrained.pt')
            torch.save({
                'X_val': val_tensor_data_dict['constrained']['X_base'],
                'y_val': val_tensor_data_dict['constrained']['y_base'],
                'metadata': {
                    'shard_type': 'validation_constrained',
                    'masking_type': 'constrained',
                    'max_var_count': k_anonymity_config['max_var_count_constrained'],
                    'augmentation_factor': k_anonymity_config['augmentation_factor'],
                    'household_count': val_puprns_count,
                    'sequence_count': val_tensor_data_dict['constrained']['X_base'].shape[0]
                }
            }, val_constrained_path)
            self._log(f"Saved: {val_constrained_path}")
            shard_count += 1

        # Unconstrained validation
        if val_tensor_data_dict['unconstrained'] is not None:
            val_unconstrained_path = os.path.join(shard_output_dir, 'val_unconstrained.pt')
            torch.save({
                'X_val': val_tensor_data_dict['unconstrained']['X_base'],
                'y_val': val_tensor_data_dict['unconstrained']['y_base'],
                'metadata': {
                    'shard_type': 'validation_unconstrained',
                    'masking_type': 'unconstrained',
                    'augmentation_factor': k_anonymity_config['augmentation_factor'],
                    'household_count': val_puprns_count,
                    'sequence_count': val_tensor_data_dict['unconstrained']['X_base'].shape[0]
                }
            }, val_unconstrained_path)
            self._log(f"Saved: {val_unconstrained_path}")
            shard_count += 1

        # Unmasked validation
        if val_tensor_data_dict['unmasked'] is not None:
            val_unmasked_path = os.path.join(shard_output_dir, 'val_unmasked.pt')
            torch.save({
                'X_val': val_tensor_data_dict['unmasked']['X_base'],
                'y_val': val_tensor_data_dict['unmasked']['y_base'],
                'metadata': {
                    'shard_type': 'validation_unmasked',
                    'masking_type': 'unmasked',
                    'household_count': val_puprns_count,
                    'sequence_count': val_tensor_data_dict['unmasked']['X_base'].shape[0]
                }
            }, val_unmasked_path)
            self._log(f"Saved: {val_unmasked_path}")
            shard_count += 1
        self._log(f"Validation: {shard_count} shards")

        # Save test shards (3 types: constrained, unconstrained, unmasked)
        self._log("Saving test shards...")
        test_puprns_count = len(self.get_puprns_for_split('test'))
        shard_count = 0
        # Constrained test
        if test_tensor_data_dict['constrained'] is not None:
            test_constrained_path = os.path.join(shard_output_dir, 'test_constrained.pt')
            torch.save({
                'X_test': test_tensor_data_dict['constrained']['X_base'],
                'y_test': test_tensor_data_dict['constrained']['y_base'],
                'metadata': {
                    'shard_type': 'test_constrained',
                    'masking_type': 'constrained',
                    'max_var_count': k_anonymity_config['max_var_count_constrained'],
                    'augmentation_factor': k_anonymity_config['augmentation_factor'],
                    'household_count': test_puprns_count,
                    'sequence_count': test_tensor_data_dict['constrained']['X_base'].shape[0]
                }
            }, test_constrained_path)
            self._log(f"Saved: {test_constrained_path}")
            shard_count += 1

        # Unconstrained test
        if test_tensor_data_dict['unconstrained'] is not None:
            test_unconstrained_path = os.path.join(shard_output_dir, 'test_unconstrained.pt')
            torch.save({
                'X_test': test_tensor_data_dict['unconstrained']['X_base'],
                'y_test': test_tensor_data_dict['unconstrained']['y_base'],
                'metadata': {
                    'shard_type': 'test_unconstrained',
                    'masking_type': 'unconstrained',
                    'augmentation_factor': k_anonymity_config['augmentation_factor'],
                    'household_count': test_puprns_count,
                    'sequence_count': test_tensor_data_dict['unconstrained']['X_base'].shape[0]
                }
            }, test_unconstrained_path)
            self._log(f"Saved: {test_unconstrained_path}")
            shard_count += 1

        # Unmasked test
        if test_tensor_data_dict['unmasked'] is not None:
            test_unmasked_path = os.path.join(shard_output_dir, 'test_unmasked.pt')
            torch.save({
                'X_test': test_tensor_data_dict['unmasked']['X_base'],
                'y_test': test_tensor_data_dict['unmasked']['y_base'],
                'metadata': {
                    'shard_type': 'test_unmasked',
                    'masking_type': 'unmasked',
                    'household_count': test_puprns_count,
                    'sequence_count': test_tensor_data_dict['unmasked']['X_base'].shape[0]
            }
        }, test_unmasked_path)
        self._log(f"Saved: {test_unmasked_path}")
        shard_count += 1
        self._log(f"Test: {shard_count} shards")
        # Training shards saved during creation
        
        # Count how many training shards have constrained files
        constrained_count = sum(1 for s in training_shard_metadata if s['constrained_file'] is not None)
        self._log(f"Training: {len(training_shard_metadata)} unconstrained shards, {constrained_count} constrained shards")
    
    def validate_shards(self) -> bool:
        """
        Validate all created shards for correctness.

        Performs 4 key validation checks:
        1. Balanced representation (household_count × augmentation_factor = sequence_count)
        2. Var count distribution (constrained shards respect max_var_count)
        3. Feature shape (all shards have 21 features)
        4. Unmasked shards have no mask tokens

        Returns:
            bool: True if all validations pass, False otherwise
        """
        shard_output_dir = self.config['sharding']['shard_output_dir']
        k_anonymity_config = self._get_k_anonymity_config()

        self._log("=" * 60)
        self._log("Shard Validation")
        self._log("=" * 60)

        validation_passed = True
        validation_errors = []

        # Determine expected number of features based on masking configuration
        data_filtering_config = self.config.get('sharding', {}).get('data_filtering', {})
        use_masked_tokens = data_filtering_config.get('use_masked_tokens', False)

        if use_masked_tokens:
            # Masked approach: 14 conditioning variables
            expected_num_features = 21  # 7 temporal + 14 conditioning
            conditioning_var_names = [
                'region', 'imd', 'heating_type', 'num_rooms', 'total_floor_area',
                'epc_rating', 'household_income_band', 'managing_financially', 'num_occs',
                'building_type', 'age_built', 'tenure', 'has_pv', 'has_ev'
            ]
        else:
            # Unmasked approach: exclude region and has_pv (used for filtering/metadata only)
            conditioning_variables = data_filtering_config.get('conditioning_variables', [])
            training_vars = [v for v in conditioning_variables if v not in ['region', 'has_pv']]
            expected_num_features = 7 + len(training_vars)  # 7 temporal + training conditioning vars
            conditioning_var_names = training_vars

        self._log(f"Expected features for validation: {expected_num_features} (7 temporal + {len(conditioning_var_names)} conditioning)")

        create_constrained = self._should_create_constrained_shards()

        # Collect all shard paths
        import glob
        all_shard_paths = []

        if use_masked_tokens:
            # Masked approach: look for constrained/unconstrained files
            if create_constrained:
                all_shard_paths.extend(glob.glob(os.path.join(shard_output_dir, 'train_constrained_*.pt')))
                all_shard_paths.extend(glob.glob(os.path.join(shard_output_dir, 'val_constrained.pt')))
                all_shard_paths.extend(glob.glob(os.path.join(shard_output_dir, 'test_constrained.pt')))

            all_shard_paths.extend(glob.glob(os.path.join(shard_output_dir, 'train_unconstrained_*.pt')))
            all_shard_paths.extend(glob.glob(os.path.join(shard_output_dir, 'val_unconstrained.pt')))
            all_shard_paths.extend(glob.glob(os.path.join(shard_output_dir, 'test_unconstrained.pt')))
        else:
            # Unmasked approach: look for unmasked files
            all_shard_paths.extend(glob.glob(os.path.join(shard_output_dir, 'train_unmasked_*.pt')))
            all_shard_paths.extend(glob.glob(os.path.join(shard_output_dir, 'val_unmasked.pt')))
            all_shard_paths.extend(glob.glob(os.path.join(shard_output_dir, 'test_unmasked.pt')))
        
        if not all_shard_paths:
            self._log("✗ VALIDATION FAILED: No shards found", level='error')
            return False

        self._log(f"Validating {len(all_shard_paths)} shards...")

        # CHECK 1: Balanced Representation & Feature Shape
        self._log("\n[CHECK 1] Balanced Representation & Feature Shape")
        for shard_path in all_shard_paths:
            shard_name = os.path.basename(shard_path)
            try:
                shard = torch.load(shard_path, map_location='cpu')
                metadata = shard['metadata']

                # Determine tensor key based on shard type (check prefix to avoid matching 'train' in 'constrained')
                if shard_name.startswith('train'):
                    X_key = 'X_train'
                elif shard_name.startswith('val'):
                    X_key = 'X_val'
                elif shard_name.startswith('test'):
                    X_key = 'X_test'
                else:
                    raise ValueError(f"Unknown shard type: {shard_name}")

                X = shard[X_key]

                # Check balanced representation (only for shards with augmentation metadata)
                if metadata.get('augmentation_factor'):
                    augmentation_factor = metadata['augmentation_factor']
                    household_count = metadata['household_count']
                    actual_sequences = X.shape[0]
                    masking_type = metadata.get('masking_type', 'unknown')

                    # Calculate actual sequences per household per augmentation
                    sequences_per_household_per_aug = actual_sequences / (household_count * augmentation_factor)

                    # Determine expected sequences based on generation mode
                    if masking_type == 'unmasked':
                        # Month-based generation: exactly 12 sequences per household per year
                        # (one sequence per month, regardless of context_length)
                        expected_sequence_number = 12
                        tolerance = 0.1  # Allow small variance for partial years
                    else:
                        # Sliding window generation: sequences = total_readings / context_length
                        # For 1 year half-hourly data: 17520 readings total (365 days * 48 half-hours)
                        context_length = self.config.get('training',{}).get('model',{}).get('context_length', 48)
                        expected_sequence_number = 17520 / context_length
                        tolerance = 0.2  # 20% tolerance for different data lengths

                    # Check if within expected range
                    lower_bound = expected_sequence_number * (1 - tolerance)
                    upper_bound = expected_sequence_number * (1 + tolerance)

                    if lower_bound <= sequences_per_household_per_aug <= upper_bound:
                        self._log(f"  ✓ {shard_name}: Balanced ({actual_sequences} sequences, ~{sequences_per_household_per_aug:.1f} per household per augmentation)")
                    else:
                        error_msg = f"  ✗ {shard_name}: Unexpected sequence count! {actual_sequences} sequences = ~{sequences_per_household_per_aug:.1f} per household per augmentation (expected ~{expected_sequence_number:.1f} for {masking_type} mode)"
                        validation_errors.append(error_msg)
                        self._log(error_msg, level='error')
                        validation_passed = False

                # Check feature shape (must match expected_num_features)
                if X.shape[2] != expected_num_features:
                    error_msg = f"  ✗ {shard_name}: Wrong feature count! Expected {expected_num_features}, got {X.shape[2]}"
                    validation_errors.append(error_msg)
                    self._log(error_msg, level='error')
                    validation_passed = False
                else:
                    self._log(f"  ✓ {shard_name}: Feature shape correct ({expected_num_features} features)")

            except Exception as e:
                error_msg = f"  ✗ {shard_name}: Failed to load or validate: {str(e)}"
                validation_errors.append(error_msg)
                self._log(error_msg, level='error')
                validation_passed = False

        # CHECK 2: Var Count Distribution (constrained shards only)
        if create_constrained:
            self._log("\n[CHECK 2] Var Count Distribution (Constrained Shards)")
            constrained_shard_paths = glob.glob(os.path.join(shard_output_dir, '*_constrained*.pt'))

            if constrained_shard_paths:
                max_var_count_config = k_anonymity_config['max_var_count_constrained']

                for shard_path in constrained_shard_paths:
                    shard_name = os.path.basename(shard_path)
                    try:
                        shard = torch.load(shard_path, map_location='cpu')
                        metadata = shard['metadata']

                        max_var_count = metadata.get('max_var_count')
                        var_count_dist = metadata.get('var_count_distribution', {})

                        # Check that metadata max_var_count matches config
                        if max_var_count != max_var_count_config:
                            error_msg = f"  ✗ {shard_name}: max_var_count mismatch! Config: {max_var_count_config}, Shard: {max_var_count}"
                            validation_errors.append(error_msg)
                            self._log(error_msg, level='error')
                            validation_passed = False

                        # Check that all var_counts in distribution are <= max_var_count
                        if var_count_dist:
                            exceeding_counts = [vc for vc in var_count_dist.keys() if vc > max_var_count]
                            if exceeding_counts:
                                error_msg = f"  ✗ {shard_name}: var_counts exceed max! Found: {exceeding_counts}"
                                validation_errors.append(error_msg)
                                self._log(error_msg, level='error')
                                validation_passed = False
                            else:
                                self._log(f"  ✓ {shard_name}: Var counts within limit (max {max(var_count_dist.keys())} <= {max_var_count})")
                        else:
                            self._log(f"  ⚠ {shard_name}: No var_count_distribution in metadata", level='warning')

                    except Exception as e:
                        error_msg = f"  ✗ {shard_name}: Failed to validate var counts: {str(e)}"
                        validation_errors.append(error_msg)
                        self._log(error_msg, level='error')
                        validation_passed = False
            else:
                self._log("  ⚠ No constrained shards found to validate", level='warning')

        # CHECK 3: Unmasked Shards - Verify Structure Only
        # Note: Unmasked shards may contain mask tokens representing genuinely missing data (NaN)
        # This is correct behavior and should not be flagged as an error
        self._log("\n[CHECK 3] Unmasked Shards (Structure Verification)")
        unmasked_shard_paths = glob.glob(os.path.join(shard_output_dir, '*_unmasked.pt'))

        if unmasked_shard_paths and self.conditioning_variable_tokenisers:
            for shard_path in unmasked_shard_paths:
                shard_name = os.path.basename(shard_path)
                try:
                    shard = torch.load(shard_path, map_location='cpu')

                    # Determine tensor key (check prefix)
                    if shard_name.startswith('val'):
                        X_key = 'X_val'
                    elif shard_name.startswith('test'):
                        X_key = 'X_test'
                    else:
                        raise ValueError(f"Unknown unmasked shard type: {shard_name}")

                    X = shard[X_key]

                    # Check conditioning variables (positions 7-20) for mask tokens
                    mask_tokens_found = []
                    for i, feature_name in enumerate(conditioning_var_names):
                        feature_idx = 7 + i  # Conditioning vars start at position 7

                        if feature_name in self.conditioning_variable_tokenisers:
                            mask_token = self.conditioning_variable_tokenisers[feature_name]['mask_token']

                            # Only check if mask token is defined (not None)
                            if mask_token is not None:
                                # Check if mask token appears in this feature
                                if (X[:, :, feature_idx] == mask_token).any():
                                    mask_tokens_found.append(feature_name)

                    if mask_tokens_found:
                        # Informational only - these represent genuinely missing data (NaN), not privacy masking
                        self._log(f"  ℹ {shard_name}: Contains mask tokens for missing data: {mask_tokens_found}")

                    self._log(f"  ✓ {shard_name}: Structure verified")

                except Exception as e:
                    error_msg = f"  ✗ {shard_name}: Failed to validate masking: {str(e)}"
                    validation_errors.append(error_msg)
                    self._log(error_msg, level='error')
                    validation_passed = False
        else:
            if not unmasked_shard_paths:
                self._log("  ⚠ No unmasked shards found to validate", level='warning')

        # Final validation summary
        self._log("\n" + "=" * 60)
        if validation_passed:
            self._log("✓ ALL VALIDATIONS PASSED", level='info')
            self._log("=" * 60)
            return True
        else:
            self._log("✗ VALIDATION FAILED", level='error')
            self._log(f"Total errors: {len(validation_errors)}", level='error')
            self._log("=" * 60)
            return False

    def save_shard_metadata(self, val_puprns, test_puprns, training_shard_metadata):
        """Generate and save shard metadata and statistics."""
        import json

        shard_output_dir = self.config['sharding']['shard_output_dir']

        # Create shard metadata
        shard_metadata = {
            'sharding_config': self.config.get('sharding', {}),
            'random_seed': self.random_seed,
            'total_households': len(self.metadata_catalog),
            'validation_shards': {
                'unconstrained': {
                    'household_count': len(val_puprns),
                    'puprn_list': val_puprns,
                    'file_path': 'val_unconstrained.pt'
                },
                'unmasked': {
                    'household_count': len(val_puprns),
                    'puprn_list': val_puprns,
                    'file_path': 'val_unmasked.pt'
                }
            },
            'test_shards': {
                'unconstrained': {
                    'household_count': len(test_puprns),
                    'puprn_list': test_puprns,
                    'file_path': 'test_unconstrained.pt'
                },
                'unmasked': {
                    'household_count': len(test_puprns),
                    'puprn_list': test_puprns,
                    'file_path': 'test_unmasked.pt'
                }
            },
            'training_shards': [],
        }
        # Only add constrained if it was created
        if any(s['constrained_file'] is not None for s in training_shard_metadata):
            shard_metadata['validation_shards']['constrained'] = {
                'household_count': len(val_puprns),
                'puprn_list': val_puprns,
                'file_path': 'val_constrained.pt'
            }
            shard_metadata['test_shards']['constrained'] = {
                'household_count': len(test_puprns),
                'puprn_list': test_puprns,
                'file_path': 'test_constrained.pt'
            }

        # Add Training shards metadata
        for shard_meta in training_shard_metadata:
            entry = {
                'shard_index': shard_meta['shard_index'],
                'household_count': shard_meta['household_count'],
                'puprn_list': shard_meta['puprns'],
                'unconstrained_file': shard_meta['unconstrained_file'],
            }
            if shard_meta['constrained_file'] is not None:
                entry['constrained_file'] = shard_meta['constrained_file']
        
            shard_metadata['training_shards'].append(entry)
        
        # Save metadata
        metadata_path = os.path.join(shard_output_dir, 'shard_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(shard_metadata, f, indent=2)
        self._log(f"Saved shard metadata: {metadata_path}")
        
        # Generate statistical analysis
        if self.config.get('sharding', {}).get('generate_statistics', True):
            self._log("Generating shard statistics...")
            stats = self.generate_shard_statistics(shard_metadata)
            
            # Save statistics
            stats_path = os.path.join(shard_output_dir, 'shard_statistics.json')
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
            self._log(f"Saved shard statistics: {stats_path}")
        else:
            self._log("Skipping statistical analysis (generate_statistics=False)")
            
    def generate_shard_statistics(self, shard_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Generate statistical analysis of shard distribution and potential biases."""
        self._log("Analyzing shard distributions and potential biases...")
        
        stats = {}
        
        # 1. Household count by split statistics
        stats['split_distribution'] = self._analyze_split_distribution()
        
        # 2. Shard size distribution statistics  
        stats['shard_size_distribution'] = self._analyze_shard_size_distribution(shard_metadata)
        
        # 3. Survey response distribution statistics (if survey data available)
        stats['survey_distribution'] = self._analyze_survey_distribution()
        
        return stats
    
    def _analyze_split_distribution(self) -> Dict[str, Any]:
        """Analyze household distribution across train/val/test splits."""
        # Use metadata_catalog which has the actual split assignments
        if self.metadata_catalog is None or self.metadata_catalog.empty:
            return {'status': 'error', 'reason': 'No metadata catalog available'}
        
        split_counts = self.metadata_catalog['split'].value_counts()
        total_households = len(self.metadata_catalog)
        
        # Convert to native Python ints for JSON serialization
        train_count = int(split_counts.get('train', 0))
        val_count = int(split_counts.get('val', 0))
        test_count = int(split_counts.get('test', 0))
        
        return {
            'total_households': total_households,
            'train_households': train_count,
            'val_households': val_count, 
            'test_households': test_count,
            'train_percentage': round(train_count / total_households * 100, 2),
            'val_percentage': round(val_count / total_households * 100, 2),
            'test_percentage': round(test_count / total_households * 100, 2)
        }
    
    def _analyze_shard_size_distribution(self, shard_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze distribution of household counts per training shard."""
        if self.metadata_catalog is None or self.metadata_catalog.empty:
            return {'status': 'error', 'reason': 'No metadata catalog available'}

        # Get actual shard sizes from the created training shards
        training_shards = shard_metadata.get('training_shards', [])

        if not training_shards:
            return {'num_shards': 0, 'status': 'no_training_shards'}

        # All training shards have same household count (one per shard, not separate constrained/unconstrained)
        shard_sizes = [shard['household_count'] for shard in training_shards]

        # Also validate against metadata_catalog
        train_households_total = int((self.metadata_catalog['split'] == 'train').sum())
        sharded_total = sum(shard_sizes)

        # Check how many shards have constrained files (optional with new feature)
        num_constrained = sum(1 for shard in training_shards if shard.get('constrained_file') is not None)
        num_unconstrained = len(training_shards)  # All shards have unconstrained files

        result = {
            'total_training_households_expected': train_households_total,
            'num_shards': len(training_shards),
            'total_training_households_sharded': sharded_total,
            'households_match': train_households_total == sharded_total,
            'actual_shard_sizes': shard_sizes,
            'constrained_shards_created': num_constrained > 0,
            'num_constrained_shards': num_constrained,
            'num_unconstrained_shards': num_unconstrained
        }

        # Add statistics if shards exist
        if shard_sizes:
            result.update({
                'min_households': min(shard_sizes),
                'max_households': max(shard_sizes),
                'mean_households': round(sum(shard_sizes) / len(shard_sizes), 2),
                'balanced': len(set(shard_sizes)) == 1,
                'size_variance': max(shard_sizes) - min(shard_sizes)
            })

        return result
    
    def _analyze_survey_distribution(self) -> Dict[str, Any]:
        """Analyze survey response distribution across splits to detect bias."""
        try:
            # Use existing metadata_catalog which already contains splits and survey data
            if self.metadata_catalog is None or self.metadata_catalog.empty:
                return {'status': 'skipped', 'reason': 'No metadata catalog available'}
            
            # metadata_catalog already has 'split' column with train/val/test assignments
            survey_df = self.metadata_catalog.copy()
            
            # Analyze B5 (number of rooms) distribution
            b5_analysis = self._analyze_feature_distribution(survey_df, 'B5', 'Number of rooms')
            
            # Analyze C1_new (number of occupants) distribution  
            c1_analysis = self._analyze_feature_distribution(survey_df, 'C1_new', 'Number of occupants')
            
            return {
                'status': 'completed',
                'total_households_with_survey': len(survey_df),
                'B5_rooms_distribution': b5_analysis,
                'C1_new_occupants_distribution': c1_analysis
            }
            
        except Exception as e:
            return {'status': 'error', 'reason': f'Failed to analyze survey data: {str(e)}'}
    
    def _analyze_feature_distribution(self, survey_df: pd.DataFrame, feature: str, feature_name: str) -> Dict[str, Any]:
        """Analyze distribution of a survey feature across splits."""
        # Filter out invalid values (negative numbers typically indicate missing/invalid data)
        valid_df = survey_df[survey_df[feature] > 0]
        
        if len(valid_df) == 0:
            return {'status': 'no_valid_data', 'feature': feature_name}
        
        # Get distribution by split
        distribution = {}
        splits = ['train', 'val', 'test']
        
        for split in splits:
            split_df = valid_df[valid_df['split'] == split]
            if len(split_df) > 0:
                value_counts = split_df[feature].value_counts().sort_index()
                # Convert to native Python types for JSON serialization
                distribution[split] = {
                    'count': int(len(split_df)),
                    'values': {int(k): int(v) for k, v in value_counts.items()},
                    'mean': round(float(split_df[feature].mean()), 2),
                    'percentages': {int(k): round(float(v)/len(split_df)*100, 1) for k, v in value_counts.items()}
                }
            else:
                distribution[split] = {'count': 0, 'values': {}, 'mean': None, 'percentages': {}}
        
        # Calculate potential bias indicators
        total_valid = int(len(valid_df))
        expected_train_pct = distribution['train']['count'] / total_valid * 100
        expected_val_pct = distribution['val']['count'] / total_valid * 100  
        expected_test_pct = distribution['test']['count'] / total_valid * 100
        
        return {
            'feature': feature_name,
            'total_valid_responses': total_valid,
            'distribution_by_split': distribution,
            'expected_split_percentages': {
                'train': round(expected_train_pct, 1),
                'val': round(expected_val_pct, 1),
                'test': round(expected_test_pct, 1)
            }
        }
