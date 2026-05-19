"""
Synthetic data generation utilities.
"""
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import json
import random
import time
import pickle
import calendar
from typing import Dict, List, Any, Tuple
from models.gpt_model import GPTSmartMeterModel
from utils.universal_tokeniser_naive import detokenise_data, TokenVocabulary
from utils.device_utils import get_device_from_config, move_to_device, log_device_info
from preprocessing.sharding import DataSharder
from training.training_utils import SimpleResourceMonitor


class SyntheticDataGenerator:
    """
    Handles synthetic data generation using dual transformer models.

    This generator uses separate electricity and gas models to generate synthetic
    smart meter data. Both models receive the same input context (weather, calendar,
    conditioning variables) but generate their respective output variables independently.
    This dual-model architecture eliminates multi-task learning interference and allows
    each model to specialize in its target variable.

    The generation process:
    1. Load conditioning variables and temporal template from training data
    2. Create input tensors with SOS tokens and conditioning context
    3. Generate electricity and gas values in parallel at each timestep
    4. Decode tokens to physical values and save in wide format

    Attributes:
        electricity_model: Trained GPT model for electricity generation
        gas_model: Trained GPT model for gas generation
        elec_boundaries: Quantile boundaries for electricity tokenization
        gas_boundaries: Quantile boundaries for gas tokenization (zero-aware)
        device: Computation device (cpu, cuda, or mps)
    """

    def __init__(
        self,
        electricity_model: GPTSmartMeterModel,
        gas_model: GPTSmartMeterModel,
        elec_boundaries: np.ndarray,
        gas_boundaries: np.ndarray,
        temp_boundaries: np.ndarray,
        solar_boundaries: np.ndarray,
        config: Dict[str, Any],
        logger=None,
        elec_bin_means: list = None,
        gas_bin_means: list = None,
        value_transform: str = "none"
    ):
        """
        Initialize SyntheticDataGenerator with dual electricity and gas models.

        Args:
            electricity_model: Trained GPT model for electricity generation. Must be
                a GPTSmartMeterModel configured for single-output electricity prediction.
            gas_model: Trained GPT model for gas generation. Must be a GPTSmartMeterModel
                configured for single-output gas prediction with zero-aware tokenization.
            elec_boundaries: Numpy array of quantile boundaries for electricity tokenization.
                Used for decoding generated tokens back to Wh values.
            gas_boundaries: Numpy array of quantile boundaries for gas tokenization.
                Uses zero-aware scheme where token 0 represents exact zero consumption.
            temp_boundaries: Numpy array of quantile boundaries for temperature tokenization.
            solar_boundaries: Numpy array of quantile boundaries for solar tokenization.
            config: Configuration dictionary containing inference parameters:
                - batch_size: Number of households per conditioning sample
                - num_samples: Number of conditioning combinations to sample
                - random_seed: Seed for reproducibility
                - block_missing_tokens: Whether to prevent missing/error token generation
                - generated_data_path: Output directory for generated data
            logger: Optional structured logger for inference tracking and metrics.
            elec_bin_means: Optional list of empirical bin means for electricity.
                If provided, uses bin means instead of bin midpoints for decoding,
                improving fidelity to training distribution.
            gas_bin_means: Optional list of empirical bin means for gas.
                If provided, uses bin means instead of bin midpoints for decoding.
            value_transform: "none" or "log1p". If "log1p", applies expm1() during
                decoding to convert from log space. Must match the transform used
                during tokeniser creation. Stored in tokeniser pickle (version 3+).
        """
        # Set up device
        self.device = get_device_from_config(config, verbose=True)
        self.logger = logger
        log_device_info(self.device, logger)

        # Move model to device
        self.electricity_model = move_to_device(electricity_model, self.device)
        self.gas_model = move_to_device(gas_model, self.device)
        self.config = config

        # Capture model memory (fixed cost) before initializing monitor
        self.model_gpu_memory_mb = 0.0
        if torch.cuda.is_available():
            self.model_gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # MPS doesn't have memory_allocated, so we can't measure this
            self.model_gpu_memory_mb = 0.0

        # Initialize resource monitoring (resets peak tracking - will capture generation only)
        self.resource_monitor = SimpleResourceMonitor()

        # Store value_transform for passing to detokenise_data
        self.value_transform = value_transform

        # Create TokenVocabulary objects - single source of truth for token schemes
        self.vocab = {
            'electricity': TokenVocabulary(
                boundaries=list(elec_boundaries),
                bin_means=list(elec_bin_means) if elec_bin_means else None,
                zero_aware=False,
                value_transform=value_transform
            ),
            'gas': TokenVocabulary(
                boundaries=list(gas_boundaries),
                bin_means=list(gas_bin_means) if gas_bin_means else None,
                zero_aware=True,
                value_transform=value_transform
            ),
            'temperature': TokenVocabulary(
                boundaries=list(temp_boundaries),
                bin_means=None,
                zero_aware=False
                # No value_transform for temperature
            ),
            'solar': TokenVocabulary(
                boundaries=list(solar_boundaries),
                bin_means=None,
                zero_aware=False
                # No value_transform for solar
            ),
        }

        # Backward compatibility aliases (used by some methods)
        self.elec_boundaries = elec_boundaries
        self.gas_boundaries = gas_boundaries
        self.temp_boundaries = temp_boundaries
        self.solar_boundaries = solar_boundaries
        self.elec_bin_means = elec_bin_means
        self.gas_bin_means = gas_bin_means

        # Convenience aliases for special tokens (used for token blocking)
        self.elec_missing_token = self.vocab['electricity'].missing_token
        self.elec_sos_token = self.vocab['electricity'].sos_token
        self.gas_missing_token = self.vocab['gas'].missing_token
        self.gas_sos_token = self.vocab['gas'].sos_token

        # Sampling temperature (Decision 039: T<1.0 suppresses upper-tail overproduction)
        self.temperature = config.get('temperature', 1.0)
        self._log(f"Sampling temperature: {self.temperature}")

        # Check if missing token blocking is enabled
        self.block_missing_tokens = config.get('block_missing_tokens', False)
        if self.block_missing_tokens:
            self._log(f"Missing token blocking ENABLED - will prevent generation of missing/SOS tokens")
            self._log(f"  Elec missing/SOS: {self.elec_missing_token}, {self.elec_sos_token}")
            self._log(f"  Gas missing/SOS: {self.gas_missing_token}, {self.gas_sos_token}")
        else:
            self._log(f"Missing token blocking DISABLED - model may generate missing/SOS tokens during synthesis")

        # Log bin means availability
        if self.vocab['electricity'].bin_means or self.vocab['gas'].bin_means:
            self._log("Using empirical bin means for improved decoding fidelity")

    def _log(self, message: str, level: str = 'info') -> None:
        """Log message using logger if available, otherwise print with [INFERENCE] prefix."""
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
            print(f"[INFERENCE] {message}")

    def generate_sequences_unmasked(
        self,
        path_to_shard_data: str,
        path_to_tokenised_original_data: str,
        conditioning_variable_tokenisers: Dict[str, Any],
        conditioning_var_order: List[str],  # Order must match training config
    ) -> Dict[str, Any]:
        """
        Generate synthetic households using batched unmasked generation.

        Flow:
        1. Load metadata_tokenised_filtered.pkl (train split only, k≥3 filtered)
        2. Sample num_samples unique conditioning variable combinations
        3. For each sample:
           - Sample random train household for temporal template (weather, calendar)
           - Create input tensor [seq_len, num_features] with SOS tokens + temporal + conditioning
           - Generate batch_size households using model.generate_batched()
           - Decode tokens to human-readable values
           - Append region (from weather) and group_size (k-anonymity)
           - Save as wide format pickle

        Args:
            path_to_shard_data: Path to shard data directory (contains metadata_tokenised_filtered.pkl)
            path_to_tokenised_original_data: Path to tokenized household data files (for temporal template)
            conditioning_variable_tokenisers: Tokenizer config for conditioning variables
            conditioning_var_order: List of conditioning variable names in the order used during training.

        Returns:
            Dict with generation statistics
        """
        self._log("\n" + "="*80)
        self._log("UNMASKED BATCHED GENERATION")
        self._log("="*80)

        # Store conditioning tokenisers
        self.conditioning_variable_tokenisers = conditioning_variable_tokenisers

        # Get config parameters
        batch_size = self.config.get('batch_size', 50)
        num_samples = self.config.get('num_samples', 10)
        sampling_strategy = self.config.get('sampling_strategy', 'without_replacement')
        random_seed = self.config.get('random_seed', 42)

        # Set random seeds
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        total_households = num_samples * batch_size
        self._log(f"[CONFIG] Batch size: {batch_size} households per batch")
        self._log(f"[CONFIG] Num samples: {num_samples} conditioning combinations")
        self._log(f"[CONFIG] Total households: {total_households}")
        self._log(f"[CONFIG] Sampling strategy: {sampling_strategy}")
        self._log(f"[CONFIG] Random seed: {random_seed}")

        # === STEP 1: Load both metadata files ===
        # Load human-readable metadata catalog (for sampling, includes k_group_size)
        catalog_path = os.path.join(path_to_shard_data, 'metadata_catalog_filtered.pkl')
        self._log(f"\n[LOADING] Metadata catalog from: {catalog_path}")

        if not os.path.exists(catalog_path):
            raise FileNotFoundError(f"metadata_catalog_filtered.pkl not found: {catalog_path}")

        metadata_catalog_filtered = pd.read_pickle(catalog_path)
        self._log(f"[LOADING] Loaded {len(metadata_catalog_filtered)} households from catalog")

        # Overwrite 'split' column using shard_metadata.json as the authoritative source.
        # metadata_catalog_filtered.pkl is saved during sharding before assign_splits() runs,
        # so its 'split' column cannot be trusted.
        shard_metadata_path = os.path.join(path_to_shard_data, 'shard_metadata.json')
        if not os.path.exists(shard_metadata_path):
            raise FileNotFoundError(f"shard_metadata.json not found: {shard_metadata_path}")

        with open(shard_metadata_path) as f:
            shard_metadata = json.load(f)

        val_puprns = set(shard_metadata['validation_shards']['unmasked']['puprn_list'])
        test_puprns = set(shard_metadata['test_shards']['unmasked']['puprn_list'])
        train_puprns = set(p for shard in shard_metadata['training_shards'] for p in shard['puprn_list'])

        def _assign_split(puprn):
            if puprn in val_puprns:
                return 'val'
            if puprn in test_puprns:
                return 'test'
            if puprn in train_puprns:
                return 'train'
            return 'unassigned'

        metadata_catalog_filtered['split'] = metadata_catalog_filtered['puprn'].map(_assign_split)
        split_counts = metadata_catalog_filtered['split'].value_counts().to_dict()
        self._log(f"[LOADING] Split assignments from shard_metadata.json: {split_counts}")

        # Load tokenised metadata (for creating input tensors)
        tokenised_path = os.path.join(path_to_shard_data, 'metadata_tokenised_filtered.pkl')
        self._log(f"[LOADING] Tokenised metadata from: {tokenised_path}")

        if not os.path.exists(tokenised_path):
            raise FileNotFoundError(f"metadata_tokenised_filtered.pkl not found: {tokenised_path}")

        metadata_tokenised_filtered = pd.read_pickle(tokenised_path)
        self._log(f"[LOADING] Loaded {len(metadata_tokenised_filtered)} households from tokenised")

        # === STEP 2: Build conditioning pool (train only, or train+test — Decision 048) ===
        if 'split' not in metadata_catalog_filtered.columns:
            raise ValueError("metadata_catalog_filtered missing 'split' column")

        conditioning_pool = self.config.get('conditioning_pool', 'train_test')
        pool_splits = ['train'] if conditioning_pool == 'train' else ['train', 'test']
        sample_catalog = metadata_catalog_filtered[
            metadata_catalog_filtered['split'].isin(pool_splits)
        ].copy()
        self._log(f"\n[FILTER] Conditioning pool ({conditioning_pool}): {len(sample_catalog)} households")

        if len(sample_catalog) == 0:
            raise ValueError(f"No households found in conditioning pool ({conditioning_pool})")

        # === STEP 3: Determine conditioning variable order ===
        # CRITICAL: Use training config order to match model feature ordering!
        # IMPORTANT: Exclude region and has_pv - they're used for metadata/filtering only, not as training features
        all_conditioning_vars = conditioning_var_order  # Use training config order
        conditioning_vars = [v for v in all_conditioning_vars if v not in ['region', 'has_pv']]

        self._log(f"\n[CONDITIONING] All variables (from training config): {all_conditioning_vars}")
        self._log(f"[CONDITIONING] Training variables (excluding region, has_pv): {conditioning_vars}")

        # Create mapping from conditioning variable names to original column names
        # (catalog uses original column names like 'B5', 'building_type_descriptive', etc.)
        var_to_original_col = {}
        for var in all_conditioning_vars:
            orig_col = conditioning_variable_tokenisers[var]['original_variable_name']
            var_to_original_col[var] = orig_col

        self._log(f"\n[MAPPING] Conditioning variable → original column name:")
        for var, orig_col in var_to_original_col.items():
            self._log(f"  {var} → {orig_col}")

        # Verify all conditioning vars present in catalog (using original column names)
        missing_vars = []
        for var in conditioning_vars:
            orig_col = var_to_original_col[var]
            if orig_col not in sample_catalog.columns:
                missing_vars.append(f"{var} ({orig_col})")

        if missing_vars:
            raise ValueError(f"Conditioning variables missing from catalog: {missing_vars}")

        # Verify k_group_size column exists
        if 'k_group_size' not in sample_catalog.columns:
            raise ValueError("metadata_catalog_filtered missing 'k_group_size' column")

        self._log(f"\n[SAMPLING] Sampling {num_samples} households from conditioning pool ({conditioning_pool})...")
        self._log(f"[SAMPLING] Available households: {len(sample_catalog)}")

        # Sample num_samples rows directly from sample catalog
        replace = (sampling_strategy == 'with_replacement')
        if num_samples > len(sample_catalog) and not replace:
            self._log(f"Requested {num_samples} samples but only {len(sample_catalog)} households available", level='warning')
            self._log("Sampling with replacement instead", level='warning')
            replace = True

        sampled_households = sample_catalog.sample(n=num_samples, replace=replace, random_state=random_seed)
        sampled_households = sampled_households.reset_index(drop=True)
        self._log(f"[SAMPLING] Sampled {len(sampled_households)} households")

        # === STEP 4: Initialize generation tracking ===
        generation_start_time = time.time()
        num_generated = 0
        num_failed = 0
        saved_files = []

        # Initialize resource monitoring
        initial_resources = self.resource_monitor.get_current_usage()
        if self.logger:
            self.logger.log_custom_metric('inference_initial_resources', initial_resources)
            self.logger.log_custom_metric('model_gpu_memory_mb', self.model_gpu_memory_mb)

        # === STEP 5: Generate batches ===
        self._log(f"\n[GENERATION] Starting batch generation...")
        if self.model_gpu_memory_mb > 0:
            self._log(f"[RESOURCES] Models loaded: {self.model_gpu_memory_mb:.1f} MB GPU memory (fixed cost)")
        self._log("="*80)

        for sample_idx, household_row in sampled_households.iterrows():
            try:
                self._log(f"\n[BATCH {sample_idx+1}/{num_samples}] Generating {batch_size} households...")

                # Get PUPRN for this household
                household_puprn = household_row['puprn']
                self._log(f"[BATCH {sample_idx+1}/{num_samples}] Sample household: {household_puprn}")

                # Log conditioning variables (from catalog - human-readable, using original column names)
                conditioning_str = ", ".join([
                    f"{var}={household_row[var_to_original_col[var]]}"
                    for var in conditioning_vars
                ])
                self._log(f"[BATCH {sample_idx+1}/{num_samples}] Conditioning: {conditioning_str}")

                # Log k-anonymity group size
                k_group_size = household_row['k_group_size']
                self._log(f"[BATCH {sample_idx+1}/{num_samples}] k-anonymity group_size: {k_group_size}")

                # === STEP 5a: Sample random train household for temporal template ===
                # Sample different household for temporal template (independent sampling)
                # Offset seed by 10000 to avoid collision with conditioning household sampling (which uses random_seed)
                template_household = sample_catalog.sample(n=1, random_state=random_seed + 10000 + sample_idx)
                template_puprn = template_household['puprn'].values[0]
                self._log(f"[BATCH {sample_idx+1}/{num_samples}] Template household: {template_puprn}")

                # Extract region from template household metadata (catalog, using original column name)
                region_col = var_to_original_col.get('region', 'Region')
                region = template_household[region_col].iloc[0] if region_col in template_household.columns else 'unknown'
                self._log(f"[BATCH {sample_idx+1}/{num_samples}] Region: {region}")

                # Load temporal data from template household
                template_file = os.path.join(path_to_tokenised_original_data, f"{template_puprn}_tokenised.pkl")
                if not os.path.exists(template_file):
                    self._log(f"Template household file not found: {template_file}", level='error')
                    num_failed += batch_size
                    continue

                template_data = pd.read_pickle(template_file)

                # === STEP 5b: Get tokenised conditioning values for this household ===
                # Look up tokenised values from metadata_tokenised_filtered using PUPRN
                tokenised_row = metadata_tokenised_filtered[
                    metadata_tokenised_filtered['puprn'] == household_puprn
                ]
                if len(tokenised_row) == 0:
                    self._log(f"PUPRN {household_puprn} not found in tokenised metadata", level='error')
                    num_failed += batch_size
                    continue

                tokenised_row = tokenised_row.iloc[0]

                all_months = self.generate_year_chunked(weather_data=template_data,
                                           conditioning_vars=conditioning_vars,
                                           conditioning_row=tokenised_row,
                                           batch_size=batch_size,
                                           year=2023)

                # === STEP 5e: Save each month as separate wide-format file ===
                output_dir = self.config.get('generated_data_path', 'generated_data')
                os.makedirs(output_dir, exist_ok=True)

                for month_result in all_months:
                    month_num = month_result['month']
                    generated_batch = month_result['data']

                    # Extract weather for this month
                    month_weather = self._extract_month_weather(template_data, 2023, month_num)

                    # Transform to wide format (SOS row removed inside method)
                    df_wide = self._month_batch_to_wide_dataframe(
                        generated_batch=generated_batch,
                        month_weather=month_weather,
                        conditioning_row=tokenised_row,
                        conditioning_vars=conditioning_vars,
                        conditioning_tokenisers=conditioning_variable_tokenisers,
                        region=region,
                        k_group_size=k_group_size,
                        month=month_num,
                    )

                    # Save month file
                    output_file = os.path.join(output_dir, f"batch_{sample_idx+1:04d}_month_{month_num:02d}.pkl")
                    df_wide.to_pickle(output_file)
                    saved_files.append(output_file)

                self._log(f"[BATCH {sample_idx+1}/{num_samples}] Saved 12 month files")

                # === STEP 5f: Append to conditioning summary ===
                self._append_conditioning_summary(
                    output_dir=output_dir,
                    batch_idx=sample_idx + 1,
                    conditioning_row=tokenised_row,
                    conditioning_vars=conditioning_vars,
                    conditioning_tokenisers=conditioning_variable_tokenisers,
                    region=region,
                    k_group_size=k_group_size,
                )

                num_generated += batch_size

                # Resource monitoring
                if self.logger and (sample_idx + 1) % max(1, num_samples // 10) == 0:
                    current_resources = self.resource_monitor.get_current_usage()
                    self.logger.log_custom_metric(f'batch_{sample_idx+1}_resources', current_resources)
                    self._log(f"[RESOURCES] Batch {sample_idx+1}: Memory {current_resources['memory_mb']:.1f}MB, "
                          f"GPU Memory {current_resources['gpu_memory_mb']:.1f}MB")

                # Cleanup
                del all_months
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()

            except Exception as e:
                self._log(f"Batch {sample_idx+1} failed: {str(e)}", level='error')
                import traceback
                traceback.print_exc()
                num_failed += batch_size
                continue

        # === STEP 6: Finalize ===
        generation_time = time.time() - generation_start_time
        final_resources = self.resource_monitor.get_summary()

        if self.logger:
            self.logger.log_custom_metric('inference_resource_summary', final_resources)
            self.logger.log_custom_metric('inference_generation_time', generation_time)

        self._log("\n" + "="*80)
        self._log("GENERATION COMPLETE")
        self._log("="*80)
        self._log(f"[RESULTS] Total households generated: {num_generated}/{total_households}")
        self._log(f"[RESULTS] Failed: {num_failed}")
        self._log(f"[RESULTS] Batches saved: {len(saved_files)}")
        self._log(f"[RESULTS] Generation time: {generation_time:.1f} seconds")
        self._log(f"[RESULTS] Time per household: {generation_time/max(1, num_generated):.2f} seconds")
        self._log(f"[RESOURCES] Peak CPU memory: {final_resources['peak_memory_mb']:.1f} MB")

        # GPU memory breakdown: model (fixed) + generation (variable) = total
        generation_gpu_peak = final_resources['peak_gpu_memory_mb']
        total_gpu_peak = self.model_gpu_memory_mb + generation_gpu_peak

        if self.model_gpu_memory_mb > 0 or generation_gpu_peak > 0:
            self._log(f"[RESOURCES] GPU memory - Models (fixed): {self.model_gpu_memory_mb:.1f} MB")
            self._log(f"[RESOURCES] GPU memory - Generation (variable): {generation_gpu_peak:.1f} MB")
            self._log(f"[RESOURCES] GPU memory - Total peak: {total_gpu_peak:.1f} MB")

        # Log GPU utilization availability status (like training does)
        if final_resources.get('gpu_utilization_available', False):
            self._log(f"[RESOURCES] Peak GPU utilization: {final_resources['peak_gpu_utilization_percent']:.1f}%")
        elif torch.cuda.is_available() or (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            self._log("[RESOURCES] GPU utilization monitoring unavailable (requires nvidia-smi for CUDA)")

        # Add GPU memory breakdown to logged metrics
        if self.logger:
            self.logger.log_custom_metric('inference_gpu_memory_breakdown', {
                'model_gpu_memory_mb': self.model_gpu_memory_mb,
                'generation_gpu_peak_mb': generation_gpu_peak,
                'total_gpu_peak_mb': total_gpu_peak,
            })

        return {
            'num_requested': total_households,
            'num_generated': num_generated,
            'num_failed': num_failed,
            'num_batches': len(saved_files),
            'generation_time_seconds': generation_time,
            'saved_files': saved_files,
            'model_gpu_memory_mb': self.model_gpu_memory_mb,
            'generation_gpu_peak_mb': generation_gpu_peak,
            'total_gpu_peak_mb': total_gpu_peak,
        }
    
    def generate_batched(self,
                        electricity_model: GPTSmartMeterModel,
                        gas_model: GPTSmartMeterModel,
                        input_tensor: torch.Tensor, temperature: float = 1.0, batch_size: int = 1,
                        block_missing_tokens: bool = False, elec_missing_token: int = None,
                        elec_sos_token: int = None, gas_missing_token: int = None,
                        gas_sos_token: int = None) -> torch.Tensor:
        """
        Generate synthetic electricity and gas values using dual models in parallel.

        This method generates batch_size synthetic households using the same conditioning
        variables. At each timestep, both the electricity and gas models independently
        predict their respective output tokens, which are then combined into the output
        sequence. This parallel generation preserves cross-fuel context while eliminating
        multi-task learning interference.

        Args:
            electricity_model: Trained GPTSmartMeterModel for electricity prediction.
                Receives the full input context and outputs electricity token logits.
            gas_model: Trained GPTSmartMeterModel for gas prediction.
                Receives the full input context and outputs gas token logits.
            input_tensor: Input tensor of shape [max_length, num_input_features] containing:
                - Features 0-1: Electricity and gas SOS tokens (to be generated)
                - Features 2-6: Temporal features (temp, solar, hh, dow, month)
                - Features 7+: Conditioning variables (constant across sequence)
            temperature: Sampling temperature for token generation. Higher values (>1.0)
                increase randomness, lower values (<1.0) make outputs more deterministic.
                Default 1.0 samples from the learned distribution.
            batch_size: Number of synthetic households to generate in parallel.
                All households share the same conditioning variables but have
                independent stochastic generation.
            block_missing_tokens: If True, sets logits for missing/SOS tokens to -inf
                before sampling, preventing the model from generating these tokens.
            elec_missing_token: Token ID for electricity missing values. Required if
                block_missing_tokens=True.
            elec_sos_token: Token ID for electricity SOS values. Required if
                block_missing_tokens=True.
            gas_missing_token: Token ID for gas missing values. Required if
                block_missing_tokens=True.
            gas_sos_token: Token ID for gas SOS values. Required if
                block_missing_tokens=True.

        Returns:
            Generated sequence tensor of shape [batch_size, max_length, num_input_features].
            Features 0 and 1 contain the generated electricity and gas tokens respectively.
            Other features are copied from the input tensor.
        """
        with torch.no_grad():
            self.electricity_model.eval()
            self.gas_model.eval()
            device = next(self.electricity_model.parameters()).device
            
            # Ensure input tensor is on the same device as model
            if input_tensor.device != device:
                input_tensor = input_tensor.to(device)
            
            # initialise batch with first timestep from input_tensor
            first_timestep = input_tensor[0].unsqueeze(0) # -> shape [1, num_input_features]
            # create batched_first_timestep shape [batch_size, 1, num_input_features]
            input_sequence = first_timestep.unsqueeze(0).repeat(batch_size, 1, 1)  # shape [batch_size, 1, num_input_features]
            generated_sequence = input_sequence.clone()
            output_sequence = input_sequence.clone()
            max_generate_steps = input_tensor.size(0) # should be equivalent to max_length
            
            # Generate from step 1 to max_steps-1 (excludes only the first SOS timestep)
            for step in range(1, max_generate_steps):
                # Electricity model first
                # Forward pass
                electricity_output = electricity_model.forward(generated_sequence) # shape of electricity_output: [batch_size, seq_len, vocab_size_electricity]
                
                # Get predictions for the last timestep
                elec_logits = electricity_output[:, -1, :] / temperature # shape of elec_logits: [batch_size, vocab_size_electricity]

                # Block missing/SOS tokens if configured
                if block_missing_tokens:
                    if elec_missing_token is not None:
                        elec_logits[:, elec_missing_token] = -float('inf')
                    if elec_sos_token is not None:
                        elec_logits[:, elec_sos_token] = -float('inf')

                # Sample from probability distributions
                elec_probs = F.softmax(elec_logits, dim=-1) # shape of elec_probs: [batch_size, vocab_size_electricity]

                electricity_prediction = torch.multinomial(elec_probs, num_samples=1) # shape [batch_size, 1]

                # Gas model second
                gas_output = gas_model.forward(generated_sequence)

                gas_logits = gas_output[:, -1, :] / temperature

                if block_missing_tokens:
                    if gas_missing_token is not None:
                        gas_logits[:, gas_missing_token] = -float('inf')
                    if gas_sos_token is not None:
                        gas_logits[:, gas_sos_token] = -float('inf')

                gas_probs = F.softmax(gas_logits, dim=-1)
                
                gas_prediction = torch.multinomial(gas_probs, num_samples=1)
                
                # Create new feature vector with predicted outputs
                new_feature_vector = input_tensor[step].clone().unsqueeze(0).repeat(batch_size, 1, 1) # shape [batch_size, 1, num_input_features]
                # update electricity variable (index 0) and gas variable (index 1) with predictions
                new_feature_vector[:,:,0] = electricity_prediction
                new_feature_vector[:,:,1] = gas_prediction
                # Other features (temp, solar, contextual) remain from original input
                
                # Update sequences
                generated_sequence = torch.cat([generated_sequence, new_feature_vector], dim=1) # shape of generated sequence : [batch_size, seq_len+1, num_input_features] 
                output_sequence = torch.cat([output_sequence, new_feature_vector], dim=1)
                
                # Maintain rolling context window
                if generated_sequence.size(1) > self.electricity_model.context_length:
                    generated_sequence = generated_sequence[:, 1:, :]
                    
        return output_sequence
        
    def _create_input_tensor(
        self,
        household_df: pd.DataFrame,
        conditioning_row: pd.Series,
        vocab_size_elec: int,
        vocab_size_gas: int,
        vocab_size_temp: int,
        vocab_size_solar: int
    ) -> torch.Tensor:
        """
        Create input tensor [household_len + 1, 21] from household data + k10 conditioning.

        IMPORTANT: First timestep uses SOS tokens for ALL temporal features (0-6)
        to match training distribution where households start with SOS tokens.
        Remaining timesteps use real weather/calendar data from the household.

        The tensor is intentionally one timestep longer than household_df to accommodate:
        - Position 0: SOS tokens (seed for generation)
        - Positions 1 to household_len: All household weather/calendar data (no waste!)

        This allows model.generate() to produce household_len output timesteps using
        all available weather/calendar context.

        Args:
            household_df: Real household DataFrame with columns:
                         temp_token, solar_token, hh, dow, month, elec_token, gas_token
                         Typically 17520 rows (1 year of half-hourly data)
            conditioning_row: Single row from masked_conditioning_table with 14 conditioning vars
            vocab_size_elec: Vocabulary size for electricity (for SOS token)
            vocab_size_gas: Vocabulary size for gas (for SOS token)
            vocab_size_temp: Vocabulary size for temperature (for SOS token)
            vocab_size_solar: Vocabulary size for solar (for SOS token)

        Returns:
            Input tensor of shape [household_len + 1, 21] ready for model.generate()

            Feature ordering:
                Timestep 0 (Seed):
                    0-1: elec_token, gas_token (SOS)
                    2-6: temp_token, solar_token, hh, dow, month (SOS)
                    7-20: conditioning variables (real values)

                Timesteps 1 to household_len (All household data):
                    0-1: elec_token, gas_token (SOS - to be generated)
                    2-6: temp_token, solar_token, hh, dow, month (from household_df)
                    7-20: conditioning variables (real values)

        Example:
            If household_df has 17520 rows:
            - input_tensor shape: [17521, 21]
            - model.generate() output: [17520, 21] (full year of generated data)
        """
        household_len = len(household_df)
        # Create tensor with +1 length to accommodate SOS at position 0 + all household data
        input_tensor = torch.zeros((household_len + 1, 21), dtype=torch.long)

        # SOS tokens for electricity and gas (last token in vocab) and others
        elec_sos = vocab_size_elec - 1
        gas_sos = vocab_size_gas - 1
        temp_sos = vocab_size_temp - 1
        solar_sos = vocab_size_solar - 1
        hh_sos = 0
        dow_sos = 7
        month_sos = 0

        # Fill features 0-1: SOS tokens for outputs (all timesteps)
        input_tensor[:, 0] = elec_sos
        input_tensor[:, 1] = gas_sos

        # Fill features 2-6 for FIRST timestep: SOS tokens
        input_tensor[0, 2] = temp_sos
        input_tensor[0, 3] = solar_sos
        input_tensor[0, 4] = hh_sos
        input_tensor[0, 5] = dow_sos
        input_tensor[0, 6] = month_sos

        # Fill features 2-6 for REMAINING timesteps: Real temporal/weather data from household
        # Now we can use ALL household data (no waste!)
        input_tensor[1:, 2] = torch.tensor(household_df['temp_token'].values, dtype=torch.long)
        input_tensor[1:, 3] = torch.tensor(household_df['solar_token'].values, dtype=torch.long)
        input_tensor[1:, 4] = torch.tensor(household_df['hh'].values, dtype=torch.long)
        input_tensor[1:, 5] = torch.tensor(household_df['dow'].values, dtype=torch.long)
        input_tensor[1:, 6] = torch.tensor(household_df['month'].values, dtype=torch.long)

        # Fill features 7-20: K10 masked conditioning variables (constant across sequence)
        # Order matches gpt_model.py documentation
        input_tensor[:, 7] = int(conditioning_row['region'])
        input_tensor[:, 8] = int(conditioning_row['imd'])
        input_tensor[:, 9] = int(conditioning_row['heating_type'])
        input_tensor[:, 10] = int(conditioning_row['num_rooms'])
        input_tensor[:, 11] = int(conditioning_row['total_floor_area'])
        input_tensor[:, 12] = int(conditioning_row['epc_rating'])
        input_tensor[:, 13] = int(conditioning_row['household_income_band'])
        input_tensor[:, 14] = int(conditioning_row['managing_financially'])
        input_tensor[:, 15] = int(conditioning_row['num_occs'])
        input_tensor[:, 16] = int(conditioning_row['building_type'])
        input_tensor[:, 17] = int(conditioning_row['age_built'])
        input_tensor[:, 18] = int(conditioning_row['tenure'])
        input_tensor[:, 19] = int(conditioning_row['has_pv'])
        input_tensor[:, 20] = int(conditioning_row['has_ev'])

        return input_tensor
    
    def decode_conditioning_token(self, token_value, var_config):
        """
        Decode a conditioning variable token to its original value.

        Handles 4 different variable types:
        - categorical/ordinal/cardinal: Direct lookup via itov dictionary
        - continuous: Decode using quantile boundaries (like energy values)
        - Unknown types: Raises ValueError

        Args:
            token_value: Integer token to decode
            var_config: Dictionary containing tokenizer configuration with keys:
                - 'type': Variable type ('categorical', 'ordinal', 'cardinal', 'continuous')
                - 'itov': Token-to-value mapping dict (for categorical/ordinal/cardinal)
                - 'boundaries': Quantile boundaries (for continuous variables)
                - 'mask_token': Token ID representing masked/missing values

        Returns:
            Decoded value in original format (str, int, float, or np.nan for missing)

        Raises:
            ValueError: If variable type is not recognized

        Examples:
            >>> # Categorical variable (region)
            >>> decode_conditioning_token(0, {'type': 'categorical', 'itov': {0: 'EAST MIDLANDS', ...}})
            'EAST MIDLANDS'

            >>> # Continuous variable (total_floor_area)
            >>> decode_conditioning_token(5, {'type': 'continuous', 'boundaries': [30, 50, ...], ...})
            65.3  # Midpoint of quantile bin
        """
        data_type = var_config.get('type', '')
        if data_type in ['cardinal','ordinal','categorical','boolean']:
            itov = var_config.get('itov', {})
            decoded_value = itov.get(token_value, np.nan)

            # Apply original dtype to ensure consistency with real data
            # This prevents float/int mismatches in fidelity evaluation
            original_dtype = var_config.get('original_dtype', '')
            if not pd.isna(decoded_value):  # Only convert non-NaN values
                if original_dtype == 'int64':
                    decoded_value = int(decoded_value)
                elif original_dtype == 'bool':
                    decoded_value = bool(decoded_value)

        elif data_type == 'continuous':
            # Create temporary TokenVocabulary for decoding continuous conditioning variables
            boundaries = var_config.get('boundaries', [])
            mask_token = var_config.get('mask_token', -1)
            temp_vocab = TokenVocabulary(boundaries=list(boundaries), zero_aware=False)
            # Use vocab decode, but map the mask_token from var_config
            if token_value == mask_token:
                decoded_value = None
            else:
                decoded_value = temp_vocab.decode(token_value)
        else:
            raise ValueError(f"Unknown data type '{data_type}' for conditioning variable")
        return decoded_value
    
    def _sequence_to_dataframe_v2(
        self,
        generated_sequence: torch.Tensor,
        conditioning_row: pd.Series
    ) -> pd.DataFrame:
        """
        Convert generated 21-feature tensor to DataFrame.

        Args:
            generated_sequence: Generated tensor of shape [seq_len, 21]
            conditioning_row: Row from masked_conditioning_table with conditioning vars

        Returns:
            DataFrame with all features including conditioning variables
        """
        # Extract electricity and gas tokens (generated)
        elec_tokens = pd.Series(generated_sequence[:, 0].tolist(), name='elec_token').astype('int')
        gas_tokens = pd.Series(generated_sequence[:, 1].tolist(), name='gas_token').astype('int')

        # Extract temp and solar tokens (from input)
        temp_tokens = pd.Series(generated_sequence[:, 2].tolist(), name='temp_token').astype('int')
        solar_tokens = pd.Series(generated_sequence[:, 3].tolist(), name='solar_token').astype('int')

        # Decode half-hour
        decoded_hh = np.array(generated_sequence[:, 4].tolist())
        generated_hh = pd.Series(
            (np.where(decoded_hh == 0, np.nan, decoded_hh)).astype(float),
            name='Readings_from_midnight_local'
        )

        # Decode day of week
        decoded_dow = np.array(generated_sequence[:, 5].tolist())
        generated_dow = pd.Series(
            (np.where(decoded_dow == 7, np.nan, decoded_dow)).astype(float),
            name='dayofweek'
        )

        # Decode month
        decoded_month = np.array(generated_sequence[:, 6].tolist())
        generated_month = pd.Series(
            (np.where(decoded_month == 0, np.nan, decoded_month)).astype(float),
            name='month'
        )

        # Create tokens DataFrame
        df_tokens = pd.DataFrame({
            elec_tokens.name: elec_tokens,
            gas_tokens.name: gas_tokens,
            temp_tokens.name: temp_tokens,
            solar_tokens.name: solar_tokens,
            generated_hh.name: generated_hh,
            generated_dow.name: generated_dow,
            generated_month.name: generated_month
        })

        # Detokenize electricity and gas values
        df_generated = detokenise_data(
            df_tokens,
            elec_boundaries=self.elec_boundaries,
            gas_boundaries=self.gas_boundaries,
            temp_boundaries=self.temp_boundaries,
            solar_boundaries=self.solar_boundaries,
            elec_bin_means=self.elec_bin_means,
            gas_bin_means=self.gas_bin_means,
            value_transform=self.value_transform
        )

        # Select and rename columns
        df_generated = df_generated[[
            'month', 'dayofweek', 'Readings_from_midnight_local', 'temp_C', 'surface_solar_radiation_downwards',
            'elec_token', 'elec_value', 'gas_token', 'gas_value'
        ]]
        df_generated = df_generated.rename(columns={
            'elec_value': 'Clean_elec_net_Wh',
            'gas_value': 'Clean_gas_Wh'
        })

        # Add all 14 conditioning variables from conditioning_row
        for var_name, var_config in self.conditioning_variable_tokenisers.items():
            token_value = int(conditioning_row[var_name])
            decoded_value = self.decode_conditioning_token(token_value, var_config)
            df_generated[var_name] = decoded_value

        return df_generated

    def _sample_conditioning_variables(
        self, 
        conditioning_contexts: List[Tuple[int, int]], 
        num_households_generate: int
    ) -> List[Tuple[int, int]]:
        """
        Sample conditioning variables (num_rooms, num_occs) for generation.
        
        Args:
            conditioning_contexts: Available (B5, C1_new) conditioning variable pairs
            num_households_generate: Number of households to generate
            
        Returns:
            List of (num_rooms, num_occs) tuples
        """
        # Log available context distribution
        context_counts = {}
        for context in conditioning_contexts:
            context_counts[context] = context_counts.get(context, 0) + 1
        
        if self.logger:
            self.logger.info(f"Available context distribution: {context_counts}")
        
        # Sample with replacement if we need more households than available contexts
        if num_households_generate <= len(conditioning_contexts):
            # Sample without replacement
            sampled_contexts = random.sample(conditioning_contexts, num_households_generate)
        else:
            # Sample with replacement
            sampled_contexts = random.choices(conditioning_contexts, k=num_households_generate)
        
        # Log sampled context distribution
        sampled_counts = {}
        for context in sampled_contexts:
            sampled_counts[context] = sampled_counts.get(context, 0) + 1
        
        if self.logger:
            self.logger.info(f"Sampled context distribution for generation: {sampled_counts}")
        
        return sampled_contexts

    def _create_input_tensor_unmasked(
        self,
        template_data: pd.DataFrame,
        conditioning_row: pd.Series,
        conditioning_vars: List[str],
    ) -> torch.Tensor:
        """
        Create input tensor for unmasked batched generation.

        IMPORTANT: First timestep uses SOS tokens for ALL temporal features (0-6)
        to match training distribution where households start with SOS tokens.
        Remaining timesteps use real weather/calendar data from the template household.

        The tensor is intentionally one timestep longer than template_data to accommodate:
        - Position 0: SOS tokens (seed for generation)
        - Positions 1 to household_len: All household weather/calendar data (no waste!)

        Input tensor structure [household_len + 1, num_features]:
        - Timestep 0 (Seed):
            Features 0-1: electricity, gas (SOS tokens = vocab_size - 1)
            Features 2-6: temp, solar, hh, dow, month (SOS tokens)
            Features 7+: conditioning variables (tokenized values)
        - Timesteps 1 to household_len:
            Features 0-1: electricity, gas (SOS tokens = vocab_size - 1, to be generated)
            Features 2-6: temp, solar, hh, dow, month (real data from template)
            Features 7+: conditioning variables (tokenized values)

        Args:
            template_data: DataFrame with temporal data (temp, solar, calendar) from real household
            conditioning_row: Series with tokenized conditioning variable values
            conditioning_vars: List of conditioning variable names

        Returns:
            Input tensor of shape [household_len + 1, num_features]
        """
        household_len = len(template_data)

        # Get vocab sizes and SOS tokens from TokenVocabulary objects
        elec_sos = self.vocab['electricity'].sos_token
        gas_sos = self.vocab['gas'].sos_token
        temp_sos = self.vocab['temperature'].sos_token
        solar_sos = self.vocab['solar'].sos_token
        hh_sos = 0  # Calendar SOS tokens
        dow_sos = 7
        month_sos = 0

        # Extract temporal features from template
        temp_tokens = template_data['temp_token'].values
        solar_tokens = template_data['solar_token'].values
        hh_tokens = template_data['hh'].values  # half-hour (1-48)
        dow_tokens = template_data['dow'].values  # day of week (0-6)
        month_tokens = template_data['month'].values  # month (1-12)

        # Get conditioning variable tokens from conditioning_row
        conditioning_tokens = []
        for var in conditioning_vars:
            token_value = int(conditioning_row[var])
            conditioning_tokens.append(token_value)

        # Build input tensor (household_len + 1 to accommodate SOS at position 0)
        num_features = 2 + 5 + len(conditioning_vars)  # elec, gas, temp, solar, hh, dow, month, + conditioning
        input_tensor = torch.zeros((household_len + 1, num_features), dtype=torch.long)

        # Fill features 0-1: SOS tokens for outputs (all timesteps)
        input_tensor[:, 0] = elec_sos
        input_tensor[:, 1] = gas_sos

        # Fill features 2-6 for FIRST timestep: SOS tokens
        input_tensor[0, 2] = temp_sos
        input_tensor[0, 3] = solar_sos
        input_tensor[0, 4] = hh_sos
        input_tensor[0, 5] = dow_sos
        input_tensor[0, 6] = month_sos

        # Fill features 2-6 for REMAINING timesteps: Real temporal/weather data from template
        input_tensor[1:, 2] = torch.tensor(temp_tokens, dtype=torch.long)
        input_tensor[1:, 3] = torch.tensor(solar_tokens, dtype=torch.long)
        input_tensor[1:, 4] = torch.tensor(hh_tokens, dtype=torch.long)
        input_tensor[1:, 5] = torch.tensor(dow_tokens, dtype=torch.long)
        input_tensor[1:, 6] = torch.tensor(month_tokens, dtype=torch.long)

        # Fill conditioning variables (same for all timesteps)
        for idx, token_value in enumerate(conditioning_tokens):
            input_tensor[:, 7 + idx] = token_value

        return input_tensor

    def _extract_month_weather(
        self,
        weather_data: pd.DataFrame,
        year: int,
        month: int,
    ) -> pd.DataFrame:
        """
        Extract weather/calendar data for a specific month.

        Args:
            weather_data: Full year DataFrame with columns:
                temp_token, solar_token, hh, dow, month
            year: Year (used for calculating expected timesteps)
            month: Month to extract (1-12)

        Returns:
            DataFrame with weather data for the specified month,
            sorted by original index to maintain temporal order.
        """
        month_data = weather_data[weather_data['month'] == month].copy()

        # Validate expected number of timesteps
        days_in_month = calendar.monthrange(year, month)[1]
        expected_timesteps = days_in_month * 48

        if len(month_data) != expected_timesteps:
            self._log(
                f"Month {month} has {len(month_data)} timesteps, expected {expected_timesteps}",
                level='warning'
            )

        return month_data.reset_index(drop=True)

    def _create_month_input_tensor(
        self,
        month_weather: pd.DataFrame,  # Weather/calendar for this month
        conditioning_row: pd.Series,   # Tokenized conditioning variables                                  
        conditioning_vars: List[str],                                                                      
    ) -> torch.Tensor:                                                                                     
        """                                                                                                
        Create input tensor for a single month with offset conditioning.                                   
                                                                                                            
        Position 0: SOS_elec, SOS_gas, dummy conditioning (temp_sos, hh=0, dow=7, month=0)                 
        Position 1 to N: placeholder elec/gas, actual conditioning for half-hours 0 to N-1                 
                                                                                                            
        This ensures the first generated value (at position 1) corresponds to                              
        the first half-hour of the month.                                                                  
        """                                                                                                
        timesteps = len(month_weather)  # e.g., 1488 for 31-day month                                      

        # Get SOS token IDs from TokenVocabulary objects
        elec_sos = self.vocab['electricity'].sos_token
        gas_sos = self.vocab['gas'].sos_token

        # Input tensor has timesteps + 1 positions                                                         
        num_features = 2 + 5 + len(conditioning_vars)                                                      
        input_tensor = torch.zeros((timesteps + 1, num_features), dtype=torch.long)                        
                                                                                                            
        # Position 0: SOS for elec/gas, dummy/SOS for temporal                                             
        input_tensor[0, 0] = elec_sos                                                                      
        input_tensor[0, 1] = gas_sos                                                                       
        input_tensor[0, 2] = month_weather['temp_token'].values[0]     # duplicate value from first time step                                                            
        input_tensor[0, 3] = month_weather['solar_token'].values[0]                                                             
        input_tensor[0, 4] = month_weather['hh'].values[0]                                                                     
        input_tensor[0, 5] = month_weather['dow'].values[0]                                                                   
        input_tensor[0, 6] = month_weather['month'].values[0]                                                         
                                                                                                            
        # Position 1 to timesteps: actual conditioning for half-hours 0 to N-1
        input_tensor[1:, 0] = elec_sos  # Placeholder (to be generated)
        input_tensor[1:, 1] = gas_sos   # Placeholder (to be generated)
        input_tensor[1:, 2] = torch.tensor(month_weather['temp_token'].values, dtype=torch.long)
        input_tensor[1:, 3] = torch.tensor(month_weather['solar_token'].values, dtype=torch.long)
        input_tensor[1:, 4] = torch.tensor(month_weather['hh'].values, dtype=torch.long)
        input_tensor[1:, 5] = torch.tensor(month_weather['dow'].values, dtype=torch.long)
        input_tensor[1:, 6] = torch.tensor(month_weather['month'].values, dtype=torch.long)                                                
                                                                                                            
        # Conditioning variables (same for all positions)                                                  
        for idx, var in enumerate(conditioning_vars):                                                      
            input_tensor[:, 7 + idx] = int(conditioning_row[var])                                          
                                                                                                            
        return input_tensor         
    
    def generate_year_chunked(                                                                             
        self,                                                                                              
        weather_data: pd.DataFrame,     # Full year of weather (tokenized)                                 
        conditioning_row: pd.Series,    # Tokenized conditioning variables                                 
        conditioning_vars: List[str],                                                                      
        batch_size: int,                                                                                   
        year: int = 2023,                                                                                  
    ) -> List[Dict]:                                                                                       
        """                                                                                                
        Generate a full year using month-chunk approach with hard reset.                                   
                                                                                                            
        Each month is generated independently with fresh SOS tokens.                                       
        """                                                                                                
        all_months = []                                                                                    
                                                                                                            
        for month in range(1, 13):                                                                         
            # Get month metadata                                                                           
            days_in_month = calendar.monthrange(year, month)[1]                                            
                                                                                                            
            # Extract weather for this month                                                               
            month_weather = self._extract_month_weather(weather_data, year, month)                         
                                                                                                            
            # Create input tensor with offset conditioning                                                 
            input_tensor = self._create_month_input_tensor(                                                
                month_weather, conditioning_row, conditioning_vars                                         
            )        

            # Move to device
            input_tensor = move_to_device(input_tensor, self.device)

            month_output = self.generate_batched(
                    electricity_model=self.electricity_model,
                    gas_model=self.gas_model,
                    input_tensor=input_tensor,
                    temperature=self.temperature,
                    batch_size=batch_size,
                    block_missing_tokens=self.block_missing_tokens,
                    elec_missing_token=self.elec_missing_token,
                    elec_sos_token=self.elec_sos_token,
                    gas_missing_token=self.gas_missing_token,
                    gas_sos_token=self.gas_sos_token
                )    
                                                                                                        
            all_months.append({                                                                            
                'month': month,                                                                            
                'data': month_output,                                                                      
                'timesteps': days_in_month * 48                                                            
            })                                                                                             
                                                                                                            
        return all_months   
    
    def _batch_to_wide_dataframe(
        self,
        generated_batch: torch.Tensor,
        conditioning_row: pd.Series,
        conditioning_vars: List[str],
        conditioning_tokenisers: Dict[str, Any],
        region: str,
        template_data: pd.DataFrame,
        k_group_size: int
    ) -> pd.DataFrame:
        """
        Transform batched generation output to wide format DataFrame.

        Wide format structure:
        - Conditioning variables (decoded): building_type, age_built, num_rooms, num_occs
        - Metadata: region, group_size (k-anonymity)
        - Calendar: datetime (from template)
        - Batch households: hh001_elec_net_Wh, hh001_gas_Wh, hh001_temp_C, hh001_solar,
                           hh002_elec_net_Wh, hh002_gas_Wh, ...

        Args:
            generated_batch: Generated tensor of shape [batch_size, seq_len, num_features]
            conditioning_row: Series with tokenized conditioning variable values
            conditioning_vars: List of conditioning variable names
            conditioning_tokenisers: Tokenizer config for conditioning variables
            region: Region name from template household
            template_data: Template household data (for datetime)
            k_group_size: k-anonymity group size from metadata_catalog_filtered

        Returns:
            Wide format DataFrame
        """
        batch_size = generated_batch.shape[0]

        # Initialize output dictionary
        output_dict = {}

        # === Add conditioning variables (decoded) ===
        for var in conditioning_vars:
            token_value = int(conditioning_row[var])
            var_config = conditioning_tokenisers[var]
            decoded_value = self.decode_conditioning_token(token_value, var_config)
            output_dict[var] = decoded_value

        # === Add metadata ===
        output_dict['region'] = region
        output_dict['group_size'] = int(k_group_size)

        # === Add shared temporal data (all households share same temporal context) ===
        # Extract temporal data from first household (same for all)
        # NOTE: Skip first timestep (index 0) which is the SOS token used for generation prompt
        temp_tokens = generated_batch[0, 1:, 2].cpu().numpy()  # Skip SOS row
        solar_tokens = generated_batch[0, 1:, 3].cpu().numpy()  # Skip SOS row
        hh_tokens = generated_batch[0, 1:, 4].cpu().numpy()  # Skip SOS row
        dow_tokens = generated_batch[0, 1:, 5].cpu().numpy()  # Skip SOS row
        month_tokens = generated_batch[0, 1:, 6].cpu().numpy()  # Skip SOS row

        # Detokenize temperature and solar using TokenVocabulary (shared across batch)
        temp_values = np.array([self.vocab['temperature'].decode(int(t)) for t in temp_tokens])
        solar_values = np.array([self.vocab['solar'].decode(int(t)) for t in solar_tokens])

        # Add shared temporal data
        output_dict['temp_C'] = temp_values
        output_dict['solar'] = solar_values
        output_dict['half_hour'] = hh_tokens
        output_dict['dow'] = dow_tokens
        output_dict['month'] = month_tokens

        # Add datetime if available in template
        if 'datetime' in template_data.columns:
            output_dict['datetime'] = template_data['datetime'].values[1:]  # Skip SOS row

        # === Add batch households (wide format - only elec and gas vary per household) ===
        for batch_idx in range(batch_size):
            household_prefix = f"hh{batch_idx+1:03d}"

            # Extract tokens for this household
            # NOTE: Skip first timestep (index 0) which is the SOS token used for generation prompt
            elec_tokens = generated_batch[batch_idx, 1:, 0].cpu().numpy()  # Skip SOS row
            gas_tokens = generated_batch[batch_idx, 1:, 1].cpu().numpy()  # Skip SOS row

            # Detokenize using TokenVocabulary (handles bin means automatically)
            elec_values = np.array([self.vocab['electricity'].decode(int(t)) for t in elec_tokens])
            gas_values = np.array([self.vocab['gas'].decode(int(t)) for t in gas_tokens])

            # Add to output dictionary (only elec and gas per household)
            output_dict[f"{household_prefix}_elec_net_Wh"] = elec_values
            output_dict[f"{household_prefix}_gas_Wh"] = gas_values

        # Create DataFrame
        df_wide = pd.DataFrame(output_dict)

        return df_wide

    def _month_batch_to_wide_dataframe(
        self,
        generated_batch: torch.Tensor,
        month_weather: pd.DataFrame,
        conditioning_row: pd.Series,
        conditioning_vars: List[str],
        conditioning_tokenisers: Dict[str, Any],
        region: str,
        k_group_size: int,
        month: int,
    ) -> pd.DataFrame:
        """
        Transform single-month generation output to wide format DataFrame.

        Args:
            generated_batch: Generated tensor of shape [batch_size, timesteps+1, num_features]
                where timesteps+1 includes the SOS row at position 0
            month_weather: Weather DataFrame for this month (timesteps rows)
            conditioning_row: Series with tokenized conditioning variable values
            conditioning_vars: List of conditioning variable names
            conditioning_tokenisers: Tokenizer config for conditioning variables
            region: Region name
            k_group_size: k-anonymity group size
            month: Month number (1-12)

        Returns:
            Wide format DataFrame with SOS row removed (timesteps rows)
        """
        batch_size = generated_batch.shape[0]

        # Initialize output dictionary
        output_dict = {}

        # === Add conditioning variables (decoded) ===
        for var in conditioning_vars:
            token_value = int(conditioning_row[var])
            var_config = conditioning_tokenisers[var]
            decoded_value = self.decode_conditioning_token(token_value, var_config)
            output_dict[var] = decoded_value

        # === Add metadata ===
        output_dict['region'] = region
        output_dict['group_size'] = int(k_group_size)
        output_dict['month'] = month

        # === Add shared temporal data from month_weather (no SOS row in weather data) ===
        temp_tokens = month_weather['temp_token'].values
        solar_tokens = month_weather['solar_token'].values
        hh_tokens = month_weather['hh'].values
        dow_tokens = month_weather['dow'].values

        # Detokenize temperature and solar using TokenVocabulary
        temp_values = np.array([self.vocab['temperature'].decode(int(t)) for t in temp_tokens])
        solar_values = np.array([self.vocab['solar'].decode(int(t)) for t in solar_tokens])

        # Add shared temporal data
        output_dict['temp_C'] = temp_values
        output_dict['solar'] = solar_values
        output_dict['half_hour'] = hh_tokens
        output_dict['dow'] = dow_tokens

        # Add datetime if available in month_weather
        if 'datetime' in month_weather.columns:
            output_dict['datetime'] = month_weather['datetime'].values

        # === Add batch households (wide format - only elec and gas vary per household) ===
        for batch_idx in range(batch_size):
            household_prefix = f"hh{batch_idx+1:03d}"

            # Extract tokens for this household, skipping SOS row (position 0)
            elec_tokens = generated_batch[batch_idx, 1:, 0].cpu().numpy()
            gas_tokens = generated_batch[batch_idx, 1:, 1].cpu().numpy()

            # Detokenize using TokenVocabulary (handles bin means automatically)
            elec_values = np.array([self.vocab['electricity'].decode(int(t)) for t in elec_tokens])
            gas_values = np.array([self.vocab['gas'].decode(int(t)) for t in gas_tokens])

            output_dict[f"{household_prefix}_elec_net_Wh"] = elec_values
            output_dict[f"{household_prefix}_gas_Wh"] = gas_values

        return pd.DataFrame(output_dict)

    def _append_conditioning_summary(
        self,
        output_dir: str,
        batch_idx: int,
        conditioning_row: pd.Series,
        conditioning_vars: List[str],
        conditioning_tokenisers: Dict[str, Any],
        region: str,
        k_group_size: int,
    ) -> None:
        """
        Append conditioning variables for a batch to the cumulative summary file.

        Args:
            output_dir: Directory for output files
            batch_idx: Batch index (1-indexed)
            conditioning_row: Series with tokenized conditioning variable values
            conditioning_vars: List of conditioning variable names
            conditioning_tokenisers: Tokenizer config for conditioning variables
            region: Region name
            k_group_size: k-anonymity group size
        """
        summary_file = os.path.join(output_dir, "conditioning_summary.pkl")

        # Build row data
        row_data = {'batch_id': batch_idx}

        # Decode conditioning variables
        for var in conditioning_vars:
            token_value = int(conditioning_row[var])
            var_config = conditioning_tokenisers[var]
            decoded_value = self.decode_conditioning_token(token_value, var_config)
            row_data[var] = decoded_value

        row_data['region'] = region
        row_data['group_size'] = int(k_group_size)

        # Load existing summary or create new
        if os.path.exists(summary_file):
            summary_df = pd.read_pickle(summary_file)
            summary_df = pd.concat([summary_df, pd.DataFrame([row_data])], ignore_index=True)
        else:
            summary_df = pd.DataFrame([row_data])

        summary_df.to_pickle(summary_file)

    def _save_household_data(self, df_generated: pd.DataFrame, puprn: str):
        """Save generated data for a specific household."""
        os.makedirs(self.config['generated_data_path'], exist_ok=True)
        generated_data_path = os.path.join(
            self.config['generated_data_path'], 
            f'generated_data_{puprn}.pkl'
        )
        df_generated.to_pickle(generated_data_path)
        print(f"[SAVE] Generated data -> {generated_data_path}")

def load_model_for_inference(
    model: GPTSmartMeterModel,
    model_path: str,
    output_variable: str,
    model_type: str = 'val',
    device: torch.device = None,
) -> GPTSmartMeterModel:
    """
    Load trained model weights for inference.

    Loads checkpoint weights from the specified training run directory into
    the provided model instance. Supports loading either the best validation
    or best training loss checkpoint.

    Args:
        model: GPTSmartMeterModel instance to load weights into. Should be
            initialized with the same architecture as the trained model.
        model_path: Path to training run directory containing saved checkpoints.
            Expected to contain files like 'electricity_best_val_model.pth'.
        output_variable: The output variable this model was trained for.
            Either 'electricity' or 'gas'. Used to construct the checkpoint
            filename (e.g., 'electricity_best_val_model.pth').
        model_type: Which checkpoint to load. Options:
            - 'val': Load best validation loss checkpoint (default, recommended)
            - 'train': Load best training loss checkpoint
        device: Device to load model weights to. If None, uses the model's
            current device. Handles cross-device loading (e.g., GPU to CPU).

    Returns:
        Model with loaded weights, ready for inference.

    Note:
        Prints warnings if checkpoint file is not found at expected path.
    """
    if model_type == 'val':
        weights_path = os.path.join(model_path, f'{output_variable}_best_val_model.pth')
    else:
        weights_path = os.path.join(model_path, f'{output_variable}_best_train_model.pth')
    
    if os.path.exists(weights_path):
        # Determine device for loading
        if device is None:
            device = next(model.parameters()).device
        
        # Load with proper device mapping
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
        # Note: This function doesn't have access to logger, keeping basic behavior
        print(f'[MODEL] Loaded best {model_type} model weights from {weights_path} to {device}')
    else:
        print(f'[WARNING] Model weights not found at {weights_path}')
    
    return model