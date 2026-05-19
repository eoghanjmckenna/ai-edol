"""
Training infrastructure for GPT models.
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import glob
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
from torch.utils.data import DataLoader
import os
from utils.device_utils import get_device_from_config, move_to_device, log_device_info
from preprocessing.datasets import MultiFeatureDataset


class ModelTrainer:
    """
    Handles single-output model training with early stopping and checkpointing.

    This trainer is designed for the separate models architecture where electricity
    and gas are trained independently. Each ModelTrainer instance handles one
    output variable (either 'electricity' or 'gas').

    Features:
        - Shard-based training for large datasets
        - Early stopping with configurable patience
        - Best-model checkpoint saving
        - Support for masked and unmasked validation
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        config: Dict[str, Any],
        run_path: str,
        output_variable: str = 'electricity',
        logger=None,
        bin_means: list = None
    ):
        """
        Initialize ModelTrainer for single-output model training.

        Args:
            model: PyTorch model to train (single-output GPTSmartMeterModel)
            optimizer: Optimizer for training (e.g., AdamW)
            loss_fn: Loss function (e.g., CrossEntropyLoss)
            config: Training configuration dictionary containing:
                - batch_size: Batch size for training
                - epochs: Maximum number of training epochs
                - patience: Early stopping patience
                - eval_iters: Number of validation iterations per epoch
                - random_seed: Base random seed for reproducibility
                - device: Device specification ('auto', 'cpu', 'cuda', 'mps')
            run_path: Path to save checkpoints, logs, and diagnostics
            output_variable: Target variable - 'electricity' or 'gas'
            logger: Optional JSON logger for structured logging
        """
        # Set output_variable first (needed for _log method)
        self.output_variable = output_variable

        # Set up device
        self.device = get_device_from_config(config, verbose=True)
        log_device_info(self.device, logger)

        # Move model and loss function to device
        self.model = move_to_device(model, self.device)
        self.optimizer = optimizer
        self.loss_fn = move_to_device(loss_fn, self.device)
        self.config = config
        self.run_path = run_path
        self.logger = logger

        # Log device info for batch size planning
        if self.device.type == 'cuda':
            gpu_memory_gb = torch.cuda.get_device_properties(self.device).total_memory / 1024**3
            self._log(f"GPU memory available: {gpu_memory_gb:.1f}GB - batch size {config.get('batch_size', 32)} (configured)")
        
        # Training state
        self.best_val_loss = float('inf')
        self.best_train_loss = float('inf')
        self.best_epoch = np.nan
        self.epochs_without_improvement = 0

        # Metrics tracking
        self.epoch_train_losses = []
        self.epoch_val_losses_masked = []  # Primary validation metric (masked data)
        self.epoch_val_losses_unmasked = []  # Secondary validation metric (unmasked data)
        self.step_train_losses = []
        self.epochs = []

        # Wh-space fidelity metrics (Tiers 1 & 2)
        if bin_means is not None:
            self.bin_means_tensor = torch.tensor(bin_means, dtype=torch.float32, device=self.device)
            self.wh_metrics_enabled = True
        else:
            self.bin_means_tensor = None
            self.wh_metrics_enabled = False
        self.epoch_train_token_bias = []
        self.epoch_train_token_mad = []
        self.epoch_train_wh_mae = []
        self.epoch_train_wh_mean_ratio = []
        self.epoch_val_token_bias = []
        self.epoch_val_token_mad = []
        self.epoch_val_wh_mae = []
        self.epoch_val_wh_mean_ratio = []

        # Checkpoint paths
        self.best_val_model_path = os.path.join(run_path, f'{output_variable}_best_val_model.pth')
        self.best_train_model_path = os.path.join(run_path, f'{output_variable}_best_train_model.pth')

    def _log(self, message: str, level: str = 'info') -> None:
        """Log message using logger if available, otherwise print with prefix.

        The prefix includes the output_variable (electricity/gas) to make it clear
        which model is being trained when running sequential training.
        """
        # Include output_variable in prefix for clarity during sequential training
        prefix = f"[TRAINING:{self.output_variable}]"
        prefixed_message = f"{self.output_variable}: {message}"

        if self.logger:
            if level == 'info':
                self.logger.info(prefixed_message)
            elif level == 'warning':
                self.logger.warning(prefixed_message)
            elif level == 'error':
                self.logger.error(prefixed_message)
            elif level == 'debug':
                self.logger.debug(prefixed_message)
        else:
            print(f"{prefix} {message}", flush=True)

    def train_with_shards(
        self,
        train_shards_dir: str,
        batch_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Train the model using sharded data with automatic mode detection.

        Automatically detects training mode based on available shards:
        - UNMASKED: Uses train_unmasked_*.pt shards (simplified conditioning)
        - MASKED: Uses train_unconstrained_*.pt shards (k-anonymity masking)
        - LEGACY: Falls back to train_shard_*.pt if others not found

        Args:
            train_shards_dir: Directory containing training shard .pt files.
                Expected to contain validation shards (val_unmasked.pt or
                val_unconstrained.pt) in the same directory.
            batch_size: Optional batch size override. If None, uses config value.

        Returns:
            Dictionary containing training metrics:
                - best_train_loss: Best training loss achieved
                - best_val_loss: Best validation loss achieved
                - best_epoch: Epoch where best validation loss occurred
                - train_time: Total training time in seconds
                - epoch_train_losses: List of training losses per epoch
                - epoch_val_losses_masked: List of primary validation losses
                - epoch_val_losses_unmasked: List of secondary validation losses
                - epochs: List of epoch indices
                - training_mode: UNMASKED, MASKED, or LEGACY
                - num_training_shards: Number of training shards used
                - unmasked_validation_enabled: Whether secondary validation was used
        """
        if batch_size is None:
            batch_size = self.config.get('batch_size', 32)

        self._log(f"Starting shard-based training for {self.config['epochs']} epochs...")
        self._log(f"Training shards directory: {train_shards_dir}")

        start_time = time.time()

        # DETECT TRAINING MODE FROM AVAILABLE SHARDS (curriculum learning removed)
        # Priority: unmasked (default) → masked → legacy

        # Check for unmasked shards first (preferred for simplified conditioning approach)
        unmasked_shard_paths = sorted(glob.glob(
            os.path.join(train_shards_dir, "train_unmasked_*.pt")
        ))

        # Check for masked shards (legacy/optional)
        masked_shard_paths = sorted(glob.glob(
            os.path.join(train_shards_dir, "train_unconstrained_*.pt")
        ))

        # Determine training mode and configure shards
        if unmasked_shard_paths:
            training_mode = "UNMASKED"
            train_shard_paths = unmasked_shard_paths
            val_shard_path = os.path.join(train_shards_dir, 'val_unmasked.pt')
            enable_unmasked_eval = False  # Already training on unmasked data
            self._log("Mode: UNMASKED (simplified conditioning)")
        elif masked_shard_paths:
            training_mode = "MASKED"
            train_shard_paths = masked_shard_paths
            val_shard_path = os.path.join(train_shards_dir, 'val_unconstrained.pt')
            enable_unmasked_eval = True  # Enable unmasked validation as secondary metric
            self._log("Mode: MASKED (k-anonymity masking)")
            self._log("Unmasked validation: ENABLED (secondary metric)")
        else:
            # Legacy fallback
            train_shard_paths = sorted(glob.glob(
                os.path.join(train_shards_dir, "train_shard_*.pt")
            ))
            if not train_shard_paths:
                raise ValueError(f"No training shards found in {train_shards_dir}")
            training_mode = "LEGACY"
            val_shard_path = os.path.join(train_shards_dir, 'val_unconstrained.pt')
            enable_unmasked_eval = True
            self._log("Mode: LEGACY")

        if not os.path.exists(val_shard_path):
            raise ValueError(f"Validation shard not found: {val_shard_path}")

        self._log(f"Found {len(train_shard_paths)} training shards")

        # Log configuration
        if self.logger:
            self.logger.log_custom_metric('shard_training_config', {
                'training_mode': training_mode,
                'num_training_shards': len(train_shard_paths),
                'unmasked_validation_enabled': enable_unmasked_eval,
                'batch_size': batch_size
            })

        # Load primary validation shard
        self._log(f"Loading primary validation shard: {val_shard_path}")
        val_tensor_data = torch.load(val_shard_path, map_location=self.device)
        val_dataloader = self._create_validation_dataloader(val_tensor_data, batch_size)
        self._log(f"Primary validation shard loaded: {len(val_dataloader.dataset)} sequences")

        # Load unmasked validation shard (if enabled for secondary evaluation)
        val_dataloader_unmasked = None
        if enable_unmasked_eval:
            val_unmasked_path = os.path.join(train_shards_dir, 'val_unmasked.pt')

            if os.path.exists(val_unmasked_path):
                self._log(f"Loading unmasked validation shard: {val_unmasked_path}")
                try:
                    val_tensor_data_unmasked = torch.load(val_unmasked_path, map_location=self.device)
                    val_dataloader_unmasked = self._create_validation_dataloader(val_tensor_data_unmasked, batch_size)
                    self._log(f"Unmasked validation shard loaded: {len(val_dataloader_unmasked.dataset)} sequences")
                except Exception as e:
                    self._log(f"Warning: Failed to load unmasked validation shard: {e}", level='warning')
                    enable_unmasked_eval = False
                    val_dataloader_unmasked = None
            else:
                self._log(f"Unmasked validation shard not found at {val_unmasked_path}")
                enable_unmasked_eval = False

        # Training loop: config['epochs'] is the total epoch count.
        for epoch in range(self.config['epochs']):
            epoch_start = time.time()

            self._log(f"Epoch {epoch+1}/{self.config['epochs']}: Training... (Mode: {training_mode})")

            # Training phase with training shards
            train_loss, train_wh_metrics = self._train_epoch_with_shards(
                train_shard_paths, batch_size, epoch,
                output_variable=self.output_variable)

            # Validation phase - Primary metric
            # Use appropriate label based on training mode
            primary_val_type = 'unmasked' if training_mode == 'UNMASKED' else 'masked'
            val_loss_masked, val_wh_metrics = self._validate_epoch(
                val_dataloader, validation_type=primary_val_type,
                output_variable=self.output_variable)

            # Secondary validation (unmasked) if available
            val_loss_unmasked = None
            if enable_unmasked_eval and val_dataloader_unmasked is not None:
                val_loss_unmasked, _ = self._validate_epoch(
                    val_dataloader_unmasked, validation_type='unmasked',
                    output_variable=self.output_variable)

            # Store Wh-space fidelity metrics
            if train_wh_metrics is not None:
                self.epoch_train_token_bias.append(train_wh_metrics['token_bias'])
                self.epoch_train_token_mad.append(train_wh_metrics['token_mad'])
                self.epoch_train_wh_mae.append(train_wh_metrics['wh_mae'])
                self.epoch_train_wh_mean_ratio.append(train_wh_metrics['wh_mean_ratio'])
            if val_wh_metrics is not None:
                self.epoch_val_token_bias.append(val_wh_metrics['token_bias'])
                self.epoch_val_token_mad.append(val_wh_metrics['token_mad'])
                self.epoch_val_wh_mae.append(val_wh_metrics['wh_mae'])
                self.epoch_val_wh_mean_ratio.append(val_wh_metrics['wh_mean_ratio'])

            # Log epoch metrics
            epoch_time = time.time() - epoch_start

            # Log Wh fidelity metrics to console
            if train_wh_metrics is not None and val_wh_metrics is not None:
                self._log(
                    f"Epoch {epoch+1}/{self.config['epochs']}: "
                    f"train_token_bias={train_wh_metrics['token_bias']:.2f}, "
                    f"val_token_bias={val_wh_metrics['token_bias']:.2f}, "
                    f"train_wh_mae={train_wh_metrics['wh_mae']:.1f}, "
                    f"val_wh_mae={val_wh_metrics['wh_mae']:.1f}, "
                    f"train_wh_ratio={train_wh_metrics['wh_mean_ratio']:.3f}, "
                    f"val_wh_ratio={val_wh_metrics['wh_mean_ratio']:.3f}"
                )

            if self.logger:
                self.logger.log_training_epoch(
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss_masked,
                    learning_rate=self.optimizer.param_groups[0]['lr'],
                    epoch_time=epoch_time
                )
                # Log training mode
                self.logger.log_custom_metric(f'epoch_{epoch}_training_mode', training_mode)
                # Log secondary unmasked validation if available
                if val_loss_unmasked is not None:
                    self.logger.log_custom_metric(f'epoch_{epoch}_val_loss_unmasked', val_loss_unmasked)
                # Log Wh fidelity metrics
                if val_wh_metrics is not None:
                    self.logger.log_custom_metric(f'epoch_{epoch}_wh_metrics', {
                        'train': train_wh_metrics, 'val': val_wh_metrics
                    })

            # Update best models and save checkpoints (using masked loss as primary metric)
            self._update_best_models(train_loss, val_loss_masked, epoch)

            # Early stopping check
            if self._should_stop_early():
                self._log(f"Early stopping after {self.config['patience']} epochs without improvement.")
                break

        train_time = time.time() - start_time
        self._log(f"Shard-based training completed in {train_time/60:.1f} minutes. Best val loss: {self.best_val_loss:.4f}")

        # Save loss diagnostics (CSV and plot)
        self._save_loss_diagnostics()
        self._save_wh_fidelity_diagnostics()

        return {
            'best_train_loss': self.best_train_loss,
            'best_val_loss': self.best_val_loss,  # Primary metric
            'best_epoch': self.best_epoch,
            'train_time': train_time,
            'epoch_train_losses': self.epoch_train_losses,
            'epoch_val_losses': self.epoch_val_losses_masked,  # Primary validation metric
            'epoch_val_losses_masked': self.epoch_val_losses_masked,
            'epoch_val_losses_unmasked': self.epoch_val_losses_unmasked,  # Secondary metric (if enabled)
            'epochs': self.epochs,
            'training_mode': training_mode,  # UNMASKED, MASKED, or LEGACY
            'num_training_shards': len(train_shard_paths),
            'unmasked_validation_enabled': enable_unmasked_eval,
        }
    
    def _create_validation_dataloader(
        self,
        val_tensor_data: Dict[str, torch.Tensor],
        batch_size: int
    ) -> DataLoader:
        """
        Create validation dataloader from loaded tensor data.

        Args:
            val_tensor_data: Dictionary containing 'X_val' and 'y_val' tensors
            batch_size: Batch size for dataloader

        Returns:
            DataLoader for validation
        """
        val_dataset = MultiFeatureDataset(
            val_tensor_data['X_val'],
            val_tensor_data['y_val']
        )
        val_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=False  # No need to shuffle validation data
        )
        return val_dataloader

    def _train_epoch_with_shards(
        self,
        train_shard_paths: List[str],
        batch_size: int,
        epoch: int,
        output_variable: str = 'electricity'
    ) -> Tuple[float, Optional[Dict[str, float]]]:
        """
        Train for one epoch using sharded data.

        Processes shards in shuffled order (reproducible via epoch-specific seed).
        Each shard is loaded, processed, and cleared from memory before loading
        the next to manage memory usage with large datasets.

        Args:
            train_shard_paths: List of paths to training shard .pt files
            batch_size: Batch size for DataLoader
            epoch: Current epoch number (used for reproducible shuffling)
            output_variable: Target variable ('electricity' or 'gas')

        Returns:
            Tuple of (average_loss, wh_metrics_dict_or_None)
        """
        self.model.train()
        total_loss = 0
        total_batches = 0
        self.epochs.append(epoch)

        # set target index in tensor depending on output variable
        if output_variable == 'electricity':
            target_index = 0
        elif output_variable == 'gas':
            target_index = 1
        else:
            raise ValueError(f"Unsupported output variable: {output_variable}")
        
        # Reset random seed with epoch-specific offset for reproducible shard shuffling
        # This ensures that each epoch's shuffle is deterministic and reproducible
        base_seed = self.config.get('random_seed', 42)
        epoch_seed = base_seed + epoch
        np.random.seed(epoch_seed)
        torch.manual_seed(epoch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(epoch_seed)

        # Shuffle shard order each epoch to ensure variety (now reproducible)
        shuffled_shard_paths = train_shard_paths.copy()
        np.random.shuffle(shuffled_shard_paths)

        # Wh metrics accumulators
        if self.wh_metrics_enabled:
            wh_accum = {
                'signed_diffs_sum': 0.0, 'abs_diffs_sum': 0.0,
                'valid_count': 0, 'wh_abs_error_sum': 0.0,
                'wh_expected_sum': 0.0, 'wh_actual_sum': 0.0,
            }

        self._log(f"Epoch {epoch+1}: Processing {len(shuffled_shard_paths)} training shards...")

        for shard_idx, shard_path in enumerate(shuffled_shard_paths):
            self._log(f"Epoch {epoch+1}: Loading shard {shard_idx+1}/{len(shuffled_shard_paths)}: {os.path.basename(shard_path)}")
            
            # Load current shard directly to configured device for efficiency
            shard_tensor_data = torch.load(shard_path, map_location=self.device)

            # Create dataset and dataloader for this shard
            train_dataset = MultiFeatureDataset(
                shard_tensor_data['X_train'],
                shard_tensor_data['y_train'],
                shuffle=True  # Break up household blocks for heterogeneous batches (uses epoch seed)
            )

            train_dataloader = DataLoader(
                dataset=train_dataset,
                batch_size=batch_size,
                shuffle=True  # Vary batch order each iteration (uses epoch seed)
            )
            
            # Process all batches in this shard
            shard_loss = 0
            shard_batches = 0

            for X, y in train_dataloader:
                # Move data to device
                X = move_to_device(X, self.device)
                y = move_to_device(y, self.device)

                # Forward pass (single output model)
                output = self.model(X)

                # Calculate loss for single output
                loss = self._calculate_loss(output, y, target_index=target_index)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Track metrics
                loss_value = loss.item()
                self.step_train_losses.append(loss_value)
                shard_loss += loss_value
                shard_batches += 1

                # Accumulate Wh-space fidelity metrics
                if self.wh_metrics_enabled:
                    batch_wh = self._compute_wh_metrics(output.detach(), y, target_index)
                    for key in wh_accum:
                        wh_accum[key] += batch_wh[key]
            
            # Clear shard data from memory
            del shard_tensor_data, train_dataset, train_dataloader
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            total_loss += shard_loss
            total_batches += shard_batches

            avg_shard_loss = shard_loss / shard_batches if shard_batches > 0 else 0
            self._log(f"Epoch {epoch+1}: Shard {shard_idx+1} avg loss: {avg_shard_loss:.4f} ({shard_batches} batches)")

        avg_epoch_loss = total_loss / total_batches if total_batches > 0 else 0
        self.epoch_train_losses.append(avg_epoch_loss)

        self._log(f"Epoch {epoch+1}/{self.config['epochs']}: Overall train loss: {avg_epoch_loss:.4f} ({total_batches} total batches)")

        # Compute epoch-level Wh metrics from accumulated sums
        wh_metrics = None
        if self.wh_metrics_enabled and wh_accum['valid_count'] > 0:
            count = wh_accum['valid_count']
            wh_metrics = {
                'token_bias': wh_accum['signed_diffs_sum'] / count,
                'token_mad': wh_accum['abs_diffs_sum'] / count,
                'wh_mae': wh_accum['wh_abs_error_sum'] / count,
                'wh_mean_ratio': wh_accum['wh_expected_sum'] / max(wh_accum['wh_actual_sum'], 1e-8),
            }

        return avg_epoch_loss, wh_metrics
        
    def _validate_epoch(self, dataloader: DataLoader, validation_type: str = 'masked',
                        output_variable: str = 'electricity') -> Tuple[float, Optional[Dict[str, float]]]:
        """
        Validate for one epoch.

        Runs validation on a subset of batches (controlled by config['eval_iters'])
        to balance validation accuracy with training speed.

        Args:
            dataloader: Validation DataLoader
            validation_type: Either 'masked' (primary) or 'unmasked' (secondary).
                Controls which loss tracking list gets updated.
            output_variable: Target variable ('electricity' or 'gas').
                Determines which column of y tensor to use as target.

        Returns:
            Tuple of (average_val_loss, wh_metrics_dict_or_None)
        """
        self.model.eval()
        # set target index in tensor depending on output variable
        if output_variable == 'electricity':
            target_index = 0
        elif output_variable == 'gas':
            target_index = 1
        else:
            raise ValueError(f"Unsupported output variable: {output_variable}")

        val_losses = []

        # Wh metrics accumulators
        if self.wh_metrics_enabled:
            wh_accum = {
                'signed_diffs_sum': 0.0, 'abs_diffs_sum': 0.0,
                'valid_count': 0, 'wh_abs_error_sum': 0.0,
                'wh_expected_sum': 0.0, 'wh_actual_sum': 0.0,
            }

        with torch.no_grad():
            val_iter = iter(dataloader)
            for _ in range(self.config['eval_iters']):
                try:
                    X_val, y_val = next(val_iter)

                    # Move data to device
                    X_val = move_to_device(X_val, self.device)
                    y_val = move_to_device(y_val, self.device)

                    # Forward pass
                    output = self.model(X_val)

                    # Calculate loss for single output
                    val_loss = self._calculate_loss(output, y_val, target_index=target_index)
                    val_losses.append(val_loss.item())

                    # Accumulate Wh-space fidelity metrics
                    if self.wh_metrics_enabled:
                        batch_wh = self._compute_wh_metrics(output, y_val, target_index)
                        for key in wh_accum:
                            wh_accum[key] += batch_wh[key]

                except StopIteration:
                    break

        avg_val_loss = np.mean(val_losses)

        # Track loss in appropriate list based on validation type
        if validation_type == 'masked':
            self.epoch_val_losses_masked.append(avg_val_loss)
        elif validation_type == 'unmasked':
            self.epoch_val_losses_unmasked.append(avg_val_loss)

        val_type_label = f'{validation_type.upper()} validation'
        self._log(f"Epoch {len(self.epochs)}/{self.config['epochs']}: {val_type_label} loss: {avg_val_loss:.4f}")

        # Compute epoch-level Wh metrics from accumulated sums
        wh_metrics = None
        if self.wh_metrics_enabled and wh_accum['valid_count'] > 0:
            count = wh_accum['valid_count']
            wh_metrics = {
                'token_bias': wh_accum['signed_diffs_sum'] / count,
                'token_mad': wh_accum['abs_diffs_sum'] / count,
                'wh_mae': wh_accum['wh_abs_error_sum'] / count,
                'wh_mean_ratio': wh_accum['wh_expected_sum'] / max(wh_accum['wh_actual_sum'], 1e-8),
            }

        return avg_val_loss, wh_metrics
    
    def _calculate_loss(
        self,
        output: torch.Tensor,
        y: torch.Tensor,
        target_index: int
    ) -> torch.Tensor:
        """
        Calculate cross-entropy loss for single-output model.

        Args:
            output: Model output logits [batch, seq_len, vocab_size]
            y: Target tensor [batch, seq_len, num_targets] where num_targets >= 2
            target_index: Index of target column in y (0 for electricity, 1 for gas)

        Returns:
            Cross-entropy loss scalar tensor
        """
        # Flatten outputs for CrossEntropyLoss
        output = output.view(-1, output.size(-1))
        
        target = y[:, :, target_index].view(-1)
        
        loss = self.loss_fn(output, target)
        
        return loss

    def _compute_wh_metrics(self, output: torch.Tensor, y: torch.Tensor,
                            target_index: int) -> Dict[str, float]:
        """Compute Wh-space fidelity metrics from logits and targets.

        Computes Tier 1 (token bias, MAD) and Tier 2 (Wh MAE, mean ratio)
        metrics. Returns raw sums so the caller can aggregate across batches
        and compute final means at epoch end.

        Args:
            output: Model output logits [batch, seq_len, vocab_size] (detached)
            y: Target tensor [batch, seq_len, num_targets]
            target_index: Index of target column in y (0=electricity, 1=gas)

        Returns:
            Dict with accumulated sums: signed_diffs_sum, abs_diffs_sum,
            valid_count, wh_abs_error_sum, wh_expected_sum, wh_actual_sum
        """
        n_bins = len(self.bin_means_tensor)

        predicted_tokens = output.argmax(dim=-1)  # [batch, seq_len]
        actual_tokens = y[:, :, target_index].long()  # [batch, seq_len]

        # Exclude special tokens (missing, SOS, error) — indices >= n_bins
        valid_mask = actual_tokens < n_bins

        valid_count = valid_mask.sum().item()
        if valid_count == 0:
            return {
                'signed_diffs_sum': 0.0, 'abs_diffs_sum': 0.0,
                'valid_count': 0, 'wh_abs_error_sum': 0.0,
                'wh_expected_sum': 0.0, 'wh_actual_sum': 0.0,
            }

        # Tier 1: Token-level bias and MAD
        pred_valid = predicted_tokens[valid_mask].float()
        actual_valid = actual_tokens[valid_mask].float()
        diffs = pred_valid - actual_valid

        # Tier 2: Wh-space MAE and mean ratio
        probs = torch.softmax(output, dim=-1)  # [batch, seq_len, vocab_size]
        value_probs = probs[:, :, :n_bins]  # [batch, seq_len, n_bins]
        # Renormalise over value bins only (exclude special token probability mass)
        value_probs = value_probs / value_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        expected_wh = torch.matmul(value_probs, self.bin_means_tensor)  # [batch, seq_len]
        actual_wh = self.bin_means_tensor[actual_tokens.clamp(max=n_bins - 1)]  # [batch, seq_len]

        expected_valid = expected_wh[valid_mask]
        actual_wh_valid = actual_wh[valid_mask]

        return {
            'signed_diffs_sum': diffs.sum().item(),
            'abs_diffs_sum': diffs.abs().sum().item(),
            'valid_count': valid_count,
            'wh_abs_error_sum': (expected_valid - actual_wh_valid).abs().sum().item(),
            'wh_expected_sum': expected_valid.sum().item(),
            'wh_actual_sum': actual_wh_valid.sum().item(),
        }

    def _save_loss_diagnostics(self):
        """
        Save loss diagnostics (CSV and plot) for the current output variable.

        Creates files prefixed with output_variable name (e.g., 'electricity_'):
        - {output_variable}_loss_diagnostics.csv: Epoch-by-epoch train/val losses
        - {output_variable}_loss_diagnostics.png: Visual plot of loss curves

        The plot marks the best epoch with a vertical line for easy identification
        of optimal checkpoint.
        """
        # Build DataFrame with all tracked losses
        num_epochs = len(self.epochs)

        # Create data dictionary
        data = {
            'epoch': list(range(1, num_epochs + 1)),
            'train_loss_total': self.epoch_train_losses[:num_epochs],
        }

        # Add validation losses (they should have same length as training)
        # Use the primary validation metric (either masked or unmasked depending on training mode)
        if self.epoch_val_losses_masked:
            val_losses_to_use = self.epoch_val_losses_masked
        elif self.epoch_val_losses_unmasked:
            val_losses_to_use = self.epoch_val_losses_unmasked
        else:
            val_losses_to_use = []

        if len(val_losses_to_use) == num_epochs:
            data['val_loss_total'] = val_losses_to_use

        # Add Wh-space fidelity metric columns if available
        if self.wh_metrics_enabled and len(self.epoch_train_token_bias) == num_epochs:
            data['train_token_bias'] = self.epoch_train_token_bias[:num_epochs]
            data['val_token_bias'] = self.epoch_val_token_bias[:num_epochs]
            data['train_token_mad'] = self.epoch_train_token_mad[:num_epochs]
            data['val_token_mad'] = self.epoch_val_token_mad[:num_epochs]
            data['train_wh_mae'] = self.epoch_train_wh_mae[:num_epochs]
            data['val_wh_mae'] = self.epoch_val_wh_mae[:num_epochs]
            data['train_wh_mean_ratio'] = self.epoch_train_wh_mean_ratio[:num_epochs]
            data['val_wh_mean_ratio'] = self.epoch_val_wh_mean_ratio[:num_epochs]

        # Create DataFrame
        df = pd.DataFrame(data)

        # Save to CSV
        csv_path = os.path.join(self.run_path, f'{self.output_variable}_loss_diagnostics.csv')
        df.to_csv(csv_path, index=False)
        self._log(f"Loss diagnostics saved to: {csv_path}")

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))

        epochs = df['epoch']

        # Plot training losses
        ax.plot(epochs, df['train_loss_total'], 'b-', linewidth=2, label='Train Total', alpha=0.8)

        # Plot validation losses if available
        if 'val_loss_total' in df.columns:
            ax.plot(epochs, df['val_loss_total'], 'r-', linewidth=2, label='Val Total', alpha=0.8)

        # Mark best epoch
        if not np.isnan(self.best_epoch):
            ax.axvline(x=self.best_epoch + 1, color='gray', linestyle='--', alpha=0.5, label=f'Best Epoch ({int(self.best_epoch + 1)})')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'Training and Validation Losses: {self.output_variable}')
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)

        # Save plot
        plot_path = os.path.join(self.run_path, f'{self.output_variable}_loss_diagnostics.png')
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        self._log(f"Loss diagnostics plot saved to: {plot_path}")

    def _save_wh_fidelity_diagnostics(self):
        """Save 2x2 Wh-space fidelity diagnostics plot.

        Creates {output_variable}_wh_fidelity_diagnostics.png with subplots:
        - Top-left: Token Bias vs Epoch (train/val, y=0 ref)
        - Top-right: Token MAD vs Epoch (train/val)
        - Bottom-left: Wh MAE vs Epoch (train/val)
        - Bottom-right: Wh Mean Ratio vs Epoch (train/val, y=1.0 ref)

        Each subplot marks the best CE loss epoch with a vertical dashed line.
        """
        if not self.wh_metrics_enabled or not self.epoch_train_token_bias:
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        epochs = list(range(1, len(self.epoch_train_token_bias) + 1))
        best_epoch_1idx = int(self.best_epoch + 1) if not np.isnan(self.best_epoch) else None

        # Top-left: Token Bias
        ax = axes[0, 0]
        ax.plot(epochs, self.epoch_train_token_bias, 'b-', label='Train', alpha=0.8)
        ax.plot(epochs, self.epoch_val_token_bias, 'r-', label='Val', alpha=0.8)
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        if best_epoch_1idx:
            ax.axvline(x=best_epoch_1idx, color='gray', linestyle='--', alpha=0.5,
                       label=f'Best CE Epoch ({best_epoch_1idx})')
        ax.set_title('Token Bias')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Mean Signed Token Diff')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

        # Top-right: Token MAD
        ax = axes[0, 1]
        ax.plot(epochs, self.epoch_train_token_mad, 'b-', label='Train', alpha=0.8)
        ax.plot(epochs, self.epoch_val_token_mad, 'r-', label='Val', alpha=0.8)
        if best_epoch_1idx:
            ax.axvline(x=best_epoch_1idx, color='gray', linestyle='--', alpha=0.5,
                       label=f'Best CE Epoch ({best_epoch_1idx})')
        ax.set_title('Token MAD')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Mean Absolute Token Diff')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

        # Bottom-left: Wh MAE
        ax = axes[1, 0]
        ax.plot(epochs, self.epoch_train_wh_mae, 'b-', label='Train', alpha=0.8)
        ax.plot(epochs, self.epoch_val_wh_mae, 'r-', label='Val', alpha=0.8)
        if best_epoch_1idx:
            ax.axvline(x=best_epoch_1idx, color='gray', linestyle='--', alpha=0.5,
                       label=f'Best CE Epoch ({best_epoch_1idx})')
        ax.set_title('Wh MAE')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Mean Absolute Error (Wh)')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

        # Bottom-right: Wh Mean Ratio
        ax = axes[1, 1]
        ax.plot(epochs, self.epoch_train_wh_mean_ratio, 'b-', label='Train', alpha=0.8)
        ax.plot(epochs, self.epoch_val_wh_mean_ratio, 'r-', label='Val', alpha=0.8)
        ax.axhline(y=1.0, color='gray', linestyle='-', alpha=0.3)
        if best_epoch_1idx:
            ax.axvline(x=best_epoch_1idx, color='gray', linestyle='--', alpha=0.5,
                       label=f'Best CE Epoch ({best_epoch_1idx})')
        ax.set_title('Wh Mean Ratio')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('E[Wh_predicted] / E[Wh_actual]')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

        fig.suptitle(f'Wh-Space Fidelity Metrics: {self.output_variable}', fontsize=14)
        plt.tight_layout()

        plot_path = os.path.join(self.run_path, f'{self.output_variable}_wh_fidelity_diagnostics.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        self._log(f"Wh fidelity diagnostics plot saved to: {plot_path}")

    def _save_model(self, model_path: str, model_type: str = "checkpoint") -> bool:
        """Save the full model to disk.

        Args:
            model_path: Path to save the model
            model_type: Type of checkpoint (for logging)

        Returns:
            bool: True if the save succeeded, False otherwise
        """
        try:
            torch.save(self.model, model_path)
            self._log(f"Saved {model_type} model: {model_path}")
            return True
        except Exception as e:
            self._log(f"Failed to save {model_type} model: {e}", level='error')
            if self.logger:
                self.logger.log_error(f'Model save failed: {model_type}', {
                    'error': str(e), 'path': model_path,
                })
            return False

    def _update_best_models(self, train_loss: float, val_loss: float, epoch: int):
        """
        Update best models and save checkpoints if losses improved.

        Saves checkpoints when:
        - Training loss improves (saves best_train checkpoint)
        - Validation loss improves (saves best_val checkpoint, resets early stopping)

        Args:
            train_loss: Training loss for current epoch
            val_loss: Validation loss for current epoch
            epoch: Current epoch number
        """
        # Store current epoch for error logging
        self.current_epoch = epoch

        # Update best training loss
        if train_loss < self.best_train_loss:
            self.best_train_loss = train_loss
            success = self._save_model(self.best_train_model_path, "best_train")
            if not success:
                self._log("Best training model checkpoint failed - continuing training", level='warning')

        # Update best validation loss
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.epochs_without_improvement = 0
            self.best_epoch = epoch
            success = self._save_model(self.best_val_model_path, "best_val")
            if success:
                self._log("Validation loss improved, saved best model weights.")
            else:
                self._log("Validation loss improved but failed to save checkpoint - continuing training", level='warning')
        else:
            self.epochs_without_improvement += 1
            self._log(f"No improvement in validation for {self.epochs_without_improvement} epochs.")
    
    def _should_stop_early(self) -> bool:
        """
        Check if early stopping criteria are met.

        Returns:
            True if epochs without validation improvement >= patience threshold
        """
        return self.epochs_without_improvement >= self.config['patience']
    
    def load_best_model(self, model_type: str = 'val'):
        """
        Load the best model weights back into self.model.

        Args:
            model_type: Which checkpoint to load:
                - 'val': Best validation loss checkpoint (recommended for inference)
                - 'train': Best training loss checkpoint

        Note:
            Loads weights with proper device mapping to self.device.
            Logs warning if checkpoint file not found.
        """
        if model_type == 'val':
            model_path = self.best_val_model_path
            self._log('Loading best validation model weights')
        else:
            model_path = self.best_train_model_path
            self._log('Loading best training model weights')

        if os.path.exists(model_path):
            # Load with proper device mapping
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self._log(f"Loaded best {model_type} model weights from {model_path} to {self.device}")
        else:
            self._log(f"Model weights not found at {model_path}", level='warning')


def create_optimizer_and_loss(model: nn.Module, config: Dict[str, Any]) -> Tuple[torch.optim.Optimizer, nn.Module]:
    """
    Create optimizer and loss function.
    
    Args:
        model: PyTorch model
        config: Configuration dictionary
        
    Returns:
        Tuple of (optimizer, loss_function)
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['optimizer']['learning_rate'])
    loss_fn = nn.CrossEntropyLoss()
    
    return optimizer, loss_fn