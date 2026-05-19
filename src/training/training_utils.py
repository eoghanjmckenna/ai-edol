"""
Training utilities and helper functions.
"""
import os
import json
import time
import random
import subprocess
import numpy as np
import psutil
import torch
from typing import Dict, Any


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def save_experiment_config(config: Dict[str, Any], run_path: str):
    """
    Save experiment configuration to JSON file.

    Args:
        config: Configuration dictionary
        run_path: Path to save configuration
    """
    with open(os.path.join(run_path, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)


class SimpleResourceMonitor:
    """Lightweight resource monitoring for training and inference processes."""

    def __init__(self):
        """Initialize resource monitor."""
        self.process = psutil.Process()
        self.start_time = time.time()
        self.baseline_cpu = 0.0
        self.peak_memory = 0.0
        self.peak_gpu_memory = 0.0
        self.peak_gpu_utilization = 0.0

        # Initial CPU reading (psutil needs time between calls)
        self.process.cpu_percent()

        # Reset PyTorch's peak memory tracking for accurate measurement
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Test GPU utilization availability
        self.gpu_util_available = self._test_gpu_utilization_availability()

    def _test_gpu_utilization_availability(self) -> bool:
        """Test if GPU utilization monitoring is available."""
        try:
            if torch.cuda.is_available():
                # Try nvidia-smi for CUDA
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=2
                )
                return result.returncode == 0
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                # powermetrics requires sudo in practice; rely on memory monitoring for MPS
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            pass
        return False

    def _get_gpu_utilization(self) -> float:
        """Get current GPU utilization percentage."""
        if not self.gpu_util_available:
            return 0.0

        try:
            if torch.cuda.is_available():
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=1
                )
                if result.returncode == 0:
                    gpu_utils = result.stdout.strip().split('\n')
                    if gpu_utils and gpu_utils[0]:
                        return float(gpu_utils[0])
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError, ValueError):
            pass
        return 0.0

    def get_current_usage(self) -> Dict[str, float]:
        """
        Get current resource usage.

        Returns:
            Dictionary with current resource metrics
        """
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        current_gpu_memory = 0.0
        peak_gpu_memory_allocated = 0.0

        if torch.cuda.is_available():
            current_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            peak_gpu_memory_allocated = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB

        current_gpu_utilization = self._get_gpu_utilization()

        # Update peaks
        self.peak_memory = max(self.peak_memory, current_memory)
        self.peak_gpu_memory = max(self.peak_gpu_memory, peak_gpu_memory_allocated)
        self.peak_gpu_utilization = max(self.peak_gpu_utilization, current_gpu_utilization)

        return {
            'cpu_percent': self.process.cpu_percent(),
            'memory_mb': current_memory,
            'gpu_memory_mb': current_gpu_memory,
            'gpu_utilization_percent': current_gpu_utilization,
            'peak_memory_mb': self.peak_memory,
            'peak_gpu_memory_mb': self.peak_gpu_memory,
            'peak_gpu_utilization_percent': self.peak_gpu_utilization,
            'elapsed_time': time.time() - self.start_time
        }

    def get_summary(self) -> Dict[str, float]:
        """
        Get resource usage summary.

        Returns:
            Dictionary with summary resource metrics
        """
        final_usage = self.get_current_usage()

        return {
            'total_time_seconds': final_usage['elapsed_time'],
            'peak_memory_mb': self.peak_memory,
            'peak_gpu_memory_mb': self.peak_gpu_memory,
            'peak_gpu_utilization_percent': self.peak_gpu_utilization,
            'final_cpu_percent': final_usage['cpu_percent'],
            'final_memory_mb': final_usage['memory_mb'],
            'final_gpu_memory_mb': final_usage['gpu_memory_mb'],
            'final_gpu_utilization_percent': final_usage['gpu_utilization_percent'],
            'gpu_utilization_available': self.gpu_util_available
        }

    def reset(self):
        """Reset monitoring counters."""
        self.start_time = time.time()
        self.peak_memory = 0.0
        self.peak_gpu_memory = 0.0
        self.peak_gpu_utilization = 0.0
        self.process.cpu_percent()  # Reset CPU counter
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
