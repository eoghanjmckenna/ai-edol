"""
Device detection and management utilities for GPU acceleration.

Handles device selection across different platforms:
- macOS: MPS (Metal Performance Shaders) for Apple Silicon
- Linux/Windows: CUDA for NVIDIA GPUs  
- Fallback: CPU for all systems

Configuration:
- "auto": Automatically detect best available device (default)
- "cpu": Force CPU usage
- "cuda": Force CUDA (fails if not available)
- "mps": Force MPS (fails if not available)
"""
import torch
import logging
from typing import Dict, Any


def detect_best_device(verbose: bool = False) -> str:
    """
    Detect the best available device for PyTorch operations.
    
    Priority order:
    1. CUDA (if available and functional)
    2. MPS (if available and functional) 
    3. CPU (always available)
    
    Args:
        verbose: Whether to log device detection details
        
    Returns:
        String device name: "cuda", "mps", or "cpu"
    """
    # Test CUDA availability and functionality
    if torch.cuda.is_available():
        try:
            # Test basic CUDA operation
            test_tensor = torch.tensor([1.0], device='cuda')
            _ = test_tensor + 1
            if verbose:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                logging.info(f"CUDA available: {gpu_name} ({gpu_memory:.1f}GB)")
            return "cuda"
        except Exception as e:
            logging.warning(f"CUDA available but non-functional: {e}")
    
    # Test MPS availability and functionality (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        try:
            # Test basic MPS operation
            test_tensor = torch.tensor([1.0], device='mps')
            _ = test_tensor + 1
            if verbose:
                logging.info("MPS available and functional (Apple Silicon)")
            return "mps"
        except Exception as e:
            logging.warning(f"MPS available but non-functional: {e}")
    
    # CPU fallback
    if verbose:
        logging.info("Using CPU (no GPU acceleration available)")
    return "cpu"


def get_device_from_config(config: Dict[str, Any], verbose: bool = False) -> torch.device:
    """
    Get PyTorch device based on configuration.
    
    Args:
        config: Configuration dictionary containing 'device' key
        verbose: Whether to log device selection details
        
    Returns:
        torch.device object
        
    Raises:
        RuntimeError: If requested device is not available
    """
    device_config = config.get('device', 'auto')
    
    if device_config == 'auto':
        device_str = detect_best_device(verbose=verbose)
    elif device_config == 'cpu':
        device_str = 'cpu'
        if verbose:
            logging.info("Using CPU (forced by configuration)")
    elif device_config == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        device_str = 'cuda'
        if verbose:
            gpu_name = torch.cuda.get_device_name(0)
            logging.info(f"Using CUDA (forced): {gpu_name}")
    elif device_config == 'mps':
        if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            raise RuntimeError("MPS requested but not available") 
        device_str = 'mps'
        if verbose:
            logging.info("Using MPS (forced): Apple Silicon")
    else:
        raise ValueError(f"Invalid device configuration: {device_config}")
    
    return torch.device(device_str)


def get_memory_info(device: torch.device) -> Dict[str, float]:
    """
    Get memory information for the specified device.
    
    Args:
        device: PyTorch device
        
    Returns:
        Dictionary with memory statistics (in MB)
    """
    if device.type == 'cuda':
        # CUDA memory statistics
        return {
            'total_memory_mb': torch.cuda.get_device_properties(device).total_memory / 1024**2,
            'allocated_memory_mb': torch.cuda.memory_allocated(device) / 1024**2,
            'cached_memory_mb': torch.cuda.memory_reserved(device) / 1024**2,
        }
    elif device.type == 'mps':
        # MPS doesn't provide detailed memory stats, return basic info
        return {
            'total_memory_mb': -1,  # Unknown
            'allocated_memory_mb': -1,  # Unknown  
            'cached_memory_mb': -1,  # Unknown
        }
    else:
        # CPU memory (return empty dict)
        return {}



def move_to_device(obj, device: torch.device):
    """
    Move tensor or model to specified device with error handling.
    
    Args:
        obj: PyTorch tensor, model, or other object with .to() method
        device: Target device
        
    Returns:
        Object moved to device
        
    Raises:
        RuntimeError: If move fails (including CPU fallback failure)
    """
    try:
        return obj.to(device)
    except RuntimeError as e:
        if device.type != 'cpu':
            logging.warning(f"Failed to move object to {device}: {e}. Attempting CPU fallback.")
            try:
                return obj.to('cpu')
            except RuntimeError as cpu_e:
                logging.error(f"CPU fallback also failed: {cpu_e}")
                raise RuntimeError(f"Failed to move object to both {device} and CPU. "
                                 f"Original error: {e}. CPU error: {cpu_e}") from e
        else:
            # Already tried CPU and it failed
            logging.error(f"Failed to move object to CPU: {e}")
            raise RuntimeError(f"Insufficient memory to load model/data. "
                             f"Try reducing model size or batch size. Error: {e}") from e


def log_device_info(device: torch.device, logger=None):
    """
    Log comprehensive device information.
    
    Args:
        device: PyTorch device
        logger: Optional JSON logger for structured logging
    """
    device_info = {
        'device_type': device.type,
        'device_index': device.index if device.index is not None else 0,
    }
    
    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(device)
        device_info.update({
            'gpu_name': props.name,
            'gpu_memory_gb': props.total_memory / 1024**3,
            'compute_capability': f"{props.major}.{props.minor}",
            'multiprocessor_count': props.multi_processor_count,
        })
    elif device.type == 'mps':
        device_info.update({
            'gpu_name': 'Apple Silicon (MPS)',
            'gpu_memory_gb': -1,  # Not available through PyTorch
        })
    
    # Log to both standard logging and JSON logger
    logging.info(f"Using device: {device} ({device_info.get('gpu_name', 'CPU')})")
    
    if logger:
        logger.log_custom_metric('device_info', device_info)