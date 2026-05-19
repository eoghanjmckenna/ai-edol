"""
PyTorch Dataset classes and DataLoader utilities.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, Any


def shuffle_tensors(tensor1: torch.Tensor, tensor2: torch.Tensor, axis: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Shuffle two tensors consistently along specified axis.
    
    Args:
        tensor1: First tensor
        tensor2: Second tensor  
        axis: Axis along which to shuffle
        
    Returns:
        Tuple of shuffled tensors
    """
    # Generate random permutation of indices on same device as tensor1
    indices = torch.randperm(tensor1.size(axis), device=tensor1.device)
    
    # Apply permutation to both tensors (move indices to tensor2's device if needed)
    shuffled_tensor1 = tensor1.index_select(axis, indices)
    shuffled_tensor2 = tensor2.index_select(axis, indices.to(tensor2.device))
    
    return shuffled_tensor1, shuffled_tensor2


class MultiFeatureDataset(Dataset):
    """Dataset for multi-feature sequence data."""
    
    def __init__(self, input_data: torch.Tensor, target_data: torch.Tensor, shuffle: bool = False):
        """
        Initialize dataset.
        
        Args:
            input_data: Input tensor of shape [n_sequences, context_length, n_features]
            target_data: Target tensor of shape [n_sequences, context_length, n_targets]
            shuffle: If True, shuffle the data during initialization
        """
        if shuffle:
            # Shuffle both tensors consistently
            self.input_data, self.target_data = shuffle_tensors(input_data, target_data, axis=0)
        else:
            self.input_data = input_data
            self.target_data = target_data
    
    def __len__(self) -> int:
        return len(self.input_data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.input_data[idx], self.target_data[idx]


def create_dataloaders(
    tensor_data: Dict[str, torch.Tensor], 
    batch_size: int
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for training, validation, and test sets.
    
    Args:
        tensor_data: Dictionary containing split tensor data
        batch_size: Batch size for DataLoaders
        
    Returns:
        Dictionary of DataLoaders
    """
    # Create datasets
    datasets = {
        'train': MultiFeatureDataset(tensor_data['X_train'], tensor_data['y_train']),
        'val': MultiFeatureDataset(tensor_data['X_val'], tensor_data['y_val']),
        'test': MultiFeatureDataset(tensor_data['X_test'], tensor_data['y_test']),
        'final_train_and_val': MultiFeatureDataset(
            tensor_data['X_final_train_and_val'], 
            tensor_data['y_final_train_and_val']
        ),
    }
    
    # Create DataLoaders
    dataloaders = {}
    for name, dataset in datasets.items():
        dataloaders[name] = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True
        )
    
    return dataloaders