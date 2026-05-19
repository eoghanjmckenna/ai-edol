"""
Data preprocessing utilities for sequence creation and tokenization.
"""
import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Tuple, Any
import pickle


def create_sequences(data: torch.Tensor, context_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create sequences from continuous data for training.
    
    Args:
        data: Input tensor
        context_length: Length of sequences to create
        
    Returns:
        Tuple of (input sequences, target sequences)
    """
    # Calculate number of complete sequences possible
    n_sequences = (len(data)) // context_length 
    
    # Truncate data to fit complete sequences
    usable_length = n_sequences * context_length + 1
    truncated_data = data[:usable_length]
    
    x = truncated_data[:-1].view(-1, context_length)
    y = truncated_data[1:].view(-1, context_length)
    
    return x, y

def create_month_sequences(
        household_data: pd.DataFrame,
        context_length: int,
        random_seed: int,
        vocab_size_elec: int,
        vocab_size_gas: int,
) -> List[Dict]:
    """
    Create training sequences from a single household's data, one sequence per calendar month.

    Each sequence has:
    - Fixed length of context_length tokens
    - SOS token at position 0 for elec/gas only (conditioning uses actual values)
    - Random start offset within valid range (may span month boundaries)

    Args:
        household_data: DataFrame with columns for elec_token, gas_token,
            temp_token, solar_token, hh (1-48), dow (0-6), month (1-12)
        context_length: Fixed sequence length (e.g., 1488 = 31 days × 48 half-hours)
        random_seed: Seed for reproducible random start times (can be None)
        vocab_size_elec: Vocabulary size for electricity (SOS token = vocab_size - 1)
        vocab_size_gas: Vocabulary size for gas (SOS token = vocab_size - 1)

    Returns:
        List of sequence dictionaries, each containing:
        - '*_base_x': input tensors of shape (context_length,)
        - '*_base_y': target tensors of shape (context_length,) shifted by 1
        - 'month': calendar month (1-12)
        - 'start_idx': global start index in household_data
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    sequences = []
    n_total = len(household_data)

    # Need at least context_length + 1 timesteps (extra 1 for target shift)
    if n_total < context_length + 1:
        return sequences

    # Get unique months in the data
    months = household_data['month'].unique()

    for month in months:
        month_mask = household_data['month'] == month
        month_indices = np.where(month_mask)[0]

        if len(month_indices) == 0:
            continue

        # Calculate valid start range: sequence should be centered on this month
        # but can extend into adjacent months to fill context_length
        month_start = month_indices[0]

        # Start can be anywhere from (month_start - context_length + 1) to month_start
        # but must be >= 0 and allow full context_length + 1 (extra 1 for target shift)
        earliest_start = max(0, month_start - context_length + 1)
        latest_start = min(month_start, n_total - context_length - 1)

        if latest_start < earliest_start:
            continue  # Not enough data around this month

        # Random start within valid range
        start_idx = np.random.randint(earliest_start, latest_start + 1)

        # Extract sequence data: need context_length + 1 rows for x and shifted y
        seq_data = household_data.iloc[start_idx:start_idx + context_length + 1].copy()
        seq_data = seq_data.reset_index(drop=True)

        # Set SOS for first timestep for elec and gas only
        # Conditioning variables (temp, solar, hh, dow, month) keep actual values
        seq_data.loc[0, 'elec_token'] = vocab_size_elec - 1
        seq_data.loc[0, 'gas_token'] = vocab_size_gas - 1

        # x = first context_length elements (indices 0 to context_length-1)
        # y = last context_length elements (indices 1 to context_length), shifted by 1
        sequences.append({
            'elec_base_x': torch.tensor(seq_data['elec_token'].iloc[:-1].values, dtype=torch.long),
            'gas_base_x': torch.tensor(seq_data['gas_token'].iloc[:-1].values, dtype=torch.long),
            'temp_base_x': torch.tensor(seq_data['temp_token'].iloc[:-1].values, dtype=torch.long),
            'solar_base_x': torch.tensor(seq_data['solar_token'].iloc[:-1].values, dtype=torch.long),
            'hh_base_x': torch.tensor(seq_data['hh'].iloc[:-1].values, dtype=torch.long),
            'dow_base_x': torch.tensor(seq_data['dow'].iloc[:-1].values, dtype=torch.long),
            'month_base_x': torch.tensor(seq_data['month'].iloc[:-1].values, dtype=torch.long),
            'elec_base_y': torch.tensor(seq_data['elec_token'].iloc[1:].values, dtype=torch.long),
            'gas_base_y': torch.tensor(seq_data['gas_token'].iloc[1:].values, dtype=torch.long),
            'month': month,
            'start_idx': start_idx,
        })

    return sequences


def process_multi_household_data(
    list_of_mod_1_data: List[pd.DataFrame], 
    training_config: Dict[str, Any], 
    vocab_size_elec: int, 
    vocab_size_gas: int,
    vocab_size_temp: int,
    vocab_size_solar: int
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    Process multiple household data into training tensors.
    Expects pre-tokenized temp_token and solar_token columns from universal tokenizer.
    
    Args:
        list_of_mod_1_data: List of household dataframes with pre-tokenized temp/solar data
        training_config: Configuration dictionary
        vocab_size_elec: Vocabulary size for electricity (from tokenizer)
        vocab_size_gas: Vocabulary size for gas (from tokenizer)
        vocab_size_temp: Vocabulary size for temperature (from tokenizer)
        vocab_size_solar: Vocabulary size for solar (from tokenizer)
        
    Returns:
        Tuple of (processed tensors dict, vocab info dict)
    """
    # Add end-of-sequence tokens to each household
    for mod1_data in list_of_mod_1_data:
        mod1_data.loc[len(mod1_data)] = pd.Series({
            'elec_token': vocab_size_elec - 1,
            'gas_token': vocab_size_gas - 1,
            'temp_token': vocab_size_temp - 1,
            'solar_token': vocab_size_solar - 1,
            'dow': 7,
            'month': 0,
            'hh': 0,
            'B5': 0,
            'C1': 0,
        })
    
    # Concatenate all household data
    multih_df = pd.concat(list_of_mod_1_data, axis=0, ignore_index=True)
    
    # Create base tensors
    base_tensors = _create_base_tensors(multih_df)
    
    # Create sequences
    sequence_data = _create_sequence_data(base_tensors, training_config['model']['context_length'])
    
    # Stack tensors (shuffling will be done on-demand in DataLoader)
    processed_tensors = _stack_base_tensors(sequence_data)
    
    # Split into train/val/test
   # split_tensors = _split_tensors(processed_tensors) # No longer used due to sharding approach
    
    # return split_tensors

    return processed_tensors

def process_multi_household_data_with_conditioning(
    household_data_pairs: List[Tuple[pd.DataFrame, pd.Series]],
    config: dict,
    vocab_size_elec: int,
    vocab_size_gas: int,
) -> Dict[str, torch.Tensor]:
    """
    Process multiple households with their conditioning contexts into training tensors.

    Creates month-based sequences where each calendar month produces one sequence per household.
    Each sequence has SOS tokens for elec/gas at position 0, with actual values for conditioning.

    Dynamically handles variable numbers of conditioning variables based on config.

    For masked approach (14 vars): num_features = 7 + 14 = 21
    For unmasked approach (e.g., 6 vars): num_features = 7 + 6 = 13

    Args:
        household_data_pairs: List of (energy_data DataFrame, conditioning_vars Series) tuples
        config: Training config dict with 'model.context_length' and optional 'random_seed'
        vocab_size_elec: Vocabulary size for electricity (SOS token = vocab_size - 1)
        vocab_size_gas: Vocabulary size for gas (SOS token = vocab_size - 1)

    Returns:
        {
            'X_base': [N, context_length, num_features],
            'y_base': [N, context_length, 2]
        }

    Raises:
        ValueError: If no household data pairs provided or no sequences could be created
    """
    X_sequences = []
    y_sequences = []

    context_length = config['model']['context_length']
    base_random_seed = config.get('random_seed', None)

    # Determine number of conditioning variables from first household
    if not household_data_pairs:
        raise ValueError("No household data pairs provided")

    first_conditioning_vars = household_data_pairs[0][1]
    num_conditioning_vars = len(first_conditioning_vars)
    num_features = 7 + num_conditioning_vars  # 7 temporal + conditioning vars

    for idx, (energy_data, conditioning_vars) in enumerate(household_data_pairs):
        # Compute random seed for this household (None if base seed is None)
        random_seed = base_random_seed + idx if base_random_seed is not None else None

        # Create sequences for this household
        sequences = create_month_sequences(
            household_data=energy_data,
            context_length=context_length,
            vocab_size_elec=vocab_size_elec,
            vocab_size_gas=vocab_size_gas,
            random_seed=random_seed,
        )

        for seq in sequences:
            # Build feature tensor [context_length, num_features]
            features = torch.zeros(context_length, num_features, dtype=torch.long)

            # Positions 0-6: Energy/temporal features
            features[:, 0] = seq['elec_base_x']
            features[:, 1] = seq['gas_base_x']
            features[:, 2] = seq['temp_base_x']
            features[:, 3] = seq['solar_base_x']
            features[:, 4] = seq['hh_base_x']
            features[:, 5] = seq['dow_base_x']
            features[:, 6] = seq['month_base_x']

            # Positions 7+: Conditioning variables (constant across sequence, dynamically assigned)
            for var_idx, var_value in enumerate(conditioning_vars):
                features[:, 7 + var_idx] = var_value

            # Targets [context_length, 2]
            targets = torch.stack([seq['elec_base_y'], seq['gas_base_y']], dim=1)

            X_sequences.append(features)
            y_sequences.append(targets)

    if not X_sequences:
        raise ValueError(
            "No sequences could be created from the provided household data. "
            "This may indicate that households have insufficient data "
            f"(need at least {context_length + 1} timesteps per household)."
        )

    return {
        'X_base': torch.stack(X_sequences, dim=0),  # [n_sequences, context_length, num_features]
        'y_base': torch.stack(y_sequences, dim=0),  # [n_sequences, context_length, num_targets]
    }

def _create_base_tensors(multih_df: pd.DataFrame) -> Dict[str, torch.Tensor]:
    """Create base tensors from processed dataframe with pre-tokenized temp/solar data."""
    return {
        'elec_base': torch.tensor(multih_df['elec_token'], dtype=torch.long),
        'gas_base': torch.tensor(multih_df['gas_token'], dtype=torch.long),
        'temp_base': torch.tensor(multih_df['temp_token'], dtype=torch.long),
        'solar_base': torch.tensor(multih_df['solar_token'], dtype=torch.long),
        'hh_base': torch.tensor(multih_df['hh'], dtype=torch.long),
        'dow_base': torch.tensor(multih_df['dow'], dtype=torch.long),
        'month_base': torch.tensor(multih_df['month'], dtype=torch.long),
        'B5_base': torch.tensor(multih_df['B5'], dtype=torch.long),
        'C1_base': torch.tensor(multih_df['C1'], dtype=torch.long),
    }


def _create_sequence_data(base_tensors: Dict[str, torch.Tensor], context_length: int) -> Dict[str, torch.Tensor]:
    """Create sequences from base tensors."""
    sequence_data = {}
    
    # Create sequences for each tensor
    for name, tensor in base_tensors.items():
        x, y = create_sequences(data=tensor, context_length=context_length)
        sequence_data[f'{name}_x'] = x
        if name in ['elec_base', 'gas_base']:  # Only need targets for output features
            sequence_data[f'{name}_y'] = y
    
    return sequence_data


def _stack_base_tensors(sequence_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Stack input tensors WITHOUT pre-shuffling for memory efficiency."""
    # Stack input features
    X_base = torch.stack([
        sequence_data['elec_base_x'],   # 0
        sequence_data['gas_base_x'],    # 1
        sequence_data['temp_base_x'],   # 2
        sequence_data['solar_base_x'],  # 3
        sequence_data['hh_base_x'],     # 4
        sequence_data['dow_base_x'],    # 5
        sequence_data['month_base_x'],  # 6
        sequence_data['B5_base_x'],     # 7
        sequence_data['C1_base_x'],     # 8
    ], dim=2)
    
    # Stack target features
    y_base = torch.stack([
        sequence_data['elec_base_y'],
        sequence_data['gas_base_y'],
    ], dim=2)
    
    return {
        'X_base': X_base,
        'y_base': y_base,
    }

# _split_tensors no longer used in sharding approach
# def _split_tensors(processed_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
#     """Split tensors into train/val/test sets."""
#     X_shuffled = processed_tensors['X_base_shuffled']
#     y_shuffled = processed_tensors['y_base_shuffled']
    
#     n1 = int(X_shuffled.size(0) * 0.8)
#     n2 = int(X_shuffled.size(0) * 0.9)
    
#     return {
#         'X_train': X_shuffled[:n1],
#         'X_val': X_shuffled[n1:n2],
#         'X_test': X_shuffled[n2:],
#         'X_final_train_and_val': torch.clone(X_shuffled),
#         'X_gen': torch.clone(processed_tensors['X_base']),
#         'y_train': y_shuffled[:n1],
#         'y_val': y_shuffled[n1:n2],
#         'y_test': y_shuffled[n2:],
#         'y_final_train_and_val': torch.clone(y_shuffled),
#         'y_gen': torch.clone(processed_tensors['y_base']),
#     }