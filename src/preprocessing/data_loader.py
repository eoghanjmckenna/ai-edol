"""
Data loading utilities for SERL survey data and household energy data.
"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional


def load_serl_survey_data(config: Dict[str, Any], puprn: int) -> pd.DataFrame:
    """
    Load SERL survey data for a specific household.
    
    Args:
        config: Configuration dictionary containing paths and settings
        puprn: Household identifier
        
    Returns:
        DataFrame with SERL survey data for the household
    """
    df = pd.read_csv(config['data']['path_to_serl_survey_data'])
    mask = (df['PUPRN'] == puprn)
    my_serl_survey_data = df.loc[mask, config['serl_survey_context']]
    return my_serl_survey_data

def load_mod1_data(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Load module 1 data for a random household.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        DataFrame with household energy data
    """
    import random
    import os
    
    logging.info('Selecting data for random PUPRN.')
    pickle_files = [f for f in os.listdir(config['data']['path_to_module_1_data']) if f.endswith('.pkl')]

    continue_to_load_data = True
    while continue_to_load_data:
        random_file = random.choice(pickle_files)
        
        my_puprn = random_file.split('.')[0]
        
        file_path = os.path.join(config['data']['path_to_module_1_data'], random_file)
        
        mod1_data = pd.read_pickle(file_path)
        
        logging.info(f'PUPRN {my_puprn} selected and data loaded.')

        logging.info('Checking for too much missing data...')

        too_much_missing = any((mod1_data[['Clean_elec_net_Wh','Clean_gas_Wh']].isna().sum() / len(mod1_data)) > config['data']['missing_data_threshold'])
        if too_much_missing:
            logging.info(f'{my_puprn} has too much missing data, going to next choice...')
        else:
            logging.info(f'Missing data check passed for {my_puprn}')
            described = mod1_data[['Clean_elec_net_Wh','Clean_gas_Wh']].describe()
            count_elec = described.loc['count','Clean_elec_net_Wh']
            count_gas = described.loc['count','Clean_gas_Wh']
            percent_elec = count_elec / len(mod1_data)
            percent_gas = count_gas / len(mod1_data)
            mean_elec = described.loc['mean','Clean_elec_net_Wh']
            mean_gas = described.loc['mean','Clean_gas_Wh']
            logging.info(f'Electricity data: count {count_elec}, percent not nan {percent_elec*100}%, mean {mean_elec:.3f}')
            logging.info(f'Gas data: count {count_gas}, percent not nan {percent_gas*100}%, mean {mean_gas:.3f}')
            continue_to_load_data = False
    return mod1_data

def _process_household_data(mod1_data: pd.DataFrame, num_rooms: int, num_occs: int) -> pd.DataFrame:
    """Process household data with calendar and context variables."""
    
    # Create calendar variables
    mod1_data['dow'] = pd.to_datetime(mod1_data['Read_date_effective_local'], format="%Y-%m-%d").dt.dayofweek.values
    mod1_data['month'] = pd.to_datetime(mod1_data['Read_date_effective_local'], format="%Y-%m-%d").dt.month.values
    mod1_data['hh'] = mod1_data['Readings_from_midnight_local']
    
    # Create context variables
    mod1_data['B5'] = num_rooms
    mod1_data['C1'] = num_occs
    
    return mod1_data