"""
Household stratification utilities for balanced sampling.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any


def stratify_sample(my_serl_survey_data: pd.DataFrame, target_sample: Dict[str, List[int]]) -> bool:
    """
    Determine if a household should be excluded from the sample based on stratification requirements.
    
    Args:
        my_serl_survey_data: DataFrame containing household survey data
        target_sample: Dictionary with stratification targets for B5 and C1
        
    Returns:
        True if household should be excluded, False if it should be included
    """
    exclude_from_sample = True
    B5 = my_serl_survey_data.loc[:, 'B5'].values[0]
    C1 = my_serl_survey_data.loc[:, 'C1_new'].values[0]
    
    # Check if values are within target range
    target_B5_list = target_sample['B5']
    target_C1_list = target_sample['C1']
    
    if B5 > len(target_B5_list) - 1:
        return exclude_from_sample
    if C1 > len(target_C1_list) - 1:
        return exclude_from_sample
    
    # Check if we still need households with these characteristics
    remaining_to_fill_B5 = target_B5_list[B5]
    remaining_to_fill_C1 = target_C1_list[C1]
    
    if remaining_to_fill_B5 > 0:
        exclude_from_sample = False
        target_B5_list[B5] += -1
    if remaining_to_fill_C1 > 0:
        exclude_from_sample = False
        target_C1_list[C1] += -1
    
    return exclude_from_sample