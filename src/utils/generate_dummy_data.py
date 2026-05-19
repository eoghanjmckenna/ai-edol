#!/usr/bin/env python3
"""
AI-EDOL Dummy Data Generation Script

Generates dummy SERL Observatory data for local development with structural fidelity.
Creates EPC data, SERL survey data, and Module 1 time series data.

Usage:
    python src/utils/generate_dummy_data.py --num_households 100 --year 2023 --config path_to_config.yaml
"""

import argparse
import random
import string
import yaml
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import numpy as np


def generate_puprn() -> str:
    """Generate 8-character alphanumeric PUPRN identifier."""
    chars = string.ascii_uppercase + string.digits
    return ''.join(random.choices(chars, k=8))


def generate_unique_puprns(count: int) -> List[str]:
    """Generate list of unique PUPRN identifiers."""
    puprns = set()
    while len(puprns) < count:
        puprns.add(generate_puprn())
    return list(puprns)


def load_epc_schema() -> List[str]:
    """Load EPC field names from glossary."""
    # Based on actual EPC CSV header from cleaned_certificates.csv
    # Note: Using camelCase naming to match actual TRE data format
    epc_fields = [
        'PUPRN', 'currentEnergyRating', 'POTENTIAL_ENERGY_RATING',
        'CURRENT_ENERGY_EFFICIENCY', 'POTENTIAL_ENERGY_EFFICIENCY',
        'PROPERTY_TYPE', 'BUILT_FORM', 'INSPECTION_DATE', 'LOCAL_AUTHORITY',
        'CONSTITUENCY', 'COUNTY', 'LODGEMENT_DATE', 'TRANSACTION_TYPE',
        'ENVIRONMENT_IMPACT_CURRENT', 'ENVIRONMENT_IMPACT_POTENTIAL',
        'ENERGY_CONSUMPTION_CURRENT', 'ENERGY_CONSUMPTION_POTENTIAL',
        'CO2_EMISSIONS_CURRENT', 'CO2_EMISS_CURR_PER_FLOOR_AREA',
        'CO2_EMISSIONS_POTENTIAL', 'LIGHTING_COST_CURRENT',
        'LIGHTING_COST_POTENTIAL', 'HEATING_COST_CURRENT',
        'HEATING_COST_POTENTIAL', 'HOT_WATER_COST_CURRENT',
        'HOT_WATER_COST_POTENTIAL', 'totalFloorArea', 'ENERGY_TARIFF',
        'MAINS_GAS_FLAG', 'FLOOR_LEVEL', 'FLAT_TOP_STOREY',
        'FLAT_STOREY_COUNT', 'MAIN_HEATING_CONTROLS', 'MULTI_GLAZE_PROPORTION',
        'GLAZED_TYPE', 'GLAZED_AREA', 'EXTENSION_COUNT', 'NUMBER_HABITABLE_ROOMS',
        'NUMBER_HEATED_ROOMS', 'LOW_ENERGY_LIGHTING', 'NUMBER_OPEN_FIREPLACES',
        'HOTWATER_DESCRIPTION', 'HOT_WATER_ENERGY_EFF', 'HOT_WATER_ENV_EFF',
        'FLOOR_DESCRIPTION', 'FLOOR_ENERGY_EFF', 'FLOOR_ENV_EFF',
        'WINDOWS_DESCRIPTION', 'WINDOWS_ENERGY_EFF', 'WINDOWS_ENV_EFF',
        'WALLS_DESCRIPTION', 'WALLS_ENERGY_EFF', 'WALLS_ENV_EFF',
        'SECONDHEAT_DESCRIPTION', 'SHEATING_ENERGY_EFF', 'SHEATING_ENV_EFF',
        'ROOF_DESCRIPTION', 'ROOF_ENERGY_EFF', 'ROOF_ENV_EFF',
        'MAINHEAT_DESCRIPTION', 'MAINHEAT_ENERGY_EFF', 'MAINHEAT_ENV_EFF',
        'MAINHEATCONT_DESCRIPTION', 'MAINHEATC_ENERGY_EFF', 'MAINHEATC_ENV_EFF',
        'LIGHTING_DESCRIPTION', 'LIGHTING_ENERGY_EFF', 'LIGHTING_ENV_EFF',
        'MAIN_FUEL', 'WIND_TURBINE_COUNT', 'HEAT_LOSS_CORRIDOR',
        'UNHEATED_CORRIDOR_LENGTH', 'FLOOR_HEIGHT', 'PHOTO_SUPPLY',
        'SOLAR_WATER_HEATING_FLAG', 'MECHANICAL_VENTILATION',
        'LOCAL_AUTHORITY_LABEL', 'CONSTITUENCY_LABEL', 'POSTTOWN',
        'CONSTRUCTION_AGE_BAND', 'LODGEMENT_DATETIME', 'TENURE',
        'FIXED_LIGHTING_OUTLETS_COUNT', 'LOW_ENERGY_FIXED_LIGHT_COUNT'
    ]
    return epc_fields


def generate_epc_data(puprns: List[str]) -> pd.DataFrame:
    """Generate EPC data for all households."""
    epc_fields = load_epc_schema()
    
    # Define value ranges and options
    energy_ratings = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    property_types = ['House', 'Flat', 'Maisonette', 'Bungalow']
    built_forms = ['Detached', 'Semi-Detached', 'Mid-Terrace', 'End-Terrace']
    efficiency_ratings = ['Very Good', 'Good', 'Average', 'Poor', 'Very Poor']
    yes_no_flags = ['Y', 'N']
    fuel_types = ['mains gas (not community)', 'electricity', 'oil', 'solid fuel']
    tenure_types = ['Owner-occupied', 'Rented (private)', 'Rented (social)']
    
    data = []
    for puprn in puprns:
        row = {'PUPRN': puprn}
        
        for field in epc_fields[1:]:  # Skip PUPRN as it's already set
            if 'RATING' in field and 'EFFICIENCY' not in field:
                if random.random() < 0.3:
                    row[field] = np.nan
                else:
                    row[field] = random.choice(energy_ratings)
            elif 'EFFICIENCY' in field:
                row[field] = random.randint(20, 100)
            elif field == 'PROPERTY_TYPE':
                row[field] = random.choice(property_types)
            elif field == 'BUILT_FORM':
                row[field] = random.choice(built_forms)
            elif 'DATE' in field:
                base_date = datetime(2020, 1, 1)
                random_days = random.randint(0, 1000)
                row[field] = (base_date + timedelta(days=random_days)).strftime('%Y-%m-%d')
            elif field == 'totalFloorArea':
                if random.random() < 0.3:
                    row[field] = np.nan
                else:
                    row[field] = round(random.uniform(30.0, 300.0), 1)
            elif field == 'FLOOR_HEIGHT':
                row[field] = round(random.uniform(2.0, 3.5), 2)
            elif 'COST' in field:
                row[field] = random.randint(50, 2000)
            elif 'EMISSIONS' in field:
                row[field] = round(random.uniform(0.5, 10.0), 1)
            elif 'CONSUMPTION' in field:
                row[field] = random.randint(50, 500)
            elif 'ENVIRONMENT_IMPACT' in field:
                row[field] = random.randint(20, 100)
            elif 'FLAG' in field:
                row[field] = random.choice(yes_no_flags)
            elif 'ENERGY_EFF' in field or 'ENV_EFF' in field:
                row[field] = random.choice(efficiency_ratings)
            elif field == 'MAIN_FUEL':
                row[field] = random.choice(fuel_types)
            elif field == 'TENURE':
                row[field] = random.choice(tenure_types)
            elif 'COUNT' in field or 'NUMBER' in field:
                row[field] = random.randint(0, 10)
            elif 'PROPORTION' in field or 'SUPPLY' in field:
                row[field] = round(random.uniform(0.0, 100.0), 1)
            elif 'DESCRIPTION' in field:
                row[field] = f"Sample {field.lower().replace('_', ' ')}"
            elif 'LABEL' in field or field in ['COUNTY', 'POSTTOWN']:
                row[field] = f"Sample {field.lower()}"
            elif field == 'CONSTRUCTION_AGE_BAND':
                age_bands = ['before 1900', '1900-1929', '1930-1949', '1950-1966', 
                           '1967-1975', '1976-1982', '1983-1990', '1991-1995',
                           '1996-2002', '2003-2006', '2007-2011', '2012 onwards']
                row[field] = random.choice(age_bands)
            elif 'DATETIME' in field:
                base_date = datetime(2020, 1, 1)
                random_days = random.randint(0, 1000)
                row[field] = (base_date + timedelta(days=random_days)).strftime('%Y-%m-%d %H:%M:%S')
            else:
                # Default to string for unknown fields
                row[field] = f"Sample_{field}"
        
        data.append(row)
    
    return pd.DataFrame(data)


def load_serl_survey_schema() -> List[str]:
    """Return the SERL survey field schema used for dummy data generation.

    A self-contained list covering the survey variables the pipeline uses:
    conditioning contexts (rooms, occupants, building type/age) and the
    error/edit flags consumed during sharding context validation.
    """
    return [
        'PUPRN', 'Survey_version', 'Recorded_date', 'Collection_method', 'Language',
        'A1', 'A2', 'B1', 'B4', 'B5', 'B5_err', 'B9', 'C1', 'C1_new', 'C5',
        'D1', 'D2', 'D4',
        'A301', 'A302', 'A303', 'A304', 'A305', 'A306', 'A307', 'A308', 'A309', 'A310',
        'A3_Other', 'A3_sum', 'A3_err', 'A301_edit',
    ]


def generate_serl_survey_data(puprns: List[str]) -> pd.DataFrame:
    """Generate SERL survey data for all households."""
    survey_fields = load_serl_survey_schema()
    
    # Define response options
    survey_versions = ['Wave1', 'Wave2', 'Wave3']
    collection_methods = ['Online', 'Postal']
    languages = ['English', 'Welsh', 'Unknown']
    binary_responses = [0, 1]  # No/Yes or similar binary
    multi_choice_responses = [1, 2, 3, 4, 5]  # Various multi-choice scales
    missing_codes = [-1, -2, -9]  # Don't know, No answer, Not applicable
    
    data = []
    for puprn in puprns:
        row = {'PUPRN': puprn}
        
        for field in survey_fields[1:]:  # Skip PUPRN as it's already set
            # Handle specific known fields
            if field == 'Survey_version':
                row[field] = random.choice(survey_versions)
            elif field == 'Recorded_date':
                base_date = datetime(2021, 1, 1)
                random_days = random.randint(0, 730)
                row[field] = (base_date + timedelta(days=random_days)).strftime('%Y-%m-%d')
            elif field == 'Collection_method':
                row[field] = random.choice(collection_methods)
            elif field == 'Language':
                row[field] = random.choice(languages)
            elif field == 'B1':
                # type of accommodation
                row[field] = random.choice([-2,1,2,3,4,5,6])
            elif field == 'B4':
                # tenure
                row[field] = random.choice([-2,1,2,3,4,5])
            elif field == 'B5':
                # Number of rooms (balanced for stratification with 100 households)
                row[field] = random.choice([2, 3, 4])  # 3 values: 2, 3, or 4 rooms
            elif field == 'B5_err':
                # Error flag for number of rooms (no error for dev testing)
                row[field] = 0  # Always 0 for simplicity in dev testing
            elif field == 'B9':
                # when accommodation built
                row[field] = random.randint(-2, 7)
            elif field == 'C1_new':
                # Number of occupants (balanced for stratification)
                row[field] = random.choice([1, 2, 3])  # 3 values: 1, 2, or 3 occupants
            elif field == 'C5':
                # have EV?
                row[field] = random.randint(-2, 2)
            elif field == 'D4':
                # how are you managing financially?
                row[field] = random.randint(-3,5)
            elif field.startswith('A3'):
                if not field.endswith('_err'):
                    # A301 - A310 type of heating related questions
                    row[field] = random.choice([0,1])
                else:
                    # A3_err - error in central heating type questions
                    if random.random() < 0.1:
                        row[field] = True
                    else:
                        row[field] = False
            elif '_text' in field.lower() or '_other' in field.lower():
                # Free text fields
                row[field] = f"Sample text for {field}"
            elif field.endswith('_sum') or field.endswith('_diff'):
                # Derived numeric fields
                row[field] = random.randint(0, 10)
            elif field.endswith('_err') or field.endswith('_edit'):
                # Boolean error/edit flags
                row[field] = random.choice([True, False])
            elif field.startswith('A') and field[1:].isdigit():
                # A-series questions (energy/heating related)
                if random.random() < 0.1:  # 10% missing data
                    row[field] = random.choice(missing_codes)
                else:
                    row[field] = random.choice(multi_choice_responses)
            elif field.startswith('B') and field[1:].isdigit():
                # B-series questions (property related)
                if random.random() < 0.05:  # 5% missing data
                    row[field] = random.choice(missing_codes)
                else:
                    row[field] = random.choice(multi_choice_responses)
            elif field.startswith('C') and field[1:].isdigit():
                # C-series questions (household composition)
                if random.random() < 0.05:  # 5% missing data
                    row[field] = random.choice(missing_codes)
                else:
                    row[field] = random.randint(0, 15)  # Can be larger numbers for people counts
            elif field.startswith('D') and field[1:].isdigit():
                # D-series questions (demographics)
                if random.random() < 0.15:  # 15% missing data (more sensitive)
                    row[field] = random.choice(missing_codes)
                else:
                    row[field] = random.choice(multi_choice_responses)
            elif field.endswith('01') or field.endswith('02') or field.endswith('03'):
                # Multi-part questions (often binary)
                row[field] = random.choice(binary_responses)
            else:
                # Default handling for any other fields
                if random.random() < 0.08:  # 8% missing data overall
                    row[field] = random.choice(missing_codes)
                else:
                    row[field] = random.choice(multi_choice_responses)
        
        data.append(row)
    
    return pd.DataFrame(data)

def generate_participant_summary_data(puprns: List[str]) -> pd.DataFrame:
    """Generate participant summary data for all households"""
    fields = ['PUPRN', 'Region', 'IMD_quintile']

    # define response options
    regions = ['NORTH EAST',
                'NORTH WEST',
                'YORKSHIRE AND THE HUMBER',
                'EAST MIDLANDS',
                'WEST MIDLANDS',
                'EAST OF ENGLAND',
                'LONDON',
                'SOUTH EAST',
                'SOUTH WEST',
                'SCOTLAND',
                'WALES']
    data = []
    for puprn in puprns:
        row = {'PUPRN': puprn}
        for field in fields[1:]:
            if field == 'Region':
                row[field] = random.choice(regions)
            elif field == 'IMD_quintile':
                row[field] = random.randint(1,5)
        data.append(row)
    return pd.DataFrame(data)

def generate_follow_up_survey(puprns: List[str]) -> pd.DataFrame:
    """Generate SERL follow-up survey"""
    fields = ['PUPRN', 
              'A1_corr_C', # During this winter, to what temperature do you set your heating controller for late afternoons or evenings?
              'A1_err',
              'B3_1_yes', # Does your accommodation have solar panels for electricity
              'B3_4_yes', # Does your accommodation have EV charging point
              'C1', # Select option that corresponds with your recent gross household annual income?
              'D4', # Does anyone in your household work in the following ways?
              'D5', # How many plug-in electric vehicles does your household have?
              ]
    # define response options
    incomes = ['Below £10,000',
     '£10,001 to £20,000',
     '£20,001 to £30,000',
     '£30,001 to £40,000',
     '£40,001 to £50,000',
      '£50,001 to £60,000',
      '£60,001 to £70,000',
      '£70,001 to £80,000',
      '£80,001 to £90,000',
       '£90,0001 to £100,000',
       'Above £100,000',
               'Prefer not to answer']
    yes_or_no = ['Yes', 'No', 'No response']
    wfh = ['Always work from home',
     'Sometimes work from home',
     'Never work from home',
     'Not applicable /prefer not to say']
    evs = ['0',
     '1',
     '2',
     '3 or more',
     "Don't know"]

    data = []
    for puprn in puprns:
        row = {'PUPRN': puprn}
        for field in fields[1:]:
            if field == 'A1_corr_C':
                if random.random() < 0.4:
                    row[field] = np.nan
                else:
                    row[field] = round(random.uniform(15,25), 1)
            elif field == 'A1_err':
                if random.random() < 0.1:
                    row[field] = True 
                else:
                    row[field] = False
            elif field.startswith('B3'):
                if random.random() < 0.4:
                    row[field] = np.nan
                else:
                    row[field] = random.choice(yes_or_no)
            elif field == 'C1':
                if random.random() < 0.4:
                    row[field] = np.nan
                else:
                    row[field] = random.choice(incomes)
            elif field == 'D4':
                if random.random() < 0.4:
                    row[field] = np.nan
                else:
                    row[field] = random.choice(wfh)
            elif field == 'D5':
                if random.random() < 0.4:
                    row[field] = np.nan
                else:
                    row[field] = random.choice(evs)
        data.append(row)
    return pd.DataFrame(data)

def generate_list_of_exporters(puprns: List[str]) -> pd.DataFrame:
    data = []
    for puprn in puprns:
        if random.random() < 0.07:
            data.append(puprn)
        
    return pd.DataFrame(data)

def generate_module1_data(puprn: str, year: int, household_profile: dict = None) -> pd.DataFrame:
    """
    Generate Module 1 time series data for a single household with realistic patterns.

    The generated data includes:
    - Seasonal patterns (higher electricity in winter for lighting, higher gas in winter for heating)
    - Daily patterns (morning and evening peaks for electricity)
    - Realistic mean and variance matching typical UK household consumption
    - Weather correlated with season

    Args:
        puprn: Household identifier
        year: Year to generate data for
        household_profile: Optional dict with household-specific parameters:
            - base_elec_mean: Base electricity consumption (default 175 Wh)
            - base_gas_mean: Base gas consumption when heating is on (default 1500 Wh)
            - elec_variance_scale: Multiplier for electricity variance (default 1.0)

    Returns:
        DataFrame with half-hourly data for the year
    """
    # Default household profile
    if household_profile is None:
        household_profile = {}

    # Household-specific parameters (with some random variation)
    base_elec_mean = household_profile.get('base_elec_mean', 175 + random.gauss(0, 50))
    base_gas_mean = household_profile.get('base_gas_mean', 1500 + random.gauss(0, 300))
    elec_variance_scale = household_profile.get('elec_variance_scale', 0.8 + random.random() * 0.4)

    # Generate all half-hourly timestamps for the year
    start_date = datetime(year, 1, 1, 0, 30, tzinfo=timezone.utc)  # First reading at 00:30

    data = []
    index = []
    current_time = start_date
    reading_num = 1

    # Generate 17,520 records (365 days * 48 half-hours per day)
    for day in range(365):
        # Calculate month (1-12) and day of year for seasonal patterns
        current_date = start_date + timedelta(days=day)
        month = current_date.month
        day_of_year = day + 1

        # Seasonal multipliers using sinusoidal pattern
        # Electricity: higher in winter (more lighting), peak in Dec/Jan
        # Using cosine so peak is at day 0 (Jan 1)
        elec_seasonal = 1.0 + 0.3 * np.cos(2 * np.pi * (day_of_year - 15) / 365)

        # Gas: much higher in winter (heating), zero/minimal in summer
        # Peak heating in Jan, minimal Jun-Aug
        gas_seasonal_raw = np.cos(2 * np.pi * (day_of_year - 15) / 365)
        # Transform: positive in winter, zero in summer
        gas_seasonal = max(0, gas_seasonal_raw) * 2  # 0 in summer, ~2 in winter peak

        # Temperature: correlated with season (cold in winter, warm in summer)
        base_temp = 10 + 8 * np.cos(2 * np.pi * (day_of_year - 200) / 365)  # Peak warmth ~July
        daily_temp_variation = random.gauss(0, 3)  # Day-to-day variation

        # Solar radiation: correlated with season and time of day
        base_solar = max(0, 400 + 350 * np.cos(2 * np.pi * (day_of_year - 172) / 365))  # Peak ~summer solstice

        for half_hour in range(48):
            # Handle midnight edge case for effective date
            if current_time.hour == 0 and current_time.minute == 0:
                effective_date = (current_time - timedelta(days=1)).date()
            else:
                effective_date = current_time.date()

            hour = current_time.hour + current_time.minute / 60

            # === ELECTRICITY GENERATION ===
            # Daily pattern: morning peak (7-9), evening peak (17-21), low overnight
            if 7 <= hour < 9:
                daily_elec_mult = 1.4  # Morning peak
            elif 17 <= hour < 21:
                daily_elec_mult = 1.6  # Evening peak (highest)
            elif 9 <= hour < 17:
                daily_elec_mult = 0.9  # Daytime (people at work)
            elif 21 <= hour < 23:
                daily_elec_mult = 1.2  # Late evening
            else:
                daily_elec_mult = 0.5  # Overnight (0-7)

            # Combine seasonal and daily patterns
            elec_mean = base_elec_mean * elec_seasonal * daily_elec_mult

            # Add noise with realistic variance (std proportional to mean, ~50-80% of mean)
            elec_std = elec_mean * 0.6 * elec_variance_scale
            elec_value = max(0, random.gauss(elec_mean, elec_std))

            # Occasional high values (appliance usage spikes)
            if random.random() < 0.02:  # 2% chance of spike
                elec_value += random.uniform(500, 2000)

            # === GAS GENERATION ===
            # Gas follows heating pattern: on in winter, off in summer
            if gas_seasonal > 0.1:
                # Heating season: higher in morning and evening
                if 6 <= hour < 9:
                    daily_gas_mult = 1.5  # Morning heating
                elif 17 <= hour < 22:
                    daily_gas_mult = 1.3  # Evening heating
                elif 22 <= hour or hour < 6:
                    daily_gas_mult = 0.8  # Overnight (lower thermostat)
                else:
                    daily_gas_mult = 0.6  # Daytime

                gas_mean = base_gas_mean * gas_seasonal * daily_gas_mult
                gas_std = gas_mean * 0.4
                gas_value = max(0, random.gauss(gas_mean, gas_std))
            else:
                # Summer: mostly zero, occasional hot water usage
                if random.random() < 0.15:  # 15% chance of hot water
                    gas_value = random.uniform(50, 300)
                else:
                    gas_value = 0

            # === TEMPERATURE ===
            # Daily variation: cooler at night, warmer in afternoon
            hour_temp_offset = -3 * np.cos(2 * np.pi * (hour - 14) / 24)  # Peak at 2pm
            temp_value = base_temp + daily_temp_variation + hour_temp_offset + random.gauss(0, 1)

            # === SOLAR RADIATION ===
            # Only during daylight, peak at midday
            sunrise_offset = 2 * np.cos(2 * np.pi * day_of_year / 365)  # Earlier sunrise in summer
            sunrise = 6 - sunrise_offset
            sunset = 18 + sunrise_offset

            if sunrise <= hour <= sunset:
                # Bell curve centered on midday
                solar_daily = np.sin(np.pi * (hour - sunrise) / (sunset - sunrise))
                solar_value = base_solar * solar_daily * (0.5 + 0.5 * random.random())  # Add clouds
            else:
                solar_value = 0

            row = {
                'PUPRN': puprn,
                'Read_date_time_local': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                'Read_date_effective_local': effective_date.strftime('%Y-%m-%d'),
                'Readings_from_midnight_local': reading_num,
                'Clean_elec_net_Wh': round(elec_value, 1),
                'Clean_gas_Wh': round(gas_value, 1),
                'temp_C': round(temp_value, 1),
                'surface_solar_radiation_downwards': round(solar_value, 1),
            }

            index.append(current_time)
            data.append(row)

            # Move to next half hour
            current_time += timedelta(minutes=30)
            reading_num += 1
            if reading_num > 48:
                reading_num = 1

    return pd.DataFrame(data, index=index)


def load_config_from_yaml(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config.get('dummy_data', {})

def main():
    """Main function to generate dummy data."""
    parser = argparse.ArgumentParser(description='Generate SERL Observatory dummy data')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML configuration file')
    parser.add_argument('--num_households', type=int, default=None,
                        help='Number of dummy households to generate')
    parser.add_argument('--year', type=int, default=None,
                        help='Year for time series data')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Load configuration
    config = {}
    if args.config:
        config = load_config_from_yaml(args.config)
        print(f"[CONFIG] Loaded configuration from: {args.config}")
    
    # Override config with command line arguments (if provided)
    num_households = args.num_households if args.num_households is not None else config.get('num_households', 100)
    
    # Handle date_range from config
    if 'date_range' in config:
        start_date = datetime.strptime(config['date_range']['start'], '%Y-%m-%d')
        end_date = datetime.strptime(config['date_range']['end'], '%Y-%m-%d')
        year = start_date.year  # Use start year
    else:
        year = args.year if args.year is not None else 2023
    
    # Get output directory
    if args.output_dir:
        output_dir_str = args.output_dir
    elif 'output' in config and 'folder' in config['output']:
        output_dir_str = config['output']['folder']
    else:
        output_dir_str = 'data/dummy_data'  # Updated default path
    
    # Get seed
    seed = args.seed if args.seed is not None else config.get('seed', 42)
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Create output directory
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating dummy data for {num_households} households...")
    print(f"Output directory: {output_dir}")
    print(f"Year: {year}")
    print(f"Random seed: {seed}")
    
    # Generate unique PUPRNs
    print("Generating unique PUPRN identifiers...")
    puprns = generate_unique_puprns(num_households)
    
    # Generate EPC data
    print("Generating EPC data...")
    epc_data = generate_epc_data(puprns)
    epc_file = output_dir / 'epc_data.csv'
    epc_data.to_csv(epc_file, index=False)
    print(f"  Saved: {epc_file}")
    
    # Generate SERL survey data
    print("Generating SERL survey data...")
    survey_data = generate_serl_survey_data(puprns)
    survey_file = output_dir / 'serl_survey_data.csv'
    survey_data.to_csv(survey_file, index=False)
    print(f"  Saved: {survey_file}")
    
    # Generate SERL participant summary data
    print("Generating SERL participant summary data...")
    summary_data = generate_participant_summary_data(puprns)
    summary_file = output_dir / 'serl_participant_summary_data.csv'
    summary_data.to_csv(summary_file, index=False)
    print(f"  Saved: {summary_file}")
    
    # Generate SERL follow-on survey data
    print("Generating SERL follow-on survey data...")
    followup_data = generate_follow_up_survey(puprns)
    followup_file = output_dir / 'serl_follow_up_survey_2023_data.csv'
    followup_data.to_csv(followup_file, index=False, encoding='latin-1')
    print(f"  Saved: {followup_file}")
    
    # Generate list of exporter puprns
    print("Generating list of exporter puprns data...")
    exporter_data = generate_list_of_exporters(puprns)
    exporter_file = output_dir / 'Elec_2023_list_of_exporter_puprns.csv'
    exporter_data.to_csv(exporter_file, index=False)
    print(f"  Saved: {exporter_file}")
    
    # Generate Module 1 time series data for each household
    print("Generating Module 1 time series data...")
    for i, puprn in enumerate(puprns):
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{len(puprns)} households")
        
        module1_data = generate_module1_data(puprn, year)
        module1_file = output_dir / f'{puprn}.pkl'
        module1_data.to_pickle(module1_file)
    
    print(f"\nDummy data generation complete!")
    print(f"Generated files:")
    print(f"  - {len(puprns)} household records in epc_data.csv")
    print(f"  - {len(puprns)} household records in serl_survey_data.csv")
    print(f"  - {len(puprns)} module 1 .pkl files with time series data")
    print(f"  - Total time series records: {len(puprns) * 17520}")


if __name__ == '__main__':
    main()