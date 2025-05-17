from pathlib import Path
import pandas as pd
from math import sin, cos, sqrt, atan2, radians
import numpy as np


def load_data(path):
    path = Path(path)  # Ensures compatibility on all OS
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")
    return pd.read_csv(path).reset_index(drop=True)


def process_tournament_data(mm_data: pd.DataFrame, coach_data: pd.DataFrame) -> pd.DataFrame:
    # Select relevant columns from coach data
    coach_cols = [
        'game_id', 'team1_pt_overall_ncaa', 'team1_pt_overall_s16', 'team1_pt_overall_ff',
        'team1_pt_career_school_wins', 'team2_pt_overall_ncaa', 'team2_pt_overall_s16',
        'team2_pt_overall_ff', 'team2_pt_career_school_wins'
    ]
    coach_data = coach_data[coach_cols]
    
    # Merge coach data into main match data
    mm_data = pd.merge(mm_data, coach_data, how='left', on='game_id')
    mm_data = mm_data.fillna(0)
    
    # Coach experience scores
    mm_data['team1_coach_experience_score'] = (
        0.5 * mm_data['team1_pt_overall_ncaa'] +
        1.0 * mm_data['team1_pt_overall_s16'] +
        2.0 * mm_data['team1_pt_overall_ff'] +
        0.25 * mm_data['team1_pt_career_school_wins']
    )
    
    mm_data['team2_coach_experience_score'] = (
        0.5 * mm_data['team2_pt_overall_ncaa'] +
        1.0 * mm_data['team2_pt_overall_s16'] +
        2.0 * mm_data['team2_pt_overall_ff'] +
        0.25 * mm_data['team2_pt_career_school_wins']
    )

    # Point difference
    mm_data['point_diff'] = abs(mm_data['team1_score'] - mm_data['team2_score'])

    # Adjusted Efficiency Margin (AdjEM)
    mm_data['team1_AdjEM'] = mm_data['team1_adjoe'] - mm_data['team1_adjde']
    mm_data['team2_AdjEM'] = mm_data['team2_adjoe'] - mm_data['team2_adjde']

    # Seed difference
    mm_data['SeedDiff'] = mm_data['team1_seed'] - mm_data['team2_seed']

    # Effective Field Goal Percentage (eFG)
    mm_data['team1_eFG'] = (mm_data['team1_fg2pct'] * 2 + mm_data['team1_fg3pct'] * 3) / 5
    mm_data['team2_eFG'] = (mm_data['team2_fg2pct'] * 2 + mm_data['team2_fg3pct'] * 3) / 5

    # Turnover Margin
    mm_data['TurnoverMargin'] = (
        (mm_data['team1_stlrate'] - mm_data['team1_oppstlrate']) -
        (mm_data['team2_stlrate'] - mm_data['team2_oppstlrate'])
    )

    # Free Throw Rate (FTR)
    mm_data['team1_FTR'] = mm_data['team1_ftpct'] / (mm_data['team1_fg2pct'] + mm_data['team1_fg3pct'])
    mm_data['team2_FTR'] = mm_data['team2_ftpct'] / (mm_data['team2_fg2pct'] + mm_data['team2_fg3pct'])

    return mm_data


def distance(lat1, lon1, lat2, lon2):

    # approximate radius of earth in km
    R = 6373.0

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    
    return distance


def write_data(df, filename, file_format='csv'):
    """
    Save a DataFrame to the data/processed/ folder.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        filename (str): Output file name (e.g., 'cleaned_data.csv').
        file_format (str): Format to save ('csv' or 'xlsx').

    Returns:
        str: Full path to the saved file.
    """
    processed_path = Path('data/processed')
    processed_path.mkdir(parents=True, exist_ok=True)  # Ensure folder exists

    full_path = processed_path / filename

    if file_format == 'csv':
        df.to_csv(full_path, index=False)
    elif file_format == 'xlsx':
        df.to_excel(full_path, index=False)
    else:
        raise ValueError("Unsupported file format. Use 'csv' or 'xlsx'.")

    print(f"Data written to: {full_path.resolve()}")
    return str(full_path)

