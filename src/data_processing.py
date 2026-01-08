import numpy as np
import pandas as pd

def preprocess_data(df: pd.DataFrame, fill_strategy: str) -> pd.DataFrame:
    """
    Clean and preprocess the input DataFrame.

    Parameters:
    ----------
    df : pd.DataFrame
        Raw input data to be processed.
    fill_strategy : str, optional
        Strategy to fill missing values ('mean','median','mode').

    Returns:
    ----------
    pd.DataFrame
        Cleaned and preprocessed DataFrame.

    Raises:
    ----------
    ValueError
        If fill_strategy is not recognized.
    """
    try:
        df = df.copy()
        # Rename target for consistency
        if 'median_house_value' in df.columns:
            df = df.rename(columns={'median_house_value': 'Price'})

        # Handle Missing Values
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].isnull().any():
                if fill_strategy == 'mean':
                    df[col] = df[col].fillna(df[col].mean())
                elif fill_strategy == 'median':
                    df[col] = df[col].fillna(df[col].median())
                elif fill_strategy == 'mode':
                    df[col] = df[col].fillna(df[col].mode()[0])
                else:
                    raise ValueError(f"Fill strategy '{fill_strategy}' not recognized.")

        # Create Derived Features
        df['Rooms_Per_Household'] = df['total_rooms'] / df['households']
        df['Bedrooms_Per_Room'] = df['total_bedrooms'] / df['total_rooms']

        return df
    except Exception as e:
        print(f"Error in preprocessing data: {e}")
        raise



