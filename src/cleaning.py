"""
Data cleaning and transformation utilities.
Handles log transformations, dummy creation, and lagged variables.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional


def add_well_prepared_dummy(
    df: pd.DataFrame,
    df_low_prep: pd.DataFrame,
    country_col: str = 'country',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Add 'well_prepared' dummy variable to complete dataset.
    
    well_prepared = 1 if country is in df_low (below median WRI 2001)
    well_prepared = 0 otherwise
    
    Parameters
    ----------
    df_all : pd.DataFrame
        Complete dataset
    df_low : pd.DataFrame
        Dataset with well-prepared countries only
    country_col : str, default "country"
        Name of country column
    
    Returns
    -------
    pd.DataFrame
        df_all with added 'well_prepared' column
    """
    df_result = df.copy()
    
    # Get list of well-prepared countries
    well_prepared_countries = df_low_prep[country_col].unique()
    
    # Create dummy
    df_result['well_prepared'] = df_result[country_col].isin(well_prepared_countries).astype(int)
    
    if verbose:
        print(f"Added 'well_prepared' dummy:")
        print(f"  - Well-prepared countries (1): {df_result['well_prepared'].sum():,} obs")
        print(f"  - Not well-prepared (0): {(df_result['well_prepared'] == 0).sum():,} obs")
    
    return df_result


def log_transform_in_place(
    df: pd.DataFrame,
    special_vars: Optional[List[str]] = None,
    skip_cols: Optional[List[str]] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Apply log transformations IN PLACE to numeric columns.
    
    Rules:
    - Special vars (deaths, wri_v): ln(x + 1) to handle zeros
    - Other numeric continuous: ln(x) only if x > 0
    - Skip: year, binary dummies (0/1), and columns in skip_cols
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    special_vars : list of str, optional
        Variables to transform as ln(x+1). Default: ['deaths', 'wri_v']
    skip_cols : list of str, optional
        Additional columns to skip
    
    Returns
    -------
    pd.DataFrame
        Dataframe with log-transformed columns (in place)
    """
    df = df.copy()
    
    if special_vars is None:
        special_vars = ['deaths', 'wri_v']
    
    # Detect numeric columns
    num_cols = df.select_dtypes(include='number').columns.tolist()
    
    # Build skip list
    skip = []
    if 'year' in df.columns:
        skip.append('year')
    
    # Skip binary dummies (0/1)
    for c in num_cols:
        vals = df[c].dropna().unique()
        if len(vals) <= 2 and set(vals).issubset({0, 1}):
            skip.append(c)
    
    # Add user-specified skip columns
    if skip_cols:
        skip.extend(skip_cols)
    
    skip = list(set(skip))  # Remove duplicates
    
    # Apply transformations
    transformed = []
    for c in num_cols:
        if c in skip:
            continue
        
        x = pd.to_numeric(df[c], errors='coerce')
        
        if c in special_vars:
            # ln(x + 1) for deaths, wri_v
            df[c] = np.where(x >= 0, np.log(x + 1), np.nan)
            transformed.append(f"{c} -> ln({c}+1)")
        else:
            # ln(x) for other continuous vars
            df[c] = np.where(x > 0, np.log(x), np.nan)
            transformed.append(f"{c} -> ln({c})")
    
    if verbose:
        print(f"Log transformations applied ({len(transformed)} variables):")
        for t in transformed:
            print(f"  - {t}")
    
    return df


def lag_variable_by_country_year(
    df: pd.DataFrame,
    col: str,
    country_col: str = "country",
    year_col: str = "year",
    lag: int = 1,
    fill_first: str = "keep"
) -> pd.DataFrame:
    """
    Create lagged variable within each country.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe (must be sorted by country, year)
    col : str
        Column to lag
    country_col : str, default "country"
        Country identifier column
    year_col : str, default "year"
        Year column
    lag : int, default 1
        Number of periods to lag
    fill_first : str, default "keep"
        How to handle first observation per country:
        - "keep": keep original value
        - "nan": set to NaN
        - "drop": drop rows with missing lags
    
    Returns
    -------
    pd.DataFrame
        Dataframe with added lagged column named '{col}_lag{lag}'
    """
    df = df.copy()
    
    # Sort by country and year
    df = df.sort_values([country_col, year_col])
    
    # Create lag
    lag_col = f"{col}_lag{lag}"
    df[lag_col] = df.groupby(country_col)[col].shift(lag)
    
    # Handle first observations
    if fill_first == "nan":
        pass  # Already NaN
    elif fill_first == "keep":
        # Fill NaN lags with original value
        df[lag_col] = df[lag_col].fillna(df[col])
    elif fill_first == "drop":
        df = df.dropna(subset=[lag_col])
    else:
        raise ValueError(f"Unknown fill_first option: {fill_first}")
    
    print(f"Created lagged variable: {lag_col} (lag={lag}, fill_first='{fill_first}')")
    
    return df


def check_panel_structure(
    df: pd.DataFrame,
    country_col: str = "country",
    year_col: str = "year",
    verbose: bool = True
) -> Dict:
    """
    Check panel data structure and report issues.
    
    Parameters
    ----------
    df : pd.DataFrame
        Panel dataset
    country_col : str, default "country"
        Country identifier
    year_col : str, default "year"
        Time identifier
    verbose : bool, default True
        Print detailed report
    
    Returns
    -------
    dict
        Dictionary with panel structure info
    """
    info = {}
    
    # Basic counts
    info['n_countries'] = df[country_col].nunique()
    info['n_years'] = df[year_col].nunique()
    info['n_obs'] = len(df)
    info['year_min'] = df[year_col].min()
    info['year_max'] = df[year_col].max()
    
    # Check duplicates
    duplicates = df.duplicated(subset=[country_col, year_col], keep=False)
    info['n_duplicates'] = duplicates.sum()
    
    # Check balance
    obs_per_country = df.groupby(country_col).size()
    info['balanced'] = (obs_per_country.nunique() == 1)
    info['min_obs_per_country'] = obs_per_country.min()
    info['max_obs_per_country'] = obs_per_country.max()
    info['mean_obs_per_country'] = obs_per_country.mean()
    
    # Missing values
    info['missing_by_col'] = df.isnull().sum().to_dict()
    
    if verbose:
        print(f"\n{'='*60}")
        print("Panel Structure Check")
        print(f"{'='*60}")
        print(f"Countries: {info['n_countries']}")
        print(f"Years: {info['n_years']} ({info['year_min']}-{info['year_max']})")
        print(f"Total observations: {info['n_obs']:,}")
        print(f"\nBalanced: {'Yes' if info['balanced'] else 'No'}")
        print(f"  Min obs/country: {info['min_obs_per_country']}")
        print(f"  Max obs/country: {info['max_obs_per_country']}")
        print(f"  Mean obs/country: {info['mean_obs_per_country']:.1f}")
        
        if info['n_duplicates'] > 0:
            print(f"\n⚠️  Warning: {info['n_duplicates']} duplicate (country, year) pairs found!")
        
        missing_cols = {k: v for k, v in info['missing_by_col'].items() if v > 0}
        if missing_cols:
            print(f"\nMissing values:")
            for col, count in missing_cols.items():
                pct = 100 * count / info['n_obs']
                print(f"  - {col}: {count:,} ({pct:.1f}%)")
    
    return info
