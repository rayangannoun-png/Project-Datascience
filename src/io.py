"""
Data loading utilities for Tourism & Natural Disasters project.
Handles robust CSV loading with automatic separator detection.
"""

import pandas as pd
from pathlib import Path
from typing import Tuple, Optional


def load_all_datasets(
    path_all: str = "Final_dataset_v.csv",
    path_low: str = "Final_dataset_FIXEDGROUP_wri_vBelowMedian2001.csv",
    path_high: str = "Final_dataset_FIXEDGROUP_wri_vAboveEqMedian2001.csv",
    sep: Optional[str] = None,
    data_dir: str = "data"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all three datasets with robust separator detection.
    
    Parameters
    ----------
    path_all : str
        Filename for complete dataset
    path_low : str
        Filename for well-prepared countries (below median WRI 2001)
    path_high : str
        Filename for not well-prepared countries (above/equal median WRI 2001)
    sep : str, optional
        CSV separator. If None, tries ';' first, then ','
    data_dir : str, default "data"
        Directory containing the CSV files (relative to project root)
    
    Returns
    -------
    df_all : pd.DataFrame
        Complete dataset
    df_low : pd.DataFrame
        Well-prepared countries (below median baseline)
    df_high : pd.DataFrame
        Not well-prepared countries (above/equal median baseline)
    """
    # Build full paths
    base_path = Path(data_dir)
    
    paths = {
        'all': base_path / path_all,
        'low': base_path / path_low,
        'high': base_path / path_high
    }
    
    # Check all files exist
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Dataset '{name}' not found at: {path}")
    
    # Load datasets with auto-detected separators
    if sep is None:
        df_all = pd.read_csv(paths['all'], sep=_detect_separator(paths['all']))
        df_low = pd.read_csv(paths['low'], sep=_detect_separator(paths['low']))
        df_high = pd.read_csv(paths['high'], sep=_detect_separator(paths['high']))
    else:
        df_all = pd.read_csv(paths['all'], sep=sep)
        df_low = pd.read_csv(paths['low'], sep=sep)
        df_high = pd.read_csv(paths['high'], sep=sep)
    
    # Standardize column names
    df_all = _standardize_columns(df_all)
    df_low = _standardize_columns(df_low)
    df_high = _standardize_columns(df_high)
    
    # Basic validation
    _validate_dataset(df_all, "complete dataset")
    _validate_dataset(df_low, "well-prepared dataset")
    _validate_dataset(df_high, "not well-prepared dataset")
    
    return df_all, df_low, df_high


def _detect_separator(filepath: Path) -> str:
    """Detect CSV separator by trying common options and validating."""
    # Try both separators and pick the one that gives most columns
    with open(filepath, 'r') as f:
        first_line = f.readline().strip()
    
    semicolon_count = first_line.count(';')
    comma_count = first_line.count(',')
    
    # Use the separator that appears more frequently
    if semicolon_count > comma_count:
        return ';'
    elif comma_count > 0:
        return ','
    else:
        raise ValueError(f"Could not detect separator in {filepath}")


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names for easier access."""
    df = df.copy()
    
    # Rename problematic column name
    if "tourists ['000]" in df.columns:
        df = df.rename(columns={"tourists ['000]": "tourists"})
    
    # Strip whitespace from all column names
    df.columns = df.columns.str.strip()
    
    return df


def _validate_dataset(df: pd.DataFrame, name: str) -> None:
    """Perform basic validation checks on loaded dataset."""
    if df.empty:
        raise ValueError(f"{name} is empty")
    
    # Check for required columns
    required_cols = ['country', 'year']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")
    
    # Check for duplicates on (country, year)
    duplicates = df.duplicated(subset=['country', 'year'], keep=False)
    if duplicates.any():
        n_dup = duplicates.sum()
        print(f"Warning: {name} has {n_dup} duplicate (country, year) pairs")


def get_dataset_info(df: pd.DataFrame, name: str = "Dataset") -> None:
    """Print useful information about a dataset."""
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nYear range: {df['year'].min()} - {df['year'].max()}")
    print(f"Countries: {df['country'].nunique()} unique")
    print(f"\nMissing values:")
    missing = df.isnull().sum()
    if missing.any():
        print(missing[missing > 0])
    else:
        print("  None")
