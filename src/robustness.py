"""
Robustness Analysis Module

This module provides functions for testing the robustness of regression results
by identifying and removing outlier countries, then re-running the analysis.

Functions
---------
identify_outliers_iqr : Identify outliers using the IQR method
remove_outlier_countries : Remove outlier countries from dataset
compare_robust_results : Compare original vs robust regression results
run_robustness_analysis : Main function to run complete robustness test
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from .models import run_baseline_specifications


def identify_outliers_iqr(
    df: pd.DataFrame,
    variables: List[str],
    groupby: str = 'country'
) -> Tuple[List[str], Dict[str, Tuple[float, float]]]:
    """
    Identify outlier countries using the IQR method on country-level means.
    
    The IQR (Interquartile Range) method identifies outliers as values that fall
    outside the range [Q1 - 1.5*IQR, Q3 + 1.5*IQR], where:
    - Q1 = 25th percentile
    - Q3 = 75th percentile
    - IQR = Q3 - Q1
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with panel data
    variables : list of str
        List of variables to check for outliers (e.g., ['ln_deaths', 'ln_tourists'])
    groupby : str, default='country'
        Column to group by (typically 'country')
    
    Returns
    -------
    outlier_countries : list of str
        List of unique country names identified as outliers
    bounds : dict
        Dictionary with variable names as keys and (lower_bound, upper_bound) tuples as values
    
    Examples
    --------
    >>> outliers, bounds = identify_outliers_iqr(df, ['ln_deaths', 'ln_tourists'])
    >>> print(f"Found {len(outliers)} outlier countries")
    >>> print(f"Bounds for ln_deaths: {bounds['ln_deaths']}")
    """
    # Calculate country-level means
    df_means = df.groupby(groupby)[variables].mean().reset_index()
    
    outlier_countries = []
    bounds = {}
    
    for var in variables:
        # Calculate quartiles and IQR
        Q1 = df_means[var].quantile(0.25)
        Q3 = df_means[var].quantile(0.75)
        IQR = Q3 - Q1
        
        # Calculate bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        bounds[var] = (lower_bound, upper_bound)
        
        # Identify outliers
        outliers = df_means[
            (df_means[var] < lower_bound) | (df_means[var] > upper_bound)
        ][groupby].tolist()
        
        outlier_countries.extend(outliers)
    
    # Return unique countries
    outlier_countries = list(set(outlier_countries))
    
    return outlier_countries, bounds


def identify_outliers_studentized(
    df: pd.DataFrame,
    y_col: str = 'tourists',
    x_col: str = 'deaths',
    entity_col: str = 'country',
    time_col: str = 'year',
    controls: Optional[List[str]] = None
) -> Tuple[List[str], pd.DataFrame]:
    """
    Identify outlier countries using studentized residuals with 2/sqrt(n) threshold.
    
    This method:
    1. Fits a panel FE regression model
    2. Calculates studentized residuals for each observation
    3. Identifies countries with |studentized residual| > 2/sqrt(n)
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe (should be log-transformed)
    y_col : str, default='tourists'
        Dependent variable
    x_col : str, default='deaths'
        Main independent variable
    entity_col : str, default='country'
        Entity identifier
    time_col : str, default='year'
        Time identifier
    controls : list of str, optional
        Control variables
    
    Returns
    -------
    outlier_countries : list of str
        List of countries with outlier observations
    outlier_df : pd.DataFrame
        DataFrame with outlier details (country, year, residual, threshold)
    
    Examples
    --------
    >>> outliers, details = identify_outliers_studentized(df)
    >>> print(f"Found {len(outliers)} countries with outlier observations")
    """
    from .models import fit_panel_fe
    
    # Fit the model
    result = fit_panel_fe(
        df,
        y_col=y_col,
        x_col=x_col,
        entity_col=entity_col,
        time_col=time_col,
        controls=controls
    )
    
    # Get residuals
    residuals = result.resids
    
    # Calculate studentized residuals
    # For panel data, we use the standard error of residuals
    sigma = np.sqrt(result.resid_ss / result.df_resid)
    
    # Studentized residuals (simplified for panel data)
    studentized_resids = residuals / sigma
    
    # Calculate threshold: 2/sqrt(n)
    n = len(residuals)
    threshold = 2 / np.sqrt(n)
    
    # Identify outliers
    outlier_mask = np.abs(studentized_resids) > threshold
    
    # Create DataFrame with outlier information
    outlier_data = []
    for idx in residuals[outlier_mask].index:
        country = idx[0] if isinstance(idx, tuple) else df.loc[df.index == idx, entity_col].iloc[0]
        year = idx[1] if isinstance(idx, tuple) else df.loc[df.index == idx, time_col].iloc[0]
        resid = studentized_resids.loc[idx]
        
        outlier_data.append({
            'country': country,
            'year': year,
            'studentized_residual': resid,
            'threshold': threshold
        })
    
    outlier_df = pd.DataFrame(outlier_data)
    
    # Get unique countries
    outlier_countries = outlier_df['country'].unique().tolist() if len(outlier_df) > 0 else []
    
    return outlier_countries, outlier_df


def identify_outliers_studentized(
    df: pd.DataFrame,
    y_col: str = 'tourists',
    x_col: str = 'deaths',
    entity_col: str = 'country',
    time_col: str = 'year',
    controls: Optional[List[str]] = None
) -> Tuple[List[str], pd.DataFrame]:
    """
    Identify outlier countries using studentized residuals with 2/sqrt(n) threshold.
    
    This method:
    1. Fits a panel FE regression model
    2. Calculates studentized residuals for each observation
    3. Identifies countries with |studentized residual| > 2/sqrt(n)
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe (should be log-transformed)
    y_col : str, default='tourists'
        Dependent variable
    x_col : str, default='deaths'
        Main independent variable
    entity_col : str, default='country'
        Entity identifier
    time_col : str, default='year'
        Time identifier
    controls : list of str, optional
        Control variables
    
    Returns
    -------
    outlier_countries : list of str
        List of countries with outlier observations
    outlier_df : pd.DataFrame
        DataFrame with outlier details (country, year, residual, threshold)
    """
    from .models import fit_panel_fe
    
    # Fit the model
    result = fit_panel_fe(
        df,
        y_col=y_col,
        x_col=x_col,
        entity_col=entity_col,
        time_col=time_col,
        controls=controls
    )
    
    # Get residuals
    residuals = result.resids
    
    # Calculate studentized residuals
    sigma = np.sqrt(result.resid_ss / result.df_resid)
    studentized_resids = residuals / sigma
    
    # Calculate threshold: 2/sqrt(n)
    n = len(residuals)
    threshold = 2 / np.sqrt(n)
    
    # Identify outliers
    outlier_mask = np.abs(studentized_resids) > threshold
    
    # Create DataFrame with outlier information
    outlier_data = []
    for idx in residuals[outlier_mask].index:
        country = idx[0] if isinstance(idx, tuple) else df.loc[df.index == idx, entity_col].iloc[0]
        year = idx[1] if isinstance(idx, tuple) else df.loc[df.index == idx, time_col].iloc[0]
        resid = studentized_resids.loc[idx]
        
        outlier_data.append({
            'country': country,
            'year': year,
            'studentized_residual': resid,
            'threshold': threshold
        })
    
    outlier_df = pd.DataFrame(outlier_data)
    
    # Get unique countries
    outlier_countries = outlier_df['country'].unique().tolist() if len(outlier_df) > 0 else []
    
    return outlier_countries, outlier_df


def remove_outlier_countries(
    df: pd.DataFrame,
    outlier_countries: List[str],
    country_col: str = 'country'
) -> pd.DataFrame:
    """
    Remove outlier countries from the dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    outlier_countries : list of str
        List of country names to remove
    country_col : str, default='country'
        Name of the country column
    
    Returns
    -------
    pd.DataFrame
        Dataframe with outlier countries removed
    
    Examples
    --------
    >>> df_robust = remove_outlier_countries(df, ['China', 'India'])
    >>> print(f"Removed {len(df) - len(df_robust)} observations")
    """
    df_robust = df[~df[country_col].isin(outlier_countries)].copy()
    return df_robust


def compare_robust_results(
    results_original: Dict,
    results_robust: Dict,
    coef_name: str = 'ln_deaths'
) -> pd.DataFrame:
    """
    Compare regression coefficients between original and robust datasets.
    
    Parameters
    ----------
    results_original : dict
        Dictionary of regression results from original dataset
        Keys are specification names, values are regression result objects
    results_robust : dict
        Dictionary of regression results from robust dataset (outliers removed)
    coef_name : str, default='ln_deaths'
        Name of the coefficient to compare
    
    Returns
    -------
    pd.DataFrame
        Comparison table with columns:
        - Specification: name of the regression specification
        - Original: coefficient from original dataset
        - Robust: coefficient from robust dataset
        - Difference: absolute difference
        - Pct_Change: percentage change
    
    Examples
    --------
    >>> comparison = compare_robust_results(results_orig, results_robust)
    >>> print(comparison)
    """
    comparison_data = []
    
    for spec_name in results_original.keys():
        coef_orig = results_original[spec_name].params[coef_name]
        coef_robust = results_robust[spec_name].params[coef_name]
        diff = coef_robust - coef_orig
        pct_change = 100 * diff / coef_orig if coef_orig != 0 else 0
        
        comparison_data.append({
            'Specification': spec_name,
            'Original': coef_orig,
            'Robust': coef_robust,
            'Difference': diff,
            'Pct_Change': pct_change
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    return comparison_df


def run_robustness_analysis(
    df: pd.DataFrame,
    outlier_vars: List[str] = None,
    verbose: bool = True
) -> Dict:
    """
    Run complete robustness analysis by removing outlier countries and re-running regressions.
    
    This is the main function that orchestrates the entire robustness test:
    1. Identifies outlier countries based on specified variables
    2. Removes outlier countries from the dataset
    3. Runs baseline regressions on both original and robust datasets
    4. Compares the results
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe (should already be log-transformed)
    outlier_vars : list of str, optional
        Variables to use for outlier detection. Default: ['ln_deaths', 'ln_tourists']
    verbose : bool, default=True
        Whether to print progress messages
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'outlier_countries': list of outlier country names
        - 'bounds': dict of (lower, upper) bounds for each variable
        - 'df_robust': robust dataset with outliers removed
        - 'results_original': regression results from original dataset
        - 'results_robust': regression results from robust dataset
        - 'comparison': comparison dataframe
        - 'n_original': number of observations in original dataset
        - 'n_robust': number of observations in robust dataset
        - 'n_countries_original': number of countries in original dataset
        - 'n_countries_robust': number of countries in robust dataset
    
    Examples
    --------
    >>> results = run_robustness_analysis(df)
    >>> print(f"Removed {len(results['outlier_countries'])} countries")
    >>> print(results['comparison'])
    """
    if outlier_vars is None:
        outlier_vars = ['ln_deaths', 'ln_tourists']
    
    if verbose:
        print("=" * 60)
        print("ROBUSTNESS ANALYSIS: Outlier Detection & Removal")
        print("=" * 60)
    
    # Step 1: Identify outliers
    if verbose:
        print(f"\n1️⃣ Identifying outliers based on: {', '.join(outlier_vars)}")
    
    outlier_countries, bounds = identify_outliers_iqr(df, outlier_vars)
    
    if verbose:
        print(f"\n   Found {len(outlier_countries)} outlier countries:")
        for country in sorted(outlier_countries):
            print(f"   - {country}")
        
        print("\n   Bounds (IQR method):")
        for var, (lb, ub) in bounds.items():
            print(f"   - {var}: [{lb:.2f}, {ub:.2f}]")
    
    # Step 2: Remove outliers
    if verbose:
        print(f"\n2️⃣ Removing outlier countries from dataset")
    
    df_robust = remove_outlier_countries(df, outlier_countries)
    
    n_original = len(df)
    n_robust = len(df_robust)
    n_countries_original = df['country'].nunique()
    n_countries_robust = df_robust['country'].nunique()
    
    if verbose:
        print(f"\n   Original: {n_original:,} obs, {n_countries_original} countries")
        print(f"   Robust:   {n_robust:,} obs, {n_countries_robust} countries")
        print(f"   Removed:  {n_original - n_robust:,} obs ({100 * (n_original - n_robust) / n_original:.1f}%)")
    
    # Step 3: Run regressions on both datasets
    if verbose:
        print(f"\n3️⃣ Running baseline regressions on both datasets")
    
    results_original = run_baseline_specifications(df)
    results_robust = run_baseline_specifications(df_robust)
    
    if verbose:
        print(f"   ✓ Original dataset: {len(results_original)} specifications")
        print(f"   ✓ Robust dataset: {len(results_robust)} specifications")
    
    # Step 4: Compare results
    if verbose:
        print(f"\n4️⃣ Comparing regression coefficients")
    
    comparison = compare_robust_results(results_original, results_robust)
    
    if verbose:
        print("\n   Coefficient comparison (ln_deaths):")
        print(comparison.to_string(index=False))
    
    # Package results
    results = {
        'outlier_countries': outlier_countries,
        'bounds': bounds,
        'df_robust': df_robust,
        'results_original': results_original,
        'results_robust': results_robust,
        'comparison': comparison,
        'n_original': n_original,
        'n_robust': n_robust,
        'n_countries_original': n_countries_original,
        'n_countries_robust': n_countries_robust
    }
    
    if verbose:
        print("\n" + "=" * 60)
        print("✅ ROBUSTNESS ANALYSIS COMPLETE")
        print("=" * 60)
    
    return results


def plot_outlier_detection(
    df: pd.DataFrame,
    outlier_vars: List[str],
    outlier_countries: List[str],
    bounds: Dict[str, Tuple[float, float]],
    figsize: Tuple[int, int] = (14, 6)
) -> plt.Figure:
    """
    Create boxplots showing outlier detection for country-level means.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    outlier_vars : list of str
        Variables used for outlier detection
    outlier_countries : list of str
        List of identified outlier countries
    bounds : dict
        Dictionary with (lower_bound, upper_bound) for each variable
    figsize : tuple, default=(14, 6)
        Figure size
    
    Returns
    -------
    matplotlib.figure.Figure
        Figure object with boxplots
    """
    # Calculate country means
    df_means = df.groupby('country')[outlier_vars].mean().reset_index()
    
    # Create figure
    n_vars = len(outlier_vars)
    fig, axes = plt.subplots(1, n_vars, figsize=figsize)
    
    if n_vars == 1:
        axes = [axes]
    
    fig.suptitle('Country-Level Means: Identifying Outliers (IQR Method)', 
                 fontsize=14, fontweight='bold')
    
    colors = ['lightcoral', 'lightskyblue', 'lightgreen', 'lightyellow', 'lightpink']
    
    for i, var in enumerate(outlier_vars):
        ax = axes[i]
        
        # Create boxplot
        bp = ax.boxplot(
            df_means[var].dropna(), 
            vert=True, 
            patch_artist=True,
            boxprops=dict(facecolor=colors[i % len(colors)], alpha=0.7),
            medianprops=dict(color='darkred', linewidth=2),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black'),
            flierprops=dict(marker='o', markerfacecolor='darkred', markersize=8, alpha=0.7)
        )
        
        # Add bounds
        lb, ub = bounds[var]
        ax.axhline(y=lb, color='blue', linestyle='--', linewidth=1.5, 
                   label=f'Lower: {lb:.2f}')
        ax.axhline(y=ub, color='blue', linestyle='--', linewidth=1.5, 
                   label=f'Upper: {ub:.2f}')
        
        # Labels
        ax.set_title(var, fontsize=12, fontweight='bold')
        ax.set_ylabel('Country Mean', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig
