"""
Difference-in-Differences: Analysis & Visualization

This module contains functions for running DiD analysis:
- Time series plots (treated vs control countries)
- DiD TWFE regressions with fixed effects
- Results comparison and visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict
from linearmodels.panel import PanelOLS


def plot_did_timeseries(
    df: pd.DataFrame,
    treated_country: str,
    control_countries: List[str],
    treat_year: int,
    pre_start: int = 2001,
    post_years: int = 4,
    use_log: bool = False,
    country_col: str = "country",
    year_col: str = "year",
    tour_col: str = None,
    figsize: Tuple[float, float] = (11, 5.5)
) -> plt.Figure:
    """
    Plot time series of tourists for treated vs control countries.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataset
    treated_country : str
        Name of treated country
    control_countries : list of str
        List of control countries
    treat_year : int
        Treatment year (disaster year)
    pre_start : int, default=2001
        Start year of pre-treatment period
    post_years : int, default=4
        Number of years after treatment to include
    use_log : bool, default=False
        Whether to use ln(tourists) instead of raw tourists
    country_col : str, default='country'
        Country column name
    year_col : str, default='year'
        Year column name
    tour_col : str, optional
        Tourists column name (auto-detected if None)
    figsize : tuple, default=(11, 5.5)
        Figure size
    
    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    # Auto-detect tourists column
    if tour_col is None:
        tour_col = next((c for c in df.columns if "tourist" in c.strip().lower()), None)
    
    if tour_col is None:
        raise KeyError("Tourists column not found")
    
    # Define time window
    year_min = pre_start
    year_max = treat_year + post_years
    
    # Filter data
    all_countries = [treated_country] + control_countries
    d = df[[country_col, year_col, tour_col]].copy()
    d[year_col] = pd.to_numeric(d[year_col], errors="coerce")
    d[tour_col] = pd.to_numeric(d[tour_col], errors="coerce")
    d = d.dropna(subset=[country_col, year_col, tour_col])
    
    # Filter countries and years
    d = d[d[country_col].isin(all_countries)].copy()
    d = d[(d[year_col] >= year_min) & (d[year_col] <= year_max)].copy()
    d[year_col] = d[year_col].astype(int)
    
    # Apply log transformation if requested
    if use_log:
        d = d[d[tour_col] > 0].copy()
        d["value"] = np.log(d[tour_col])
        value_col = "value"
        ylabel = "ln(Tourists)"
    else:
        value_col = tour_col
        ylabel = "Tourists"
    
    # Pivot to wide format
    years = list(range(year_min, year_max + 1))
    wide = (
        d.pivot(index=year_col, columns=country_col, values=value_col)
        .reindex(years)
        .sort_index()
    )
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot treated country (bold line)
    if treated_country in wide.columns:
        ax.plot(wide.index, wide[treated_country], marker="o", linewidth=3, 
                label=f"{treated_country} (Treated)", color="red", zorder=3)
    
    # Plot control countries (thinner lines)
    colors = plt.cm.tab10(range(len(control_countries)))
    for i, cc in enumerate(control_countries):
        if cc in wide.columns:
            ax.plot(wide.index, wide[cc], marker="o", linewidth=2, 
                    label=f"{cc} (Control)", color=colors[i], alpha=0.8)
    
    # Add pre/post shading
    ax.axvspan(year_min, treat_year, alpha=0.08, color='blue', label='Pre-treatment')
    ax.axvspan(treat_year, year_max, alpha=0.14, color='orange', label='Post-treatment')
    
    # Add treatment cutoff line
    ax.axvline(treat_year, linestyle="--", linewidth=2, color='black', 
               label=f'Treatment ({treat_year})')
    
    # Labels and formatting
    title = f"{ylabel} over time — {treated_country} vs Controls"
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(years)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True, loc='best')
    
    plt.tight_layout()
    
    return fig


def run_did_twfe(
    df: pd.DataFrame,
    treated_country: str,
    control_countries: List[str],
    treat_year: int,
    pre_start: int = 2001,
    post_years: int = 4,
    country_col: str = "country",
    year_col: str = "year",
    tour_col: str = None,
    verbose: bool = True
) -> Tuple[PanelOLS, pd.DataFrame]:
    """
    Run DiD TWFE regression with country and year fixed effects.
    
    Model: ln(tourists) ~ DiD + Country FE + Year FE
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataset
    treated_country : str
        Name of treated country
    control_countries : list of str
        List of control countries
    treat_year : int
        Treatment year
    pre_start : int, default=2001
        Start year of analysis
    post_years : int, default=4
        Number of post-treatment years
    country_col : str, default='country'
        Country column name
    year_col : str, default='year'
        Year column name
    tour_col : str, optional
        Tourists column name (auto-detected if None)
    verbose : bool, default=True
        Whether to print progress
    
    Returns
    -------
    tuple of (PanelOLS result, pd.DataFrame)
        (regression result, prepared dataframe)
    """
    # Auto-detect tourists column
    if tour_col is None:
        tour_col = next((c for c in df.columns if "tourist" in c.strip().lower()), None)
    
    if tour_col is None:
        raise KeyError("Tourists column not found")
    
    # Define time window
    year_min = pre_start
    year_max = treat_year + post_years
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"DiD TWFE Regression: {treated_country} vs {', '.join(control_countries)}")
        print(f"{'='*70}")
        print(f"Treatment year: {treat_year}")
        print(f"Pre-period: {year_min}–{treat_year-1}")
        print(f"Post-period: {treat_year}–{year_max}")
    
    # Filter data
    all_countries = [treated_country] + control_countries
    df_did = df[df[country_col].isin(all_countries)].copy()
    df_did[year_col] = pd.to_numeric(df_did[year_col], errors="coerce")
    df_did = df_did[(df_did[year_col] >= year_min) & (df_did[year_col] <= year_max)].copy()
    
    # Convert to numeric
    df_did[tour_col] = pd.to_numeric(df_did[tour_col], errors="coerce")
    
    # Create ln(tourists)
    df_did["ln_tourists"] = np.where(df_did[tour_col] > 0, np.log(df_did[tour_col]), np.nan)
    
    # Create DiD variables
    df_did["treated"] = (df_did[country_col] == treated_country).astype(int)
    df_did["post"] = (df_did[year_col] >= treat_year).astype(int)
    df_did["did"] = df_did["treated"] * df_did["post"]
    
    # Set panel index
    df_did = df_did.set_index([country_col, year_col])
    
    # Prepare regression dataset
    df_reg = df_did[["ln_tourists", "did"]].dropna().copy()
    
    if verbose:
        print(f"Observations: {len(df_reg)}")
        print(f"Countries: {df_reg.index.get_level_values(0).nunique()}")
        print(f"Years: {df_reg.index.get_level_values(1).nunique()}")
    
    # Run TWFE DiD regression
    y = df_reg["ln_tourists"]
    X = df_reg[["did"]]
    
    model = PanelOLS(y, X, entity_effects=True, time_effects=True)
    result = model.fit(cov_type="robust")
    
    if verbose:
        print(f"\n{'='*70}")
        print("REGRESSION RESULTS")
        print(f"{'='*70}")
        print(result.summary)
        print(f"\n{'='*70}")
        print(f"DiD coefficient: {result.params['did']:.6f}")
        print(f"Standard error: {result.std_errors['did']:.6f}")
        print(f"P-value: {result.pvalues['did']:.4f}")
        print(f"R² (within): {result.rsquared_within:.4f}")
        print(f"{'='*70}\n")
    
    return result, df_reg


def compare_did_results(
    results: Dict[str, Tuple],
    labels: Dict[str, str] = None
) -> pd.DataFrame:
    """
    Compare DiD coefficients across multiple specifications.
    
    Parameters
    ----------
    results : dict
        Dictionary mapping spec_name -> (result, df_reg) tuple
    labels : dict, optional
        Custom labels for specifications
    
    Returns
    -------
    pd.DataFrame
        Comparison table with coefficients, SE, p-values
    """
    if labels is None:
        labels = {k: k for k in results.keys()}
    
    comparison = []
    
    for spec_name, (result, _) in results.items():
        did_coef = result.params['did']
        did_se = result.std_errors['did']
        did_pval = result.pvalues['did']
        r2_within = result.rsquared_within
        n_obs = result.nobs
        
        # Calculate 95% CI
        ci_low = did_coef - 1.96 * did_se
        ci_high = did_coef + 1.96 * did_se
        
        # Significance stars
        if did_pval < 0.01:
            stars = "***"
        elif did_pval < 0.05:
            stars = "**"
        elif did_pval < 0.10:
            stars = "*"
        else:
            stars = ""
        
        comparison.append({
            "Specification": labels[spec_name],
            "DiD Coefficient": did_coef,
            "Std. Error": did_se,
            "P-value": did_pval,
            "Significance": stars,
            "95% CI Lower": ci_low,
            "95% CI Upper": ci_high,
            "R² (within)": r2_within,
            "N": int(n_obs)
        })
    
    df_comp = pd.DataFrame(comparison)
    
    return df_comp


def run_did_analysis_pair(
    df: pd.DataFrame,
    treated_country: str,
    control_countries: List[str],
    treat_year: int,
    pre_start: int = 2001,
    post_years: int = 4,
    show_plots: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Run complete DiD analysis for one treated-control pair.
    
    This function orchestrates:
    1. Time series plots (raw and log)
    2. DiD TWFE regression
    3. Results summary
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataset
    treated_country : str
        Name of treated country
    control_countries : list of str
        List of control countries
    treat_year : int
        Treatment year
    pre_start : int, default=2001
        Start year
    post_years : int, default=4
        Post-treatment years
    show_plots : bool, default=True
        Whether to create and show plots
    verbose : bool, default=True
        Whether to print progress
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'result': regression result
        - 'df_reg': prepared regression dataframe
        - 'fig_raw': raw tourists plot (if show_plots=True)
        - 'fig_log': log tourists plot (if show_plots=True)
    """
    if verbose:
        print("\n" + "="*70)
        print(f"DiD ANALYSIS: {treated_country} vs {', '.join(control_countries)}")
        print("="*70)
    
    output = {}
    
    # Step 1: Time series plots
    if show_plots:
        if verbose:
            print("\n1️⃣ Creating time series plots...")
        
        # Raw tourists
        fig_raw = plot_did_timeseries(
            df, treated_country, control_countries, treat_year,
            pre_start=pre_start, post_years=post_years, use_log=False
        )
        output['fig_raw'] = fig_raw
        
        # Log tourists
        fig_log = plot_did_timeseries(
            df, treated_country, control_countries, treat_year,
            pre_start=pre_start, post_years=post_years, use_log=True
        )
        output['fig_log'] = fig_log
        
        if verbose:
            print("   ✓ Plots created")
    
    # Step 2: DiD regression
    if verbose:
        print("\n2️⃣ Running DiD TWFE regression...")
    
    result, df_reg = run_did_twfe(
        df, treated_country, control_countries, treat_year,
        pre_start=pre_start, post_years=post_years, verbose=verbose
    )
    
    output['result'] = result
    output['df_reg'] = df_reg
    
    if verbose:
        print("\n" + "="*70)
        print("✅ ANALYSIS COMPLETE")
        print("="*70 + "\n")
    
    return output


def run_event_study(
    df: pd.DataFrame,
    treated_country: str,
    control_countries: List[str],
    treat_year: int,
    pre_start: int = 2001,
    post_years: int = 4,
    country_col: str = "country",
    year_col: str = "year",
    tour_col: str = None,
    verbose: bool = True
) -> Tuple[PanelOLS, pd.DataFrame]:
    """
    Run event-study (dynamic DiD) regression with leads and lags.
    
    This creates a coefficient for each year relative to treatment,
    allowing us to test parallel trends and see dynamic effects.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataset
    treated_country : str
        Name of treated country
    control_countries : list of str
        List of control countries
    treat_year : int
        Treatment year
    pre_start : int, default=2001
        Start year of analysis
    post_years : int, default=4
        Number of post-treatment years
    country_col : str, default='country'
        Country column name
    year_col : str, default='year'
        Year column name
    tour_col : str, optional
        Tourists column name (auto-detected if None)
    verbose : bool, default=True
        Whether to print progress
    
    Returns
    -------
    tuple of (PanelOLS result, pd.DataFrame with coefficients)
        (regression result, event-time coefficients DataFrame)
    """
    # Auto-detect tourists column
    if tour_col is None:
        tour_col = next((c for c in df.columns if "tourist" in c.strip().lower()), None)
    
    if tour_col is None:
        raise KeyError("Tourists column not found")
    
    # Define time window
    year_min = pre_start
    year_max = treat_year + post_years
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Event-Study (Dynamic DiD): {treated_country}")
        print(f"{'='*70}")
        print(f"Treatment year: {treat_year}")
        print(f"Window: {year_min}–{year_max}")
    
    # Filter data
    all_countries = [treated_country] + control_countries
    df_did = df[df[country_col].isin(all_countries)].copy()
    df_did[year_col] = pd.to_numeric(df_did[year_col], errors="coerce")
    df_did = df_did[(df_did[year_col] >= year_min) & (df_did[year_col] <= year_max)].copy()
    
    # Convert to numeric
    df_did[tour_col] = pd.to_numeric(df_did[tour_col], errors="coerce")
    
    # Create ln(tourists)
    df_did["ln_tourists"] = np.where(df_did[tour_col] > 0, np.log(df_did[tour_col]), np.nan)
    
    # Create treated dummy
    df_did["treated"] = (df_did[country_col] == treated_country).astype(int)
    
    # Create event-time variable (relative to treatment year)
    df_did["event_time"] = df_did[year_col] - treat_year
    
    # Set panel index
    df_did = df_did.set_index([country_col, year_col])
    
    # Create event-time dummies (exclude -1 as reference period)
    event_times = sorted(df_did["event_time"].unique())
    event_dummies = []
    
    for t in event_times:
        if t != -1:  # -1 is the reference period
            dummy_name = f"event_t{t:+d}"
            df_did[dummy_name] = ((df_did["event_time"] == t) & (df_did["treated"] == 1)).astype(int)
            event_dummies.append(dummy_name)
    
    # Prepare regression dataset
    df_reg = df_did[["ln_tourists"] + event_dummies].dropna().copy()
    
    if verbose:
        print(f"Observations: {len(df_reg)}")
        print(f"Event-time periods: {len(event_dummies)}")
    
    # Run event-study regression
    y = df_reg["ln_tourists"]
    X = df_reg[event_dummies]
    
    model = PanelOLS(y, X, entity_effects=True, time_effects=True)
    result = model.fit(cov_type="robust")
    
    # Extract coefficients for plotting
    coefs_data = []
    for dummy in event_dummies:
        t = int(dummy.replace("event_t", ""))
        coef = result.params[dummy]
        se = result.std_errors[dummy]
        ci_low = coef - 1.96 * se
        ci_high = coef + 1.96 * se
        
        coefs_data.append({
            "event_time": t,
            "coefficient": coef,
            "se": se,
            "ci_low": ci_low,
            "ci_high": ci_high
        })
    
    # Add reference period (-1) with coefficient = 0
    coefs_data.append({
        "event_time": -1,
        "coefficient": 0.0,
        "se": 0.0,
        "ci_low": 0.0,
        "ci_high": 0.0
    })
    
    coefs_df = pd.DataFrame(coefs_data).sort_values("event_time")
    
    if verbose:
        print(f"\n{'='*70}")
        print("Event-study coefficients:")
        print(coefs_df.to_string(index=False))
        print(f"{'='*70}\n")
    
    return result, coefs_df


def plot_event_study(
    coefs_df: pd.DataFrame,
    title: str = None,
    figsize: Tuple[float, float] = (10, 6)
) -> plt.Figure:
    """
    Plot event-study coefficients with confidence intervals.
    
    Parameters
    ----------
    coefs_df : pd.DataFrame
        DataFrame with columns: event_time, coefficient, ci_low, ci_high
    title : str, optional
        Plot title
    figsize : tuple, default=(10, 6)
        Figure size
    
    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot coefficients
    ax.plot(coefs_df["event_time"], coefs_df["coefficient"], 
            marker="o", linewidth=2, markersize=8, color="steelblue", label="Coefficient")
    
    # Plot confidence intervals
    ax.fill_between(coefs_df["event_time"], 
                     coefs_df["ci_low"], 
                     coefs_df["ci_high"],
                     alpha=0.2, color="steelblue", label="95% CI")
    
    # Add zero line
    ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    
    # Add vertical line at treatment (event_time = 0)
    ax.axvline(0, color="red", linestyle="--", linewidth=2, alpha=0.7, label="Treatment")
    
    # Labels and formatting
    if title is None:
        title = "Event-Study: Dynamic Treatment Effects"
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Years Relative to Treatment", fontsize=12)
    ax.set_ylabel("Coefficient on ln(Tourists)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, loc="best")
    
    plt.tight_layout()
    
    return fig


def run_all_did_analyses(
    df: pd.DataFrame,
    specifications: Dict[str, Dict],
    show_plots: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Run DiD analyses for multiple specifications.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataset
    specifications : dict
        Dictionary mapping spec_name -> spec_config
        Each spec_config should have:
        - 'treated': str (treated country)
        - 'controls': list of str (control countries)
        - 'treat_year': int (treatment year)
    show_plots : bool, default=True
        Whether to create plots
    verbose : bool, default=True
        Whether to print progress
    
    Returns
    -------
    dict
        Dictionary mapping spec_name -> analysis results
    
    Examples
    --------
    >>> specs = {
    ...     'Brazil': {
    ...         'treated': 'Brazil',
    ...         'controls': ['Uruguay', 'Paraguay'],
    ...         'treat_year': 2011
    ...     },
    ...     'Netherlands': {
    ...         'treated': 'Netherlands',
    ...         'controls': ['Luxembourg', 'Sweden'],
    ...         'treat_year': 2006
    ...     }
    ... }
    >>> results = run_all_did_analyses(df, specs)
    """
    all_results = {}
    
    for spec_name, spec_config in specifications.items():
        if verbose:
            print("\n" + "="*80)
            print(f"SPECIFICATION: {spec_name}")
            print("="*80)
        
        results = run_did_analysis_pair(
            df,
            treated_country=spec_config['treated'],
            control_countries=spec_config['controls'],
            treat_year=spec_config['treat_year'],
            show_plots=show_plots,
            verbose=verbose
        )
        
        all_results[spec_name] = (results['result'], results['df_reg'])
        
        # Store plots separately if created
        if show_plots:
            all_results[f"{spec_name}_fig_raw"] = results.get('fig_raw')
            all_results[f"{spec_name}_fig_log"] = results.get('fig_log')
    
    # Create comparison table
    if verbose:
        print("\n" + "="*80)
        print("COMPARISON ACROSS SPECIFICATIONS")
        print("="*80 + "\n")
    
    # Filter only regression results (exclude plots)
    regression_results = {k: v for k, v in all_results.items() if not k.endswith('_fig_raw') and not k.endswith('_fig_log')}
    comparison_df = compare_did_results(regression_results)
    
    if verbose:
        print(comparison_df.to_string(index=False))
        print("\n" + "="*80 + "\n")
    
    all_results['comparison'] = comparison_df
    
    return all_results


def run_did_different_controls(
    df: pd.DataFrame,
    treated_country: str,
    control_a: str,
    control_b: str,
    treat_year: int,
    pre_start: int = 2001,
    post_years: int = 4,
    verbose: bool = True
) -> Dict:
    """
    Run DiD with different control group combinations for robustness.
    
    Tests three specifications:
    1. Treated vs Control A only
    2. Treated vs Control B only
    3. Treated vs Pooled controls (A+B)
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataset
    treated_country : str
        Name of treated country
    control_a : str
        First control country
    control_b : str
        Second control country
    treat_year : int
        Treatment year
    pre_start : int, default=2001
        Start year
    post_years : int, default=4
        Post-treatment years
    verbose : bool, default=True
        Whether to print progress
    
    Returns
    -------
    dict
        Dictionary with results for each specification
    """
    if verbose:
        print("\n" + "="*70)
        print(f"ROBUSTNESS: Different Control Groups")
        print(f"Treated: {treated_country} ({treat_year})")
        print("="*70)
    
    results = {}
    
    # Spec 1: Control A only
    if verbose:
        print(f"\n1️⃣ {treated_country} vs {control_a} only")
    result_a, _ = run_did_twfe(
        df, treated_country, [control_a], treat_year,
        pre_start=pre_start, post_years=post_years, verbose=False
    )
    results[f'{treated_country} vs {control_a}'] = result_a
    
    # Spec 2: Control B only
    if verbose:
        print(f"\n2️⃣ {treated_country} vs {control_b} only")
    result_b, _ = run_did_twfe(
        df, treated_country, [control_b], treat_year,
        pre_start=pre_start, post_years=post_years, verbose=False
    )
    results[f'{treated_country} vs {control_b}'] = result_b
    
    # Spec 3: Pooled controls
    if verbose:
        print(f"\n3️⃣ {treated_country} vs {control_a} + {control_b} (pooled)")
    result_pooled, _ = run_did_twfe(
        df, treated_country, [control_a, control_b], treat_year,
        pre_start=pre_start, post_years=post_years, verbose=False
    )
    results[f'{treated_country} vs {control_a}+{control_b}'] = result_pooled
    
    # Create comparison table
    comparison = []
    for spec_name, result in results.items():
        did_coef = result.params['did']
        did_se = result.std_errors['did']
        did_pval = result.pvalues['did']
        
        comparison.append({
            'Specification': spec_name,
            'DiD Coefficient': did_coef,
            'Std. Error': did_se,
            'P-value': did_pval,
            'N': int(result.nobs)
        })
    
    comparison_df = pd.DataFrame(comparison)
    
    if verbose:
        print("\n" + "="*70)
        print("COMPARISON TABLE")
        print("="*70)
        print(comparison_df.to_string(index=False))
        print("="*70 + "\n")
    
    return {
        'results': results,
        'comparison': comparison_df
    }


def run_did_different_windows(
    df: pd.DataFrame,
    treated_country: str,
    control_countries: List[str],
    treat_year: int,
    verbose: bool = True
) -> Dict:
    """
    Run DiD with different time windows for robustness.
    
    Tests different specifications:
    1. Pre: 2001, Post: D+2
    2. Pre: 2001, Post: D+3
    3. Pre: 2001, Post: D+4 (baseline)
    4. Pre: 2003, Post: D+4
    5. Pre: 2005, Post: D+4
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataset
    treated_country : str
        Name of treated country
    control_countries : list of str
        List of control countries
    treat_year : int
        Treatment year
    verbose : bool, default=True
        Whether to print progress
    
    Returns
    -------
    dict
        Dictionary with results for each window specification
    """
    if verbose:
        print("\n" + "="*70)
        print(f"ROBUSTNESS: Different Time Windows")
        print(f"Treated: {treated_country} ({treat_year})")
        print("="*70)
    
    results = {}
    
    # Different post lengths (pre fixed at 2001)
    for post_len in [2, 3, 4]:
        spec_name = f"Pre:2001, Post:D+{post_len}"
        if verbose:
            print(f"\n📅 {spec_name}")
        result, _ = run_did_twfe(
            df, treated_country, control_countries, treat_year,
            pre_start=2001, post_years=post_len, verbose=False
        )
        results[spec_name] = result
    
    # Different pre starts (post fixed at D+4)
    for pre_start in [2003, 2005]:
        spec_name = f"Pre:{pre_start}, Post:D+4"
        if verbose:
            print(f"\n📅 {spec_name}")
        result, _ = run_did_twfe(
            df, treated_country, control_countries, treat_year,
            pre_start=pre_start, post_years=4, verbose=False
        )
        results[spec_name] = result
    
    # Create comparison table
    comparison = []
    for spec_name, result in results.items():
        did_coef = result.params['did']
        did_se = result.std_errors['did']
        did_pval = result.pvalues['did']
        
        comparison.append({
            'Window': spec_name,
            'DiD Coefficient': did_coef,
            'Std. Error': did_se,
            'P-value': did_pval,
            'N': int(result.nobs)
        })
    
    comparison_df = pd.DataFrame(comparison)
    
    if verbose:
        print("\n" + "="*70)
        print("COMPARISON TABLE")
        print("="*70)
        print(comparison_df.to_string(index=False))
        print("="*70 + "\n")
    
    return {
        'results': results,
        'comparison': comparison_df
    }
