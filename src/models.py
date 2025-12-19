"""
Regression models and result tables.
Supports OLS with clustered SE and Fixed Effects (FE) panel models.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
from typing import List, Tuple, Optional, Dict
from .utils import p_to_stars, format_cell_html, display_html_table, get_t_critical


def fit_ols_clustered(
    df: pd.DataFrame,
    y_col: str,
    x_col: str,
    cluster_col: str,
    controls: Optional[List[str]] = None
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Fit OLS regression with cluster-robust standard errors.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset
    y_col : str
        Dependent variable
    x_col : str
        Main independent variable
    cluster_col : str
        Clustering variable (typically country)
    controls : list of str, optional
        Control variables to include
    
    Returns
    -------
    RegressionResultsWrapper
        Fitted OLS model with cluster-robust SE
    """
    # Select columns
    cols = [y_col, x_col, cluster_col]
    if controls:
        cols.extend(controls)
    
    d = df[cols].copy()
    
    # Convert to numeric
    for c in [y_col, x_col] + (controls or []):
        d[c] = pd.to_numeric(d[c], errors='coerce')
    
    d = d.dropna()
    
    # Prepare X and y
    x_vars = [x_col] + (controls or [])
    y = d[y_col]
    X = sm.add_constant(d[x_vars])
    
    # Fit with cluster-robust SE
    model = sm.OLS(y, X)
    result = model.fit(cov_type='cluster', cov_kwds={'groups': d[cluster_col]})
    
    return result


def fit_panel_fe(
    df: pd.DataFrame,
    y_col: str,
    x_col: str,
    entity_col: str,
    time_col: str,
    controls: Optional[List[str]] = None,
    entity_effects: bool = True,
    time_effects: bool = True
) -> 'PanelEffectsResults':
    """
    Fit Fixed Effects (FE) panel regression with cluster-robust SE.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset
    y_col : str
        Dependent variable
    x_col : str
        Main independent variable
    entity_col : str
        Entity identifier (typically country)
    time_col : str
        Time identifier (typically year)
    controls : list of str, optional
        Control variables
    entity_effects : bool, default True
        Include entity (country) fixed effects
    time_effects : bool, default True
        Include time (year) fixed effects
    
    Returns
    -------
    PanelEffectsResults
        Fitted FE model with cluster-robust SE
    """
    # Select columns
    cols = [y_col, x_col, entity_col, time_col]
    if controls:
        cols.extend(controls)
    
    d = df[cols].copy()
    
    # Convert to numeric
    for c in [y_col, x_col] + (controls or []):
        d[c] = pd.to_numeric(d[c], errors='coerce')
    
    d = d.dropna()
    
    # Set panel index
    d = d.set_index([entity_col, time_col])
    
    # Prepare variables
    x_vars = [x_col] + (controls or [])
    
    # Fit FE model
    model = PanelOLS(
        d[y_col],
        d[x_vars],
        entity_effects=entity_effects,
        time_effects=time_effects
    )
    
    result = model.fit(cov_type='clustered', cluster_entity=True)
    
    return result


def build_regression_table(
    models: List,
    model_names: List[str],
    var_labels: Optional[Dict[str, str]] = None,
    digits: int = 6
) -> pd.DataFrame:
    """
    Build formatted regression table from multiple models.
    
    Parameters
    ----------
    models : list
        List of fitted models (statsmodels or linearmodels)
    model_names : list of str
        Names for each model column
    var_labels : dict, optional
        Custom labels for variables {var_name: display_name}
    digits : int, default 6
        Number of decimal places
    
    Returns
    -------
    pd.DataFrame
        Formatted regression table with coefficients, SE, and stats
    """
    if var_labels is None:
        var_labels = {}
    
    # Collect all variables
    all_vars = set()
    for model in models:
        all_vars.update(model.params.index)
    
    # Order: const first, then others
    if 'const' in all_vars:
        rows = ['const'] + sorted([v for v in all_vars if v != 'const'])
    else:
        rows = sorted(all_vars)
    
    # Build coefficient table
    table = pd.DataFrame(index=rows, columns=model_names, dtype=object)
    
    for model, col_name in zip(models, model_names):
        params = model.params
        
        # Get standard errors (different attribute names)
        if hasattr(model, 'bse'):  # statsmodels
            ses = model.bse
        elif hasattr(model, 'std_errors'):  # linearmodels
            ses = model.std_errors
        else:
            raise ValueError("Cannot extract standard errors from model")
        
        pvals = model.pvalues
        
        for var in rows:
            if var in params.index:
                coef = params[var]
                se = ses[var]
                p = pvals[var]
                table.loc[var, col_name] = format_cell_html(coef, se, p, digits)
            else:
                table.loc[var, col_name] = ""
    
    # Apply variable labels
    if var_labels:
        table.index = [var_labels.get(v, v) for v in table.index]
    
    # Add model statistics
    stats = _build_stats_rows(models, model_names)
    final_table = pd.concat([table, stats], axis=0)
    
    return final_table


def _build_stats_rows(
    models: List,
    model_names: List[str]
) -> pd.DataFrame:
    """Build statistics rows for regression table."""
    stats_data = {
        'N': [],
        '# Countries': [],
        '# Years': [],
        'R² (OLS) / R² within (FE)': [],
        'Country FE': [],
        'Year FE': [],
        'SE': []
    }
    
    for model in models:
        # Sample size
        if hasattr(model, 'nobs'):
            stats_data['N'].append(str(int(model.nobs)))
        else:
            stats_data['N'].append("")
        
        # R-squared
        if hasattr(model, 'rsquared'):  # OLS
            stats_data['R² (OLS) / R² within (FE)'].append(f"{model.rsquared:.6f}")
        elif hasattr(model, 'rsquared_within'):  # FE
            stats_data['R² (OLS) / R² within (FE)'].append(f"{model.rsquared_within:.6f}")
        else:
            stats_data['R² (OLS) / R² within (FE)'].append("")
        
        # FE indicators and panel dimensions
        # Check if it's a panel FE model (has rsquared_within attribute)
        if hasattr(model, 'rsquared_within'):
            stats_data['Country FE'].append("Yes")
            stats_data['Year FE'].append("Yes")
            
            # Panel dimensions
            idx = model.model.dependent.index
            stats_data['# Countries'].append(str(idx.get_level_values(0).nunique()))
            stats_data['# Years'].append(str(idx.get_level_values(1).nunique()))
        else:
            stats_data['Country FE'].append("No")
            stats_data['Year FE'].append("No")
            stats_data['# Countries'].append("")
            stats_data['# Years'].append("")
        
        # SE type
        stats_data['SE'].append("Cluster (country)")
    
    stats_df = pd.DataFrame(stats_data, index=model_names).T
    return stats_df


def extract_beta_with_ci(
    model,
    var: str,
    alpha: float = 0.05
) -> Tuple[float, float, float, int]:
    """
    Extract coefficient, confidence interval, and number of clusters.
    
    Parameters
    ----------
    model
        Fitted regression model
    var : str
        Variable name
    alpha : float, default 0.05
        Significance level for CI
    
    Returns
    -------
    beta : float
        Coefficient estimate
    ci_low : float
        Lower bound of CI
    ci_high : float
        Upper bound of CI
    n_clusters : int
        Number of clusters
    """
    beta = float(model.params[var])
    
    # Get SE
    if hasattr(model, 'bse'):
        se = float(model.bse[var])
    else:
        se = float(model.std_errors[var])
    
    # Get number of clusters
    if hasattr(model, 'model') and hasattr(model.model, 'dependent'):
        # FE model
        n_clusters = model.model.dependent.index.get_level_values(0).nunique()
    else:
        # Try to infer from OLS
        n_clusters = 50  # Conservative default
    
    # Compute CI
    t_crit = get_t_critical(n_clusters, alpha)
    ci_low = beta - t_crit * se
    ci_high = beta + t_crit * se
    
    return beta, ci_low, ci_high, n_clusters


def run_baseline_specifications(
    df: pd.DataFrame,
    y_col: str = "tourists",
    x_col: str = "deaths",
    entity_col: str = "country",
    time_col: str = "year",
    controls: Optional[List[str]] = None
) -> Tuple[List, List[str]]:
    """
    Run standard baseline specifications for correlation analysis.
    
    Specifications:
    1. OLS (no FE, no controls) with cluster SE
    2. FE (country + year, no controls) with cluster SE
    3. FE (country + year + controls) with cluster SE
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset (already log-transformed)
    y_col : str, default "tourists"
        Dependent variable
    x_col : str, default "deaths"
        Main independent variable
    entity_col : str, default "country"
        Entity identifier
    time_col : str, default "year"
        Time identifier
    controls : list of str, optional
        Control variables. Default: ['wri_v', 'pop_density', 'gdp_pc']
    
    Returns
    -------
    models : list
        List of fitted models
    model_names : list of str
        Names for each specification
    """
    if controls is None:
        controls = ['wri_v', 'pop_density', 'gdp_pc']
    
    models = []
    model_names = []
    
    # (1) OLS no FE, no controls
    m1 = fit_ols_clustered(df, y_col, x_col, entity_col, controls=None)
    models.append(m1)
    model_names.append(f"(1) OLS: ln_{y_col} ~ ln_{x_col}")
    
    # (2) FE country+year, no controls
    m2 = fit_panel_fe(df, y_col, x_col, entity_col, time_col, controls=None)
    models.append(m2)
    model_names.append(f"(2) FE (country+year): ln_{y_col} ~ ln_{x_col}")
    
    # (3) FE country+year + controls
    m3 = fit_panel_fe(df, y_col, x_col, entity_col, time_col, controls=controls)
    models.append(m3)
    ctrl_str = " + ".join(controls)
    model_names.append(f"(3) FE (country+year) + controls: + {ctrl_str}")
    
    return models, model_names


def fit_interaction_model(
    df: pd.DataFrame,
    y_col: str,
    x_col: str,
    interaction_var: str,
    entity_col: str,
    time_col: str,
    controls: Optional[List[str]] = None
) -> 'PanelEffectsResults':
    """
    Fit FE model with interaction term.
    
    Model: y ~ x + x*interaction_var + FE + controls
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset
    y_col : str
        Dependent variable
    x_col : str
        Main independent variable
    interaction_var : str
        Variable to interact with x (e.g., 'well_prepared')
    entity_col : str
        Entity identifier
    time_col : str
        Time identifier
    controls : list of str, optional
        Control variables
    
    Returns
    -------
    PanelEffectsResults
        Fitted FE model with interaction
    """
    # Create interaction term
    interaction_name = f"{x_col}_x_{interaction_var}"
    df = df.copy()
    df[interaction_name] = pd.to_numeric(df[x_col], errors='coerce') * pd.to_numeric(df[interaction_var], errors='coerce')
    
    # Prepare X variables
    X_cols = [x_col, interaction_name]
    if controls:
        X_cols.extend(controls)
    
    # Prepare panel data
    cols = [entity_col, time_col, y_col] + X_cols
    panel = df[cols].dropna().copy()
    panel = panel.set_index([entity_col, time_col])
    
    # Fit FE model
    model = PanelOLS(
        panel[y_col],
        panel[X_cols],
        entity_effects=True,
        time_effects=True
    )
    
    result = model.fit(cov_type='clustered', cluster_entity=True)
    
    return result


def extract_group_effects(
    result,
    x_col: str,
    interaction_var: str
) -> Dict[str, Tuple[float, float]]:
    """
    Extract group-specific effects from an interaction model.
    
    For a model: y ~ x + x*group + controls
    Returns the effect of x for each group value (0 and 1).
    
    Parameters
    ----------
    result : PanelResults
        Fitted interaction model result
    x_col : str
        Name of the main effect variable (without 'ln_' prefix)
    interaction_var : str
        Name of the interaction variable
    
    Returns
    -------
    dict
        Dictionary with keys 'group_0' and 'group_1', each containing (beta, se) tuple
    
    Examples
    --------
    >>> result = fit_interaction_model(df, 'tourists', 'deaths', 'well_prepared')
    >>> effects = extract_group_effects(result, 'deaths', 'well_prepared')
    >>> beta_0, se_0 = effects['group_0']
    >>> beta_1, se_1 = effects['group_1']
    """
    # Get coefficients
    params = result.params
    cov = result.cov
    
    # Build variable names (use actual column name, not ln_ prefix)
    x_var = x_col
    interaction_term = f'{x_var}_x_{interaction_var}'
    
    # Extract betas
    beta_x = params[x_var]
    beta_interaction = params[interaction_term]
    
    # Group 0: beta = beta_x
    beta_0 = beta_x
    se_0 = np.sqrt(cov.loc[x_var, x_var])
    
    # Group 1: beta = beta_x + beta_interaction
    beta_1 = beta_x + beta_interaction
    
    # SE for group 1: sqrt(Var(beta_x) + Var(beta_int) + 2*Cov(beta_x, beta_int))
    var_x = cov.loc[x_var, x_var]
    var_int = cov.loc[interaction_term, interaction_term]
    cov_x_int = cov.loc[x_var, interaction_term]
    se_1 = np.sqrt(var_x + var_int + 2 * cov_x_int)
    
    return {
        'group_0': (beta_0, se_0),
        'group_1': (beta_1, se_1)
    }


def run_control_comparison(
    df: pd.DataFrame,
    y_col: str = 'tourists',
    x_col: str = 'deaths',
    entity_col: str = 'country',
    time_col: str = 'year',
    control_specs: Optional[List[Tuple[str, List[str]]]] = None,
    verbose: bool = True
) -> Dict:
    """
    Run multiple FE regressions with progressive addition of control variables.
    
    This function fits a series of panel FE models, each adding one more control variable,
    to show how the coefficient on the main variable changes as controls are added.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe (should be log-transformed)
    y_col : str, default='tourists'
        Dependent variable name (without 'ln_' prefix)
    x_col : str, default='deaths'
        Main independent variable (without 'ln_' prefix)
    entity_col : str, default='country'
        Entity identifier for fixed effects
    time_col : str, default='year'
        Time identifier for fixed effects
    control_specs : list of tuples, optional
        List of (name, controls) tuples. Each tuple is (model_name, list_of_controls).
        Default: Progressive addition of wri_v, pop_density, gdp_pc
    verbose : bool, default=True
        Whether to print progress messages
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'results': dict of {model_name: regression_result}
        - 'betas': DataFrame with beta, se, ci_low, ci_high for each model
        - 'model_names': list of model names
        - 'n_clusters': number of clusters (countries)
    
    Examples
    --------
    >>> # Default: progressive addition of controls
    >>> results = run_control_comparison(df)
    >>> 
    >>> # Custom control specs
    >>> specs = [
    ...     ("Model 1", []),
    ...     ("Model 2", ["wri_v"]),
    ...     ("Model 3", ["wri_v", "pop_density"])
    ... ]
    >>> results = run_control_comparison(df, control_specs=specs)
    """
    if control_specs is None:
        control_specs = [
            ("(1) FE: deaths", []),
            ("(2) FE: + wri_v", ["wri_v"]),
            ("(3) FE: + wri_v + pop_density", ["wri_v", "pop_density"]),
            ("(4) FE: + wri_v + pop_density + gdp_pc", ["wri_v", "pop_density", "gdp_pc"]),
        ]
    
    if verbose:
        print("=" * 60)
        print("CONTROL VARIABLE COMPARISON")
        print("=" * 60)
        print(f"Y: ln({y_col})")
        print(f"X: ln({x_col})")
        print(f"Number of specifications: {len(control_specs)}")
        print("=" * 60)
    
    # Fit all models
    results = {}
    model_names = []
    
    for i, (name, controls) in enumerate(control_specs, 1):
        if verbose:
            ctrl_str = ", ".join([f"ln({c})" for c in controls]) if controls else "none"
            print(f"\n{i}. {name}")
            print(f"   Controls: {ctrl_str}")
        
        # Fit FE model
        result = fit_panel_fe(
            df,
            y_col=y_col,
            x_col=x_col,
            entity_col=entity_col,
            time_col=time_col,
            controls=controls
        )
        
        results[name] = result
        model_names.append(name)
        
        if verbose:
            beta = result.params[x_col]
            se = result.std_errors[x_col]
            print(f"   β = {beta:.6f} (SE = {se:.6f})")
    
    # Extract betas and CIs
    betas_data = []
    n_clusters = None
    
    for name in model_names:
        result = results[name]
        
        beta = float(result.params[x_col])
        se = float(result.std_errors[x_col])
        
        # Get number of clusters for t-critical
        G = result.model.dependent.index.get_level_values(0).nunique()
        if n_clusters is None:
            n_clusters = G
        
        # Calculate 95% CI
        try:
            from scipy.stats import t
            t_crit = t.ppf(0.975, df=max(G - 1, 1))
        except Exception:
            t_crit = 1.96
        
        ci_low = beta - t_crit * se
        ci_high = beta + t_crit * se
        
        betas_data.append({
            'model': name,
            'beta': beta,
            'se': se,
            'ci_low': ci_low,
            'ci_high': ci_high
        })
    
    betas_df = pd.DataFrame(betas_data)
    
    if verbose:
        print("\n" + "=" * 60)
        print("✅ All models fitted successfully")
        print(f"   Number of clusters: {n_clusters}")
        print("=" * 60)
    
    return {
        'results': results,
        'betas': betas_df,
        'model_names': model_names,
        'n_clusters': n_clusters,
        'y_col': y_col,
        'x_col': x_col
    }


def run_heterogeneity_analysis(
    df: pd.DataFrame,
    y_col: str = 'tourists',
    x_col: str = 'deaths',
    interaction_var: str = 'well_prepared',
    entity_col: str = 'country',
    time_col: str = 'year',
    controls: Optional[List[str]] = None,
    group_labels: Optional[Dict[int, str]] = None,
    verbose: bool = True
) -> Dict:
    """
    Run complete heterogeneity analysis: fit interaction model, extract effects, and prepare plots.
    
    This is a wrapper function that orchestrates the entire heterogeneity analysis workflow:
    1. Fits an interaction model with fixed effects
    2. Extracts group-specific effects
    3. Prepares data for plotting
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe (should be log-transformed)
    y_col : str, default='tourists'
        Dependent variable name (without 'ln_' prefix)
    x_col : str, default='deaths'
        Main independent variable (without 'ln_' prefix)
    interaction_var : str, default='well_prepared'
        Binary interaction variable (0/1)
    entity_col : str, default='country'
        Entity identifier for fixed effects
    time_col : str, default='year'
        Time identifier for fixed effects
    controls : list of str, optional
        Control variables (without 'ln_' prefix)
    group_labels : dict, optional
        Labels for groups {0: 'label0', 1: 'label1'}
    verbose : bool, default=True
        Whether to print progress messages
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'result': regression result object
        - 'effects': group-specific effects dict
        - 'group_effects': formatted effects for plotting
        - 'n_clusters': number of clusters
        - 'controls_used': list of controls used
        - 'group_labels': group labels used
    
    Examples
    --------
    >>> # With WRI control
    >>> results_wri = run_heterogeneity_analysis(
    ...     df, controls=['wri_v', 'pop_density', 'gdp_pc']
    ... )
    >>> 
    >>> # Without WRI control
    >>> results_no_wri = run_heterogeneity_analysis(
    ...     df, controls=['pop_density', 'gdp_pc']
    ... )
    """
    if controls is None:
        controls = ['wri_v', 'pop_density', 'gdp_pc']
    
    if group_labels is None:
        group_labels = {0: 'Not well prepared', 1: 'Well prepared'}
    
    if verbose:
        print("=" * 60)
        print(f"HETEROGENEITY ANALYSIS")
        print("=" * 60)
        print(f"Y: ln({y_col})")
        print(f"X: ln({x_col})")
        print(f"Interaction: {interaction_var}")
        print(f"Controls: {', '.join(['ln(' + c + ')' for c in controls])}")
        print("=" * 60)
    
    # Step 1: Fit interaction model
    if verbose:
        print("\n1️⃣ Fitting interaction model...")
    
    result = fit_interaction_model(
        df,
        y_col=y_col,
        x_col=x_col,
        interaction_var=interaction_var,
        entity_col=entity_col,
        time_col=time_col,
        controls=controls
    )
    
    if verbose:
        print("   ✓ Model fitted successfully")
    
    # Step 2: Extract group-specific effects
    if verbose:
        print("\n2️⃣ Extracting group-specific effects...")
    
    effects = extract_group_effects(result, x_col, interaction_var)
    
    beta_0, se_0 = effects['group_0']
    beta_1, se_1 = effects['group_1']
    
    if verbose:
        print(f"   ✓ {group_labels[0]}: β = {beta_0:.6f} (SE = {se_0:.6f})")
        print(f"   ✓ {group_labels[1]}: β = {beta_1:.6f} (SE = {se_1:.6f})")
    
    # Step 3: Prepare data for plotting
    group_effects = {
        group_labels[0]: effects['group_0'],
        group_labels[1]: effects['group_1']
    }
    
    # Get number of clusters
    n_clusters = result.model.dependent.index.get_level_values(0).nunique()
    
    if verbose:
        print(f"\n3️⃣ Analysis complete")
        print(f"   ✓ Number of clusters: {n_clusters}")
        print("=" * 60)
    
    # Package results
    return {
        'result': result,
        'effects': effects,
        'group_effects': group_effects,
        'n_clusters': n_clusters,
        'controls_used': controls,
        'group_labels': group_labels,
        'y_col': y_col,
        'x_col': x_col,
        'interaction_var': interaction_var
    }
