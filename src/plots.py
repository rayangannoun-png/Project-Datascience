"""
Plotting functions for correlation analysis.
Includes scatter plots with CI bands, FE-residualized plots, and beta variation charts.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
from typing import Optional, Tuple, List, Dict
from .utils import get_t_critical


def plot_correlation_with_ci(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    cluster_col: str,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 7),
    sample_frac: Optional[float] = None,
    seed: int = 42
) -> plt.Figure:
    """
    Scatter plot with OLS regression line and cluster-robust 95% CI band.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset
    x_col : str
        X variable (independent)
    y_col : str
        Y variable (dependent)
    cluster_col : str
        Clustering variable for robust SE
    title : str, optional
        Plot title
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label
    figsize : tuple, default (12, 7)
        Figure size
    sample_frac : float, optional
        Fraction of data to plot (e.g., 0.3 for 30%). Useful for large datasets.
    seed : int, default 42
        Random seed for sampling
    
    Returns
    -------
    matplotlib.Figure
        Figure object
    """
    # Clean data
    d = df[[cluster_col, x_col, y_col]].copy()
    d[x_col] = pd.to_numeric(d[x_col], errors='coerce')
    d[y_col] = pd.to_numeric(d[y_col], errors='coerce')
    d = d.dropna()
    
    # Sample if requested
    if sample_frac is not None and 0 < sample_frac < 1:
        d = d.sample(frac=sample_frac, random_state=seed)
    
    # Fit OLS with cluster-robust SE
    y = d[y_col]
    X = sm.add_constant(d[[x_col]])
    model = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': d[cluster_col]})
    
    b0 = float(model.params['const'])
    b1 = float(model.params[x_col])
    
    # Get t-critical value
    n_clusters = d[cluster_col].nunique()
    t_crit = get_t_critical(n_clusters)
    
    # Compute CI band for mean prediction
    V = model.cov_params().loc[['const', x_col], ['const', x_col]].values
    
    xg = np.linspace(d[x_col].min(), d[x_col].max(), 200)
    yhat = b0 + b1 * xg
    se_pred = np.sqrt(V[0, 0] + 2*xg*V[0, 1] + (xg**2)*V[1, 1])
    
    ci_low = yhat - t_crit * se_pred
    ci_high = yhat + t_crit * se_pred
    
    # Plot
    sns.set(style="white", context="talk")
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.scatter(d[x_col], d[y_col], alpha=0.8, s=55, color='blue', label='Data')
    ax.plot(xg, yhat, linewidth=2.5, color='orange', label='OLS fit')
    ax.fill_between(xg, ci_low, ci_high, color='orange', alpha=0.15, label='95% CI (cluster-robust)')
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Relationship between {x_col} and {y_col}\n95% CI clustered by {cluster_col}")
    
    ax.set_xlabel(xlabel or x_col)
    ax.set_ylabel(ylabel or y_col)
    ax.legend(loc='best')
    ax.grid(False)
    
    plt.tight_layout()
    
    return fig


def plot_fe_residualized(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    entity_col: str,
    time_col: str,
    controls: Optional[List[str]] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 7),
    sample_frac: Optional[float] = None,
    seed: int = 42,
    show_ci: bool = True
) -> plt.Figure:
    """
    FE-residualized scatter plot (Frisch-Waugh-Lovell).
    
    Residualizes both x and y with respect to:
    - Entity (country) and time (year) fixed effects
    - Control variables (if provided)
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset
    x_col : str
        X variable
    y_col : str
        Y variable
    entity_col : str
        Entity identifier (country)
    time_col : str
        Time identifier (year)
    controls : list of str, optional
        Control variables to partial out
    title : str, optional
        Plot title
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label
    figsize : tuple, default (12, 7)
        Figure size
    sample_frac : float, optional
        Fraction of data to plot
    seed : int, default 42
        Random seed for sampling
    show_ci : bool, default True
        Show 95% CI band
    
    Returns
    -------
    matplotlib.Figure
        Figure object
    """
    # Clean data
    cols = [entity_col, time_col, x_col, y_col]
    if controls:
        cols.extend(controls)
    
    d = df[cols].copy()
    for c in [x_col, y_col] + (controls or []):
        d[c] = pd.to_numeric(d[c], errors='coerce')
    d = d.dropna()
    
    # Two-way FE demeaning
    def twoway_demean(s, ent, tim, max_iter=200, tol=1e-12):
        z = s.astype(float).copy()
        z = z - z.mean()
        for _ in range(max_iter):
            z_old = z.copy()
            z = z - z.groupby(ent).transform('mean')
            z = z - z.groupby(tim).transform('mean')
            if np.nanmax(np.abs(z - z_old)) < tol:
                break
        return z
    
    d['y_fe'] = twoway_demean(d[y_col], d[entity_col], d[time_col])
    d['x_fe'] = twoway_demean(d[x_col], d[entity_col], d[time_col])
    
    # Partial out controls (FWL)
    if controls:
        for c in controls:
            d[f'{c}_fe'] = twoway_demean(d[c], d[entity_col], d[time_col])
        
        Z = d[[f'{c}_fe' for c in controls]]
        y_res = sm.OLS(d['y_fe'], sm.add_constant(Z)).fit().resid
        x_res = sm.OLS(d['x_fe'], sm.add_constant(Z)).fit().resid
    else:
        y_res = d['y_fe']
        x_res = d['x_fe']
    
    # Sample if requested
    if sample_frac is not None and 0 < sample_frac < 1:
        sample_idx = y_res.sample(frac=sample_frac, random_state=seed).index
        y_res = y_res.loc[sample_idx]
        x_res = x_res.loc[sample_idx]
        d_sampled = d.loc[sample_idx]
    else:
        d_sampled = d
    
    # Fit regression on residualized vars (no intercept)
    model = sm.OLS(y_res, x_res).fit(
        cov_type='cluster',
        cov_kwds={'groups': d_sampled[entity_col]}
    )
    
    beta = float(model.params[0])
    se_beta = float(model.bse[0])
    
    # Get t-critical
    n_clusters = d_sampled[entity_col].nunique()
    t_crit = get_t_critical(n_clusters)
    
    # Plot
    sns.set(style="white", context="talk")
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.scatter(x_res, y_res, alpha=0.8, s=55, color='blue', label='FE-residualized data')
    
    xg = np.linspace(x_res.min(), x_res.max(), 200)
    yhat = beta * xg
    
    ax.plot(xg, yhat, linewidth=2.5, color='orange', label=f'Slope = {beta:.4f}')
    
    if show_ci:
        band = t_crit * np.abs(xg) * se_beta
        ax.fill_between(xg, yhat - band, yhat + band, color='orange', alpha=0.15, label='95% CI')
    
    if title:
        ax.set_title(title)
    else:
        ctrl_str = f" + controls ({', '.join(controls)})" if controls else ""
        ax.set_title(f"FE-adjusted: {y_col} ~ {x_col}{ctrl_str}\n95% CI clustered by {entity_col}")
    
    ax.set_xlabel(xlabel or f"{x_col} (FE-adjusted)")
    ax.set_ylabel(ylabel or f"{y_col} (FE-adjusted)")
    ax.legend(loc='best')
    ax.grid(False)
    
    plt.tight_layout()
    
    return fig


def plot_beta_variation(
    models: List,
    model_names: List[str],
    var: str,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: Tuple[int, int] = (10.5, 5.8),
    colors: Optional[dict] = None
) -> plt.Figure:
    """
    Plot coefficient estimates with 95% CI across specifications.
    
    Parameters
    ----------
    models : list
        List of fitted models
    model_names : list of str
        Names for each model
    var : str
        Variable to plot (e.g., 'deaths')
    title : str, optional
        Plot title
    ylabel : str, optional
        Y-axis label
    figsize : tuple, default (10.5, 5.8)
        Figure size
    colors : dict, optional
        Color mapping for each model
    
    Returns
    -------
    matplotlib.Figure
        Figure object
    """
    from .models import extract_beta_with_ci
    
    # Extract coefficients and CIs
    rows = []
    for model, name in zip(models, model_names):
        if var in model.params.index:
            beta, ci_low, ci_high, _ = extract_beta_with_ci(model, var)
            rows.append([name, beta, ci_low, ci_high])
    
    coef_df = pd.DataFrame(rows, columns=['model', 'beta', 'ci_low', 'ci_high'])
    
    # Default colors
    if colors is None:
        colors = {
            model_names[0]: '#9ecae1',  # light blue
            model_names[1]: '#f4a6a6',  # light red
            model_names[2]: '#a1d99b',  # light green
        }
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    xpos = np.arange(len(coef_df))
    box_w = 0.62
    
    for i, r in coef_df.iterrows():
        x = xpos[i]
        lo, hi, b = float(r['ci_low']), float(r['ci_high']), float(r['beta'])
        c = colors.get(r['model'], '#cccccc')
        
        # CI rectangle
        ax.add_patch(Rectangle(
            (x - box_w/2, lo), box_w, hi - lo,
            facecolor=c, edgecolor=c, alpha=0.45, zorder=1
        ))
        
        # Dashed vertical line through CI
        ax.vlines(x, lo, hi, colors='black', linestyles='--', linewidth=1.8, zorder=2)
        
        # Coefficient as thick horizontal black line
        ax.hlines(b, x - box_w/2, x + box_w/2, colors='black', linewidth=3.0, zorder=3)
        
        # Small caps at CI ends
        cap = box_w * 0.18
        ax.hlines(lo, x - cap/2, x + cap/2, colors='black', linewidth=1.6, zorder=3)
        ax.hlines(hi, x - cap/2, x + cap/2, colors='black', linewidth=1.6, zorder=3)
    
    # Zero line
    ax.axhline(0, color='grey', linestyle='--', linewidth=1.5, alpha=0.8)
    
    # Labels
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"Model {i+1}" for i in range(len(coef_df))], fontsize=12)
    ax.set_ylabel(ylabel or f"Beta estimate (coefficient on {var})", fontsize=13)
    ax.set_title(title or "Betas across specifications (95% CI clustered by country)", fontsize=16)
    
    # Grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.35)
    ax.xaxis.grid(False)
    
    # Y-limits padding
    ymin = float(coef_df['ci_low'].min())
    ymax = float(coef_df['ci_high'].max())
    pad = 0.12 * (ymax - ymin) if ymax > ymin else 0.1
    ax.set_ylim(ymin - pad, ymax + pad)
    
    # Legend
    legend_items = [
        Patch(facecolor=colors.get(name, '#ccc'), edgecolor=colors.get(name, '#ccc'),
              alpha=0.45, label=f"{name}: CI")
        for name in model_names if name in colors
    ]
    legend_items.append(Line2D([0], [0], color='black', linewidth=3, label='Coefficient'))
    ax.legend(handles=legend_items, loc='upper right', frameon=True)
    
    plt.tight_layout()
    
    return fig


def plot_heterogeneity_comparison(
    group_effects: dict,
    group_labels: List[str],
    n_clusters: int,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: Tuple[int, int] = (10.5, 5.8),
    colors: Optional[dict] = None
) -> plt.Figure:
    """
    Plot comparison of coefficients across groups with 95% CI.
    
    Parameters
    ----------
    group_effects : dict
        Dictionary with keys matching group_labels, values are (beta, se) tuples
    group_labels : list of str
        Labels for each group (e.g., ['Not well prepared', 'Well prepared'])
    n_clusters : int
        Number of clusters for t-critical value
    title : str, optional
        Plot title
    ylabel : str, optional
        Y-axis label
    figsize : tuple, default (10.5, 5.8)
        Figure size
    colors : dict, optional
        Color mapping for each group
    
    Returns
    -------
    matplotlib.Figure
        Figure object
    """
    # Default colors
    if colors is None:
        colors = {
            group_labels[0]: '#f4a6a6',  # light red/orange
            group_labels[1]: '#a1d99b',  # light green
        }
    
    # Get t-critical value
    t_crit = get_t_critical(n_clusters)
    
    # Prepare data
    rows = []
    for label in group_labels:
        beta, se = group_effects[label]
        ci_low = beta - t_crit * se
        ci_high = beta + t_crit * se
        rows.append([label, beta, ci_low, ci_high])
    
    coef_df = pd.DataFrame(rows, columns=['group', 'beta', 'ci_low', 'ci_high'])
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    xpos = np.arange(len(coef_df))
    box_w = 0.62
    
    for i, r in coef_df.iterrows():
        x = xpos[i]
        lo, hi, b = float(r['ci_low']), float(r['ci_high']), float(r['beta'])
        c = colors.get(r['group'], '#cccccc')
        
        # CI rectangle
        ax.add_patch(Rectangle(
            (x - box_w/2, lo), box_w, hi - lo,
            facecolor=c, edgecolor=c, alpha=0.45, zorder=1
        ))
        
        # Dashed vertical line through CI
        ax.vlines(x, lo, hi, colors='black', linestyles='--', linewidth=1.8, zorder=2)
        
        # Coefficient as thick horizontal black line
        ax.hlines(b, x - box_w/2, x + box_w/2, colors='black', linewidth=3.0, zorder=3)
        
        # Small caps at CI ends
        cap = box_w * 0.18
        ax.hlines(lo, x - cap/2, x + cap/2, colors='black', linewidth=1.6, zorder=3)
        ax.hlines(hi, x - cap/2, x + cap/2, colors='black', linewidth=1.6, zorder=3)
    
    # Zero line
    ax.axhline(0, color='grey', linestyle='--', linewidth=1.5, alpha=0.8)
    
    # Labels
    ax.set_xticks(xpos)
    ax.set_xticklabels(group_labels, fontsize=12)
    ax.set_ylabel(ylabel or "Beta estimate", fontsize=13)
    ax.set_title(title or "Heterogeneity by Group (95% CI clustered)", fontsize=16)
    
    # Grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.35)
    ax.xaxis.grid(False)
    
    # Y-limits padding
    ymin = float(coef_df['ci_low'].min())
    ymax = float(coef_df['ci_high'].max())
    pad = 0.12 * (ymax - ymin) if ymax > ymin else 0.1
    ax.set_ylim(ymin - pad, ymax + pad)
    
    # Legend
    legend_items = [
        Patch(facecolor=colors.get(label, '#ccc'), edgecolor=colors.get(label, '#ccc'),
              alpha=0.45, label=f"{label}: CI")
        for label in group_labels if label in colors
    ]
    legend_items.append(Line2D([0], [0], color='black', linewidth=3, label='Coefficient'))
    ax.legend(handles=legend_items, loc='upper right', frameon=True)
    
    plt.tight_layout()
    
    return fig


def plot_heterogeneity_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: str,
    entity_col: str,
    time_col: str,
    controls: Optional[List[str]] = None,
    group_labels: Optional[dict] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 7),
    sample_frac: float = 0.30,
    seed: int = 42,
    ylim: Optional[Tuple[float, float]] = None,
    xlim: Optional[Tuple[float, float]] = None,
    line_colors: Optional[dict] = None
) -> plt.Figure:
    """
    FE-residualized scatter plot with separate regression lines by group.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset
    x_col : str
        X variable
    y_col : str
        Y variable
    group_col : str
        Grouping variable (e.g., 'well_prepared')
    entity_col : str
        Entity identifier
    time_col : str
        Time identifier
    controls : list of str, optional
        Control variables to partial out
    group_labels : dict, optional
        Labels for groups {0: 'label0', 1: 'label1'}
    title : str, optional
        Plot title
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label
    figsize : tuple, default (12, 7)
        Figure size
    sample_frac : float, default 0.30
        Fraction of data to plot
    seed : int, default 42
        Random seed
    ylim : tuple, optional
        Y-axis limits
    xlim : tuple, optional
        X-axis limits
    line_colors : dict, optional
        Colors for each group {0: 'color0', 1: 'color1'}
    
    Returns
    -------
    matplotlib.Figure
        Figure object
    """
    import statsmodels.formula.api as smf
    
    # Default labels and colors
    if group_labels is None:
        group_labels = {0: 'Group 0', 1: 'Group 1'}
    if line_colors is None:
        line_colors = {0: 'orange', 1: 'red'}
    
    def Qname(col: str) -> str:
        """Safe Patsy quoting"""
        escaped = col.replace('"', '\\"')
        return f'Q("{escaped}")'
    
    # Prepare data
    use_cols = [entity_col, time_col, y_col, x_col, group_col]
    if controls:
        use_cols.extend(controls)
    
    d = df[use_cols].dropna().copy()
    
    # Residualize y and x on FE + controls
    if controls:
        ctrl_terms = " + ".join([Qname(c) for c in controls])
    else:
        ctrl_terms = "1"
    
    f_y = f"{Qname(y_col)} ~ {ctrl_terms} + C({Qname(entity_col)}) + C({Qname(time_col)})"
    f_x = f"{Qname(x_col)} ~ {ctrl_terms} + C({Qname(entity_col)}) + C({Qname(time_col)})"
    
    d['y_resid'] = smf.ols(f_y, data=d).fit().resid
    d['x_resid'] = smf.ols(f_x, data=d).fit().resid
    
    # Fit separate lines by group
    def fit_line_with_cluster_ci(df_sub):
        X = sm.add_constant(df_sub['x_resid'].astype(float))
        y = df_sub['y_resid'].astype(float)
        g = df_sub[entity_col]
        fit = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': g})
        
        b = fit.params.values
        V = fit.cov_params().values
        
        x_min, x_max = np.quantile(df_sub['x_resid'], 0.01), np.quantile(df_sub['x_resid'], 0.99)
        x_grid = np.linspace(x_min, x_max, 200)
        Xg = np.column_stack([np.ones_like(x_grid), x_grid])
        
        y_hat = Xg @ b
        se = np.sqrt(np.einsum("ij,jk,ik->i", Xg, V, Xg))
        ci_low = y_hat - 1.96 * se
        ci_high = y_hat + 1.96 * se
        return fit, x_grid, y_hat, ci_low, ci_high
    
    # Get unique group values
    groups = sorted(d[group_col].unique())
    fits = {}
    
    for grp in groups:
        d_grp = d[d[group_col] == grp].copy()
        fits[grp] = fit_line_with_cluster_ci(d_grp)
    
    # Sample for visibility
    d_plot = d.sample(frac=sample_frac, random_state=seed)
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.scatter(d_plot['x_resid'], d_plot['y_resid'], s=28, alpha=0.8, color='blue',
              label=f'Observations ({int(sample_frac*100)}% sample)')
    
    for grp in groups:
        fit, x_grid, y_hat, ci_low, ci_high = fits[grp]
        color = line_colors.get(grp, 'black')
        label = group_labels.get(grp, f'Group {grp}')
        ax.plot(x_grid, y_hat, linewidth=2.5, color=color, label=label)
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Heterogeneity: {y_col} vs {x_col} — FE-adjusted")
    
    ax.set_xlabel(xlabel or f"{x_col} (FE-adjusted)")
    ax.set_ylabel(ylabel or f"{y_col} (FE-adjusted)")
    
    if ylim:
        ax.set_ylim(ylim)
    if xlim:
        ax.set_xlim(xlim)
    
    ax.grid(False)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
    
    plt.tight_layout()
    
    return fig


def plot_boxplots_transformation(
    df: pd.DataFrame,
    variables: List[str] = None,
    labels: Optional[dict] = None,
    figsize: Tuple[int, int] = (20, 8),
    groupby: str = 'country'
) -> plt.Figure:
    """
    Create boxplots showing distributions before and after log transformation.
    
    This function creates a 2-row grid of boxplots:
    - Row 1: Original variables (country means)
    - Row 2: Log-transformed variables (country means)
    
    This visualization justifies the use of log transformation in regressions
    by showing how it normalizes skewed distributions.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    variables : list of str, optional
        List of variables to plot. Default: ['deaths', 'tourists', 'gdp_pc', 'pop_density', 'wri_v']
    labels : dict, optional
        Dictionary mapping variable names to display labels
    figsize : tuple, default=(20, 8)
        Figure size
    groupby : str, default='country'
        Column to group by for calculating means
    
    Returns
    -------
    matplotlib.figure.Figure
        Figure object with boxplots
    
    Examples
    --------
    >>> fig = plot_boxplots_transformation(df)
    >>> plt.show()
    """
    if variables is None:
        variables = ['deaths', 'tourists', 'gdp_pc', 'pop_density', 'wri_v']
    
    if labels is None:
        labels = {
            'deaths': 'Deaths',
            'tourists': 'Tourists',
            'gdp_pc': 'GDP per Capita',
            'pop_density': 'Population Density',
            'wri_v': 'WRI Vulnerability'
        }
    
    # Calculate country-level means
    df_means = df.groupby(groupby)[variables].mean().reset_index()
    
    # Add log-transformed columns (with +1 for deaths and wri_v)
    for var in variables:
        if var in ['deaths', 'wri_v']:
            df_means[f'ln_{var}'] = np.log(df_means[var] + 1)
        else:
            df_means[f'ln_{var}'] = np.log(df_means[var])
    
    # Create figure
    fig, axes = plt.subplots(2, len(variables), figsize=figsize)
    fig.suptitle('Distribution Before and After Log Transformation (Country Means)', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    # Row 1: Original variables
    for i, var in enumerate(variables):
        ax = axes[0, i]
        ax.boxplot(
            df_means[var].dropna(), 
            vert=True, 
            patch_artist=True,
            boxprops=dict(facecolor='lightblue', alpha=0.7),
            medianprops=dict(color='red', linewidth=2),
            whiskerprops=dict(color='blue'),
            capprops=dict(color='blue'),
            flierprops=dict(marker='o', markerfacecolor='red', markersize=4, alpha=0.5)
        )
        ax.set_title(labels[var], fontsize=12, fontweight='bold')
        ax.set_ylabel('Original Scale\n(Country Means)', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # Row 2: Log-transformed variables
    for i, var in enumerate(variables):
        ax = axes[1, i]
        ax.boxplot(
            df_means[f'ln_{var}'].dropna(), 
            vert=True, 
            patch_artist=True,
            boxprops=dict(facecolor='lightgreen', alpha=0.7),
            medianprops=dict(color='darkgreen', linewidth=2),
            whiskerprops=dict(color='green'),
            capprops=dict(color='green'),
            flierprops=dict(marker='o', markerfacecolor='darkgreen', markersize=4, alpha=0.5)
        )
        
        # Add note for deaths and wri_v about +1
        title_suffix = ' (+1)' if var in ['deaths', 'wri_v'] else ''
        ax.set_title(f'ln({labels[var]}{title_suffix})', fontsize=12, fontweight='bold')
        ax.set_ylabel('Log Scale\n(Country Means)', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig


def plot_correlation_matrix(
    df: pd.DataFrame,
    variables: List[str] = None,
    labels: Optional[dict] = None,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = 'RdBu_r',
    annot: bool = True,
    fmt: str = '.3f'
) -> plt.Figure:
    """
    Create a clean and aesthetic correlation matrix heatmap.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    variables : list of str, optional
        List of variables to include. Default: ['ln_deaths', 'ln_tourists', 'ln_gdp_pc', 'ln_pop_density', 'ln_wri_v']
    labels : dict, optional
        Dictionary mapping variable names to display labels
    figsize : tuple, default=(10, 8)
        Figure size
    cmap : str, default='RdBu_r'
        Colormap (Red-Blue reversed)
    annot : bool, default=True
        Whether to show numerical values
    fmt : str, default='.3f'
        Format for numerical values
    
    Returns
    -------
    matplotlib.figure.Figure
        Figure object with correlation matrix
    
    Examples
    --------
    >>> fig = plot_correlation_matrix(df)
    >>> plt.show()
    """
    if variables is None:
        variables = ['ln_deaths', 'ln_tourists', 'ln_gdp_pc', 'ln_pop_density', 'ln_wri_v']
    
    if labels is None:
        labels = {
            'ln_deaths': 'ln(Deaths)',
            'ln_tourists': 'ln(Tourists)',
            'ln_gdp_pc': 'ln(GDP pc)',
            'ln_pop_density': 'ln(Pop Density)',
            'ln_wri_v': 'ln(WRI Vuln.)'
        }
    
    # Select and rename columns
    df_corr = df[variables].copy()
    df_corr.columns = [labels.get(v, v) for v in variables]
    
    # Calculate correlation matrix
    corr_matrix = df_corr.corr()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        corr_matrix,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=1,
        linecolor='white',
        cbar_kws={
            'label': 'Correlation Coefficient',
            'shrink': 0.8,
            'orientation': 'vertical'
        },
        annot_kws={'size': 11, 'weight': 'bold'},
        ax=ax
    )
    
    # Customize plot
    ax.set_title('Correlation Matrix: Log-Transformed Variables', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Rotate labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=11)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11)
    
    plt.tight_layout()
    
    return fig


def plot_beta_comparison(
    betas_df: pd.DataFrame,
    model_order: Optional[List[str]] = None,
    colors: Optional[Dict[str, str]] = None,
    title: str = "Betas across FE specifications (95% CI clustered by country)",
    ylabel: str = "Beta estimate (coefficient on ln(deaths+1))",
    figsize: Tuple[float, float] = (11.5, 6.2)
) -> plt.Figure:
    """
    Plot coefficient comparison across multiple model specifications.
    
    Creates a visualization showing how the coefficient on the main variable changes
    as control variables are progressively added. Each model is shown with:
    - Colored rectangle for 95% CI
    - Black horizontal line for point estimate
    - Dashed vertical line through CI
    
    Parameters
    ----------
    betas_df : pd.DataFrame
        DataFrame with columns: 'model', 'beta', 'ci_low', 'ci_high'
    model_order : list of str, optional
        Order of models on x-axis. If None, uses order from betas_df
    colors : dict, optional
        Color mapping {model_name: color}. If None, uses default palette
    title : str
        Plot title
    ylabel : str
        Y-axis label
    figsize : tuple
        Figure size (width, height)
    
    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    
    Examples
    --------
    >>> results = run_control_comparison(df)
    >>> fig = plot_beta_comparison(results['betas'])
    >>> plt.show()
    """
    import numpy as np
    from matplotlib.patches import Rectangle, Patch
    from matplotlib.lines import Line2D
    
    # Default colors (light pastel palette)
    if colors is None:
        default_colors = {
            0: "#9ecae1",  # light blue
            1: "#a1d99b",  # light green
            2: "#fdd0a2",  # light orange
            3: "#cbc9e2",  # light purple
            4: "#fdae6b",  # orange
            5: "#bcbddc",  # purple
        }
        # Map to model names
        colors = {}
        for i, model in enumerate(betas_df['model'].unique()):
            colors[model] = default_colors.get(i, "#cccccc")
    
    # Set model order
    if model_order is None:
        model_order = betas_df['model'].tolist()
    
    # Sort DataFrame by model order
    betas_df = betas_df.copy()
    betas_df['model'] = pd.Categorical(betas_df['model'], categories=model_order, ordered=True)
    betas_df = betas_df.sort_values('model').reset_index(drop=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    xpos = np.arange(len(betas_df))
    box_w = 0.62
    
    # Plot each model
    for i, row in betas_df.iterrows():
        model = row['model']
        x = xpos[i]
        lo = float(row['ci_low'])
        hi = float(row['ci_high'])
        beta = float(row['beta'])
        color = colors.get(model, "#cccccc")
        
        # CI rectangle
        ax.add_patch(Rectangle(
            (x - box_w/2, lo), box_w, hi - lo,
            facecolor=color, edgecolor=color, alpha=0.55, zorder=1
        ))
        
        # Dashed vertical line through CI
        ax.vlines(x, lo, hi, colors="black", linestyles="--", linewidth=1.8, zorder=2)
        
        # Coefficient as thick horizontal black line
        ax.hlines(beta, x - box_w/2, x + box_w/2, colors="black", linewidth=3.0, zorder=3)
        
        # Small caps at CI ends
        cap = box_w * 0.18
        ax.hlines(lo, x - cap/2, x + cap/2, colors="black", linewidth=1.6, zorder=3)
        ax.hlines(hi, x - cap/2, x + cap/2, colors="black", linewidth=1.6, zorder=3)
    
    # Zero reference line
    ax.axhline(0, color="grey", linestyle="--", linewidth=1.5, alpha=0.8)
    
    # Labels and formatting
    ax.set_xticks(xpos)
    ax.set_xticklabels(model_order, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=16)
    
    # Grid
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax.xaxis.grid(False)
    
    # Y-limits with padding
    ymin = float(betas_df['ci_low'].min())
    ymax = float(betas_df['ci_high'].max())
    pad = 0.12 * (ymax - ymin) if ymax > ymin else 0.1
    ax.set_ylim(ymin - pad, ymax + pad)
    
    # Legend (CI per model + coefficient)
    legend_items = [
        Patch(facecolor=colors[m], edgecolor=colors[m], alpha=0.55, label=f"{m}: CI")
        for m in model_order
    ]
    legend_items.append(Line2D([0], [0], color="black", linewidth=3, label="Coefficient"))
    
    # Place legend outside plot area
    ax.legend(handles=legend_items, loc="upper left", bbox_to_anchor=(1.02, 1), frameon=True)
    
    plt.tight_layout()
    
    return fig
