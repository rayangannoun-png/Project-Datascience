"""
Utility functions for formatting and display.
"""

import pandas as pd
import numpy as np
from scipy.stats import t as t_dist
from typing import Optional


def p_to_stars(p: float) -> str:
    """
    Convert p-value to significance stars.
    
    Parameters
    ----------
    p : float
        P-value
    
    Returns
    -------
    str
        Significance stars: *** (p<0.01), ** (p<0.05), * (p<0.10), or empty
    """
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def format_cell_html(
    coef: float,
    se: float,
    p: float,
    digits: int = 6
) -> str:
    """
    Format regression coefficient cell for HTML table.
    
    Format: coefficient*** (standard error)
    
    Parameters
    ----------
    coef : float
        Coefficient estimate
    se : float
        Standard error
    p : float
        P-value
    digits : int, default 6
        Number of decimal places
    
    Returns
    -------
    str
        HTML-formatted cell content
    """
    if pd.isna(coef):
        return ""
    
    stars = p_to_stars(p)
    return f"{coef:.{digits}f}{stars}<br><span style='font-size:12px;'>({se:.{digits}f})</span>"


def format_cell_latex(
    coef: float,
    se: float,
    p: float,
    digits: int = 4
) -> str:
    """
    Format regression coefficient cell for LaTeX table.
    
    Parameters
    ----------
    coef : float
        Coefficient estimate
    se : float
        Standard error
    p : float
        P-value
    digits : int, default 4
        Number of decimal places
    
    Returns
    -------
    str
        LaTeX-formatted cell content
    """
    if pd.isna(coef):
        return ""
    
    stars = p_to_stars(p)
    return f"{coef:.{digits}f}{stars} ({se:.{digits}f})"


def display_html_table(df_table: pd.DataFrame) -> None:
    """
    Display pandas DataFrame as styled HTML table (for Jupyter).
    
    Parameters
    ----------
    df_table : pd.DataFrame
        Table to display
    """
    from IPython.display import display, HTML
    
    html = df_table.to_html(escape=False, border=0)
    html = f"""
    <div style="max-width: 100%; overflow-x: auto;">
      <style>
        table {{ border-collapse: collapse; font-family: Arial, sans-serif; font-size: 14px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; text-align: center; }}
        th {{ background: #f5f5f5; font-weight: 700; }}
        td:first-child, th:first-child {{ text-align: left; font-weight: 700; }}
      </style>
      {html}
    </div>
    """
    display(HTML(html))


def get_t_critical(
    n_clusters: int,
    alpha: float = 0.05
) -> float:
    """
    Get t-critical value for cluster-robust inference.
    
    Uses df = n_clusters - 1 for cluster-robust standard errors.
    
    Parameters
    ----------
    n_clusters : int
        Number of clusters
    alpha : float, default 0.05
        Significance level (0.05 for 95% CI)
    
    Returns
    -------
    float
        t-critical value
    """
    df = max(n_clusters - 1, 1)
    try:
        return t_dist.ppf(1 - alpha/2, df=df)
    except Exception:
        return 1.96  # Fallback to normal approximation


def detect_country_column(df: pd.DataFrame) -> Optional[str]:
    """
    Auto-detect country column from common names.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset
    
    Returns
    -------
    str or None
        Name of country column, or None if not found
    """
    candidates = ["country", "Country", "ISO3", "iso3", "ISO", "Entity", "Name"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def detect_year_column(df: pd.DataFrame) -> Optional[str]:
    """
    Auto-detect year column from common names.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset
    
    Returns
    -------
    str or None
        Name of year column, or None if not found
    """
    candidates = ["year", "Year", "TIME", "time", "period", "Period"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def print_section_header(title: str, width: int = 70) -> None:
    """
    Print a formatted section header.
    
    Parameters
    ----------
    title : str
        Section title
    width : int, default 70
        Width of separator line
    """
    print(f"\n{'='*width}")
    print(f"{title}")
    print(f"{'='*width}\n")


def display_data_summary(
    df_all: pd.DataFrame,
    df_low: pd.DataFrame,
    df_high: pd.DataFrame,
    transformed_vars: Optional[list] = None,
    show_preview: bool = True,
    preview_rows: int = 10,
    show_summary: bool = True
) -> None:
    """
    Display a compact HTML summary table of loaded datasets with scrollable preview.
    
    Parameters
    ----------
    df_all : pd.DataFrame
        Complete dataset
    df_low : pd.DataFrame
        Well-prepared countries dataset
    df_high : pd.DataFrame
        Not well-prepared countries dataset
    transformed_vars : list, optional
        List of variables that were log-transformed
    show_preview : bool, default True
        Whether to show scrollable DataFrame preview
    preview_rows : int, default 10
        Number of rows to show in preview
    show_summary : bool, default True
        Whether to show the summary statistics table
    """
    from IPython.display import HTML, display
    
    if transformed_vars is None:
        transformed_vars = ['deaths', 'tourists', 'wri_v', 'pop_density', 'gdp_pc']
    
    vars_str = ', '.join(transformed_vars)
    
    # Generate DataFrame preview HTML
    preview_html = ""
    if show_preview:
        df_preview = df_all.head(preview_rows)
        preview_table = df_preview.to_html(
            index=False,
            classes='preview-table',
            border=0,
            float_format=lambda x: f'{x:.2f}' if isinstance(x, float) else x
        )
        preview_html = f"""
        <div style="margin-top: 15px;">
            <h4 style="margin-bottom: 10px;">📋 Data Preview (first {preview_rows} rows)</h4>
            <div style="max-height: 300px; overflow: auto; border: 1px solid #ddd; border-radius: 5px;">
                <style>
                    .preview-table {{
                        width: 100%;
                        border-collapse: collapse;
                        font-size: 12px;
                    }}
                    .preview-table th {{
                        position: sticky;
                        top: 0;
                        background-color: #4CAF50;
                        color: white;
                        padding: 8px;
                        text-align: left;
                        font-weight: bold;
                        border: 1px solid #ddd;
                        z-index: 10;
                    }}
                    .preview-table td {{
                        padding: 6px 8px;
                        border: 1px solid #eee;
                        text-align: left;
                    }}
                    .preview-table tr:nth-child(even) {{
                        background-color: #f9f9f9;
                    }}
                    .preview-table tr:hover {{
                        background-color: #f0f0f0;
                    }}
                </style>
                {preview_table}
            </div>
        </div>
        """
    
    # Generate summary table HTML
    summary_table_html = ""
    if show_summary:
        summary_table_html = f"""
        <h4 style="margin-top: 0;">📊 Data Loaded Successfully</h4>
        <table style="width: 100%; border-collapse: collapse;">
            <tr style="background-color: #f0f0f0;">
                <th style="padding: 8px; text-align: left; border-bottom: 2px solid #ddd;">Dataset</th>
                <th style="padding: 8px; text-align: right; border-bottom: 2px solid #ddd;">Observations</th>
                <th style="padding: 8px; text-align: right; border-bottom: 2px solid #ddd;">Countries</th>
                <th style="padding: 8px; text-align: right; border-bottom: 2px solid #ddd;">Years</th>
            </tr>
            <tr>
                <td style="padding: 8px; border-bottom: 1px solid #eee;"><b>Complete Dataset</b></td>
                <td style="padding: 8px; text-align: right; border-bottom: 1px solid #eee;">{df_all.shape[0]:,}</td>
                <td style="padding: 8px; text-align: right; border-bottom: 1px solid #eee;">{df_all['country'].nunique()}</td>
                <td style="padding: 8px; text-align: right; border-bottom: 1px solid #eee;">{df_all['year'].min()}–{df_all['year'].max()}</td>
            </tr>
            <tr>
                <td style="padding: 8px; border-bottom: 1px solid #eee;">Well-prepared (low WRI)</td>
                <td style="padding: 8px; text-align: right; border-bottom: 1px solid #eee;">{df_low.shape[0]:,}</td>
                <td style="padding: 8px; text-align: right; border-bottom: 1px solid #eee;">{df_low['country'].nunique()}</td>
                <td style="padding: 8px; text-align: right; border-bottom: 1px solid #eee;">{df_low['year'].min()}–{df_low['year'].max()}</td>
            </tr>
            <tr>
                <td style="padding: 8px;">Not well-prepared (high WRI)</td>
                <td style="padding: 8px; text-align: right;">{df_high.shape[0]:,}</td>
                <td style="padding: 8px; text-align: right;">{df_high['country'].nunique()}</td>
                <td style="padding: 8px; text-align: right;">{df_high['year'].min()}–{df_high['year'].max()}</td>
            </tr>
        </table>
        <p style="margin-bottom: 0; margin-top: 10px; font-size: 0.9em; color: #666;">
            ✓ Log transformations applied to: {vars_str}
        </p>
        """
    
    summary_html = f"""
    <div style="border: 1px solid #ddd; padding: 10px; border-radius: 5px;">
        {summary_table_html}
        {preview_html}
    </div>
    """
    display(HTML(summary_html))
