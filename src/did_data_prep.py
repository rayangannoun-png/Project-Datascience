"""
Difference-in-Differences: Data Preparation

This module contains functions for preparing data for DiD analysis:
- Identifying treated countries (high disaster deaths)
- Creating tourism growth tables
- Matching treated countries with control countries
- Ranking countries by mean deaths
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from IPython.display import display, HTML


def load_did_data(url: str = None) -> pd.DataFrame:
    """
    Load data for DiD analysis with robust separator detection.
    
    Parameters
    ----------
    url : str, optional
        URL to CSV file. If None, uses default GitHub URL.
    
    Returns
    -------
    pd.DataFrame
        Loaded dataset
    """
    if url is None:
        url = "https://raw.githubusercontent.com/rayangannoun-png/Project-Datascience/refs/heads/main/Final_dataset_v.csv"
    
    # Robust CSV reading (try ; then ,)
    try:
        df = pd.read_csv(url, sep=";", engine="python")
        if df.shape[1] == 1:  # wrong separator
            df = pd.read_csv(url, sep=",", engine="python")
    except Exception:
        df = pd.read_csv(url, sep=None, engine="python")
    
    print(f"✓ Data loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    # Convert tourists from thousands to actual numbers
    tour_col = next((c for c in df.columns if "tourist" in c.strip().lower()), None)
    if tour_col:
        df[tour_col] = pd.to_numeric(df[tour_col], errors="coerce") * 1000
        print(f"✓ Converted {tour_col} from thousands to actual numbers")
    
    return df


def identify_top_deaths_events(
    df: pd.DataFrame,
    percentile: float = 0.97,
    country_col: str = None,
    year_col: str = None,
    deaths_col: str = None
) -> pd.DataFrame:
    """
    Identify country-year pairs with deaths in top percentile.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataset
    percentile : float, default=0.97
        Percentile threshold (0.97 = top 3%)
    country_col : str, optional
        Country column name (auto-detected if None)
    year_col : str, optional
        Year column name (auto-detected if None)
    deaths_col : str, optional
        Deaths column name (auto-detected if None)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with top deaths events (country, year, deaths)
    """
    # Auto-detect columns
    if country_col is None:
        country_col = next((c for c in df.columns if c.strip().lower() in ["country", "pays"]), None)
    if year_col is None:
        year_col = next((c for c in df.columns if c.strip().lower() in ["year", "année", "annee"]), None)
    if deaths_col is None:
        deaths_col = next((c for c in df.columns if "death" in c.strip().lower()), None)
    
    if country_col is None or year_col is None or deaths_col is None:
        raise KeyError(f"Columns not found. Found: country={country_col}, year={year_col}, deaths={deaths_col}")
    
    # Calculate threshold
    deaths_num = pd.to_numeric(df[deaths_col], errors="coerce")
    threshold = deaths_num.quantile(percentile)
    
    # Filter top events
    top_events = (
        df.loc[deaths_num >= threshold, [country_col, year_col, deaths_col]]
        .copy()
    )
    top_events[deaths_col] = pd.to_numeric(top_events[deaths_col], errors="coerce")
    top_events = top_events.sort_values(deaths_col, ascending=False).reset_index(drop=True)
    
    print(f"\n✓ Threshold ({percentile*100:.0f}th percentile): {threshold:,.0f} deaths")
    print(f"✓ Found {len(top_events):,} events in top {(1-percentile)*100:.0f}%")
    
    # Get unique countries
    unique_countries = top_events[country_col].dropna().unique()
    print(f"✓ Unique countries: {len(unique_countries)}")
    
    return top_events


def create_tourism_growth_table(
    df: pd.DataFrame,
    countries: List[str],
    years_by_country: Dict[str, Set[int]] = None,
    country_col: str = "country",
    year_col: str = "year",
    tour_col: str = None
) -> pd.DataFrame:
    """
    Create 2D table of tourism growth rates with highlighted disaster years.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataset
    countries : list of str
        List of countries to include
    years_by_country : dict, optional
        Dict mapping country -> set of disaster years (for highlighting)
    country_col : str, default='country'
        Country column name
    year_col : str, default='year'
        Year column name
    tour_col : str, optional
        Tourists column name (auto-detected if None)
    
    Returns
    -------
    pd.DataFrame
        Styled DataFrame with growth rates (% per year)
    """
    # Auto-detect tourists column
    if tour_col is None:
        tour_col = next((c for c in df.columns if "tourist" in c.strip().lower()), None)
    
    if tour_col is None:
        raise KeyError(f"Tourists column not found")
    
    # Prepare data
    tmp = df[[country_col, year_col, tour_col]].copy()
    tmp = tmp[tmp[country_col].isin(countries)].copy()
    
    tmp[year_col] = pd.to_numeric(tmp[year_col], errors="coerce").astype("Int64")
    tmp[tour_col] = pd.to_numeric(tmp[tour_col], errors="coerce")
    
    tmp = tmp.dropna(subset=[country_col, year_col, tour_col]).sort_values([country_col, year_col])
    
    # Calculate growth rate: (tourists_t / tourists_{t-1}) - 1
    tmp["tourists_growth_pct"] = tmp.groupby(country_col)[tour_col].pct_change() * 100
    
    # Pivot to 2D table
    growth_table = (
        tmp.pivot(index=country_col, columns=year_col, values="tourists_growth_pct")
        .reindex(countries)
        .sort_index(axis=1)
    )
    
    # Apply styling if disaster years provided
    if years_by_country is not None:
        def highlight_disasters(df):
            styles = pd.DataFrame("", index=df.index, columns=df.columns)
            for c, yrs in years_by_country.items():
                if c in styles.index:
                    for y in yrs:
                        if y in styles.columns:
                            styles.loc[c, y] = "background-color: #CCFF00; color: black; font-weight: 700;"
            return styles
        
        styled = (
            growth_table.style
            .format(lambda v: "" if pd.isna(v) else f"{v:.2f}%")
            .apply(highlight_disasters, axis=None)
        )
        return styled
    
    return growth_table


def match_control_countries(
    df: pd.DataFrame,
    treated_countries: List[str],
    years_by_country: Dict[str, Set[int]],
    pre_start: int = 2002,
    top_k: int = 5,
    min_overlap_years: int = 5,
    post_years: int = 4,
    country_col: str = "country",
    year_col: str = "year",
    tour_col: str = None,
    deaths_col: str = None
) -> pd.DataFrame:
    """
    Match each treated country with top-K control countries based on pre-treatment tourism growth.
    
    Uses MSE (Mean Squared Error) on tourism growth rates during pre-treatment period
    to find the best matching control countries.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataset
    treated_countries : list of str
        List of treated countries
    years_by_country : dict
        Dict mapping country -> set of treatment years
    pre_start : int, default=2002
        Start year of pre-treatment period
    top_k : int, default=5
        Number of control countries to match per treated country
    min_overlap_years : int, default=5
        Minimum number of overlapping years required for matching
    post_years : int, default=4
        Number of post-treatment years
    country_col : str, default='country'
        Country column name
    year_col : str, default='year'
        Year column name
    tour_col : str, optional
        Tourists column name (auto-detected if None)
    deaths_col : str, optional
        Deaths column name (auto-detected if None)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with matched pairs and statistics
    """
    # Auto-detect columns
    if tour_col is None:
        tour_col = next((c for c in df.columns if "tourist" in c.strip().lower()), None)
    if deaths_col is None:
        deaths_col = next((c for c in df.columns if "death" in c.strip().lower()), None)
    
    if tour_col is None or deaths_col is None:
        raise KeyError(f"Columns not found. tourists={tour_col}, deaths={deaths_col}")
    
    # Build growth panel for ALL countries
    dfg = df[[country_col, year_col, tour_col]].copy()
    dfg[year_col] = pd.to_numeric(dfg[year_col], errors="coerce").astype("Int64")
    dfg[tour_col] = pd.to_numeric(dfg[tour_col], errors="coerce")
    dfg = dfg.dropna(subset=[country_col, year_col, tour_col]).sort_values([country_col, year_col])
    
    dfg["growth_pct"] = dfg.groupby(country_col)[tour_col].pct_change() * 100
    
    growth_wide = (
        dfg.pivot(index=country_col, columns=year_col, values="growth_pct")
        .sort_index(axis=1)
    )
    
    # Deaths table for fast sums
    dd_deaths = df[[country_col, year_col, deaths_col]].copy()
    dd_deaths[year_col] = pd.to_numeric(dd_deaths[year_col], errors="coerce").astype("Int64")
    dd_deaths[deaths_col] = pd.to_numeric(dd_deaths[deaths_col], errors="coerce")
    dd_deaths = dd_deaths.dropna(subset=[country_col, year_col])
    
    def sum_deaths(country, y0, y1):
        """Sum deaths for a country between years y0 and y1 (inclusive)."""
        mask = (dd_deaths[country_col] == country) & (dd_deaths[year_col] >= y0) & (dd_deaths[year_col] <= y1)
        return float(dd_deaths.loc[mask, deaths_col].fillna(0).sum())
    
    # Candidate control countries (all except treated)
    all_countries = growth_wide.index.astype(str).tolist()
    candidate_countries = [c for c in all_countries if c not in set(treated_countries)]
    
    # Treatment year = first disaster year for each country
    first_treatment_year = {c: min(years_by_country[c]) for c in treated_countries}
    
    # Match each treated country
    results = []
    
    for tc in treated_countries:
        if tc not in growth_wide.index:
            continue
        
        t_year = int(first_treatment_year[tc])
        pre_years = [y for y in growth_wide.columns if (pd.notna(y) and pre_start <= int(y) <= (t_year - 1))]
        
        if len(pre_years) == 0:
            continue
        
        tv = growth_wide.loc[tc, pre_years]  # treated country's pre-treatment growth
        
        # Calculate MSE with each candidate
        matches = []
        for cc in candidate_countries:
            if cc not in growth_wide.index:
                continue
            
            cv = growth_wide.loc[cc, pre_years]  # candidate's growth
            
            # Only use overlapping non-missing years
            mask = (~tv.isna()) & (~cv.isna())
            n = int(mask.sum())
            
            min_overlap = min(min_overlap_years, len(pre_years))
            if n < min_overlap:
                continue
            
            # MSE on overlapping years
            diff = (tv[mask].astype(float) - cv[mask].astype(float))
            mse = float(np.mean(diff**2))
            
            # Correlation (optional metric)
            corr = np.nan
            if n >= 2:
                corr = float(np.corrcoef(tv[mask].astype(float), cv[mask].astype(float))[0, 1])
            
            matches.append((cc, n, mse, corr))
        
        # Sort by MSE (lower = better match)
        matches.sort(key=lambda x: x[2])
        top = matches[:top_k]
        
        # Calculate deaths for control countries
        pre_end = t_year - 1
        post_start = t_year
        post_end = t_year + (post_years - 1)
        
        for rank, (cc, n, mse, corr) in enumerate(top, start=1):
            control_pre = sum_deaths(cc, pre_start, pre_end)
            control_post = sum_deaths(cc, post_start, post_end)
            
            results.append({
                "treated_country": tc,
                "treat_year": t_year,
                "pre_start": pre_start,
                "pre_end": pre_end,
                "post_start": post_start,
                "post_end": post_end,
                "rank": rank,
                "matched_country": cc,
                "overlap_years": n,
                "mse": mse,
                "corr": corr,
                "control_deaths_pre_total": control_pre,
                "control_deaths_post_total": control_post,
                "control_deaths_pre_post_total": control_pre + control_post,
            })
    
    matches_df = (
        pd.DataFrame(results)
        .sort_values(["treated_country", "rank"])
        .reset_index(drop=True)
    )
    
    print(f"\n✓ Matched {len(treated_countries)} treated countries")
    print(f"✓ Top {top_k} controls per treated country")
    print(f"✓ Total matches: {len(matches_df)}")
    
    return matches_df


def rank_countries_by_deaths(
    df: pd.DataFrame,
    country_col: str = None,
    deaths_col: str = None,
    display_scrollable: bool = True
) -> pd.DataFrame:
    """
    Rank all countries by mean deaths (highest to lowest).
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataset
    country_col : str, optional
        Country column name (auto-detected if None)
    deaths_col : str, optional
        Deaths column name (auto-detected if None)
    display_scrollable : bool, default=True
        Whether to display as scrollable HTML table
    
    Returns
    -------
    pd.DataFrame
        Ranked countries with mean deaths
    """
    # Auto-detect columns
    if country_col is None:
        country_col = next((c for c in df.columns if c.strip().lower() == "country"), None)
    if deaths_col is None:
        deaths_col = next((c for c in df.columns if "death" in c.strip().lower()), None)
    
    if country_col is None or deaths_col is None:
        raise KeyError(f"Columns not found. country={country_col}, deaths={deaths_col}")
    
    # Calculate mean deaths per country
    tmp = df[[country_col, deaths_col]].copy()
    tmp[deaths_col] = pd.to_numeric(tmp[deaths_col], errors="coerce")
    tmp = tmp.dropna(subset=[country_col, deaths_col])
    
    ranking = (
        tmp.groupby(country_col, as_index=False)[deaths_col]
        .mean()
        .rename(columns={deaths_col: "mean_deaths"})
        .sort_values("mean_deaths", ascending=False)
        .reset_index(drop=True)
    )
    
    ranking.insert(0, "rank", ranking.index + 1)
    
    print(f"\n✓ Ranked {len(ranking)} countries by mean deaths")
    
    # Display as scrollable HTML if requested
    if display_scrollable:
        html = ranking.to_html(index=False)
        display(HTML(f"""
        <div style="max-height:500px; overflow:auto; border:1px solid #ddd; padding:6px;">
          {html}
        </div>
        """))
    
    return ranking


def get_top_bottom_countries(
    df: pd.DataFrame,
    country_col: str = None,
    deaths_col: str = None,
    top_percentile: float = 0.90,
    bottom_percentile: float = 0.10
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get top and bottom countries by mean deaths.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataset
    country_col : str, optional
        Country column name (auto-detected if None)
    deaths_col : str, optional
        Deaths column name (auto-detected if None)
    top_percentile : float, default=0.90
        Percentile for top countries (0.90 = top 10%)
    bottom_percentile : float, default=0.10
        Percentile for bottom countries (0.10 = bottom 10%)
    
    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        (top_countries, bottom_countries)
    """
    # Auto-detect columns
    if country_col is None:
        country_col = next((c for c in df.columns if c.strip().lower() == "country"), None)
    if deaths_col is None:
        deaths_col = next((c for c in df.columns if "death" in c.strip().lower()), None)
    
    if country_col is None or deaths_col is None:
        raise KeyError(f"Columns not found. country={country_col}, deaths={deaths_col}")
    
    # Calculate mean deaths per country
    tmp = df[[country_col, deaths_col]].copy()
    tmp[deaths_col] = pd.to_numeric(tmp[deaths_col], errors="coerce")
    tmp = tmp.dropna(subset=[country_col, deaths_col])
    
    mean_deaths = (
        tmp.groupby(country_col, as_index=False)[deaths_col]
        .mean()
        .rename(columns={deaths_col: "mean_deaths"})
    )
    
    # Calculate thresholds
    q_top = mean_deaths["mean_deaths"].quantile(top_percentile)
    q_bottom = mean_deaths["mean_deaths"].quantile(bottom_percentile)
    
    # Filter top and bottom
    top = mean_deaths[mean_deaths["mean_deaths"] >= q_top].sort_values("mean_deaths", ascending=False).reset_index(drop=True)
    bottom = mean_deaths[mean_deaths["mean_deaths"] <= q_bottom].sort_values("mean_deaths", ascending=True).reset_index(drop=True)
    
    print(f"\n✓ Total countries: {len(mean_deaths)}")
    print(f"✓ Top {(1-top_percentile)*100:.0f}% threshold: {q_top:,.2f} deaths")
    print(f"✓ Bottom {bottom_percentile*100:.0f}% threshold: {q_bottom:,.2f} deaths")
    print(f"✓ Top countries: {len(top)}")
    print(f"✓ Bottom countries: {len(bottom)}")
    
    return top, bottom
