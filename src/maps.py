"""
Interactive Maps for Tourism and Disaster Analysis

This module provides functions for creating interactive world maps
showing tourism, disaster exposure, and preparedness.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional


def create_world_map(
    df: pd.DataFrame,
    year: int = None,
    country_col: str = "country",
    year_col: str = "year",
    tourists_col: str = None,
    deaths_col: str = None,
    wri_col: str = "wri_v",
    title: str = None
) -> go.Figure:
    """
    Create interactive world map showing tourism, disasters, and preparedness.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataset
    year : int, optional
        Specific year to display (if None, uses mean across all years)
    country_col : str, default='country'
        Country column name
    year_col : str, default='year'
        Year column name
    tourists_col : str, optional
        Tourists column name (auto-detected if None)
    deaths_col : str, optional
        Deaths column name (auto-detected if None)
    wri_col : str, default='wri_v'
        WRI vulnerability column name
    title : str, optional
        Map title
    
    Returns
    -------
    plotly.graph_objects.Figure
        Interactive map figure
    """
    # Auto-detect columns
    if tourists_col is None:
        tourists_col = next((c for c in df.columns if "tourist" in c.strip().lower()), None)
    if deaths_col is None:
        deaths_col = next((c for c in df.columns if "death" in c.strip().lower()), None)
    
    if tourists_col is None or deaths_col is None:
        raise KeyError(f"Columns not found. tourists={tourists_col}, deaths={deaths_col}")
    
    # Filter data
    if year is not None:
        df_map = df[df[year_col] == year].copy()
        year_label = str(year)
    else:
        # Aggregate across all years (mean)
        df_map = (
            df.groupby(country_col, as_index=False)
            [[tourists_col, deaths_col, wri_col]]
            .mean()
        )
        year_label = "2001-2019 (mean)"
    
    # Convert to numeric
    df_map[tourists_col] = pd.to_numeric(df_map[tourists_col], errors="coerce")
    df_map[deaths_col] = pd.to_numeric(df_map[deaths_col], errors="coerce")
    df_map[wri_col] = pd.to_numeric(df_map[wri_col], errors="coerce")
    
    # Remove missing values
    df_map = df_map.dropna(subset=[country_col, tourists_col, deaths_col])
    
    # Classify preparedness (based on WRI vulnerability)
    # Higher WRI vulnerability = less prepared
    median_wri = df_map[wri_col].median()
    df_map["preparedness"] = df_map[wri_col].apply(
        lambda x: "Less Prepared" if x > median_wri else "More Prepared"
    )
    
    # Create hover text
    df_map["hover_text"] = (
        "<b>" + df_map[country_col] + "</b><br>" +
        "Tourists: " + df_map[tourists_col].apply(lambda x: f"{x:,.0f}") + "<br>" +
        "Deaths: " + df_map[deaths_col].apply(lambda x: f"{x:,.0f}") + "<br>" +
        "WRI Vulnerability: " + df_map[wri_col].apply(lambda x: f"{x:.2f}") + "<br>" +
        "Preparedness: " + df_map["preparedness"]
    )
    
    # Create figure
    if title is None:
        title = f"Tourism, Disaster Exposure, and Preparedness ({year_label})"
    
    fig = px.choropleth(
        df_map,
        locations=country_col,
        locationmode="country names",
        color=tourists_col,
        hover_name=country_col,
        hover_data={
            country_col: False,
            tourists_col: ":,.0f",
            deaths_col: ":,.0f",
            wri_col: ":.2f",
            "preparedness": True
        },
        color_continuous_scale="YlOrRd",
        labels={
            tourists_col: "Tourists",
            deaths_col: "Deaths",
            wri_col: "WRI Vulnerability"
        },
        title=title
    )
    
    # Add bubble markers for deaths (size = deaths, color = preparedness)
    # Prepare data for scatter
    df_scatter = df_map[df_map[deaths_col] > 0].copy()
    
    if len(df_scatter) > 0:
        # Get country coordinates (approximate - you may want to use a proper geocoding library)
        # For now, we'll use plotly's built-in country locations
        
        # Create scatter trace for deaths
        fig.add_trace(
            go.Scattergeo(
                locations=df_scatter[country_col],
                locationmode="country names",
                marker=dict(
                    size=np.sqrt(df_scatter[deaths_col]) * 0.05,  # Much smaller bubbles (10x reduction)
                    color=df_scatter["preparedness"].map({
                        "More Prepared": "purple",
                        "Less Prepared": "red"
                    }),
                    opacity=0.6,
                    line=dict(width=0.5, color="white"),
                    sizemode="diameter"
                ),
                text=df_scatter["hover_text"],
                hoverinfo="text",
                name="Disaster Deaths",
                showlegend=True
            )
        )
    
    # Update layout
    fig.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type="natural earth"
        ),
        height=600,
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    return fig


def create_interactive_disaster_map(
    df: pd.DataFrame,
    country_col: str = "country",
    year_col: str = "year",
    vul_col: str = "wri_v",
    base_year: int = 2001,
    start_year: int = 2001,
    end_year: int = 2019,
    bubble_size: int = 4
) -> go.Figure:
    """
    Create interactive world map with tourism, disasters, and preparedness.
    Includes choropleth for tourists, bubble overlay for deaths, and vulnerability tables.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataset with country, year, tourists, deaths, and vulnerability data
    country_col : str, default='country'
        Country column name
    year_col : str, default='year'
        Year column name
    vul_col : str, default='wri_v'
        Vulnerability column name
    base_year : int, default=2001
        Base year for vulnerability split
    start_year : int, default=2001
        Start year for aggregation
    end_year : int, default=2019
        End year for aggregation
    bubble_size : int, default=4
        Maximum bubble size in pixels
    
    Returns
    -------
    plotly.graph_objects.Figure
        Interactive map with tables
    """
    import pycountry
    from plotly.subplots import make_subplots
    
    # Find tourists and deaths columns
    tourists_cols = [c for c in df.columns if 'tourist' in c.lower()]
    tour_col = tourists_cols[0] if tourists_cols else "tourists"
    deaths_col = "deaths"
    
    # ISO helpers
    def to_iso3(name):
        try:
            return pycountry.countries.lookup(name).alpha_3
        except Exception:
            return None
    
    name_fix = {
        "Democratic Republic of the Congo": "Congo, The Democratic Republic of the",
        "Turkey": "Türkiye",
    }
    
    def clean_country(name):
        if pd.isna(name):
            return name
        name = str(name).strip()
        return name_fix.get(name, name)
    
    # Prep data
    df2 = df.copy()
    df2[country_col] = df2[country_col].apply(clean_country)
    
    # Base-year vulnerability + median split
    vul_base = (
        df2.loc[df2[year_col] == base_year, [country_col, vul_col]]
          .dropna(subset=[country_col, vul_col])
          .groupby(country_col, as_index=False)[vul_col].mean()
    )
    median_vul = float(vul_base[vul_col].median())
    
    vul_base["vul_group"] = np.where(
        vul_base[vul_col] > median_vul,
        "High vulnerability",
        "Low vulnerability"
    )
    
    # Aggregate totals
    df_win = df2[(df2[year_col] >= start_year) & (df2[year_col] <= end_year)].copy()
    agg = (
        df_win.groupby(country_col, as_index=False)
              .agg(total_tourists=(tour_col, "sum"),
                   total_deaths=(deaths_col, "sum"))
    )
    
    map_df = agg.merge(vul_base, on=country_col, how="left")
    map_df["iso3"] = map_df[country_col].apply(to_iso3)
    map_df = map_df.dropna(subset=["iso3"]).copy()
    
    # Top 5 / Bottom 5 tables
    v_sorted = vul_base.sort_values(vul_col, ascending=False).reset_index(drop=True)
    
    top5 = v_sorted.head(5).copy()
    top5["Rank"] = [1, 2, 3, 4, 5]
    top5[vul_col] = top5[vul_col].round(2)
    
    bottom5 = v_sorted.tail(5).sort_values(vul_col, ascending=True).copy()
    bottom5["Rank"] = [1, 2, 3, 4, 5]
    bottom5[vul_col] = bottom5[vul_col].round(2)
    
    # Bubble scaling
    max_deaths = float(map_df["total_deaths"].max()) if map_df["total_deaths"].max() > 0 else 1.0
    sizeref = 2.0 * max_deaths / (bubble_size ** 2)
    
    group_colors = {"Low vulnerability": "royalblue", "High vulnerability": "tomato"}
    
    # Build figure
    fig = make_subplots(
        rows=2, cols=2,
        specs=[
            [{"type": "geo", "colspan": 2}, None],
            [{"type": "table"}, {"type": "table"}]
        ],
        row_heights=[0.70, 0.30],
        vertical_spacing=0.06,
        horizontal_spacing=0.06
    )
    
    # Choropleth (tourists)
    fig.add_trace(
        go.Choropleth(
            locations=map_df["iso3"],
            z=map_df["total_tourists"],
            text=map_df[country_col],
            colorscale="Blues",
            marker_line_color="white",
            marker_line_width=0.35,
            colorbar=dict(
                title=f"Total tourists ('000)<br>{start_year}–{end_year}",
                orientation="h",
                y=0.30,
                x=0.5,
                xanchor="center",
                len=0.55,
                thickness=14,
                tickformat="~s"
            ),
            hovertemplate="<b>%{text}</b><br>Total tourists ('000): %{z:,.0f}<extra></extra>"
        ),
        row=1, col=1
    )
    
    # Bubble overlay
    for g in ["Low vulnerability", "High vulnerability"]:
        dfg = map_df[map_df["vul_group"] == g].copy()
        
        fig.add_trace(
            go.Scattergeo(
                locations=dfg["iso3"],
                locationmode="ISO-3",
                mode="markers",
                name=g,
                marker=dict(
                    size=dfg["total_deaths"],
                    sizemode="area",
                    sizeref=sizeref,
                    sizemin=2,
                    opacity=0.55,
                    color=group_colors[g],
                    line=dict(width=0.45, color="white"),
                ),
                customdata=np.stack([
                    dfg[country_col],
                    dfg["total_deaths"],
                    dfg["total_tourists"],
                    dfg[vul_col],
                ], axis=-1),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    f"Disaster deaths ({start_year}–{end_year}): "+"%{customdata[1]:,.0f}<br>"
                    f"Tourists ('000, {start_year}–{end_year}): "+"%{customdata[2]:,.0f}<br>"
                    f"Vulnerability (base {base_year}): "+"%{customdata[3]:.2f}"
                    "<extra></extra>"
                ),
            ),
            row=1, col=1
        )
    
    # Tables
    fig.add_trace(
        go.Table(
            header=dict(values=["<b>Top 5 most vulnerable</b>", "", ""], align="left"),
            cells=dict(
                values=[
                    ["<b>Rank</b>"] + top5["Rank"].astype(str).tolist(),
                    ["<b>Country</b>"] + top5[country_col].tolist(),
                    [f"<b>{vul_col}</b>"] + top5[vul_col].astype(str).tolist(),
                ],
                align="left",
                height=24
            )
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Table(
            header=dict(values=["<b>Bottom 5 least vulnerable</b>", "", ""], align="left"),
            cells=dict(
                values=[
                    ["<b>Rank</b>"] + bottom5["Rank"].astype(str).tolist(),
                    ["<b>Country</b>"] + bottom5[country_col].tolist(),
                    [f"<b>{vul_col}</b>"] + bottom5[vul_col].astype(str).tolist(),
                ],
                align="left",
                height=24
            )
        ),
        row=2, col=2
    )
    
    # Map styling
    fig.update_geos(
        projection_type="natural earth",
        showcoastlines=True,
        showcountries=True,
        projection_scale=1,
        center=dict(lon=0, lat=20),
        row=1, col=1
    )
    
    fig.update_layout(
        height=900,
        title=(
            "Where do tourists go, and where do disasters hit hardest?<br>"
            f"<sup>Interactive map: pinch to zoom, drag to pan. "
            f"Country color = total tourists ('000) {start_year}-{end_year}; "
            f"Bubble size = total disaster deaths {start_year}-{end_year}; "
            f"Bubble color = vulnerability group (median split in {base_year}, median {vul_col} = {median_vul:.2f})</sup>"
        ),
        legend_title_text="Vulnerability group (bubble color)",
        legend=dict(
            x=0.7, y=0.95,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=1
        ),
        margin=dict(l=0, r=0, t=90, b=20),
        dragmode='pan'
    )
    
    return fig


def create_simple_world_map(
    df: pd.DataFrame,
    value_col: str,
    country_col: str = "country",
    title: str = None,
    color_scale: str = "Blues"
) -> go.Figure:
    """
    Create a simple choropleth map for a single variable.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataset (should be aggregated by country)
    value_col : str
        Column to display on map
    country_col : str, default='country'
        Country column name
    title : str, optional
        Map title
    color_scale : str, default='Blues'
        Plotly color scale name
    
    Returns
    -------
    plotly.graph_objects.Figure
        Interactive map figure
    """
    fig = px.choropleth(
        df,
        locations=country_col,
        locationmode="country names",
        color=value_col,
        hover_name=country_col,
        hover_data={value_col: ":,.2f"},
        color_continuous_scale=color_scale,
        title=title or f"World Map: {value_col}"
    )
    
    fig.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type="natural earth"
        ),
        height=500,
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    return fig
