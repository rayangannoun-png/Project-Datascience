"""
Tourism & Natural Disasters Analysis Package

Modules:
- io: Data loading utilities
- cleaning: Data transformations (logs, dummies, lags)
- models: Regression models (OLS, FE)
- plots: Visualization functions
- robustness: Robustness analysis (outlier detection, sensitivity tests)
- utils: Helper functions
"""

__version__ = "0.1.0"

# Import key functions for easier access
from .io import load_all_datasets, get_dataset_info
from .cleaning import (
    add_well_prepared_dummy,
    log_transform_in_place,
    lag_variable_by_country_year,
    check_panel_structure
)
from .models import (
    fit_ols_clustered,
    fit_panel_fe,
    build_regression_table,
    run_baseline_specifications,
    extract_beta_with_ci,
    fit_interaction_model,
    extract_group_effects,
    run_heterogeneity_analysis,
    run_control_comparison
)
from .plots import (
    plot_correlation_with_ci,
    plot_fe_residualized,
    plot_beta_variation,
    plot_heterogeneity_comparison,
    plot_heterogeneity_scatter,
    plot_boxplots_transformation,
    plot_correlation_matrix,
    plot_beta_comparison
)
from .robustness import (
    identify_outliers_iqr,
    remove_outlier_countries,
    compare_robust_results,
    run_robustness_analysis,
    plot_outlier_detection
)
from .did_data_prep import (
    load_did_data,
    identify_top_deaths_events,
    create_tourism_growth_table,
    match_control_countries,
    rank_countries_by_deaths,
    get_top_bottom_countries
)
from .did_analysis import (
    plot_did_timeseries,
    run_did_twfe,
    compare_did_results,
    run_did_analysis_pair,
    run_all_did_analyses,
    run_event_study,
    plot_event_study,
    run_did_different_controls,
    run_did_different_windows
)
from .maps import (
    create_world_map,
    create_simple_world_map,
    create_interactive_disaster_map
)
from .utils import (
    p_to_stars,
    format_cell_html,
    display_html_table,
    get_t_critical,
    print_section_header,
    display_data_summary
)

__all__ = [
    # IO
    'load_all_datasets',
    'get_dataset_info',
    # Cleaning
    'add_well_prepared_dummy',
    'log_transform_in_place',
    'lag_variable_by_country_year',
    'check_panel_structure',
    # Models
    'fit_ols_clustered',
    'fit_panel_fe',
    'build_regression_table',
    'run_baseline_specifications',
    'extract_beta_with_ci',
    'fit_interaction_model',
    'extract_group_effects',
    'run_heterogeneity_analysis',
    'run_control_comparison',
    # Plots
    'plot_correlation_with_ci',
    'plot_fe_residualized',
    'plot_beta_variation',
    'plot_heterogeneity_comparison',
    'plot_heterogeneity_scatter',
    'plot_boxplots_transformation',
    'plot_correlation_matrix',
    'plot_beta_comparison',
    # Robustness
    'identify_outliers_iqr',
    'remove_outlier_countries',
    'compare_robust_results',
    'run_robustness_analysis',
    'plot_outlier_detection',
    # DiD Data Prep
    'load_did_data',
    'identify_top_deaths_events',
    'create_tourism_growth_table',
    'match_control_countries',
    'rank_countries_by_deaths',
    'get_top_bottom_countries',
    # DiD Analysis
    'plot_did_timeseries',
    'run_did_twfe',
    'compare_did_results',
    'run_did_analysis_pair',
    'run_all_did_analyses',
    'run_event_study',
    'plot_event_study',
    'run_did_different_controls',
    'run_did_different_windows',
    # Maps
    'create_world_map',
    'create_simple_world_map',
    'create_interactive_disaster_map',
    # Utils
    'p_to_stars',
    'format_cell_html',
    'display_html_table',
    'get_t_critical',
    'print_section_header',
    'display_data_summary',
]
