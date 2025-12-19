# Pack Your Suitcase... And A First Aid Kit?
## Tourism In An Age Of Natural Disasters

**Authors:** Vanessa El Khoury, Rayan Ghannoun, Daphne Vryghem  
**Date:** December 19, 2025

---

## 📋 Overview

This project analyzes the impact of natural disasters on international tourism, with a focus on how disaster preparedness affects tourist behavior. Using panel data from 2000-2019, we examine whether deaths from natural disasters deter tourists and whether well-prepared countries are more resilient to these shocks.

### Key Findings

- **Average Effect**: Small, statistically non-significant negative relationship between disaster deaths and tourism
- **Heterogeneity by Preparedness**: 
  - Well-prepared countries show positive/neutral effects (β ≈ +0.005)
  - Less-prepared countries show negative effects (β ≈ -0.006)
- **Policy Implication**: Early investment in disaster preparedness helps protect tourism sectors

---

## 🗂️ Project Structure

```
datascience/
├── report.ipynb          # Main analysis notebook
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── data/                # Datasets
│   ├── final_dataset.csv
│   ├── final_dataset_low_prep.csv
│   └── final_dataset_high_prep.csv
├── assets/              # Images and figures
│   ├── Beach.jpg
│   ├── Tourists_trend.jpg
│   ├── summary.jpg
│   └── end.jpg
└── src/                 # Source code modules
    ├── __init__.py
    ├── io.py           # Data loading
    ├── cleaning.py     # Data transformations
    ├── models.py       # Regression models
    ├── plots.py        # Visualization functions
    ├── maps.py         # Interactive maps
    ├── utils.py        # Helper functions
    ├── robustness.py   # Robustness checks
    ├── did_data_prep.py
    ├── did_analysis.py
    └── diff_in_diff.py
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- Jupyter Notebook

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd datascience
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook report.ipynb
   ```

4. **Run the analysis**
   - Open `report.ipynb`
   - Click "Run All" or execute cells sequentially

---

## 📊 Data Sources

- **Natural Disasters**: EM-DAT (Emergency Events Database), 2000-2019
- **Tourism**: UN World Tourism Organization (UNWTO)
- **Preparedness**: World Risk Index (WRI) - Vulnerability component
- **GDP per capita**: World Bank
- **Population Density**: Our World in Data (HYDE, Gapminder, UN)

---

## 🔬 Methodology

### Model Specification

```
ln(tourists_{it}) = α + β·ln(deaths_{it}+1) + γ'·X_{it} + μ_i + τ_t + ε_{it}
```

Where:
- **ln(tourists)**: Log of international tourist arrivals
- **ln(deaths+1)**: Log of deaths from natural disasters
- **X**: Control variables (preparedness, population density, GDP per capita)
- **μ_i**: Country fixed effects
- **τ_t**: Time fixed effects
- **ε**: Error term

### Key Analyses

1. **Baseline Regression**: OLS and Fixed Effects models
2. **Progressive Controls**: Adding confounders step-by-step
3. **Heterogeneity Analysis**: Interaction with preparedness level
4. **Robustness Checks**: Outlier detection, sensitivity tests

---

## 📈 Main Results

### Baseline Model (Full Sample)
- **Coefficient**: β = -0.0026
- **Standard Error**: 0.0064
- **Interpretation**: 1% increase in deaths → 0.0026% decrease in tourism (not significant)

### Heterogeneity by Preparedness
- **Well-Prepared Countries**: β = +0.0048 (SE: 0.0053)
- **Less-Prepared Countries**: β = -0.0062 (SE: 0.0096)
- **Difference**: 0.011 percentage points

---

## 📦 Dependencies

Core libraries:
- `pandas >= 2.0.0` - Data manipulation
- `numpy >= 1.24.0` - Numerical computing
- `matplotlib >= 3.7.0` - Plotting
- `seaborn >= 0.12.0` - Statistical visualization
- `statsmodels >= 0.14.0` - Statistical models
- `linearmodels >= 5.3` - Panel data models
- `plotly >= 5.14.0` - Interactive visualizations

See `requirements.txt` for complete list.

---

## 📝 Notebook Structure

The `report.ipynb` notebook is organized as follows:

1. **Introduction**: Context and research question
2. **Data & Methodology**: Data sources, model specification
3. **Baseline Results**: Main regression findings
4. **Heterogeneity Analysis**: Effects by preparedness level
5. **Conclusion**: Key takeaways and policy implications
6. **Appendix**: Additional tables, figures, and robustness checks

---

## 🎯 Key Visualizations

- **Interactive World Map**: Tourism, disasters, and preparedness (2001-2019)
- **Beta Coefficient Evolution**: Effect of adding controls
- **Heterogeneity Scatter Plot**: Separate regression lines by preparedness
- **Box Plots**: Variable distributions before/after log transformation

---

## 🔧 Usage Examples

### Load Data
```python
from src import load_all_datasets, log_transform_in_place

# Load datasets
df_all, df_low_prep, df_high_prep = load_all_datasets()

# Apply log transformations
df = log_transform_in_place(df_all.copy())
```

### Run Baseline Regression
```python
from src import run_baseline_specifications, build_regression_table

# Run specifications
models, model_names = run_baseline_specifications(df)

# Build table
table = build_regression_table(models, model_names)
```

### Heterogeneity Analysis
```python
from src import add_well_prepared_dummy, run_heterogeneity_analysis

# Add preparedness dummy
df_with_prep = add_well_prepared_dummy(df_all.copy(), df_low_prep)
df_with_prep = log_transform_in_place(df_with_prep)

# Run analysis
results = run_heterogeneity_analysis(
    df_with_prep,
    controls=['wri_v', 'pop_density', 'gdp_pc']
)
```

---

## 📚 References

Key papers cited in the analysis:
- Rosselló, J., Becken, S., & Santana-Gallego, M. (2020). The effects of natural disasters on international tourism: A global analysis. *Tourism Management*, 79, 104080.
- Guimarães, W. P., Lenzi, M. A., & Quintanilha, J. A. (2025). Does community flood preparedness reduce mortality and injuries? *Natural Hazards and Earth System Sciences*, 25, 3803–3820.

See full reference list in the notebook.

---

## 👥 Authors

- **Vanessa El Khoury**
- **Rayan Ghannoun**
- **Daphne Vryghem**

---

## 📄 License

This project is for academic purposes. Please cite appropriately if using this code or methodology.

---

## 🤝 Contributing

This is an academic project. For questions or suggestions, please contact the authors.

---

## ⚠️ Notes

- Data covers 2000-2019 (pre-COVID)
- All monetary values in constant 2015 USD
- Standard errors clustered at country level
- Fixed effects control for time-invariant country characteristics and global trends

---

**Last Updated**: December 19, 2025
