import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

# ---------- helpers ----------
def _align_dropna(*series):
    """Inner-join on index & drop NaNs for all input Series, return as tuple."""
    df = pd.concat(series, axis=1).dropna()
    return tuple(df.iloc[:, i] for i in range(df.shape[1]))

# periods = observations per year (12 = monthly, 252 = daily)
# rf can be scalar per-period or a Series aligned to strat/ref

# ---------- absolute metrics on one return series ----------
def get_total_return(r: pd.Series) -> float:
    r = r.dropna()
    if r.empty: return np.nan
    return (1.0 + r).prod() - 1.0

def get_cagr(r: pd.Series, periods: int = 12) -> float:
    r = r.dropna()
    n = len(r)
    if n == 0: return np.nan
    tot = (1.0 + r).prod()
    return tot**(periods / n) - 1.0

def get_sharpe_ratio(r: pd.Series, rf=0.0, periods: int = 12) -> float:
    r = r.dropna()
    if isinstance(rf, pd.Series):
        rf = _align_dropna(r, rf)[1]
    ex = r - rf
    s = ex.std(ddof=1)
    return np.nan if s == 0 or np.isnan(s) else ex.mean() / s * np.sqrt(periods)

def get_sortino_ratio(r: pd.Series, rf=0.0, periods: int = 12) -> float:
    r = r.dropna()
    if isinstance(rf, pd.Series):
        r, rf = _align_dropna(r, rf)
    ex = r - rf
    downside = ex[ex < 0.0]
    ds = downside.std(ddof=1)
    return np.nan if ds == 0 or np.isnan(ds) else ex.mean() / ds * np.sqrt(periods)

def get_max_drawdown(r: pd.Series) -> float:
    r = r.dropna()
    if r.empty: return np.nan
    wealth = (1.0 + r).cumprod()
    peak = wealth.cummax()
    dd = wealth / peak - 1.0
    return dd.min()  # negative number

def get_hit_rate(r: pd.Series) -> float:
    r = r.dropna()
    return np.nan if r.empty else (r > 0).mean()

def get_skew(r: pd.Series) -> float:
    r = r.dropna()
    return np.nan if r.empty else skew(r, bias=False)

def get_kurtosis(r: pd.Series) -> float:
    r = r.dropna()
    # Fisher=True => kurtosis excess (normal=0). Add 3 if you prefer Pearson.
    return np.nan if r.empty else kurtosis(r, fisher=True, bias=False)

# ---------- relative metrics strat vs ref ----------
def get_beta(strat: pd.Series, ref: pd.Series, rf=0.0) -> float:
    strat, ref = _align_dropna(strat, ref)
    if isinstance(rf, pd.Series):
        strat, ref, rf = _align_dropna(strat, ref, rf)
    # excess returns
    ex_s = strat - rf
    ex_m = ref - rf
    var = ex_m.var(ddof=1)
    if var == 0 or np.isnan(var): return np.nan
    cov = np.cov(ex_s, ex_m, ddof=1)[0, 1]
    return cov / var

def get_alpha(strat: pd.Series, ref: pd.Series, rf=0.0, periods: int = 12, annualize=True) -> float:
    """CAPM alpha via OLS with intercept (excess returns)."""
    strat, ref = _align_dropna(strat, ref)
    if isinstance(rf, pd.Series):
        strat, ref, rf = _align_dropna(strat, ref, rf)
    ex_s = strat - rf
    ex_m = ref - rf
    b = get_beta(ex_s, ex_m, 0.0)  # already excess
    if np.isnan(b): return np.nan
    alpha_per_period = ex_s.mean() - b * ex_m.mean()
    return alpha_per_period * periods if annualize else alpha_per_period

def get_tracking_error(strat: pd.Series, ref: pd.Series, periods: int = 12) -> float:
    s, m = _align_dropna(strat, ref)
    diff = s - m
    te = diff.std(ddof=1)
    return np.nan if te == 0 or np.isnan(te) else te * np.sqrt(periods)

def get_information_ratio(strat: pd.Series, ref: pd.Series, periods: int = 12) -> float:
    s, m = _align_dropna(strat, ref)
    diff = s - m
    mu = diff.mean()
    sd = diff.std(ddof=1)
    if sd == 0 or np.isnan(sd): return np.nan
    # annualized IR = (mu*periods)/(sd*sqrt(periods)) = sqrt(periods)* (mu/sd)
    return (mu / sd) * np.sqrt(periods)

def get_treynor_ratio(strat: pd.Series, ref: pd.Series, rf=0.0, periods: int = 12) -> float:
    strat, ref = _align_dropna(strat, ref)
    if isinstance(rf, pd.Series):
        strat, ref, rf = _align_dropna(strat, ref, rf)
    ex_s = strat - rf
    b = get_beta(strat, ref, rf)
    if b == 0 or np.isnan(b): return np.nan
    # annualize numerator only (mean excess return)
    return ex_s.mean() * periods / b



# S&P 500 Forecasting Using Macro-Financial Variables

## Overview
This project leverages **machine learning** techniques to forecast the **weekly log returns** of the **S&P 500** using a dataset of macro-financial variables spanning over 20 years as part of **AI for Alpha**'s data challenge. The project was implemented using an **Object-Oriented Programming (OOP)** approach for better modularity and maintainability.

## Features
- **Exploratory Data Analysis:** ..
- **Data Preprocessing:** Handles missing values, outliers, and feature scaling.
- **Feature Selection:** Uses PCA, correlation analysis with bonferonni correction and permutation importance.
- **Modeling:** Implements a forecaster including time series split, ....
- **Hyperparameter Tuning:** Uses randomized search for optimal model performance.
- **Evaluation Metrics:** Custom scoring function combining RMSE and Directional Accuracy (DA).

## Installation
To set up the environment, install the required dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure
```plaintext
📂 Macro-SP500-Forecast
│── 📂 data                    # Raw datasets (S&P Daily Close Price and Macro-Financial Features)
│── 📂 src                     # Source code
│   │── processor.py           # Preprocessing module
│   │── EDA.ipynb              # Exploratory Data Analysis
│   │── skew_transformer.py    # Skew Transforer module
│   │── forecaster.py          # Machine learning models
│   │── arimaforecaster.py     # Autoregressive models
│   │── main.ipynb             # Main script to run the forecasting
│── prpoject_report            # Written Project Report
│── README.md                  # Documentation
```

## License
MIT License.

---
### Author
Solal Danan
