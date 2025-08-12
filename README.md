import numpy as np
import pandas as pd

def prepare_risk_premia(
    df,
    id_col='sedolcd',
    date_col='date',
    ret_col='totalreturn',
    price_col='raw_quoteclose',
    mcap_col='marketvalue',
    bvps_col='wsf_bps_ltm',
    sector_col='GICS_sector_name',
    add_mom=True,
    add_vol=True,
    add_sector_dummies=True,
    mom_window=12,
    vol_window=12,
    min_frac=0.75,
):
    """
    Build firm-level risk-premia exposures for cross-sectional (Fama–MacBeth) regressions.

    Outputs added:
      - ret_fwd_1m : forward 1M return  (r_{t+1})
      - SIZE       : ln(Market Cap)
      - VALUE      : ln(Book/Market) using per-share ratio BVPS/Price
      - MOM        : (optional) 12-1 momentum (t-12..t-2), log-compounded
      - VOL        : (optional) 12M rolling std of monthly returns (t-12..t-1)
      - SEC_*      : (optional) sector dummies from `sector_col`, drop_first=True

    Notes
    -----
    • Assumes `ret_col` is a monthly simple return (e.g., 0.02 for +2%).
    • For VALUE we use ln(BVPS / Price); with consistent per-share fields this equals ln(BE/ME).
    • Rolling stats use only past data via shift(1) so they are known at time t.
    • Run this AFTER filtering by currency/market to match your test universe.
    """
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])
    d = d.sort_values([id_col, date_col]).reset_index(drop=True)

    # 0) Forward 1M return (dependent variable for FMB)
    d['ret_fwd_1m'] = d.groupby(id_col)[ret_col].shift(-1)

    # 1) SIZE = ln(Market Cap)
    mc = pd.to_numeric(d[mcap_col], errors='coerce')
    d['SIZE'] = np.log(mc.replace(0, np.nan))

    # 2) VALUE = ln(Book/Market) via per-share ratio (BVPS / Price)
    px = pd.to_numeric(d[price_col], errors='coerce')
    bvps = pd.to_numeric(d[bvps_col],  errors='coerce')
    bm = (bvps / px).replace([np.inf, -np.inf], np.nan)
    bm = bm.where((bm > 0) & np.isfinite(bm))  # keep only positive, finite
    d['VALUE'] = np.log(bm)

    g = d.groupby(id_col, group_keys=False)
    minp_mom = int(mom_window * min_frac)
    minp_vol = int(vol_window * min_frac)

    # 3) MOM 12-1: cumulative from t-12..t-2 (exclude t-1)
    if add_mom:
        d['_log1p'] = np.log1p(pd.to_numeric(d[ret_col], errors='coerce'))
        # sum over last 12 months up to t-1, then subtract last month (t-1) to exclude it
        sum12_up_to_t1 = g['_log1p'].rolling(mom_window, min_periods=minp_mom).sum().shift(1)
        last1 = g['_log1p'].shift(1)
        d['MOM'] = np.expm1(sum12_up_to_t1 - last1)
        d.drop(columns=['_log1p'], inplace=True)

    # 4) VOL 12m: std over t-12..t-1, known at t
    if add_vol:
        d['VOL'] = g[ret_col].rolling(vol_window, min_periods=minp_vol).std().shift(1)

    # 5) Sector dummies (drop_first to avoid collinearity with intercept)
    sector_cols = []
    if add_sector_dummies and sector_col in d.columns:
        # Create once across the whole frame so columns are consistent across dates
        dummies = pd.get_dummies(d[sector_col], prefix='SEC', drop_first=True, dtype=float)
        d = pd.concat([d, dummies], axis=1)
        sector_cols = dummies.columns.tolist()

    # Final numeric clean-up
    for c in ['SIZE', 'VALUE', 'MOM', 'VOL', 'ret_fwd_1m']:
        if c in d.columns:
            d[c] = d[c].replace([np.inf, -np.inf], np.nan)

    controls = ['SIZE', 'VALUE']
    if add_mom: controls.append('MOM')
    if add_vol: controls.append('VOL')
    controls += sector_cols  # dummies last

    return d, controls



# utiles.py
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.linear_model import LinearRegression


def get_total_return(returns: pd.Series) -> float:
    cum = (1 + returns).cumprod()
    return cum.iloc[-1] - 1


def get_cagr(returns: pd.Series, periods: int = 12) -> float:
    cum = (1 + returns).cumprod()
    n = len(returns.dropna())
    return cum.iloc[-1]**(periods/n) - 1 if n > 0 else np.nan


def get_sharpe_ratio(returns: pd.Series, periods: int = 12) -> float:
    r = returns.dropna()
    return (r.mean()/r.std()*np.sqrt(periods)) if r.std() > 0 else np.nan


def get_sortino_ratio(returns: pd.Series, periods: int = 12) -> float:
    r = returns.dropna()
    downside = r[r < 0]
    down_std = downside.std()*np.sqrt(periods) if len(downside) > 0 else np.nan
    return (r.mean()/down_std) if down_std and down_std > 0 else np.nan


def get_max_drawdown(returns: pd.Series) -> float:
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    return (cum/peak - 1).min()


def get_hit_rate(returns: pd.Series) -> float:
    return (returns.dropna() > 0).mean()


def get_skew(returns: pd.Series) -> float:
    return skew(returns.dropna())


def get_kurtosis(returns: pd.Series) -> float:
    return kurtosis(returns.dropna())

# Factor comparison metrics
def get_alpha(strat: pd.Series, ref: pd.Series) -> float:
    X = ref.values.reshape(-1,1)
    y = strat.values
    reg = LinearRegression().fit(X, y)
    return reg.intercept_

def get_beta(strat: pd.Series, ref: pd.Series) -> float:
    X = ref.values.reshape(-1,1)
    y = strat.values
    reg = LinearRegression().fit(X, y)
    return reg.coef_[0]

def get_tracking_error(strat: pd.Series, ref: pd.Series, periods: int = 12) -> float:
    excess = strat - ref
    return excess.std()*np.sqrt(periods)

def get_information_ratio(strat: pd.Series, ref: pd.Series, periods: int = 12) -> float:
    excess = strat - ref
    if excess.std() == 0:
        return np.nan
    return (excess.mean()*periods)/(excess.std()*np.sqrt(periods))

def get_treynor_ratio(strat: pd.Series, ref: pd.Series, periods: int = 12) -> float:
    beta = get_beta(strat, ref)
    if beta == 0:
        return np.nan
    return (strat.mean()*periods)/beta

# Coverage do'nt delete


# Total benchmark constituents at each date
benchmark_counts = df_benchmark.groupby('date')['sedolcd'].nunique()

# Predicted constituents at each date (i.e., universe d'investissement)
pred_counts = df_pred.groupby('date')['sedolcd'].nunique()

# Relative coverage: % of benchmark covered by your universe
coverage_pct = (pred_counts / benchmark_counts).fillna(0)

# Afficher quelques stats
print("Coverage moyen :", coverage_pct.mean())
print("Coverage min / max :", coverage_pct.min(), "/", coverage_pct.max())

# Visualiser
import matplotlib.pyplot as plt
coverage_pct.plot(title="Coverage of Benchmark by Investment Universe")
plt.ylabel("Coverage (%)")
plt.xlabel("Date")
plt.show()




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
