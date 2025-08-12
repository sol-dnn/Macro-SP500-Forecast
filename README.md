import numpy as np
import pandas as pd

def prepare_signal_analysis_data(
    df,
    *,
    id_col='sedolcd',
    date_col='date',
    ret_col='totalreturn',          # simple monthly return, e.g. 0.02
    price_col='raw_quoteclose',
    mcap_col='marketvalue',
    bvps_col='wsf_bps_ltm',
    sector_col='GICS_sector_name',
    # windows / options
    mom_window=12,                  # MOM = 12-1
    vol_window=12,                  # VOL = 12m std
    beta_window=24,                 # BETA = 24m
    min_history_frac=0.75,          # min data inside each window
    mkt_method='equal',             # 'equal' or 'value' (for market return)
    compute_market=True,            # set False if you already merged an index as mkt_ret_m
    make_sector_dummies=True
):
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])
    d = d.sort_values([id_col, date_col]).reset_index(drop=True)

    # ---------- 0) Forward 1M return ----------
    d['ret_fwd_1m'] = d.groupby(id_col)[ret_col].shift(-1)

    # ---------- 1) SIZE: ln(Market Cap) ----------
    mc = pd.to_numeric(d[mcap_col], errors='coerce')
    d['SIZE'] = np.log(mc.replace(0, np.nan))

    # ---------- 2) VALUE: ln(Book/Market) via per-share BVPS/Price ----------
    px   = pd.to_numeric(d[price_col], errors='coerce')
    bvps = pd.to_numeric(d[bvps_col],  errors='coerce')
    bm = (bvps / px).replace([np.inf, -np.inf], np.nan)
    bm = bm.where((bm > 0) & np.isfinite(bm))
    d['VALUE'] = np.log(bm)

    g = d.groupby(id_col, group_keys=False)
    minp_mom  = int(mom_window  * min_history_frac)
    minp_vol  = int(vol_window  * min_history_frac)
    minp_beta = int(beta_window * min_history_frac)

    # ---------- 3) MOM 12-1 (t-12..t-2), log-compounded ----------
    # Build log(1+r), then:
    # sum over last 12 months up to t-1 (rolling)  MINUS the last month t-1  => exclude t-1
    d['_log1p'] = np.log1p(pd.to_numeric(d[ret_col], errors='coerce'))

    sum12_up_to_t1 = (
        g['_log1p']
        .rolling(mom_window, min_periods=minp_mom)
        .sum()
        .shift(1)
        .reset_index(level=0, drop=True)  # <<< align indices (fix)
    )
    last1 = g['_log1p'].shift(1)  # index already flat

    d['MOM'] = np.expm1(sum12_up_to_t1 - last1)
    d.drop(columns=['_log1p'], inplace=True)

    # ---------- 4) VOL 12m std (t-12..t-1), known at t ----------
    d['VOL'] = (
        g[ret_col]
        .rolling(vol_window, min_periods=minp_vol)
        .std()
        .shift(1)
        .reset_index(level=0, drop=True)   # <<< align indices (fix)
    )

    # ---------- 5) Market return per date (universe-consistent) ----------
    if compute_market or 'mkt_ret_m' not in d.columns:
        if mkt_method == 'equal':
            d['mkt_ret_m'] = d.groupby(date_col)[ret_col].transform('mean')
        elif mkt_method == 'value':
            def _vw(x):
                w = pd.to_numeric(x[mcap_col], errors='coerce').to_numpy()
                r = pd.to_numeric(x[ret_col], errors='coerce').to_numpy()
                w = np.where(np.isfinite(w) & (w>0), w, np.nan)
                return np.nan if np.nansum(w)==0 else np.nansum(w*r)/np.nansum(w)
            mkt = d.groupby(date_col).apply(_vw)
            d['mkt_ret_m'] = d[date_col].map(mkt)
        else:
            raise ValueError("mkt_method must be 'equal' or 'value'")

    # ---------- 6) BETA 24m vs that market (exposure), known at t ----------
    def _rolling_beta_for_group(sub):
        r = pd.to_numeric(sub[ret_col], errors='coerce')
        m = pd.to_numeric(sub['mkt_ret_m'], errors='coerce')
        mu_r = r.rolling(beta_window, min_periods=minp_beta).mean()
        mu_m = m.rolling(beta_window, min_periods=minp_beta).mean()
        cov  = (r*m).rolling(beta_window, min_periods=minp_beta).mean() - mu_r*mu_m
        var  = m.rolling(beta_window, min_periods=minp_beta).var()
        return (cov/var).shift(1)

    d['BETA'] = g.apply(_rolling_beta_for_group).reset_index(level=0, drop=True)

    # ---------- 7) Sector dummies ----------
    sec_cols = []
    if make_sector_dummies and sector_col in d.columns:
        dummies = pd.get_dummies(d[sector_col], prefix='SEC', drop_first=True, dtype=float)
        d = pd.concat([d, dummies], axis=1)
        sec_cols = dummies.columns.tolist()

    # ---------- 8) Optional: market-neutral forward return (diagnostics) ----------
    rm_fwd = d.groupby(date_col)['mkt_ret_m'].transform('first').shift(-1)
    d['ret_fwd_1m_MN'] = d['ret_fwd_1m'] - d['BETA'] * rm_fwd

    # ---------- 9) Clean numerics ----------
    for c in ['SIZE','VALUE','MOM','VOL','mkt_ret_m','BETA','ret_fwd_1m','ret_fwd_1m_MN']:
        if c in d.columns:
            d[c] = d[c].replace([np.inf, -np.inf], np.nan)

    controls_ff3  = ['SIZE','VALUE'] + sec_cols
    controls_full = ['SIZE','VALUE','MOM','VOL','BETA'] + sec_cols
    meta = {
        'sector_dummy_cols': sec_cols,
        'controls_ff3': controls_ff3,
        'controls_full': controls_full,
        'windows': {'MOM': mom_window, 'VOL': vol_window, 'BETA': beta_window},
        'market_method': mkt_method,
    }
    return d, meta


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
