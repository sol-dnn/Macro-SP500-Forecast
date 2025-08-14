import numpy as np
import pandas as pd

# -----------------------------
# helpers
# -----------------------------
def _as_list(x):
    if x is None: return []
    return x if isinstance(x, (list, tuple)) else [x]

def _cs_residualize(df, y_col, x_cols, group_keys, min_cs=10):
    """
    Cross-sectional residuals per group: y - X*beta with intercept.
    Returns a Series aligned to df.index.
    """
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for _, g in df.groupby(group_keys):
        if len(g) < min_cs:
            continue
        X = g[x_cols].copy()
        X = X.assign(const=1.0)
        y = g[y_col].astype(float)
        m = X.notna().all(1) & y.notna()
        if m.sum() <= len(X.columns) + 2:
            continue
        Xm = X.loc[m].values
        ym = y.loc[m].values
        beta = np.linalg.pinv(Xm.T @ Xm) @ (Xm.T @ ym)
        yhat = Xm @ beta
        resid = ym - yhat
        out.loc[m.index[m]] = resid
    return out

# -----------------------------
# main pipeline
# -----------------------------
def make_signal_versions(
    df,
    signal_cols,
    *,
    date_col='date',
    by=None,                        # e.g. 'instrmtccy' or ['instrmtccy','GICS_sector_name']
    q_low=0.01, q_high=0.99,        # winsor quantiles (cross-sectional)
    hedges=('none','market','full'),# which versions to build
    market_factor='BETA',           # used for 'market' hedge
    full_factors=('BETA','SIZE','VALUE','VOL'),  # used for 'full' hedge
    clip_at=2.0,                    # final clip on the z-scored series
    min_group=8,                    # min names per CS group to compute stats
    min_cs_resid=10                 # min names per CS group to run regression hedge
):
    """
    For each signal in `signal_cols`, build (per date × by-group):
      1) cross-sectional winsorization (q_low/q_high),
      2) hedging by regression residuals:
           - 'none'   : no hedge
           - 'market' : hedge on [1, BETA]
           - 'full'   : hedge on [1, BETA, SIZE, VALUE, VOL] (present columns only)
      3) cross-sectional z-score
      4) clip to [-clip_at, clip_at]

    Returns
    -------
    df_out : DataFrame (original + new columns)
    created_cols : dict {signal: [new_col_names]}
    """
    df_out = df.copy()
    group_keys = [date_col] + _as_list(by)
    g = df_out.groupby(group_keys, group_keys=False)

    # ensure factors exist flags
    present_full = [c for c in _as_list(full_factors) if c in df_out.columns]
    do_market = (market_factor in df_out.columns)

    # normalize hedge names
    hedge_list = [h.lower() for h in hedges]
    valid = {'none','market','full'}
    if not set(hedge_list).issubset(valid):
        raise ValueError(f"hedges must be subset of {valid}")

    created = {}

    for sig in signal_cols:
        if sig not in df_out.columns:
            raise KeyError(f"Signal '{sig}' not in DataFrame")

        # ---------- 1) CS winsor ----------
        # per-group quantiles
        ql = g[sig].transform(lambda s: s.quantile(q_low)  if s.notna().sum()>=min_group else np.nan)
        qh = g[sig].transform(lambda s: s.quantile(q_high) if s.notna().sum()>=min_group else np.nan)
        xw = df_out[sig].astype(float).clip(lower=ql, upper=qh)
        df_out[f'{sig}__w'] = xw  # temp

        new_cols = []

        # build each hedge version
        for mode in hedge_list:
            if mode == 'none':
                base = xw
                tag  = 'BASE'
            elif mode == 'market':
                if not do_market:
                    # skip if beta missing
                    continue
                resid = _cs_residualize(
                    df_out.assign(_y_=xw),
                    y_col='_y_',
                    x_cols=[market_factor],
                    group_keys=group_keys,
                    min_cs=min_cs_resid
                )
                base = resid
                tag  = 'HBETA'
            elif mode == 'full':
                xcols = [c for c in present_full]  # only those present
                if not xcols:
                    continue
                resid = _cs_residualize(
                    df_out.assign(_y_=xw),
                    y_col='_y_',
                    x_cols=xcols,
                    group_keys=group_keys,
                    min_cs=min_cs_resid
                )
                base = resid
                tag  = 'HALL'
            else:
                continue

            # ---------- 3) CS z-score on the (hedged) series ----------
            df_out['_base_'] = base
            mu  = df_out.groupby(group_keys)['_base_'].transform('mean')
            sd  = df_out.groupby(group_keys)['_base_'].transform(lambda s: s.std(ddof=1))
            z   = (df_out['_base_'] - mu) / sd.replace(0, np.nan)   # leave NaN if degenerate
            # ---------- 4) clip ----------
            zc  = z.clip(-clip_at, clip_at)

            out_col = f'{sig}_{tag}_ZC{int(clip_at) if clip_at.is_integer() else clip_at}'
            df_out[out_col] = zc
            new_cols.append(out_col)

            # clean temp
            df_out.drop(columns=['_base_'], inplace=True)

        # drop temp winsor
        df_out.drop(columns=[f'{sig}__w'], inplace=True)
        created[sig] = new_cols

    return df_out, created


SIGNALS = ['FEY','PEG_INV','SURPRISE_NTM']  # your signal columns

df_proc, cols_made = make_signal_versions(
    df_factors, SIGNALS,
    date_col='date',
    by='instrmtccy',                 # or ['instrmtccy','GICS_sector_name'] if you prefer within-sector CS
    q_low=0.01, q_high=0.99,
    hedges=('none','market','full'),
    market_factor='BETA',
    full_factors=('BETA','SIZE','VALUE','VOL'),
    clip_at=2.0,
    min_group=8,
    min_cs_resid=10
)

print(cols_made)
# {'FEY': ['FEY_BASE_ZC2', 'FEY_HBETA_ZC2', 'FEY_HALL_ZC2'], ...}




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
