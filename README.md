import numpy as np
import pandas as pd

def _as_list(x):
    if x is None: return []
    return x if isinstance(x, (list, tuple)) else [x]

def _cs_residualize(df, y_col, x_cols, group_keys, min_cs=10):
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for _, g in df.groupby(group_keys):
        if len(g) < min_cs:
            continue
        X = g[x_cols].copy().assign(const=1.0)
        y = g[y_col].astype(float)
        m = X.notna().all(1) & y.notna()
        if m.sum() <= len(X.columns) + 2:
            continue
        Xm = X.loc[m].values
        ym = y.loc[m].values
        beta = np.linalg.pinv(Xm.T @ Xm) @ (Xm.T @ ym)
        out.loc[m.index[m]] = ym - (Xm @ beta)
    return out

def make_signal_versions(
    df,
    signal_cols,
    *,
    date_col='date',
    by=None,                        # e.g. 'instrmtccy' or ['instrmtccy','GICS_sector_name']
    q_low=0.01, q_high=0.99,        # winsor quantiles
    hedges=('none','market','full'),
    market_factor='BETA',
    full_factors=('BETA','SIZE','VALUE','VOL'),
    clip_at=2.0,
    min_group=8,
    min_cs_resid=10
):
    """
    For each signal in `signal_cols`, build per (date × by-group):
      1) CS winsor (q_low/q_high)
      2) Hedge via CS residuals: 'none' | 'market'(BETA) | 'full'(BETA,SIZE,VALUE,VOL)
      3) CS z-score
      4) Clip to [-clip_at, clip_at]

    Returns
    -------
    df_out : DataFrame (original + new columns)
    created_cols : dict {signal: [new_col_names]}
    lists : dict with:
        - 'raw'      : list of the pure input signals found in df
        - 'variants' : list of all constructed columns
        - 'all'      : raw + variants (de-duplicated)
    """
    df_out = df.copy()
    group_keys = [date_col] + _as_list(by)
    g = df_out.groupby(group_keys, group_keys=False)

    present_full = [c for c in _as_list(full_factors) if c in df_out.columns]
    do_market = (market_factor in df_out.columns)

    hedge_list = [h.lower() for h in hedges]
    valid = {'none','market','full'}
    if not set(hedge_list).issubset(valid):
        raise ValueError(f"hedges must be subset of {valid}")

    created = {}
    all_variants = []

    for sig in signal_cols:
        if sig not in df_out.columns:
            raise KeyError(f"Signal '{sig}' not in DataFrame")

        # --- CS winsor ---
        ql = g[sig].transform(lambda s: s.quantile(q_low)  if s.notna().sum()>=min_group else np.nan)
        qh = g[sig].transform(lambda s: s.quantile(q_high) if s.notna().sum()>=min_group else np.nan)
        xw = df_out[sig].astype(float).clip(lower=ql, upper=qh)
        df_out[f'{sig}__w'] = xw  # temp

        new_cols = []

        for mode in hedge_list:
            if mode == 'none':
                base = xw
                tag  = 'BASE'
            elif mode == 'market':
                if not do_market:
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
                xcols = [c for c in present_full]
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

            # --- CS z-score ---
            df_out['_base_'] = base
            mu = df_out.groupby(group_keys)['_base_'].transform('mean')
            sd = df_out.groupby(group_keys)['_base_'].transform(lambda s: s.std(ddof=1))
            z  = (df_out['_base_'] - mu) / sd.replace(0, np.nan)  # leave NaN if degenerate
            zc = z.clip(-clip_at, clip_at)

            out_col = f'{sig}_{tag}_ZC{int(clip_at) if float(clip_at).is_integer() else clip_at}'
            df_out[out_col] = zc
            new_cols.append(out_col)
            all_variants.append(out_col)

            df_out.drop(columns=['_base_'], inplace=True)

        df_out.drop(columns=[f'{sig}__w'], inplace=True)
        created[sig] = new_cols

    # Build the lists you wanted
    raw_list = [s for s in signal_cols if s in df_out.columns]
    # De-dupe while preserving order
    seen = set()
    variants_list = [c for c in all_variants if not (c in seen or seen.add(c))]
    all_list = raw_list + [c for c in variants_list if c not in raw_list]

    lists = {
        'raw': raw_list,
        'variants': variants_list,
        'all': all_list,
    }
    return df_out, created, lists


SIGNALS = ['FEY','PEG_INV','SURPRISE_NTM']

df_proc, created_cols, lists = make_signal_versions(
    df_factors, SIGNALS,
    date_col='date',
    by='instrmtccy',                       # or ['instrmtccy','GICS_sector_name']
    q_low=0.01, q_high=0.99,
    hedges=('none','market','full'),
    market_factor='BETA',
    full_factors=('BETA','SIZE','VALUE','VOL'),
    clip_at=2.0
)

print(lists['raw'])       # -> ['FEY','PEG_INV','SURPRISE_NTM']
print(lists['variants'])  # -> ['FEY_BASE_ZC2','FEY_HBETA_ZC2','FEY_HALL_ZC2', ...]
print(lists['all'])       # -> raw + variants (handy to pass into IC/FMB utilities)




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
