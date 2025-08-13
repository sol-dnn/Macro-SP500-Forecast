import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# ---------- helpers ----------
def stars(p): 
    return '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else ''

def fama_macbeth(df, y_col, x_cols, date_col='date', min_cs=10):
    """
    Monthly cross-sectional OLS. Returns a table with mean coef, sd, se, t, p for each regressor.
    Drops months where CS sample < min_cs or not enough df.
    """
    rows = []
    for dt, cs in df.groupby(date_col):
        X = cs[x_cols].copy().assign(const=1.0)
        y = cs[y_col]
        m = X.notna().all(1) & y.notna()
        if m.sum() > len(x_cols) + 3 and m.sum() >= min_cs:
            Xm = X.loc[m].values
            ym = y.loc[m].values
            beta = np.linalg.pinv(Xm.T @ Xm) @ (Xm.T @ ym)
            rows.append(pd.Series(beta, index=X.columns))
    Bt = pd.DataFrame(rows)
    out = []
    for col in Bt.columns:
        vals = Bt[col].dropna()
        if vals.size >= 10:
            mean = vals.mean()
            sd   = vals.std(ddof=1)
            se   = sd / np.sqrt(vals.size)
            t    = mean / se if se > 0 else np.nan
            p    = 2 * stats.t.sf(np.abs(t), df=vals.size-1) if np.isfinite(t) else np.nan
            out.append([col, mean, sd, se, t, p, vals.size])
    res = pd.DataFrame(out, columns=['var','coef_mean','coef_sd','coef_se','t','p','T']).set_index('var')
    return res

def plot_fmb_bars(res_df, title='', sort=True):
    """
    Bar plot of mean coefficients with SD error bars and significance stars.
    Drops 'const'. Sorts by |coef| for readability (optional).
    """
    if res_df is None or res_df.empty:
        print("Empty regression result.")
        return
    res = res_df.drop(index=[i for i in res_df.index if i.lower()=='const'], errors='ignore').copy()
    if sort:
        res = res.reindex(res['coef_mean'].abs().sort_values(ascending=False).index)
    x = np.arange(len(res))
    h = res['coef_mean'].values
    e = res['coef_sd'].values

    plt.figure(figsize=(10,4))
    plt.bar(x, h)
    plt.errorbar(x, h, yerr=e, fmt='none', capsize=4)
    for xi, (hi, pi) in enumerate(zip(h, res['p'].values)):
        if np.isfinite(hi):
            plt.text(xi, hi + (0.02 if hi>=0 else -0.02), stars(pi), ha='center',
                     va='bottom' if hi>=0 else 'top', fontsize=10)
    plt.axhline(0, lw=1)
    plt.xticks(x, res.index, rotation=0)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def run_fmb_and_report(df, signal_cols, controls=None, *, 
                       date_col='date',
                       y_raw='ret_fwd_1m',
                       y_mn='ret_fwd_1m_MN',
                       use_market_neutral=False,
                       plot=False):
    """
    For each signal:
      - run univariate FMB:   y ~ signal
      - run multivariate FMB: y ~ signal + controls
    Returns:
      uni_summary:  DataFrame (rows = signals; cols = coef, t, p, stars, T)  [signal-only]
      multi_summary: DataFrame (rows = signals; cols = coef, t, p, stars, T, controls_used)
      uni_tables:   dict signal -> full regression table (all vars)
      multi_tables: dict signal -> full regression table (all vars)
    If plot=True: plots bar charts for full univariate & multivariate results per signal.
    """
    y_col = y_mn if use_market_neutral and (y_mn in df.columns) else y_raw
    if controls is None:
        controls = []  # no controls if not provided

    uni_rows, multi_rows = [], []
    uni_tables, multi_tables = {}, {}

    for sig in signal_cols:
        # --- univariate ---
        res_uni = fama_macbeth(df, y_col=y_col, x_cols=[sig], date_col=date_col)
        uni_tables[sig] = res_uni
        if sig in res_uni.index:
            r = res_uni.loc[sig]
            uni_rows.append({
                'signal': sig,
                'coef': r['coef_mean'],
                't': r['t'],
                'p': r['p'],
                'stars': stars(r['p']) if np.isfinite(r['p']) else '',
                'T': int(r['T'])
            })
        else:
            uni_rows.append({'signal': sig, 'coef': np.nan, 't': np.nan, 'p': np.nan, 'stars':'', 'T':0})

        # --- multivariate ---
        ctrls_present = [c for c in controls if c in df.columns and df[c].notna().any()]
        res_multi = fama_macbeth(df, y_col=y_col, x_cols=[sig] + ctrls_present, date_col=date_col)
        multi_tables[sig] = res_multi
        if sig in res_multi.index:
            r = res_multi.loc[sig]
            multi_rows.append({
                'signal': sig,
                'coef': r['coef_mean'],
                't': r['t'],
                'p': r['p'],
                'stars': stars(r['p']) if np.isfinite(r['p']) else '',
                'T': int(r['T']),
                'controls_used': ','.join(ctrls_present)
            })
        else:
            multi_rows.append({'signal': sig, 'coef': np.nan, 't': np.nan, 'p': np.nan, 'stars':'',
                               'T':0, 'controls_used': ','.join(ctrls_present)})

        # --- optional plots ---
        if plot:
            plot_fmb_bars(res_uni,  f'FMB — {sig} alone ({y_col})')
            plot_fmb_bars(res_multi,f'FMB — {sig} + controls ({y_col})')

    uni_summary   = pd.DataFrame(uni_rows).set_index('signal')
    multi_summary = pd.DataFrame(multi_rows).set_index('signal')
    return uni_summary, multi_summary, uni_tables, multi_tables

# choose your signals and controls
SIGNALS  = ['FWD_EY','PEG_INV','SURPRISE_NTM']       # example
CONTROLS = ['SIZE','VALUE','MOM','VOL','BETA']       # keep only those you prepared

# run with raw forward returns
uni_sum, multi_sum, uni_tbls, multi_tbls = run_fmb_and_report(
    df_factors, SIGNALS, controls=CONTROLS, use_market_neutral=False, plot=False
)

print("Univariate (signal alone):")
display(uni_sum)    # one row per signal: coef, t, p, stars, T

print("Multivariate (incremental vs controls):")
display(multi_sum)  # one row per signal: coef, t, p, stars, T, controls_used

# optional: plot full details for one signal
# plot_fmb_bars(uni_tbls['FWD_EY'],  'FMB — FWD_EY alone')
# plot_fmb_bars(multi_tbls['FWD_EY'],'FMB — FWD_EY + controls')



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
