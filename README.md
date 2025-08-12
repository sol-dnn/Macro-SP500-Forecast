import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# ===========================
# Config: column names (yours)
# ===========================
ID_COL     = 'sedolcd'
DATE_COL   = 'date'
RET_COL    = 'totalreturn'       # simple monthly total return (e.g., 0.02)
PRICE_COL  = 'raw_quoteclose'
MCAP_COL   = 'marketvalue'
BVPS_COL   = 'wsf_bps_ltm'
SECTOR_COL = 'GICS_sector_name'  # optional; not strictly required

# If you have a proper market return series per month, merge as this column name:
MKT_RET_COL = 'mkt_ret_m'        # optional; if missing, a proxy will be built


# ==========================================
# Helpers: stars and safe printing utilities
# ==========================================
def stars(p):
    return '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else ''

def safe_std(x):
    x = pd.Series(x).dropna()
    return np.nan if x.size == 0 else x.std(ddof=1)


# ==========================================
# 0) PREP: forward returns and factor exposures
# ==========================================
def prepare_data(df):
    """
    Returns a copy with:
      - ret_fwd_1m
      - SIZE, VALUE, MOM (12-1), VOL (12m), BETA (24m vs market)
      - market return series MKT_RET_COL (original or proxy)
    """
    d = df.copy()
    d[DATE_COL] = pd.to_datetime(d[DATE_COL])
    d = d.sort_values([ID_COL, DATE_COL]).reset_index(drop=True)

    # Forward 1m return
    d['ret_fwd_1m'] = d.groupby(ID_COL)[RET_COL].shift(-1)

    # SIZE
    if MCAP_COL in d.columns:
        d['SIZE'] = np.log(d[MCAP_COL].replace(0, np.nan))
    else:
        d['SIZE'] = np.nan

    # VALUE (book-to-market)
    if BVPS_COL in d.columns and PRICE_COL in d.columns:
        x = (d[BVPS_COL] / d[PRICE_COL]).replace([np.inf, -np.inf, 0], np.nan)
        d['VALUE'] = np.log(x)
    else:
        d['VALUE'] = np.nan

    # Momentum 12-1 (log-compound), exclude last month
    d['log1p_ret'] = np.log1p(d[RET_COL])
    g = d.groupby(ID_COL, group_keys=False)
    sum12 = g['log1p_ret'].rolling(12, min_periods=9).sum().shift(1)
    last1 = g['log1p_ret'].shift(1)
    d['MOM'] = np.expm1(sum12 - last1)

    # VOL: 12m rolling std, OOS with shift(1)
    d['VOL'] = g[RET_COL].rolling(12, min_periods=9).std().shift(1)

    # Market return series
    if MKT_RET_COL not in d.columns:
        # proxy: equal-weight market return across all names per date
        d[MKT_RET_COL] = d.groupby(DATE_COL)[RET_COL].transform('mean')

    # BETA: 24m rolling slope vs market (shift(1) to be known at t)
    def rolling_beta(ret, mkt, win=24):
        mu_r = ret.rolling(win, min_periods=int(win*0.7)).mean()
        mu_m = mkt.rolling(win, min_periods=int(win*0.7)).mean()
        cov  = (ret*mkt).rolling(win, min_periods=int(win*0.7)).mean() - mu_r*mu_m
        var  = mkt.rolling(win, min_periods=int(win*0.7)).var()
        return (cov / var).shift(1)

    d['BETA'] = g.apply(lambda x: rolling_beta(x[RET_COL], x[MKT_RET_COL])).reset_index(level=0, drop=True)

    # clean numerics
    for c in ['SIZE','VALUE','MOM','VOL','BETA','ret_fwd_1m']:
        if c in d.columns:
            d[c] = d[c].replace([np.inf, -np.inf], np.nan)

    return d


# ==========================================
# 1) IC (Spearman) for a list of signals
# ==========================================
def compute_ic(df, signal_cols):
    """
    Cross-sectional Spearman between signal at t and forward 1m returns.
    Returns a DataFrame: signal, IC_mean, t, p, T_months.
    """
    out = []
    for sig in signal_cols:
        ICs = []
        for d, cs in df.groupby(DATE_COL):
            s = cs[sig]; r = cs['ret_fwd_1m']
            m = s.notna() & r.notna()
            if m.sum() >= 10:
                ICs.append(stats.spearmanr(s[m], r[m]).correlation)
        ICs = pd.Series(ICs)
        if ICs.size:
            IC_mean = ICs.mean()
            IC_t    = IC_mean / (ICs.std(ddof=1) / np.sqrt(ICs.size)) if ICs.std(ddof=1) > 0 else np.nan
            IC_p    = 2 * stats.t.sf(np.abs(IC_t), df=ICs.size-1) if np.isfinite(IC_t) else np.nan
            out.append([sig, IC_mean, IC_t, IC_p, ICs.size])
    res = pd.DataFrame(out, columns=['signal','IC_mean','t','p','T_months']).set_index('signal')
    return res


# ==========================================
# 2) Fama–MacBeth regressions (univariate & multivariate)
# ==========================================
def fama_macbeth(df, y_col, x_cols):
    """
    Monthly cross-sectional OLS; returns mean coef, sd, se, t, p across time for each regressor.
    """
    rows = []
    for d, cs in df.groupby(DATE_COL):
        X = cs[x_cols].copy().assign(const=1.0)
        y = cs[y_col]
        m = X.notna().all(1) & y.notna()
        if m.sum() > len(x_cols) + 3:
            Xm = X[m].values
            ym = y[m].values
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

def run_fmb_for_signals(df, signal_cols, use_market_neutral=False, include_beta=True):
    """
    For each signal in signal_cols:
      - Univariate FMB: r_{t+1} ~ signal
      - Multivariate FMB: r_{t+1} ~ signal + SIZE, VALUE, MOM, VOL (+ BETA if include_beta=True)
    If use_market_neutral=True, uses market-neutral forward returns and removes BETA from controls.
    Returns dicts: uni_results[signal], multi_results[signal]
    """
    d = df.copy()
    y_col = 'ret_fwd_1m'
    KNOWN = ['SIZE','VALUE','MOM','VOL']
    if include_beta:
        KNOWN += ['BETA']

    if use_market_neutral:
        # market-neutral forward returns: r_{i,t+1} - beta_{i,t} * r_{m,t+1}
        # we need next month's market return; proxy via DATE-level series
        rm_fwd = d.groupby(DATE_COL)[MKT_RET_COL].transform('first').shift(-1)
        d['ret_fwd_1m_MN'] = d['ret_fwd_1m'] - d['BETA'] * rm_fwd
        y_col = 'ret_fwd_1m_MN'
        # if market-neutral, we typically drop BETA from controls
        KNOWN = ['SIZE','VALUE','MOM','VOL']

    uni, multi = {}, {}
    for sig in signal_cols:
        # Univariate
        res_uni = fama_macbeth(d, y_col, [sig])
        # Multivariate (only controls that actually exist)
        controls = [c for c in KNOWN if c in d.columns and d[c].notna().any()]
        res_multi = fama_macbeth(d, y_col, [sig] + controls)
        uni[sig], multi[sig] = res_uni, res_multi
    return uni, multi


# ==========================================
# 3) Plot: bar of coefficients with SD + stars
# ==========================================
def plot_coef_bar(res_df, title):
    """
    res_df: output from fama_macbeth (one regression result).
    Plots mean coefficients with SD error bars and significance stars.
    """
    res = res_df.drop(index=[i for i in res_df.index if i.lower()=='const'], errors='ignore')
    order = res.index.tolist()
    x = np.arange(len(order))
    h = res.loc[order, 'coef_mean'].values
    e = res.loc[order, 'coef_sd'].values

    plt.figure(figsize=(10,4))
    plt.bar(x, h)
    plt.errorbar(x, h, yerr=e, fmt='none', capsize=4)
    for xi, (hi, pi) in enumerate(zip(h, res.loc[order, 'p'].values)):
        plt.text(xi, hi + (0.02 if hi>=0 else -0.02), stars(pi),
                 ha='center', va='bottom' if hi>=0 else 'top', fontsize=10)
    plt.xticks(x, order, rotation=0)
    plt.axhline(0, lw=1)
    plt.title(title)
    plt.tight_layout()
    plt.show()


# ==========================================
# 4) Cross-sectional correlations: signal vs factor exposures
# ==========================================
def cs_correlations(df, signal_cols, factor_cols, method='spearman'):
    """
    Cross-sectional correlations per date between each signal and each factor exposure,
    averaged over time, with SD across time. Returns dict of DataFrames per signal.
    """
    out = {}
    for sig in signal_cols:
        rows = []
        for fac in factor_cols:
            vals = []
            for dte, cs in df.groupby(DATE_COL):
                s1, s2 = cs[sig], cs[fac]
                m = s1.notna() & s2.notna()
                if m.sum() >= 10:
                    if method == 'spearman':
                        vals.append(stats.spearmanr(s1[m], s2[m]).correlation)
                    else:
                        vals.append(np.corrcoef(s1[m], s2[m])[0,1])
            if len(vals):
                rows.append([fac, np.nanmean(vals), safe_std(vals), len(vals)])
        res = pd.DataFrame(rows, columns=['factor','avg_corr','sd_corr','T']).set_index('factor').sort_values('avg_corr')
        out[sig] = res
    return out

def plot_corr_bar(res_corr, sig_name, title_suffix='Avg CS correlation'):
    """
    Bar plot for one signal's correlation summary (res_corr is a DataFrame with avg_corr, sd_corr).
    """
    if res_corr is None or res_corr.empty:
        print(f"No correlation data for {sig_name}")
        return
    x = np.arange(len(res_corr))
    y = res_corr['avg_corr'].values
    e = res_corr['sd_corr'].values
    plt.figure(figsize=(8,3))
    plt.bar(x, y)
    plt.errorbar(x, y, yerr=e, fmt='none', capsize=4)
    plt.axhline(0, lw=1)
    plt.xticks(x, res_corr.index, rotation=0)
    plt.title(f'{title_suffix}: {sig_name}')
    plt.tight_layout()
    plt.show()


# ===========================
# ========== USAGE ==========
# ===========================
# 1) Start from your original DataFrame df_pred (with columns defined above).
#    Example signals to test:
# SIG_COLS = ['FEY', 'PEG_INV', 'SURPRISE_PRICE_NTM']  # put your actual signal column names
# df_all = prepare_data(df_pred)

# 2) IC for all signals
# ic_table = compute_ic(df_all, SIG_COLS)
# display(ic_table)

# 3) Fama–MacBeth regressions
#    Option A: raw forward returns, include BETA control
# uni_A, multi_A = run_fmb_for_signals(df_all, SIG_COLS, use_market_neutral=False, include_beta=True)

#    Option B: market-neutral forward returns, drop BETA control
# uni_B, multi_B = run_fmb_for_signals(df_all, SIG_COLS, use_market_neutral=True, include_beta=False)

#    Plot examples:
# for s in SIG_COLS:
#     plot_coef_bar(uni_A[s],   f'FMB raw r_1m — {s} alone')
#     plot_coef_bar(multi_A[s], f'FMB raw r_1m — {s} + controls')
#     # optional:
#     # plot_coef_bar(uni_B[s],   f'FMB MN r_1m — {s} alone')
#     # plot_coef_bar(multi_B[s], f'FMB MN r_1m — {s} + SIZE/VALUE/MOM/VOL')

# 4) Cross-sectional correlations: signal vs factor exposures (including BETA)
# known_factors = [c for c in ['SIZE','VALUE','MOM','VOL','BETA'] if c in df_all.columns and df_all[c].notna().any()]
# corr_dict = cs_correlations(df_all, SIG_COLS, known_factors, method='spearman')
# for s in SIG_COLS:
#     plot_corr_bar(corr_dict[s], s, title_suffix='Avg CS Spearman corr')
#     # print(corr_dict[s])  # to see the table too


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
