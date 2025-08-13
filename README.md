import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

def cs_corr_table(df,
                  signal_cols,
                  factor_cols,
                  *,
                  date_col='date',
                  ret_col='ret_fwd_1m',
                  method='spearman',   # 'spearman' or 'pearson'
                  min_cs_n=10,         # min names per date to compute a CS corr
                  plot=False):
    """
    Cross-sectional correlations per date between each signal and:
      - every factor in `factor_cols`
      - the forward return column `ret_col`
    Then time-average those CS correlations to a single number per pair.

    Returns: DataFrame with one row per signal.
             Columns: factor columns + ['RET_FWD_1M'] + ['T_months'].
             'T_months' is the min number of monthly CS correlations used
             across all targets for that signal (coverage sanity check).

    If plot=True: prints a barplot per signal (factors + RET_FWD_1M).
    """
    assert all(c in df.columns for c in signal_cols), "Some signals missing in df"
    assert all(c in df.columns for c in factor_cols), "Some factors missing in df"
    assert ret_col in df.columns, f"{ret_col} missing in df"

    targets = list(factor_cols) + ['RET_FWD_1M']  # label for readability
    col_map = {**{c: c for c in factor_cols}, 'RET_FWD_1M': ret_col}

    rows = []
    for sig in signal_cols:
        corrs = {}
        Ts = []
        for tgt in targets:
            tcol = col_map[tgt]
            vals = []
            for dte, cs in df.groupby(date_col):
                s = cs[sig]
                f = cs[tcol]
                m = s.notna() & f.notna()
                if m.sum() >= min_cs_n:
                    if method == 'spearman':
                        vals.append(stats.spearmanr(s[m], f[m]).correlation)
                    else:
                        vals.append(np.corrcoef(s[m], f[m])[0, 1])
            corrs[tgt] = float(np.nanmean(vals)) if len(vals) else np.nan
            Ts.append(len(vals))
        # build row
        row = {**corrs, 'T_months': int(np.nanmin(Ts) if len(Ts) else 0)}
        row['signal'] = sig
        rows.append(row)

        # optional barplot
        if plot:
            xs = [k for k in targets]  # order: factors, then forward return
            ys = [corrs[k] for k in xs]
            x = np.arange(len(xs))
            plt.figure(figsize=(8,3))
            plt.bar(x, ys)
            plt.axhline(0, lw=1)
            plt.xticks(x, xs, rotation=0)
            plt.title(f'Avg CS {method.title()} corr — {sig}  (T≥{row["T_months"]})')
            plt.tight_layout()
            plt.show()

    res = pd.DataFrame(rows).set_index('signal')[targets + ['T_months']]
    return res

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
