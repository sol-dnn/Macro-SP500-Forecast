import pandas as pd
import numpy as np

# Préparation
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# Calcul du biais analyste
df['bias'] = (
    df['target_eps_ltm_12m']
    - df['ic_estimate_eps_mean_ntm_twa']
) / df['quoteclose']

# Quintiles de valorisation par date
df['valuation_ratio'] = df['quoteclose'] / df['marketvalue']
df['valuation_quintile'] = (
    df.groupby('date')['valuation_ratio']
      .transform(lambda x: pd.qcut(x, 5, labels=False, duplicates='drop') + 1)
)

# Pré-allocation des colonnes de rolling bias
for lvl in ['global', 'val', 'sector', 'ticker']:
    for stat in ['mean', 'median']:
        df[f'bias_{lvl}_{stat}_24m'] = np.nan

# Boucle sur chaque date
for current_date in df['date'].drop_duplicates().sort_values():
    window_start = current_date - pd.DateOffset(months=24)
    mask_win = (df['date'] > window_start) & (df['date'] < current_date)
    win = df.loc[mask_win]

    #  ▶︎ ne calculer que si on a au moins 24 mois distincts
    n_months = win['date'].dt.to_period('M').nunique()
    if n_months < 24:
        continue

    # Calcul des biais sur la fenêtre valide
    g_mean = win['bias'].mean()
    g_med  = win['bias'].median()

    vb = win.groupby('valuation_quintile')['bias']
    val_mean = vb.mean();    val_med  = vb.median()

    sb = win.groupby('GICS_sector_name')['bias']
    sec_mean = sb.mean();    sec_med  = sb.median()

    tb = win.groupby('sedolcd')['bias']
    tic_mean = tb.mean();    tic_med  = tb.median()

    # Assignation des biais calculés aux lignes du mois courant
    mask_curr = df['date'] == current_date
    df.loc[mask_curr, 'bias_global_mean_24m']   = g_mean
    df.loc[mask_curr, 'bias_global_median_24m'] = g_med

    for q, m in val_mean.items():
        sel = mask_curr & (df['valuation_quintile'] == q)
        df.loc[sel, 'bias_val_mean_24m']   = m
        df.loc[sel, 'bias_val_median_24m'] = val_med.loc[q]

    for sec, m in sec_mean.items():
        sel = mask_curr & (df['GICS_sector_name'] == sec)
        df.loc[sel, 'bias_sector_mean_24m']   = m
        df.loc[sel, 'bias_sector_median_24m'] = sec_med.loc[sec]

    for tic, m in tic_mean.items():
        sel = mask_curr & (df['sedolcd'] == tic)
        df.loc[sel, 'bias_ticker_mean_24m']   = m
        df.loc[sel, 'bias_ticker_median_24m'] = tic_med.loc[tic]

# Construction des prédictions naïves corrigées
for lvl, col_suffix in [
    ('global', ''), ('val', '_val'),
    ('sector', '_sector'), ('ticker', '_ticker')
]:
    for stat in ['mean', 'median']:
        bias_col = f'bias_{lvl}_{stat}_24m'
        pred_col = f'naive_pred{col_suffix}_{stat}'
        df[pred_col] = (
            df['ic_estimate_eps_mean_ntm_twa']
            + df[bias_col] * df['quoteclose']
        )


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
