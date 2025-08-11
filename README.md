import numpy as np
import pandas as pd

df = df_pred.copy()

# requis
assert {'eps_ml_ntm','price'}.issubset(df.columns), "cols manquantes"

# tri temporel si 'date' et 'ticker' existent (utile pour révisions)
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
sort_cols = [c for c in ['ticker','date'] if c in df.columns]
if sort_cols:
    df = df.sort_values(sort_cols)

# basiques
eps_hat = df['eps_ml_ntm'].replace(0, np.nan)
P = df['price'].replace(0, np.nan)

df['FEY'] = eps_hat / P                       # forward E/P
df['FPE'] = P / eps_hat                       # forward P/E

# croissance (si EPS courant dispo)
if 'eps_ttm' in df.columns:
    eps_curr = df['eps_ttm'].replace(0, np.nan)
    df['G_1Y']     = (eps_hat / eps_curr) - 1
    df['G_1Y_LOG'] = np.log(eps_hat / eps_curr)
else:
    df['G_1Y'] = np.nan
    df['G_1Y_LOG'] = np.nan

# PEG forward (ε pour éviter explosion quand g≈0)
EPS_G_EPS = 0.01
df['PEG'] = df['FPE'] / np.maximum(df['G_1Y'], EPS_G_EPS)

# révisions des prévisions (si ticker/date)
if {'ticker','date'}.issubset(df.columns):
    df['REV_1M'] = df.groupby('ticker')['eps_ml_ntm'].pct_change(1)
    df['REV_3M'] = df.groupby('ticker')['eps_ml_ntm'].pct_change(3)
    df['REV_ACCEL'] = df['REV_1M'] - df.groupby('ticker')['REV_1M'].shift(1)
else:
    df['REV_1M'] = df['REV_3M'] = df['REV_ACCEL'] = np.nan

# divergence vs consensus (si dispo)
if 'eps_cons_ntm' in df.columns:
    cons = df['eps_cons_ntm'].replace(0, np.nan)
    df['DIV_RATIO'] = (eps_hat / cons) - 1
    if 'sigma_cons_ntm' in df.columns:
        sig = df['sigma_cons_ntm'].replace(0, np.nan)
        df['DIV_Z'] = (eps_hat - cons) / sig
    else:
        df['DIV_Z'] = np.nan
else:
    df['DIV_RATIO'] = df['DIV_Z'] = np.nan

# qualité/profitabilité anticipée (si dispo)
if 'bvps' in df.columns:
    df['ROE_FWD'] = eps_hat / df['bvps'].replace(0, np.nan)
else:
    df['ROE_FWD'] = np.nan

if 'salesps' in df.columns:
    df['PM_FWD'] = eps_hat / df['salesps'].replace(0, np.nan)
else:
    df['PM_FWD'] = np.nan

# mispricing vs multiple cible sectoriel (si 'sector' dispo)
if {'sector'}.issubset(df.columns):
    # multiple cible M* = médiane FPE par (date, secteur)
    df['MSTAR_SECTOR'] = df.groupby(['date','sector'])['FPE'].transform('median') if 'date' in df.columns else df.groupby('sector')['FPE'].transform('median')
    df['MISPRICING_MSTAR'] = (df['MSTAR_SECTOR'] * eps_hat - P) / P
else:
    # fallback global (par date si dispo)
    if 'date' in df.columns:
        mstar = df.groupby('date')['FPE'].transform('median')
    else:
        mstar = df['FPE'].median()
    df['MSTAR_SECTOR'] = mstar
    df['MISPRICING_MSTAR'] = (df['MSTAR_SECTOR'] * eps_hat - P) / P

# (optionnel) incertitude du modèle si dispo (ex: écart-type des prévisions d'ensemble)
if 'eps_ml_ntm_std' in df.columns:
    df['UNCERTAINTY'] = -df['eps_ml_ntm_std']   # plus petit = mieux
else:
    df['UNCERTAINTY'] = np.nan

# nettoyage numérique
for c in ['FEY','FPE','G_1Y','G_1Y_LOG','PEG','REV_1M','REV_3M','REV_ACCEL',
          'DIV_RATIO','DIV_Z','ROE_FWD','PM_FWD','MISPRICING_MSTAR','UNCERTAINTY']:
    df[c] = df[c].replace([np.inf, -np.inf], np.nan)

# df contient maintenant les colonnes facteurs "raw"
df_factors = df



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
