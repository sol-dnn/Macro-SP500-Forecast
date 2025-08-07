import pandas as pd
import numpy as np

# Assurer bon format date et tri
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['date'])

# Biais de prévision
df['bias'] = (df['target_eps_ltm_12m'] - df['ic_estimate_eps_mean_ntm_twa']) / df['quoteclose']

# Ratio de valorisation : prix / capitalisation
df['valuation_ratio'] = df['quoteclose'] / df['marketvalue']

# Quintiles de valorisation par date (cross-sectionnel)
df['valuation_quintile'] = df.groupby('date')['valuation_ratio'].transform(
    lambda x: pd.qcut(x, 5, labels=False, duplicates='drop') + 1
)

# --- Fonctions utilitaires rolling ---
def rolling_group_bias(df, group_cols, bias_col='bias', win=24, method='mean'):
    """
    Calcule la moyenne ou médiane du biais sur 24 mois glissants pour chaque groupe (e.g., secteur, quintile)
    """
    df = df.sort_values(['date'])
    grouped = df.groupby(group_cols)
    if method == 'mean':
        return grouped[bias_col].transform(lambda x: x.shift(1).rolling(win, min_periods=12).mean())
    elif method == 'median':
        return grouped[bias_col].transform(lambda x: x.shift(1).rolling(win, min_periods=12).median())
    else:
        raise ValueError("Method must be 'mean' or 'median'")

# --- Biais Global (mean & median) ---
df['bias_global_mean_24m'] = df['bias'].shift(1).rolling(window=24, min_periods=12).mean()
df['bias_global_median_24m'] = df['bias'].shift(1).rolling(window=24, min_periods=12).median()

# --- Biais par quintile de valorisation ---
df['bias_val_mean_24m'] = rolling_group_bias(df, ['valuation_quintile'], method='mean')
df['bias_val_median_24m'] = rolling_group_bias(df, ['valuation_quintile'], method='median')

# --- Biais par secteur ---
df['bias_sector_mean_24m'] = rolling_group_bias(df, ['GICS_sector_name'], method='mean')
df['bias_sector_median_24m'] = rolling_group_bias(df, ['GICS_sector_name'], method='median')

# --- Facultatif : Biais par titre (sedolcd) ---
df['bias_ticker_mean_24m'] = rolling_group_bias(df, ['sedolcd'], method='mean')
df['bias_ticker_median_24m'] = rolling_group_bias(df, ['sedolcd'], method='median')

# --- Génération des prédictions corrigées (6 modèles) ---
for level in ['global', 'val', 'sector', 'ticker']:
    for method in ['mean', 'median']:
        bias_col = f'bias_{level}_{method}_24m'
        pred_col = f'naive_pred_{level}_{method}'
        df[pred_col] = df['ic_estimate_eps_mean_ntm_twa'] + df[bias_col] * df['quoteclose']



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
