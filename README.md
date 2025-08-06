def _erc_weights(self,
                 df_subset: pd.DataFrame,
                 signal_col: str,
                 portfolio_type: str,
                 window_months: int = 36) -> pd.Series:
    current_date = df_subset[self.date_col].iloc[0]
    w = pd.Series(0.0, index=df_subset.index)

    # Sélection de l'historique de rendements
    window_start = current_date - pd.DateOffset(months=window_months)

    def get_returns(assets):
        hist = self.returns_history[
            (self.returns_history[self.date_col] < current_date) &
            (self.returns_history[self.date_col] >= window_start) &
            (self.returns_history[self.asset_col].isin(assets))
        ]
        ret = hist.pivot(index=self.date_col, columns=self.asset_col, values='Return')
        return ret.dropna(how='any', axis=0)

    if portfolio_type == 'long_only':
        assets = df_subset[self.asset_col].unique()
        ret = get_returns(assets)
        if ret.shape[0] >= 2:
            cov = LedoitWolf().fit(ret.values).covariance_
            ivar = 1 / np.sqrt(np.diag(cov))
            weights = ivar / ivar.sum()
            w.loc[df_subset.index] = weights

    elif portfolio_type == 'short_only':
        assets = df_subset[self.asset_col].unique()
        ret = get_returns(assets)
        if ret.shape[0] >= 2:
            cov = LedoitWolf().fit(ret.values).covariance_
            ivar = 1 / np.sqrt(np.diag(cov))
            weights = -ivar / ivar.sum()
            w.loc[df_subset.index] = weights

    elif portfolio_type == 'long_short':
        longs = df_subset[df_subset[signal_col] > 0]
        shorts = df_subset[df_subset[signal_col] < 0]

        if not longs.empty:
            ret_long = get_returns(longs[self.asset_col].unique())
            if ret_long.shape[0] >= 2:
                cov_long = LedoitWolf().fit(ret_long.values).covariance_
                ivar_long = 1 / np.sqrt(np.diag(cov_long))
                w_long = ivar_long / ivar_long.sum() * 0.5
                w.loc[longs.index] = w_long

        if not shorts.empty:
            ret_short = get_returns(shorts[self.asset_col].unique())
            if ret_short.shape[0] >= 2:
                cov_short = LedoitWolf().fit(ret_short.values).covariance_
                ivar_short = 1 / np.sqrt(np.diag(cov_short))
                w_short = -ivar_short / ivar_short.sum() * 0.5
                w.loc[shorts.index] = w_short

    elif portfolio_type == 'q5_q1':
        df_q = df_subset.copy()
        df_q['quantile'] = pd.qcut(df_q[signal_col], 5, labels=False) + 1
        top = df_q[df_q['quantile'] == df_q['quantile'].max()]
        bottom = df_q[df_q['quantile'] == df_q['quantile'].min()]

        if not top.empty:
            ret_top = get_returns(top[self.asset_col].unique())
            if ret_top.shape[0] >= 2:
                cov_top = LedoitWolf().fit(ret_top.values).covariance_
                ivar_top = 1 / np.sqrt(np.diag(cov_top))
                w_top = ivar_top / ivar_top.sum() * 0.5
                w.loc[top.index] = w_top

        if not bottom.empty:
            ret_bot = get_returns(bottom[self.asset_col].unique())
            if ret_bot.shape[0] >= 2:
                cov_bot = LedoitWolf().fit(ret_bot.values).covariance_
                ivar_bot = 1 / np.sqrt(np.diag(cov_bot))
                w_bot = -ivar_bot / ivar_bot.sum() * 0.5
                w.loc[bottom.index] = w_bot

    else:
        raise ValueError(f"Unsupported portfolio_type: {portfolio_type}")

    return w

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
