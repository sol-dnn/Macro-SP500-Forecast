import pandas as pd
import numpy as np
from sklearn.covariance import LedoitWolf

class MyStrategy:
    def __init__(self,
                 signal_data: pd.DataFrame,
                 returns_history: pd.DataFrame,
                 date_col: str = 'Date',
                 asset_col: str = 'Asset'):
        self.data = signal_data
        self.returns_history = returns_history
        self.date_col = date_col
        self.asset_col = asset_col

    def build_portfolio(self,
                        signal_col: str,
                        portfolio_type: str,
                        weight_type: str = 'equal') -> pd.DataFrame:
        df = self.data.dropna(subset=[signal_col]).copy()

        def process_group(group: pd.DataFrame) -> pd.DataFrame:
            group = group.copy()
            group['weight'] = 0.0
            # select subset based on portfolio type
            if portfolio_type == 'long_only':
                sel = group[group[signal_col] > 0]
            elif portfolio_type == 'short_only':
                sel = group[group[signal_col] < 0]
            elif portfolio_type == 'long_short':
                sel = pd.concat([group[group[signal_col] > 0], group[group[signal_col] < 0]])
            elif portfolio_type == 'q5_q1':
                group['quantile'] = pd.qcut(group[signal_col], 5, labels=False) + 1
                top = group[group['quantile'] == group['quantile'].max()]
                bottom = group[group['quantile'] == group['quantile'].min()]
                sel = pd.concat([top, bottom])
            else:
                raise ValueError(f"Unknown portfolio_type: {portfolio_type}")
            # compute weights
            weights = self._compute_weights(sel, signal_col, weight_type, portfolio_type)
            group.loc[sel.index, 'weight'] = weights.values
            return group

        result = (
            df.groupby(self.date_col)
              .apply(process_group)
              .reset_index(drop=True)
        )
        return result

    def _compute_weights(self,
                         df_subset: pd.DataFrame,
                         signal_col: str,
                         weight_type: str,
                         portfolio_type: str) -> pd.Series:
        if weight_type == 'equal':
            return pd.Series(1.0 / len(df_subset), index=df_subset.index)
        elif weight_type == 'signal':
            raw = df_subset[signal_col]
            return raw / raw.abs().sum()
        elif weight_type == 'erc':
            return self._erc_weights(df_subset, signal_col, portfolio_type)
        else:
            raise ValueError(f"Unknown weight_type: {weight_type}")

    def _erc_weights(self,
                     df_subset: pd.DataFrame,
                     signal_col: str,
                     portfolio_type: str,
                     window_months: int = 36) -> pd.Series:
        current_date = df_subset[self.date_col].iloc[0]

        # Sélection des actifs selon le portefeuille
        if portfolio_type == 'long_only':
            df_in = df_subset[df_subset[signal_col] > 0]
        elif portfolio_type == 'short_only':
            df_in = df_subset[df_subset[signal_col] < 0]
        elif portfolio_type in ['long_short', 'q5_q1']:
            df_in = df_subset.copy()
        else:
            raise ValueError(f"Unsupported portfolio_type: {portfolio_type}")

        assets = df_in[self.asset_col].unique()
        if len(assets) == 0:
            return pd.Series(0.0, index=df_subset.index)

        # Récupérer les rendements historiques passés dans la fenêtre
        window_start = current_date - pd.DateOffset(months=window_months)
        hist_returns = self.returns_history[
            (self.returns_history[self.date_col] < current_date) &
            (self.returns_history[self.date_col] >= window_start) &
            (self.returns_history[self.asset_col].isin(assets))
        ]

        # Pivot format large
        ret_wide = hist_returns.pivot(index=self.date_col,
                                      columns=self.asset_col,
                                      values='Return')
        ret_sel = ret_wide.dropna(how='any')
        if ret_sel.shape[0] < 2:
            return pd.Series(0.0, index=df_subset.index)

        # Estimation de covariance régulière
        cov = LedoitWolf().fit(ret_sel.values).covariance_

        # Approximation ERC = inverse de la volatilité
        ivar = 1 / np.sqrt(np.diag(cov))
        raw_w = ivar / np.sum(ivar)

        if portfolio_type == 'short_only':
            raw_w = -raw_w

        if portfolio_type == 'long_short':
            raw_w = np.sign(raw_w) * np.abs(raw_w) / 2

        if portfolio_type == 'q5_q1':
            top_assets = df_subset[df_subset['quantile'] == df_subset['quantile'].max()][self.asset_col]
            bot_assets = df_subset[df_subset['quantile'] == df_subset['quantile'].min()][self.asset_col]
            mask = [a in top_assets.values for a in ret_sel.columns]
            w_top = raw_w[mask]
            w_bot = raw_w[[not m for m in mask]]
            w = np.zeros_like(raw_w)
            w[mask] = w_top / np.sum(np.abs(w_top)) * 0.5
            w[[not m for m in mask]] = -w_bot / np.sum(np.abs(w_bot)) * 0.5
            raw_w = w

        w_series = pd.Series(raw_w, index=ret_sel.columns)
        return df_subset[self.asset_col].map(w_series).fillna(0.0)


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
