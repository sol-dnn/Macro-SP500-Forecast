# backtester.py
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from portfolio import PortfolioBuilder
from utils import (
    get_sharpe_ratio,
    get_sortino_ratio,
    get_max_drawdown,
    get_hit_rate,
    get_skew,
    get_kurtosis,
    get_total_return,
    get_cagr
)

class Backtester:
    """
    Runs portfolio strategies and computes P&L and key metrics.

    Parameters:
    - df: DataFrame with at least [date, asset, <signals...>, return_column] (e.g. close-to-close or total return)
    - date_col: name of date column (e.g. period end, used for rebalance)
    - asset_col: name of asset identifier (e.g. sedolcd)
    - ret_col: name of the return column to use for PnL (e.g. 'closereturn', 'totalreturn', or 'cashreturn')
    - cost_model: dict, supports:
    - asset_col: name of asset identifier
    - ret_col: name of return column (e.g. excess_return)
    - cost_model: dict, supports:
        - {'type':'bps', 'bps':10}  # transaction cost in basis points per unit turnover
    """
    def __init__(
        self,
        df: pd.DataFrame,
        date_col: str = 'date',
        asset_col: str = 'sedolcd',
        ret_col: str = 'return',  # name of the return column to use (e.g. closereturn, totalreturn)
        cost_model: Optional[Dict] = None
    ):
        self,
        df: pd.DataFrame,
        date_col: str = 'date',
        asset_col: str = 'sedolcd',
        ret_col: str = 'excess_return',
        cost_model: Optional[Dict] = None
    ):
        self.df = df.copy()
        self.date_col = date_col
        self.asset_col = asset_col
        self.ret_col = ret_col
        self.cost_model = cost_model or {}

    def _align_returns(self, df_weights: pd.DataFrame) -> pd.Series:
        # align t->t+1 returns and compute P&L
        df = df_weights.sort_values([self.asset_col, self.date_col]).copy()
        df['next_ret'] = df.groupby(self.asset_col)[self.ret_col].shift(-1)
        df = df.dropna(subset=['next_ret'])
        df['pnl'] = df['weight'] * df['next_ret']
        return df.groupby(self.date_col)['pnl'].sum().sort_index()

    def _compute_turnover(self, df_weights: pd.DataFrame) -> pd.Series:
        # turnover = 0.5 * sum(|w_t - w_{t-1}|)
        w = df_weights.pivot_table(
            index=self.date_col,
            columns=self.asset_col,
            values='weight',
            aggfunc='last'
        ).fillna(0).sort_index()
        return w.diff().abs().sum(axis=1) * 0.5

    def _apply_transaction_costs(self, pnl: pd.Series, turnover: pd.Series) -> pd.Series:
        # apply cost_model
        if self.cost_model.get('type') == 'bps':
            bps = self.cost_model.get('bps', 0) / 10000
            return pnl - turnover * bps
        return pnl

    def _compute_metrics(self, returns: pd.Series) -> Dict[str, float]:
        # calls individual utils
        return {
            'TotalReturn': get_total_return(returns),
            'CAGR': get_cagr(returns),
            'Sharpe': get_sharpe_ratio(returns),
            'Sortino': get_sortino_ratio(returns),
            'Vol': returns.std() * np.sqrt(12),
            'MaxDD': get_max_drawdown(returns),
            'HitRate': get_hit_rate(returns),
            'Skew': get_skew(returns),
            'Kurtosis': get_kurtosis(returns)
        }

    def run(
        self,
        signals: List[str],
        portfolio_type: str = 'long_only'
    ) -> pd.DataFrame:
        """
        For each signal and the given portfolio_type, build weights & compute performance.
        Returns a DataFrame report indexed by (portfolio_type, signal) with strategy metrics.
        """
        reports = []
        for sig in signals:
            # build weights
            pb = PortfolioBuilder(self.df, self.date_col, self.asset_col)
            df_w = pb.build_portfolio(sig, portfolio_type, 'equal')
            # compute gross PnL series
            gross_pnl = self._align_returns(df_w)
            # turnover
            turnover = self._compute_turnover(df_w)
            avg_to = turnover.mean()
            # net PnL after costs
            net_pnl = self._apply_transaction_costs(gross_pnl, turnover)
            # metrics
            gross_metrics = self._compute_metrics(gross_pnl)
            net_metrics = self._compute_metrics(net_pnl)
            report = {
                'signal': sig,
                'portfolio_type': portfolio_type,
                'turnover': avg_to,
                **{f'g_{k}': v for k, v in gross_metrics.items()},
                **{f'n_{k}': v for k, v in net_metrics.items()}
            }
            reports.append(report)
        return pd.DataFrame(reports).set_index(['portfolio_type', 'signal'])


# utils.py
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis


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
