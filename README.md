import pandas as pd

# 1. Créez votre liste complète de dates mensuelles
all_dates = pd.date_range(start='2014-01-01', end='2024-12-31', freq='M')

# 2. Liste unique des tickers
all_tickers = df_clean['sedolcd'].unique()

# 3. Construisez un MultiIndex complet (date, ticker)
idx = pd.MultiIndex.from_product([all_dates, all_tickers],
                                 names=['date','sedolcd'])
master = pd.DataFrame(index=idx).reset_index()

# 4. Merge avec vos totalreturn
master = master.merge(df_clean[['date','sedolcd','totalreturn']],
                      on=['date','sedolcd'],
                      how='left')

# 5. Repérez les (date, ticker) manquants
missing = master[master['totalreturn'].isna()]

# 6. Comptez ou affichez
print("Nombre total d’observations manquantes :", len(missing))
print("Tickers les plus concernés :")
print(missing['sedolcd'].value_counts().head(10))
print("Périodes manquantes les plus fréquentes :")
print(missing['date'].value_counts().head(10))



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
    Runs portfolio strategies and computes P&L, key metrics, and return series.

    Parameters:
    - df: DataFrame with at least [date, asset, <signals...>, return_column]
    - date_col: name of date column (used for grouping)
    - asset_col: name of asset identifier
    - ret_col: name of the return column (e.g. 'closereturn', 'totalreturn')
    - cost_model: dict, supports:
        {'type':'bps', 'bps':10}  # transaction cost in basis points per unit turnover
    """
    def __init__(
        self,
        df: pd.DataFrame,
        date_col: str = 'date',
        asset_col: str = 'sedolcd',
        ret_col: str = 'return',
        cost_model: Optional[Dict] = None
    ):
        self.df = df.copy()
        self.date_col = date_col
        self.asset_col = asset_col
        self.ret_col = ret_col
        self.cost_model = cost_model or {}

    def _align_returns(self, df_weights: pd.DataFrame) -> pd.Series:
        """
        Align weights at t to returns at t+1 and compute portfolio return series.
        Returns a pd.Series indexed by date.
        """
        df = df_weights.sort_values([self.asset_col, self.date_col]).copy()
        df['next_ret'] = df.groupby(self.asset_col)[self.ret_col].shift(-1)
        df = df.dropna(subset=['next_ret'])
        df['pnl'] = df['weight'] * df['next_ret']
        return df.groupby(self.date_col)['pnl'].sum().sort_index()

    def _compute_turnover(self, df_weights: pd.DataFrame) -> pd.Series:
        """
        Compute turnover=0.5*sum(|w_t - w_{t-1}|) per period.
        Returns a pd.Series indexed by date.
        """
        w = df_weights.pivot_table(
            index=self.date_col,
            columns=self.asset_col,
            values='weight',
            aggfunc='last'
        ).fillna(0).sort_index()
        return w.diff().abs().sum(axis=1) * 0.5

    def _apply_transaction_costs(self, pnl: pd.Series, turnover: pd.Series) -> pd.Series:
        """
        Subtract transaction costs from the return series.
        cost_model={'type':'bps', 'bps':10} means 10 bps per unit turnover.
        """
        if self.cost_model.get('type') == 'bps':
            bps = self.cost_model.get('bps', 0) / 10000
            return pnl - turnover * bps
        return pnl

    def _compute_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """
        Compute strategy metrics on a return series:
        TotalReturn, CAGR, Sharpe, Sortino, Volatility,
        Max Drawdown, Hit Rate, Skewness, Kurtosis.
        """
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
    ) -> (pd.DataFrame, Dict[str, Dict[str, pd.Series]]):
        """
        Run backtest for each signal and portfolio type.

        Returns:
        - report_df: DataFrame with index (portfolio_type, signal) and metric columns.
        - series_dict: dict mapping signal -> {'gross': returns, 'gross_cum': cumreturns,
                                               'net': returns, 'net_cum': cumreturns}
        """
        reports = []
        series_dict = {}

        for sig in signals:
            # 1) build weights
            pb = PortfolioBuilder(self.df, self.date_col, self.asset_col)
            df_w = pb.build_portfolio(sig, portfolio_type, 'equal')
            # 2) gross return series
            gross_ret = self._align_returns(df_w)
            gross_cum = (1 + gross_ret).cumprod()
            # 3) turnover
            turnover = self._compute_turnover(df_w)
            avg_to = turnover.mean()
            # 4) net return series after costs
            net_ret = self._apply_transaction_costs(gross_ret, turnover)
            net_cum = (1 + net_ret).cumprod()
            # 5) compute metrics
            g_metrics = self._compute_metrics(gross_ret)
            n_metrics = self._compute_metrics(net_ret)
            report = {
                'signal': sig,
                'portfolio_type': portfolio_type,
                'turnover': avg_to,
                **{f'g_{k}': v for k, v in g_metrics.items()},
                **{f'n_{k}': v for k, v in n_metrics.items()}
            }
            reports.append(report)
            series_dict[sig] = {
                'gross': gross_ret,
                'gross_cum': gross_cum,
                'net': net_ret,
                'net_cum': net_cum
            }

        report_df = pd.DataFrame(reports).set_index(['portfolio_type', 'signal'])
        return report_df, series_dict


# utils.py
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis


def get_total_return(returns: pd.Series) -> float:
    """Total return over the period: cumprod(1+r) - 1."""
    cum = (1 + returns).cumprod()
    return cum.iloc[-1] - 1


def get_cagr(returns: pd.Series, periods: int = 12) -> float:
    """Compound Annual Growth Rate from monthly returns."""
    cum = (1 + returns).cumprod()
    n = len(returns.dropna())
    return cum.iloc[-1]**(periods/n) - 1 if n > 0 else np.nan


def get_sharpe_ratio(returns: pd.Series, periods: int = 12) -> float:
    """Annualized Sharpe ratio (assumes zero risk-free)."""
    r = returns.dropna()
    return (r.mean()/r.std()*np.sqrt(periods)) if r.std() > 0 else np.nan


def get_sortino_ratio(returns: pd.Series, periods: int = 12) -> float:
    """Annualized Sortino ratio (downside vol only)."""
    r = returns.dropna()
    downside = r[r < 0]
    down_std = downside.std()*np.sqrt(periods) if len(downside) > 0 else np.nan
    return (r.mean()/down_std) if down_std and down_std > 0 else np.nan


def get_max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough drawdown."""
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    return (cum/peak - 1).min()


def get_hit_rate(returns: pd.Series) -> float:
    """Proportion of positive return periods."""
    return (returns.dropna() > 0).mean()


def get_skew(returns: pd.Series) -> float:
    """Skewness of the return distribution."""
    return skew(returns.dropna())


def get_kurtosis(returns: pd.Series) -> float:
    """Excess kurtosis of the return distribution."""
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
