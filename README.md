# backtester.py
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from portfolio import PortfolioBuilder
from utils import (
    get_sharpe_ratio,
    get_sortino_ratio,
    get_max_drawdown,
    get_hit_rate,
    get_skew,
    get_kurtosis,
    get_total_return,
    get_cagr,
    get_alpha,
    get_beta,
    get_tracking_error,
    get_information_ratio,
    get_treynor_ratio
)

class Backtester:
    """
    Runs portfolio strategies and computes P&L, key metrics, and return series.

    Parameters:
    - df: DataFrame with at least [date, asset, <signals...>, return_column]
    - date_col: name of date column (used for grouping)
    - asset_col: name of asset identifier
    - ret_col: name of the return column (e.g. 'closereturn', 'totalreturn')
    - vs: 'universe' or 'benchmark' to compare strategy against
    - benchmark_ret: optional Series of benchmark returns, required if vs='benchmark'
    - cost_model: dict, supports:
        {'type':'bps', 'bps':10}  # transaction cost in basis points per unit turnover
    """
    def __init__(
        self,
        df: pd.DataFrame,
        date_col: str = 'date',
        asset_col: str = 'sedolcd',
        ret_col: str = 'return',
        vs: str = 'universe',
        benchmark_ret: Optional[pd.Series] = None,
        cost_model: Optional[Dict] = None
    ):
        self.df = df.copy()
        self.date_col = date_col
        self.asset_col = asset_col
        self.ret_col = ret_col
        self.vs = vs
        self.benchmark_ret = benchmark_ret
        self.cost_model = cost_model or {}

    def _align_returns(self, df_weights: pd.DataFrame) -> pd.Series:
        df = df_weights.sort_values([self.asset_col, self.date_col]).copy()
        df['next_ret'] = df.groupby(self.asset_col)[self.ret_col].shift(-1)
        df = df.dropna(subset=['next_ret'])
        df['ret'] = df['weight'] * df['next_ret']
        return df.groupby(self.date_col)['ret'].sum().sort_index()

    def _compute_turnover(self, df_weights: pd.DataFrame) -> pd.Series:
        w = df_weights.pivot_table(
            index=self.date_col,
            columns=self.asset_col,
            values='weight',
            aggfunc='last'
        ).fillna(0).sort_index()
        return w.diff().abs().sum(axis=1) * 0.5

    def _apply_transaction_costs(self, returns: pd.Series, turnover: pd.Series) -> pd.Series:
        if self.cost_model.get('type') == 'bps':
            bps = self.cost_model.get('bps', 0) / 10000
            return returns - turnover * bps
        return returns

    def _compute_universe_returns(self) -> Tuple[pd.Series, pd.Series]:
        """
        Compute equal-weight strategy returns for the investment universe.
        Returns a tuple (univ_ret, univ_cum).
        """
        univ_ret = self.df.groupby(self.date_col)[self.ret_col].mean().sort_index()
        univ_cum = (1 + univ_ret).cumprod()
        return univ_ret, univ_cum

    def run(
        self,
        signals: List[str],
        portfolio_type: str = 'long_only'
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, pd.Series]]]:
        reports = []
        series_dict = {}

        # compute benchmark or universe series
        if self.vs == 'benchmark':
            if self.benchmark_ret is None:
                raise ValueError("benchmark_ret required when vs='benchmark'.")
            vs_ret = self.benchmark_ret.sort_index()
        else:
            vs_ret, vs_cum = self._compute_universe_returns()
        if self.vs == 'universe':
            series_dict['universe'] = {'gross': vs_ret, 'cum': vs_cum}

        for sig in signals:
            pb = PortfolioBuilder(self.df, self.date_col, self.asset_col)
            df_w = pb.build_portfolio(sig, portfolio_type, 'equal')
            gross = self._align_returns(df_w)
            gross_cum = (1 + gross).cumprod()
            turnover = self._compute_turnover(df_w)
            avg_to = turnover.mean()
            net = self._apply_transaction_costs(gross, turnover)
            net_cum = (1 + net).cumprod()
            g_metrics = {
                **{'turnover': avg_to},
                **self._compute_metrics(gross)
            }
            n_metrics = self._compute_metrics(net)

            # alignment for comparison
            common_idx = gross.index.intersection(vs_ret.index)
            strat = gross.loc[common_idx]
            ref = vs_ret.loc[common_idx]
            excess = strat - ref
            excess_cum = (1 + excess).cumprod()

            # factor metrics
            alpha = get_alpha(strat, ref)
            beta = get_beta(strat, ref)
            te = get_tracking_error(strat, ref)
            ir = get_information_ratio(strat, ref)
            treynor = get_treynor_ratio(strat, ref)

            report = {
                'signal': sig,
                'portfolio_type': portfolio_type,
                **{f'g_{k}': v for k, v in g_metrics.items()},
                **{f'n_{k}': v for k, v in n_metrics.items()},
                'alpha': alpha,
                'beta': beta,
                'TE': te,
                'IR': ir,
                'Treynor': treynor
            }
            reports.append(report)
            series_dict[sig] = {
                'gross': gross,
                'gross_cum': gross_cum,
                'net': net,
                'net_cum': net_cum,
                'excess': excess,
                'excess_cum': excess_cum
            }

        report_df = pd.DataFrame(reports).set_index(['portfolio_type','signal'])
        return report_df, series_dict

# utils.py
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
