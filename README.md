        def _align_returns(self, df_weights: pd.DataFrame) -> pd.Series:
        """
        Align returns at time t with weights from the previous rebalance (t-1), then compute portfolio return series.

        For each date t (period-end):
        - Use weight_{t-1} (weights computed at previous rebalance)
        - Multiply by asset returns over (t-1, t]
        - Sum across assets to get portfolio return for the period ending at t

        Returns a pd.Series indexed by the date t, where each value is the return
        earned over the previous period, i.e. weight_{t-1} * return_{t}.

        Note: benchmark returns must use the same period convention: return at t is
        the asset/index return from t-1 to t.
        """
        df = df_weights.sort_values([self.asset_col, self.date_col]).copy()
        # shift weights so that prev_w at row t is the weight from the previous rebalance
        df['prev_w'] = df.groupby(self.asset_col)['weight'].shift(1)
        # drop first date per asset where no previous weight
        df = df.dropna(subset=['prev_w', self.ret_col])
        # multiply previous weight by the return over (t-1, t]
        df['ret'] = df['prev_w'] * df[self.ret_col]
        # sum across all assets for each period-end date t
        return df.groupby(self.date_col)['ret'].sum().sort_index()


class Backtester:
    def __init__(
        self,
        df: pd.DataFrame,
        date_col: str = 'date',
        asset_col: str = 'sedolcd',
        ret_col: str = 'return',
        vs: str = 'universe',
        benchmark_ret: Optional[pd.Series] = None,
        cost_model: Optional[Dict] = None,
        currency: Optional[str] = None,          # <- new!
        mscicol: str = 'MSCI-WRLD',              # <- name of the MSCI weight column
        min_mscicol_weight: float = 0.0          # <- threshold
    ):
        # 1) copy & store parameters
        self.df = df.copy()
        self.date_col = date_col
        self.asset_col = asset_col
        self.ret_col = ret_col
        self.vs = vs
        self.benchmark_ret = benchmark_ret
        self.cost_model = cost_model or {}

        # 2) universe filtering by currency + MSCI weight
        if currency is not None:
            # keep only rows in that currency
            self.df = self.df[self.df['instrmtccy'] == currency]
        # keep only those tickers whose MSCI weight > min_mscicol_weight
        self.df = self.df[self.df[mscicol] > min_mscicol_weight]

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
        univ_ret = self.df.groupby(self.date_col)[self.ret_col].mean().sort_index()
        univ_cum = (1 + univ_ret).cumprod()
        return univ_ret, univ_cum

    def _compute_metrics(
        self,
        strat: pd.Series,
        ref: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """
        Compute full suite of metrics for strategy returns (strat) against reference (ref).
        If ref is provided, also computes alpha, beta, TE, IR, Treynor, excess return metrics.

        Always computes:
        - TotalReturn, CAGR, Sharpe, Sortino, Vol, MaxDD, HitRate, Skew, Kurtosis
        - Turnover if present in strat.index name 'turnover'

        When ref is not None:
        - ExcessReturn, ExcessCAGR, alpha, beta, TrackingError, InformationRatio, TreynorRatio
        """
        metrics = {}
        # core strategy metrics
        metrics.update({
            'TotalReturn': get_total_return(strat),
            'CAGR': get_cagr(strat),
            'Sharpe': get_sharpe_ratio(strat),
            'Sortino': get_sortino_ratio(strat),
            'Vol': strat.std() * np.sqrt(12),
            'MaxDD': get_max_drawdown(strat),
            'HitRate': get_hit_rate(strat),
            'Skew': get_skew(strat),
            'Kurtosis': get_kurtosis(strat)
        })
        # reference and excess metrics
        if ref is not None:
            common = strat.index.intersection(ref.index)
            s = strat.loc[common]
            r = ref.loc[common]
            excess = s - r
            metrics.update({
                'ExcessReturn': get_total_return(excess),
                'ExcessCAGR': get_cagr(excess),
                'alpha': get_alpha(s, r),
                'beta': get_beta(s, r),
                'TE': get_tracking_error(s, r),
                'IR': get_information_ratio(s, r),
                'Treynor': get_treynor_ratio(s, r)
            })
        return metrics

    def run(
        self,
        signals: List[str],
        portfolio_type: str = 'long_only'
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, pd.Series]]]:
        reports = []
        series_dict = {}

        # prepare reference returns
        if self.vs == 'benchmark':
            if self.benchmark_ret is None:
                raise ValueError("benchmark_ret required when vs='benchmark'.")
            vs_ret = self.benchmark_ret.sort_index()
            vs_cum = (1 + vs_ret).cumprod()
            series_dict['benchmark'] = {'gross': vs_ret, 'cum': vs_cum}
        else:
            univ_ret, univ_cum = self._compute_universe_returns()
            vs_ret, vs_cum = univ_ret, univ_cum
            series_dict['universe'] = {'gross': univ_ret, 'cum': univ_cum}

        for sig in signals:
            pb = PortfolioBuilder(self.df, self.date_col, self.asset_col)
            df_w = pb.build_portfolio(sig, portfolio_type, 'equal')

            gross = self._align_returns(df_w)
            net = self._apply_transaction_costs(gross, self._compute_turnover(df_w))

            # store series
            series_dict[sig] = {
                'gross': gross,
                'gross_cum': (1 + gross).cumprod(),
                'net': net,
                'net_cum': (1 + net).cumprod(),
                'excess': (gross - vs_ret).reindex(gross.index),
                'excess_cum': (1 + gross - vs_ret).cumprod().reindex(gross.index)
            }

            # compute metrics for gross and net
            g_met = self._compute_metrics(gross, vs_ret)
            n_met = self._compute_metrics(net, vs_ret)

            report = {
                'signal': sig,
                'portfolio_type': portfolio_type,
                **{f'g_{k}': v for k, v in g_met.items()},
                **{f'n_{k}': v for k, v in n_met.items()}
            }
            reports.append(report)

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
