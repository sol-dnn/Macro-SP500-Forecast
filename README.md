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
    - currency: optional currency code to filter by (e.g. 'USD', 'EUR', 'JPY')
    - mscicol: name of column containing MSCI World weight
    """
    def __init__(
        self,
        df: pd.DataFrame,
        date_col: str = 'date',
        asset_col: str = 'sedolcd',
        ret_col: str = 'return',
        vs: str = 'universe',
        benchmark_ret: Optional[pd.Series] = None,
        currency: Optional[str] = None,
        mscicol: str = 'MSCI-WRLD'
    ):
        # copy and basic params
        self.df = df.copy()
        self.date_col = date_col
        self.asset_col = asset_col
        self.ret_col = ret_col
        self.vs = vs
        self.benchmark_ret = benchmark_ret

        # filter by currency if given
        if currency is not None:
            self.df = self.df[self.df['instrmtccy'] == currency]
        # keep only tickers with positive MSCI weight at each date
        self.df = self.df[self.df[mscicol] > 0]

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
        df['prev_w'] = df.groupby(self.asset_col)['weight'].shift(1)
        df = df.dropna(subset=['prev_w', self.ret_col])
        df['ret'] = df['prev_w'] * df[self.ret_col]
        return df.groupby(self.date_col)['ret'].sum().sort_index()

    def _compute_turnover(self, df_weights: pd.DataFrame) -> pd.Series:
        """
        Compute turnover = 0.5 * sum(|w_t - w_{t-1}|) per period.
        Returns a pd.Series indexed by date.
        """
        w = df_weights.pivot_table(
            index=self.date_col,
            columns=self.asset_col,
            values='weight',
            aggfunc='last'
        ).fillna(0).sort_index()
        return w.diff().abs().sum(axis=1) * 0.5

    def _compute_universe_returns(self) -> Tuple[pd.Series, pd.Series]:
        """Compute equal-weight universe returns and cumulative factor."""
        univ_ret = self.df.groupby(self.date_col)[self.ret_col].mean().sort_index()
        univ_cum = (1 + univ_ret).cumprod()
        return univ_ret, univ_cum

    def _compute_metrics(
        self,
        strat: pd.Series,
        ref: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """Compute strategy metrics; includes ref-based metrics if ref provided."""
        metrics = {
            'TotalReturn': get_total_return(strat),
            'CAGR': get_cagr(strat),
            'Sharpe': get_sharpe_ratio(strat),
            'Sortino': get_sortino_ratio(strat),
            'Vol': strat.std() * np.sqrt(12),
            'MaxDD': get_max_drawdown(strat),
            'HitRate': get_hit_rate(strat),
            'Skew': get_skew(strat),
            'Kurtosis': get_kurtosis(strat)
        }
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
    - currency: optional currency code to filter by (e.g. 'USD', 'EUR', 'JPY')
    - mscicol: name of column containing MSCI World weight
    """
    def __init__(
        self,
        df: pd.DataFrame,
        date_col: str = 'date',
        asset_col: str = 'sedolcd',
        ret_col: str = 'return',
        vs: str = 'universe',
        benchmark_ret: Optional[pd.Series] = None,
        currency: Optional[str] = None,
        mscicol: str = 'MSCI-WRLD'
    ):
        # copy and basic params
        self.df = df.copy()
        self.date_col = date_col
        self.asset_col = asset_col
        self.ret_col = ret_col
        self.vs = vs
        self.benchmark_ret = benchmark_ret

                # filter by currency if given
        if currency is not None:
            self.df = self.df[self.df['instrmtccy'] == currency]
        # keep master list of assets for reindexing weights (currency-filtered)
        self.master_df = self.df[[self.date_col, self.asset_col, mscicol]].copy()  # retain mscicol for later
        # store MSCI weight column name
        self.mscicol = mscicol

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
        df['prev_w'] = df.groupby(self.asset_col)['weight'].shift(1)
        df = df.dropna(subset=['prev_w', self.ret_col])
        df['ret'] = df['prev_w'] * df[self.ret_col]
        return df.groupby(self.date_col)['ret'].sum().sort_index()

    def _compute_turnover(self, df_weights: pd.DataFrame) -> pd.Series:
        """
        Compute turnover = 0.5 * sum(|w_t - w_{t-1}|) per period.
        Returns a pd.Series indexed by date.
        """
        w = df_weights.pivot_table(
            index=self.date_col,
            columns=self.asset_col,
            values='weight',
            aggfunc='last'
        ).fillna(0).sort_index()
        return w.diff().abs().sum(axis=1) * 0.5

    def _compute_universe_returns(self) -> Tuple[pd.Series, pd.Series]:
        """Compute equal-weight universe returns and cumulative factor."""
        # equal-weight returns
        univ_ret = self.df.groupby(self.date_col)[self.ret_col].mean().sort_index()
        univ_cum = (1 + univ_ret).cumprod()
        return univ_ret, univ_cum

    def _compute_msci_universe_returns(self) -> Tuple[pd.Series, pd.Series]:
        """Compute MSCI-weighted universe returns and cumulative factor."""
        # pivot returns
        ret_wide = self.df.pivot(index=self.date_col,
                                  columns=self.asset_col,
                                  values=self.ret_col).fillna(0)
        # pivot MSCI weights from master_df
        msci_wide = self.master_df.pivot(index=self.date_col,
                                         columns=self.asset_col,
                                         values=self.mscicol).fillna(0)
        # normalize MSCI weights per date
        norm_w = msci_wide.div(msci_wide.sum(axis=1), axis=0)
        # weighted returns
        msci_ret = (norm_w * ret_wide).sum(axis=1).sort_index()
        msci_cum = (1 + msci_ret).cumprod()
        return msci_ret, msci_cum

    def _compute_metrics(
        self,
        strat: pd.Series,
        ref: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """Compute strategy metrics; includes ref-based metrics if ref provided."""
        metrics = {
            'TotalReturn': get_total_return(strat),
            'CAGR': get_cagr(strat),
            'Sharpe': get_sharpe_ratio(strat),
            'Sortino': get_sortino_ratio(strat),
            'Vol': strat.std() * np.sqrt(12),
            'MaxDD': get_max_drawdown(strat),
            'HitRate': get_hit_rate(strat),
            'Skew': get_skew(strat),
            'Kurtosis': get_kurtosis(strat)
        }
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
        """
        Run backtest for each signal and portfolio type.

        Returns:
        - report_df: DataFrame indexed by (portfolio_type, signal) with gross metrics + turnover.
        - series_dict: dict mapping 'universe_equal', 'universe_msci', and each signal to return series.
        """
        reports = []
        series_dict: Dict[str, Dict[str, pd.Series]] = {}

        # 1) Prepare reference returns
        if self.vs == 'benchmark':
            if self.benchmark_ret is None:
                raise ValueError("benchmark_ret required when vs='benchmark'.")
            # Benchmark series provided externally
            vs_ret = self.benchmark_ret.sort_index()
            vs_cum = (1 + vs_ret).cumprod()
            series_dict['benchmark'] = {'gross': vs_ret, 'cum': vs_cum}
        else:
            # Equal-weight universe
            univ_ret, univ_cum = self._compute_universe_returns()
            # MSCI-weighted universe
            msci_ret, msci_cum = self._compute_msci_universe_returns()
            series_dict['universe_equal'] = {'gross': univ_ret, 'cum': univ_cum}
            series_dict['universe_msci']  = {'gross': msci_ret, 'cum': msci_cum}
            # Use MSCI-weighted as default reference
            vs_ret, vs_cum = msci_ret, msci_cum

        # 2) Loop over each signal
        for sig in signals:
            # a) Build weights on MSCI-eligible universe
            df_signalable = self.master_df[self.master_df[self.mscicol] > 0]
            df_sig = df_signalable.drop(columns=[self.mscicol]).merge(
                self.df, on=[self.date_col, self.asset_col], how='left')
            pb = PortfolioBuilder(df_sig, self.date_col, self.asset_col)
            df_w_sub = pb.build_portfolio(sig, portfolio_type, 'equal')

            # b) Reindex weights onto full currency universe, fill missing as 0
            full = self.master_df[[self.date_col, self.asset_col]].drop_duplicates()
            df_w = (
                full
                .merge(
                    df_w_sub[[self.date_col, self.asset_col, 'weight']],
                    on=[self.date_col, self.asset_col], how='left'
                )
                .fillna({'weight': 0.0})
            )

            # c) Compute strategy return series and turnover
            gross = self._align_returns(df_w)
            turnover = self._compute_turnover(df_w)

            # d) Store series for plotting
            series_dict[sig] = {
                'gross': gross,
                'cum': (1 + gross).cumprod(),
                'turnover': turnover
            }

            # e) Compute metrics against reference
            g_met = self._compute_metrics(gross, vs_ret)
            # annualized turnover: avg monthly turnover * 12
            g_met['turnover_annualized'] = turnover.mean() * 12

            # f) Compile report
            report = {
                'signal': sig,
                'portfolio_type': portfolio_type,
                **g_met
            }
            reports.append(report)

        report_df = pd.DataFrame(reports).set_index(['portfolio_type', 'signal'])
        return report_df, series_dict

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
