def run_quintile_portfolio(
    self,
    signals: List[str],
    metric_key: str = "SR",
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Dict[str, pd.Series]]]]:
    """
    Run backtest par quintile (long-only, equiweight) pour chaque signal.
    Retourne:
      - report_df: index=signal, colonnes = {<metric>_Q1..Q5, slope, r2_adj, spearman_rho, alignment}
      - series_dict: dict[signal][f"Q{q}"] -> séries 'gross', 'cum', 'excess', 'excess_cum', 'turnover'
    """
    reports: List[Dict[str, Any]] = []
    series_dict: Dict[str, Dict[str, Dict[str, pd.Series]]] = {}

    # 1) Références (benchmark / univers égal / univers MSCI)
    if getattr(self, "benchmark_ret", None) is not None:
        bench_ret = self.benchmark_ret.sort_index()
        bench_cum = (1 + bench_ret).cumprod()
        series_dict["benchmark"] = {"gross": bench_ret, "cum": bench_cum}
    if getattr(self, "vs", None) == "benchmark":
        if getattr(self, "benchmark_ret", None) is None:
            raise ValueError("benchmark_ret required when vs='benchmark'.")
        vs_ret = bench_ret
    else:
        univ_ret, univ_cum = self._compute_universe_returns()
        msci_ret, msci_cum = self._compute_msci_universe_returns()
        series_dict["universe_equal"] = {"gross": univ_ret, "cum": univ_cum}
        series_dict["universe_msci"] = {"gross": msci_ret, "cum": msci_cum}
        # par défaut on évalue vs l'univers MSCI
        vs_ret = msci_ret

    # 2) Boucle sur chaque signal
    for sig in signals:
        # a) Univers MSCI éligible (fin de mois) + merge signal
        eligible = self.master_df[self.master_df[self.mscicol] > 0]
        df_sig = eligible.drop(columns=[self.mscicol]).merge(
            self.df_, on=[self.date_col, self.asset_col], how="left"
        )

        pb = PortfolioBuilder(
            data=df_sig,
            date_col=self.date_col,
            asset_col=self.asset_col,
            returns_history=self.returns_history,  # même param que chez toi
        )
        df_w_sub = pb.build_quintile_portfolios(signal_col=sig)

        # b) Métrique par quintile
        q_metrics = {}
        series_dict.setdefault(sig, {})

        for q in range(1, 6):
            df_q = df_w_sub[df_w_sub["quantile"] == q]

            # Reindex sur tout l'univers; poids manquants à 0
            df_w = (
                self.master_df.merge(
                    df_q[[self.date_col, self.asset_col, "weight"]],
                    on=[self.date_col, self.asset_col],
                    how="left",
                )
                .fillna({"weight": 0.0})
            )

            # c) Séries de perf et turnover
            gross = self._align_returns(df_w)
            turnover = self._compute_turnover(df_w)

            # d) Stockage pour plotting
            series_dict[sig][f"Q{q}"] = {
                "gross": gross,
                "cum": (1 + gross).cumprod(),
                "turnover": turnover,
                "excess": (gross - vs_ret).reindex(gross.index),
                "excess_cum": (1 + (gross - vs_ret)).reindex(gross.index).cumprod(),
            }

            # e) Métriques vs référence
            g_met = self._compute_metrics(gross, vs_ret)
            if metric_key not in g_met:
                raise KeyError(f"Métrique '{metric_key}' absente de _compute_metrics.")
            q_metrics[f"{metric_key}_Q{q}"] = g_met[metric_key]

        # f) Régression linéaire: métrique ~ quintile (1..5)
        x = np.arange(1, 6, dtype=float)
        y = np.array([q_metrics[f"{metric_key}_Q{i}"] for i in range(1, 6)], dtype=float)

        slope, intercept = np.polyfit(x, y, 1)
        y_hat = intercept + slope * x
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        n, p = 5, 1
        r2_adj = 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1) if n > p + 1 and np.isfinite(r2) else np.nan

        # g) Alignement (signe de la pente) + Spearman (monotonicité)
        alignment = int(np.sign(slope))  # +1 bon, -1 mauvais, 0 neutre
        spearman_rho = pd.Series(x).corr(pd.Series(y), method="spearman")

        reports.append(
            {
                "signal": sig,
                **q_metrics,
                "slope": slope,
                "r2_adj": r2_adj,
                "spearman_rho": spearman_rho,
                "alignment": alignment,
            }
        )

    report_df = pd.DataFrame(reports).set_index("signal")
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
