def run_quintile_performance(self, signals: List[str], metric_key: str = "SR") -> pd.DataFrame:
    """
    Pour chaque signal:
      - calcule la métrique (ex: Sharpe ratio) par quintile Q1..Q5 (long-only, équiweight)
      - ajuste une régression linéaire métrique ~ quintile
      - renvoie slope, R² ajusté, Spearman rho, alignement
    """
    reports = []

    # Référence
    if getattr(self, "vs", None) == "benchmark":
        if getattr(self, "benchmark_ret", None) is None:
            raise ValueError("benchmark_ret required when vs='benchmark'.")
        vs_ret = self.benchmark_ret.sort_index()
    else:
        _, _ = self._compute_universe_returns()
        msci_ret, _ = self._compute_msci_universe_returns()
        vs_ret = msci_ret

    for sig in signals:
        eligible = self.master_df[self.master_df[self.mscicol] > 0]
        df_sig = eligible.drop(columns=[self.mscicol]).merge(
            self.df_, on=[self.date_col, self.asset_col], how="left"
        )

        pb = PortfolioBuilder(
            data=df_sig,
            date_col=self.date_col,
            asset_col=self.asset_col,
            returns_history=self.returns_history,
        )
        df_w_sub = pb.build_quintile_portfolios(signal_col=sig)

        # métrique par quintile
        q_metrics = {}
        for q in range(1, 6):
            df_q = df_w_sub[df_w_sub["quantile"] == q]
            df_w = (
                self.master_df.merge(
                    df_q[[self.date_col, self.asset_col, "weight"]],
                    on=[self.date_col, self.asset_col],
                    how="left",
                )
                .fillna({"weight": 0.0})
            )
            gross = self._align_returns(df_w)
            g_met = self._compute_metrics(gross, vs_ret)
            q_metrics[f"{metric_key}_Q{q}"] = g_met[metric_key]

        # régression linéaire
        x = np.arange(1, 6, dtype=float)
        y = np.array([q_metrics[f"{metric_key}_Q{i}"] for i in range(1, 6)], dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        y_hat = intercept + slope * x
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        n, p = 5, 1
        r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 and np.isfinite(r2) else np.nan

        # alignement + Spearman
        alignment = int(np.sign(slope))
        spearman_rho = pd.Series(x).corr(pd.Series(y), method="spearman")

        reports.append({"signal": sig, **q_metrics, "slope": slope, "r2_adj": r2_adj,
                        "spearman_rho": spearman_rho, "alignment": alignment})

    return pd.DataFrame(reports).set_index("signal")


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
