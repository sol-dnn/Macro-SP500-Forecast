import numpy as np
import pandas as pd
from scipy import stats  # utilisé à la fin pour la 2e fonction

def cs_winsor_zscore_clip(
    df,
    signal_cols,
    *,
    date_col='date',
    by=None,                  # ex: 'GICS_sector_name' -> standardisation par (date, secteur)
    winsor_method='quantile', # 'quantile' ou 'iqr'
    p_low=0.01, p_high=0.99,  # pour 'quantile'
    iqr_k=3.0,                # pour 'iqr' -> médiane ± k*IQR
    clip_at=3.0,              # clip final sur le z-score
    suffix='_z',              # suffixe des nouvelles colonnes
    min_group=8               # taille mini pour calculer des stats (sinon NaN)
):
    """
    Pour chaque signal de `signal_cols` :
      (1) winsor CS par date (et optionnellement par 'by'),
      (2) z-score CS : (x - mu) / sigma,
      (3) clip final à [-clip_at, clip_at].

    Retourne: (df_new, new_cols)
    """
    d = df.copy()
    d = d.sort_values([date_col] + ([by] if by else []))
    group_keys = [date_col] + ([by] if by else [])
    g = d.groupby(group_keys, group_keys=False)

    new_cols = []

    for col in signal_cols:
        x = d[col].astype(float)

        # --- Winsor ---
        if winsor_method == 'quantile':
            q_low = g[col].transform(lambda s: s.quantile(p_low) if s.notna().sum()>=min_group else np.nan)
            q_high= g[col].transform(lambda s: s.quantile(p_high) if s.notna().sum()>=min_group else np.nan)
            x_w = x.where(x >= q_low, q_low)
            x_w = x_w.where(x_w <= q_high, q_high)
        elif winsor_method == 'iqr':
            q1 = g[col].transform(lambda s: s.quantile(0.25) if s.notna().sum()>=min_group else np.nan)
            q3 = g[col].transform(lambda s: s.quantile(0.75) if s.notna().sum()>=min_group else np.nan)
            iqr = q3 - q1
            med = g[col].transform(lambda s: s.median() if s.notna().sum()>=min_group else np.nan)
            lo = med - iqr_k*iqr
            hi = med + iqr_k*iqr
            x_w = x.where(x >= lo, lo)
            x_w = x_w.where(x_w <= hi, hi)
        else:
            raise ValueError("winsor_method must be 'quantile' or 'iqr'")

        # --- Z-score CS ---
        mu  = g.apply(lambda s: s[col].astype(float)).groupby(group_keys).transform('mean')
        # mu calculé ci-dessus sur la série brute; on peut aussi utiliser x_w pour être 100% cohérent:
        mu  = g[x_w.name].transform(lambda s: s.mean())
        # std sur le winsorisé
        sig = g[x_w.name].transform(lambda s: s.std(ddof=1))
        # éviter division par 0
        z = (x_w - mu) / sig.replace(0, np.nan)
        z = z.fillna(0.0)  # si std=0 dans un groupe minuscule -> 0

        # --- Clip final ---
        zc = z.clip(-clip_at, clip_at)
        out_col = f"{col}{suffix}"
        d[out_col] = zc
        new_cols.append(out_col)

    return d, new_cols



def cs_rank_normal_clip(
    df,
    signal_cols,
    *,
    date_col='date',
    by=None,             # ex: 'GICS_sector_name' pour within-sector
    clip_at=3.5,
    suffix='_rn',
    min_group=8
):
    """
    Pour chaque signal:
      (1) rangs CS par groupe -> quantiles u = (rank-0.5)/n
      (2) scores normaux z = Phi^{-1}(u)
      (3) clip à [-clip_at, clip_at]
    Retourne: (df_new, new_cols)
    """
    d = df.copy()
    d = d.sort_values([date_col] + ([by] if by else []))
    group_keys = [date_col] + ([by] if by else [])
    g = d.groupby(group_keys, group_keys=False)

    def _ranknorm(s: pd.Series):
        z = pd.Series(index=s.index, dtype=float)
        m = s.notna()
        n = m.sum()
        if n < min_group:
            z[m] = 0.0
            return z
        ranks = s[m].rank(method='average')  # 1..n
        u = (ranks - 0.5) / n                # (0,1)
        z[m] = stats.norm.ppf(u.clip(1e-6, 1-1e-6))
        z[~m] = np.nan
        return z

    new_cols = []
    for col in signal_cols:
        z = g[col].transform(_ranknorm)
        zr = z.clip(-clip_at, clip_at)
        out_col = f"{col}{suffix}"
        d[out_col] = zr
        new_cols.append(out_col)

    return d, new_cols
# z-score robuste aux outliers (winsor->z->clip) par (date, currency)
df_z, z_cols = cs_winsor_zscore_clip(
    df_factors, FACTS,
    by='instrmtccy',        # ou ['instrmtccy','GICS_sector_name']
    winsor_method='quantile', p_low=0.01, p_high=0.99, clip_at=3.0
)

# rank->normal par (date, currency)
df_rn, rn_cols = cs_rank_normal_clip(
    df_factors, FACTS,
    by='instrmtccy',        # ou ['instrmtccy','GICS_sector_name']
    clip_at=3.5
)


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
