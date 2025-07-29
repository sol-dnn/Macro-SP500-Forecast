import pandas as pd
import numpy as np
from typing import Literal

# Define types for clarity
PortfolioType = Literal['long_only', 'short_only', 'long_short', 'q5_q1']
WeightType = Literal['equal', 'signal']

class PortfolioBuilder:
    """
    Builds cross-sectional portfolios based on a signal column.
    Returns all original columns plus 'weight' (and 'quantile' for quantile portfolios).
    Supports multiple portfolio strategies and weighting schemes.
    """
    def __init__(self,
                 data: pd.DataFrame,
                 date_col: str = 'date',
                 asset_col: str = 'sedolcd'):
        """
        Parameters:
        - data: DataFrame with at least date, asset identifier, signal, and any other columns (e.g. returns, volatility).
        - date_col: name of the date column.
        - asset_col: name of the asset identifier column (e.g. ticker or sedol code).
        """
        self.data = data.copy()
        self.date_col = date_col
        self.asset_col = asset_col

    def build_portfolio(self,
                        signal_col: str,
                        portfolio_type: PortfolioType,
                        weight_type: WeightType) -> pd.DataFrame:
        """
        Build a cross-sectional portfolio by date using a single signal column.

        Returns a DataFrame containing all original columns plus a new 'weight' column.
        """
        # Drop rows where signal is missing
        df = self.data.dropna(subset=[signal_col]).copy()

        # Group by date and process each group
        def process_group(group: pd.DataFrame) -> pd.DataFrame:
            # 1) select assets based on portfolio type
            if portfolio_type == 'long_only':
                sel = group[group[signal_col] > 0].copy()
            elif portfolio_type == 'short_only':
                sel = group[group[signal_col] < 0].copy()
            elif portfolio_type == 'long_short':
                longs = group[group[signal_col] > 0]
                shorts = group[group[signal_col] < 0]
                sel = pd.concat([longs, shorts]).copy()
            elif portfolio_type == 'q5_q1':
                tmp = group.copy()
                tmp['quantile'] = pd.qcut(tmp[signal_col], 5, labels=False) + 1
                top = tmp[tmp['quantile'] == 5]
                bottom = tmp[tmp['quantile'] == 1]
                sel = pd.concat([top, bottom]).copy()
            else:
                raise ValueError(f"Unknown portfolio_type: {portfolio_type}")

            # 2) compute weights
            weights = self._compute_weights(sel, signal_col, weight_type, portfolio_type)
            sel['weight'] = weights.values
            return sel

        result = (
            df.groupby(self.date_col)
              .apply(process_group)
              .reset_index(drop=True)
        )
        return result

    def build_quantile_portfolios(self,
                                  signal_col: str,
                                  n_quantiles: int = 5,
                                  weight_type: WeightType = 'equal') -> pd.DataFrame:
        """
        Build portfolios for each quantile of the signal by date.

        Returns a DataFrame containing all original columns plus 'quantile' and 'weight'.
        """
        df = self.data.dropna(subset=[signal_col]).copy()
        df['quantile'] = (
            df.groupby(self.date_col)[signal_col]
              .transform(lambda x: pd.qcut(x, n_quantiles, labels=False) + 1)
        )

        def process_quantiles(group: pd.DataFrame) -> pd.DataFrame:
            parts = []
            for q in range(1, n_quantiles + 1):
                sub = group[group['quantile'] == q].copy()
                if sub.empty:
                    continue
                w = self._compute_weights(sub, signal_col, weight_type, 'q5_q1')
                sub['weight'] = w.values
                parts.append(sub)
            return pd.concat(parts) if parts else pd.DataFrame(columns=group.columns)

        result = (
            df.groupby(self.date_col)
              .apply(process_quantiles)
              .reset_index(drop=True)
        )
        return result

    def _compute_weights(self,
                         df_subset: pd.DataFrame,
                         signal_col: str,
                         weight_type: WeightType,
                         portfolio_type: PortfolioType) -> pd.Series:
        """
        Delegate to the appropriate weighting method.
        """
        if weight_type == 'equal':
            return self._equal_weights(df_subset, signal_col, portfolio_type)
        elif weight_type == 'signal':
            return self._signal_weights(df_subset, signal_col, portfolio_type)
        else:
            raise ValueError(f"Unknown weight_type: {weight_type}")

    def _equal_weights(self,
                       df_subset: pd.DataFrame,
                       signal_col: str,
                       portfolio_type: PortfolioType) -> pd.Series:
        """
        Assign equal weights:
        - long_only or short_only: weights sum to ±1
        - long_short or q5_q1: each side sums to ±0.5
        """
        n = len(df_subset)
        if n == 0:
            return pd.Series([], dtype=float)
        if portfolio_type == 'long_only':
            return pd.Series(1.0/n, index=df_subset.index)
        if portfolio_type == 'short_only':
            return pd.Series(-1.0/n, index=df_subset.index)
        # long_short or q5_q1: split ±0.5 per side
        longs = df_subset[df_subset[signal_col] > 0]
        shorts = df_subset[df_subset[signal_col] < 0]
        w = pd.Series(0.0, index=df_subset.index)
        if not longs.empty:
            w.loc[longs.index] = 0.5/len(longs)
        if not shorts.empty:
            w.loc[shorts.index] = -0.5/len(shorts)
        return w

    def _signal_weights(self,
                        df_subset: pd.DataFrame,
                        signal_col: str,
                        portfolio_type: PortfolioType) -> pd.Series:
        """
        Assign weights proportional to signal strength:
        - long_only: positive signals sum to 1
        - short_only: negative signals sum to -1
        - long_short: each side sums to ±0.5
        - q5_q1: top quantile sum to 0.5, bottom quantile sum to -0.5
        """
        s = df_subset[signal_col]
        if portfolio_type == 'long_only':
            pos = s.clip(lower=0)
            return pos/pos.sum() if pos.sum()>0 else pd.Series(0.0, index=df_subset.index)
        if portfolio_type == 'short_only':
            neg = s.clip(upper=0).abs()
            return -neg/neg.sum() if neg.sum()>0 else pd.Series(0.0, index=df_subset.index)
        if portfolio_type == 'long_short':
            pos = s.clip(lower=0)
            neg = s.clip(upper=0).abs()
            w = pd.Series(0.0, index=df_subset.index)
            if pos.sum()>0:
                w.loc[pos.index] = pos/pos.sum()*0.5
            if neg.sum()>0:
                w.loc[neg.index] = -neg/neg.sum()*0.5
            return w
        if portfolio_type == 'q5_q1':
            top = df_subset[df_subset['quantile']==max(df_subset['quantile'])][signal_col]
            bot = df_subset[df_subset['quantile']==min(df_subset['quantile'])][signal_col].abs()
            w = pd.Series(0.0, index=df_subset.index)
            if top.sum()>0:
                w.loc[top.index] = top/top.sum()*0.5
            if bot.sum()>0:
                w.loc[bot.index] = -bot/bot.sum()*0.5
            return w
        raise ValueError(f"Unsupported portfolio_type: {portfolio_type}")

# Example usage:
# builder = PortfolioBuilder(df, date_col='date', asset_col='sedolcd')
# df_weights = builder.build_portfolio('alpha', 'long_short', 'signal')
# df_quantiles = builder.build_quantile_portfolios('alpha', n_quantiles=5, weight_type='signal')


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
