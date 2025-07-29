import pandas as pd
import numpy as np
from typing import Literal

# Définition des types pour plus de clarté
PortfolioType = Literal['long_only', 'short_only', 'long_short', 'q5_q1']
WeightType = Literal['equal', 'signal']

class PortfolioBuilder:
    """
    Crée des portefeuilles cross-sectionnels basés sur une colonne signal.
    Supporte plusieurs types de portefeuilles et méthodes de pondération.
    """
    def __init__(self, data: pd.DataFrame, date_col: str = 'date', asset_col: str = 'asset'):
        self.data = data.copy()
        self.date_col = date_col
        self.asset_col = asset_col

    def build_portfolio(self,
                        signal_col: str,
                        portfolio_type: PortfolioType,
                        weight_type: WeightType) -> pd.DataFrame:
        """
        Construit les poids de manière cross-sectionnelle par date.
        Retourne un DataFrame avec les colonnes [date, asset, weight].
        """
        df = self.data[[self.date_col, self.asset_col, signal_col]].dropna()
        weights = (
            df.groupby(self.date_col)
              .apply(lambda g: self._build_on_group(g, signal_col, portfolio_type, weight_type))
              .reset_index(drop=True)
        )
        return weights

    def _build_on_group(self,
                        group: pd.DataFrame,
                        signal_col: str,
                        portfolio_type: PortfolioType,
                        weight_type: WeightType) -> pd.DataFrame:
        # Sélection en fonction du type de portefeuille
        if portfolio_type == 'long_only':
            sel = self._select_long(group, signal_col)
        elif portfolio_type == 'short_only':
            sel = self._select_short(group, signal_col)
        elif portfolio_type == 'long_short':
            sel = self._select_long_short(group, signal_col)
        elif portfolio_type == 'q5_q1':
            sel = self._select_q5_q1(group, signal_col)
        else:
            raise ValueError(f"Unknown portfolio_type: {portfolio_type}")
        # Attribution des poids
        return self.assign_weights(sel, signal_col, portfolio_type, weight_type)

    def _select_long(self, group: pd.DataFrame, signal_col: str) -> pd.DataFrame:
        return group[group[signal_col] > 0]

    def _select_short(self, group: pd.DataFrame, signal_col: str) -> pd.DataFrame:
        return group[group[signal_col] < 0]

    def _select_long_short(self, group: pd.DataFrame, signal_col: str) -> pd.DataFrame:
        return pd.concat([
            self._select_long(group, signal_col),
            self._select_short(group, signal_col)
        ])

    def _select_q5_q1(self, group: pd.DataFrame, signal_col: str) -> pd.DataFrame:
        g = group.copy()
        g['quantile'] = pd.qcut(g[signal_col], 5, labels=False) + 1
        top = g[g['quantile'] == 5]
        bottom = g[g['quantile'] == 1]
        return pd.concat([top, bottom])

    def assign_weights(self,
                       selection: pd.DataFrame,
                       signal_col: str,
                       portfolio_type: PortfolioType,
                       weight_type: WeightType) -> pd.DataFrame:
        """
        Attribue les poids selon weight_type.
        Retourne un DataFrame [date, asset, weight].
        """
        if weight_type == 'equal':
            w = self._equal_weights(selection, signal_col, portfolio_type)
        elif weight_type == 'signal':
            w = self._signal_weights(selection, signal_col, portfolio_type)
        else:
            raise ValueError(f"Unknown weight_type: {weight_type}")
        return selection[[self.date_col, self.asset_col]].assign(weight=w.values)

    def _equal_weights(self, selection: pd.DataFrame, signal_col: str, portfolio_type: PortfolioType) -> pd.Series:
        n = len(selection)
        if n == 0:
            return pd.Series([], dtype=float)
        # Poids égaux selon le type
        if portfolio_type == 'long_only':
            return pd.Series(1 / n, index=selection.index)
        if portfolio_type == 'short_only':
            return pd.Series(-1 / n, index=selection.index)
        if portfolio_type == 'long_short':
            longs = selection[selection[signal_col] > 0]
            shorts = selection[selection[signal_col] < 0]
            w = pd.Series(0, index=selection.index, dtype=float)
            if len(longs) > 0:
                w.loc[longs.index] = 0.5 / len(longs)
            if len(shorts) > 0:
                w.loc[shorts.index] = -0.5 / len(shorts)
            return w
        if portfolio_type == 'q5_q1':
            longs = selection[selection['quantile'] == 5]
            shorts = selection[selection['quantile'] == 1]
            w = pd.Series(0, index=selection.index, dtype=float)
            if len(longs) > 0:
                w.loc[longs.index] = 0.5 / len(longs)
            if len(shorts) > 0:
                w.loc[shorts.index] = -0.5 / len(shorts)
            return w
        raise ValueError(f"Unsupported portfolio_type for equal weights: {portfolio_type}")

    def _signal_weights(self, selection: pd.DataFrame, signal_col: str, portfolio_type: PortfolioType) -> pd.Series:
        s = selection[signal_col]
        # Poids proportionnels à l'intensité du signal
        if portfolio_type == 'long_only':
            pos = s.clip(lower=0)
            return pos / pos.sum() if pos.sum() != 0 else pd.Series(0, index=selection.index)
        if portfolio_type == 'short_only':
            neg = s.clip(upper=0).abs()
            return -neg / neg.sum() if neg.sum() != 0 else pd.Series(0, index=selection.index)
        if portfolio_type == 'long_short':
            pos = s.clip(lower=0)
            neg = s.clip(upper=0).abs()
            w = pd.Series(0, index=selection.index, dtype=float)
            if pos.sum() != 0:
                w.loc[pos.index] = pos / pos.sum() * 0.5
            if neg.sum() != 0:
                w.loc[neg.index] = -neg / neg.sum() * 0.5
            return w
        if portfolio_type == 'q5_q1':
            top = selection[selection['quantile'] == 5]
            bot = selection[selection['quantile'] == 1]
            w = pd.Series(0, index=selection.index, dtype=float)
            if not top.empty:
                top_s = top[signal_col]
                w.loc[top.index] = top_s / top_s.sum() * 0.5
            if not bot.empty:
                bot_s = bot[signal_col].abs()
                w.loc[bot.index] = -bot_s / bot_s.sum() * 0.5
            return w
        raise ValueError(f"Unsupported portfolio_type for signal weights: {portfolio_type}")

# Exemple d'utilisation :
# builder = PortfolioBuilder(data)
# df_weights = builder.build_portfolio(signal_col='alpha', portfolio_type='long_short', weight_type='signal')


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
