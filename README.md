from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd
from typing import List

class CrossSectionalPreprocessor(BaseEstimator, TransformerMixin):
    """
    Transformer for cross-sectional winsorization and imputation of features.

    Steps:
      1. Replace inf/-inf with NaN
      2. Winsorize numeric features per date
      3. Impute numeric by median per (date, sector)
      4. Impute numeric by median per date
      5. Impute categoricals by mode per date
      6. Fill remaining missing with a sentinel value

    Parameters:
        features: list of feature column names
        sector_col: sector grouping column name (default 'GICS_sector_name')
        cat_features: list of categorical feature names (default empty)
        lower_q, upper_q: quantile bounds for winsorization
        fill_value: value to impute remaining missing (default -11)
        verbose: if True, prints missing value report
    """
    def __init__(
        self,
        features: List[str],
        sector_col: str = 'GICS_sector_name',
        cat_features: List[str] = None,
        lower_q: float = 0.03,
        upper_q: float = 0.97,
        fill_value: float = -11,
        verbose: bool = False
    ):
        self.features = features
        self.sector_col = sector_col
        self.cat_features = cat_features or []
        self.lower_q = lower_q
        self.upper_q = upper_q
        self.fill_value = fill_value
        self.verbose = verbose

    def fit(self, X: pd.DataFrame, y=None):
        # No learning step
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        # 0) initial missing summary
        missing0 = df[self.features].isna().sum()

        # 0.b) replace infinite values
        df = df.replace([float('inf'), float('-inf')], pd.NA)
        missing_inf = df[self.features].isna().sum()

        # 1) winsorization per date
        for col in self.features:
            if col not in self.cat_features:
              if not col.startswith("ic_estimate_eps_num"):
                df[col] = df.groupby(level='date')[col].transform(
                    self._winsorize_group
                )
        missing1 = df[self.features].isna().sum()

        # 2) sector-level median imputation
        medians = (
            df.reset_index()
              .groupby(['date', self.sector_col])[self.features]
              .median()
        )
        df = df.apply(
            self._impute_sector(medians),
            axis=1
        )
        missing2 = df[self.features].isna().sum()

        # 3) date-level median imputation
        for col in self.features:
            if col not in self.cat_features:
                df[col] = df.groupby(level='date')[col].transform(
                    lambda grp: grp.fillna(grp.median())
                )
        missing3 = df[self.features].isna().sum()

        # 4) categorical mode imputation per date
        for col in self.cat_features:
            df[col] = df.groupby(level='date')[col].transform(
                self._fill_mode
            )
        missing4 = df[self.features].isna().sum()

        # 5) final fill
        df[self.features] = df[self.features].fillna(self.fill_value)
        missing5 = df[self.features].isna().sum()

        if self.verbose:
            report = pd.DataFrame({
                'before': missing0,
                'after_inf': missing_inf,
                'after_winsor': missing1,
                'after_sector': missing2,
                'after_date_median': missing3,
                'after_cat': missing4,
                'after_fill': missing5
            })
            print("Preprocessing missing value report:")
            print(report)

        return df

    def _winsorize_group(self, s: pd.Series) -> pd.Series:
        lower = s.quantile(self.lower_q)
        upper = s.quantile(self.upper_q)
        return s.clip(lower=lower, upper=upper)

    def _impute_sector(self, medians: pd.DataFrame):
        def imputer(row):
            values = row[self.features]
            if values.isna().any():
                date, sector = row.name[0], row[self.sector_col]
                try:
                    return medians.loc[(date, sector)]
                except KeyError:
                    return values
            return values
        return imputer

    def _fill_mode(self, grp: pd.Series) -> pd.Series:
        if grp.isna().all():
            return grp
        mode_vals = grp.mode()
        if len(mode_vals) == 0:
            return grp
        return grp.fillna(mode_vals.iloc[0])




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
