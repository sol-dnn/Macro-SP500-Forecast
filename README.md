import pandas as pd
import numpy as np
from scipy.stats import skew
from sklearn.base import TransformerMixin, BaseEstimator
from typing import List

class CrossSectionalPreprocessor(BaseEstimator, TransformerMixin):
    """
    Transformer for cross-sectional winsorization, skew correction, and imputation of features.

    Steps:
      0. Detect and transform skewed features (signed square root if |skew| > threshold)
      1. Replace inf/-inf with NaN
      2. Winsorize numeric features per date
      3. Impute numeric by median per (date, sector)
      4. Impute numeric by median per date
      5. Impute categorical by mode per date
      6. Fill remaining missing with a sentinel value

    Parameters:
        features: list of feature column names to process
        sector_col: sector grouping column name
        lower_q, upper_q: quantile bounds for winsorization
        skew_threshold: threshold for skew detection
        fill_value: value to impute remaining missing
        verbose: if True, prints missing value report
    """
    def __init__(
        self,
        features: List[str],
        sector_col: str = 'GICS_sector_name',
        lower_q: float = 0.03,
        upper_q: float = 0.97,
        skew_threshold: float = 1.0,
        fill_value: float = -11,
        verbose: bool = False
    ):
        self.features = features
        self.sector_col = sector_col
        self.lower_q = lower_q
        self.upper_q = upper_q
        self.skew_threshold = skew_threshold
        self.fill_value = fill_value
        self.verbose = verbose

    def fit(self, X: pd.DataFrame, y=None):
        # No fitting necessary for this transformer
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        # Determine numeric and categorical based on dtype
        num_cols = [c for c in self.features if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        cat_cols = [c for c in self.features if c in df.columns and not pd.api.types.is_numeric_dtype(df[c])]

        # 0) Skew correction for numeric features
        skew_vals = {col: skew(df[col].dropna()) for col in num_cols}
        for col, val in skew_vals.items():
            if abs(val) > self.skew_threshold:
                df[col] = self._signed_sqrt(df[col])
        missing0 = df[self.features].isna().sum()

        # 1) Replace infinite values
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        missing_inf = df[self.features].isna().sum()

        # 2) Winsorize numeric features per date
        df[num_cols] = df.groupby(level='date')[num_cols].transform(self._winsorize_group)
        missing1 = df[self.features].isna().sum()

        # 3) Sector-level median imputation for numeric features
        med = df.reset_index().groupby(['date', self.sector_col])[num_cols].median()
        def sector_imputer(row):
            date, sector = row.name[0], row[self.sector_col]
            for col in num_cols:
                if pd.isna(row[col]):
                    try:
                        row[col] = med.loc[(date, sector), col]
                    except KeyError:
                        pass
            return row
        df = df.apply(sector_imputer, axis=1)
        missing2 = df[self.features].isna().sum()

        # 4) Date-level median imputation for numeric features
        df[num_cols] = df.groupby(level='date')[num_cols].transform(lambda grp: grp.fillna(grp.median()))
        missing3 = df[self.features].isna().sum()

        # 5) Categorical mode imputation per date
        for col in cat_cols:
            df[col] = df.groupby(level='date')[col].transform(self._fill_mode)
        missing4 = df[self.features].isna().sum()

        # 6) Final fill for any remaining missing values
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
            print("Missing value report:")
            print(report)

        return df

    def _signed_sqrt(self, series: pd.Series) -> pd.Series:
        return np.sign(series) * np.sqrt(series.abs())

    def _winsorize_group(self, s: pd.Series) -> pd.Series:
        lower = s.quantile(self.lower_q)
        upper = s.quantile(self.upper_q)
        return s.clip(lower=lower, upper=upper)

    def _fill_mode(self, grp: pd.Series) -> pd.Series:
        mode_vals = grp.mode()
        if not mode_vals.empty:
            return grp.fillna(mode_vals.iloc[0])
        return grp


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
