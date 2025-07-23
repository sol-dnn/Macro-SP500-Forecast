import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator

class FeatureCreator(BaseEstimator, TransformerMixin):
    """
    Builds model features end-to-end by running several transformation steps.

    Steps executed in order:
      1. _scale_by_price
      2. _transform_date
      3. _add_idiosyncratic_metrics
      4. _add_fundamental_ratios
      5. _add_lagged_features

    No external helpers are required.
    """
    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y=None):
        # Stateless transformer
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        df = self._scale_by_price(df)
        df = self._transform_date(df)
        df = self._add_idiosyncratic_metrics(df)
        df = self._add_fundamental_ratios(df)
        df = self._add_lagged_features(df)
        return df

    def _scale_by_price(self, df: pd.DataFrame) -> pd.DataFrame:
        # Example: scale raw_var by price
        if 'raw_var' in df.columns and 'price' in df.columns:
            df['scaled_var'] = df['raw_var'] / df['price']
        return df

    def _transform_date(self, df: pd.DataFrame) -> pd.DataFrame:
        # Extract date-based features from a DateTimeIndex level 'date'
        if isinstance(df.index, pd.MultiIndex) and 'date' in df.index.names:
            df['month'] = df.index.get_level_values('date').month
            df['weekday'] = df.index.get_level_values('date').weekday
        return df

    def _add_idiosyncratic_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        # Placeholder: compute idiosyncratic volatility/residuals
        # e.g., df['idio_vol'] = ...
        return df

    def _add_fundamental_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        # Placeholder: compute P/E, P/B, ROA, etc.
        # e.g., df['pe_ratio'] = df['price'] / df['eps']
        return df

    def _add_lagged_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Placeholder: compute lagged/growth features
        # e.g., df['sales_growth_1y'] = df['sales'].pct_change(252)
        return df

class ColumnSelector(BaseEstimator, TransformerMixin):
    """
    Selects a specified subset of columns from a DataFrame.
    """
    def __init__(self, columns: list):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.columns].copy()
   
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
