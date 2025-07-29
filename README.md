    def build_quantile_portfolios(self,
                                  signal_col: str,
                                  n_quantiles: int = 5) -> pd.DataFrame:
        """
        Build long-only portfolios for each signal quantile (1 to n_quantiles) by date.
        Uses equal-weight within each bucket only (to avoid issues with mixed signal signs).
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
                weights = self._equal_weights(sub, signal_col, 'long_only')
                sub['weight'] = weights.values
                parts.append(sub)
            return pd.concat(parts) if parts else pd.DataFrame(columns=group.columns)

        result = (
            df.groupby(self.date_col)
              .apply(process_quantiles)
              .reset_index(drop=True)
        )
        return result


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
