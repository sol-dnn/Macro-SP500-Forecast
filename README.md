from src.pipeline import ModelPipeline
import pandas as pd

# 1) Load your saved pipeline
mp = ModelPipeline.load_model("artifacts/model.joblib")

# 2) Load or construct the DataFrame that has all transformations applied:
#    For example, if you precomputed transformations offline:
df_scoring = pd.read_parquet("data/processed/scoring_dataset.parquet")

# 3) Define your CV configuration exactly as in training
cv_conf = {
    "n_splits": 5,
    "embargo": 2,
    "purge": True,
    # …etc, matching your PurgedWalkForwardCV signature…
}

# 4) Call apply_model on that DataFrame
results = mp.apply_model(
    df_model=df_scoring,
    cv_conf=cv_conf,
    target="y",                # your target column name
    features=[                  # list exactly the feature columns the pipeline uses
        "feat1", "feat2", "feat3", …
    ],
    split_stage="S1",           # or "S2" if you want that stage
    get_feature_analysis=False  # turn on if you also want importances/PCA
)

# 5) Inspect the outputs
df_preds     = results["val_preds"]
ml_score     = results["ml_superiority_mean"]
mae_score    = results["mae_mean"]

print("OOS predictions shape:", df_preds.shape)
print("ML superiority:", ml_score)
print("MAE:", mae_score)


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
