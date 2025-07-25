# In apply_grid_search, before any work starts:
print("[GRID SEARCH] ▶️ Starting full grid search")

# Right before feature engineering:
print("[GRID SEARCH] 🔧 Feature engineering on raw data…")

# Right before cross‑sectional preprocessing:
print("[GRID SEARCH] 🧹 Cross‑sectional preprocessing of features…")

# At the top of the model loop:
print(f"[GRID SEARCH] 🤖 Tuning model '{model_name}'")

# At the start of each CV config:
print(f"[GRID SEARCH]   CV config: {cv_conf}")

# Inside run_inner_cv, at the top:
print("[INNER CV] ▶️  Running inner train/val splits")

# After retrieving splits:
print(f"[INNER CV]   {len(splits_inner)} splits found")

# At each split:
print(f"[INNER CV]   ▶️  Split {split_id}/{len(splits_inner)}")

# Before fitting each hyper‑parameter combo:
print(f"[INNER CV]     🔎  Params: {params}")

# After grid search completes:
print("[GRID SEARCH] ✅ Grid search finished; saving best pipeline")


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
