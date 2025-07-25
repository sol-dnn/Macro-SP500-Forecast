import pandas as pd
import numpy as np
from itertools import product
from copy import deepcopy
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from src.cv import PurgedWalkForwardCV
from src.features import FeatureCreator, ColumnSelector, CrossSectionalPreprocessor
from src.utils import compute_mae, compute_ml_superiority, compute_stats


def run_inner_cv(
    df_model: pd.DataFrame,
    cv_conf: dict,
    features: list,
    target: str,
    model_pipeline: Pipeline,
    param_grid: dict
) -> (pd.DataFrame, dict):
    """
    Perform inner train/validation CV loop for one model and CV config.

    Returns a DataFrame of per-split metrics and a dict of trained pipelines.
    """
    # collect per-parameter results
    results_all = defaultdict(list)
    val_preds_by_param = defaultdict(list)
    pipe_dict = {}

    # get inner splits for tuning
    cv = PurgedWalkForwardCV(**cv_conf)
    splits_inner = cv.get_inner_splits()

    for split_id, inner in enumerate(splits_inner, start=1):
        train_dates = inner['train_inner']
        val_dates   = inner['val_inner']

        df_train = df_model.loc[df_model.index.get_level_values('date').isin(train_dates)]
        df_val   = df_model.loc[df_model.index.get_level_values('date').isin(val_dates)]

        X_train, y_train = df_train[features], df_train[target]
        X_val,   y_val   = df_val[features],   df_val[target]

        # loop over all hyperparameter combinations
        for combo in product(*param_grid.values()):
            params = dict(zip(param_grid.keys(), combo))
            # clone and set hyperparameters
            pipe = clone(model_pipeline)
            pipe.set_params(**{f"regressor__{k}": v for k, v in params.items()})
            # fit & predict
            pipe.fit(X_train, y_train)
            y_pred_tr = pipe.predict(X_train)
            y_pred_val= pipe.predict(X_val)

            # save out-of-sample predictions
            df_val_copy = df_val.copy()
            df_val_copy['y_true'] = y_val
            df_val_copy['y_pred'] = y_pred_val
            key = tuple(sorted(params.items()))
            val_preds_by_param[key].append(df_val_copy)

            # compute metrics
            mae_tr = compute_mae(y_train, y_pred_tr)
            mae_va = compute_mae(y_val,   y_pred_val)
            _, ml_sup = compute_ml_superiority(df_val_copy, pred_col='y_pred')

            # record
            results_all[key].append({
                'split_id': split_id,
                'params': params,
                'mae_train': mae_tr,
                'mae_val': mae_va,
                'ml_superiority': ml_sup
            })
            # keep a copy of the fitted pipeline
            pipe_dict[key] = deepcopy(pipe)

    # summarize across splits
    summary = []
    for key, metrics in results_all.items():
        # extract lists
        mae_tr_list = [m['mae_train'] for m in metrics]
        mae_va_list = [m['mae_val']   for m in metrics]
        ml_list     = [m['ml_superiority'] for m in metrics]
        # compute stats
        mean_tr, std_tr, ci_low_tr, ci_high_tr = compute_stats(mae_tr_list)
        mean_va, std_va, ci_low_va, ci_high_va = compute_stats(mae_va_list)
        mean_ml, std_ml, ci_low_ml, ci_high_ml = compute_stats(ml_list)
        summary.append({
            'params': dict(key),
            'mean_mae_train': mean_tr,
            'std_mae_train': std_tr,
            'ci_low_mae_train': ci_low_tr,
            'ci_high_mae_train': ci_high_tr,
            'mean_mae_val': mean_va,
            'std_mae_val': std_va,
            'ci_low_mae_val': ci_low_va,
            'ci_high_mae_val': ci_high_va,
            'mean_ml_superiority': mean_ml,
            'std_ml_superiority': std_ml,
            'ci_low_ml_superiority': ci_low_ml,
            'ci_high_ml_superiority': ci_high_ml
        })

    results_df = pd.DataFrame(sorted(summary, key=lambda x: x['mean_mae_val']))
    return results_df, pipe_dict


def apply_grid_search(
    df_model: pd.DataFrame,
    model_grid: dict,
    cv_grid: list,
    feature_helpers: list,
    features: list,
    target: str,
    sector_col: str,
    skew_threshold: float,
    winsor_q: tuple,
    save_path: str
) -> pd.DataFrame:
    """
    Execute full grid search over (model × CV config) using run_inner_cv.
    Builds each full pipeline from raw data, aggregates results, saves best pipeline.
    """
    all_results = []
    all_pipes = {}

    for model_name, info in model_grid.items():
        ModelClass = info['model']
        param_grid = info['param_grid']

        for cv_conf in cv_grid:
            # 1) build full pipeline
            pipe = Pipeline([
                ('feat',   FeatureCreator(helpers=feature_helpers)),
                ('prep1',  ColumnSelector(columns=features + [sector_col])),
                ('prep2',  CrossSectionalPreprocessor(
                              features=features,
                              sector_col=sector_col,
                              lower_q=winsor_q[0],
                              upper_q=winsor_q[1],
                              skew_threshold=skew_threshold
                          )),
                ('select',ColumnSelector(columns=features)),
                ('regressor', ModelClass())
            ])
            # 2) run inner CV tuning
            res_df, pipes = run_inner_cv(
                df_model, cv_conf, features, target, pipe, param_grid
            )
            res_df['model'] = model_name
            res_df['cv_conf'] = str(cv_conf)
            all_results.append(res_df)
            all_pipes.update(pipes)

    # concat all results
    final_df = pd.concat(all_results, ignore_index=True)
    # pick best
    best_idx = final_df['mean_mae_val'].idxmin()
    best_params = final_df.loc[best_idx, 'params']
    best_key = tuple(sorted(best_params.items()))
    best_pipe = all_pipes[best_key]
    # save
    joblib.dump(best_pipe, save_path)

    return final_df.sort_values('mean_mae_val').reset_index(drop=True)

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
