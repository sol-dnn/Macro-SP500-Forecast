import joblib
import pandas as pd
from sklearn.base import clone
from src.cv import PurgedWalkForwardCV
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from src.utils import compute_ml_superiority, compute_mae
import os
from datetime import datetime

class ModelPipeline:
    """
    Wrapper for scikit-learn pipeline with support for purged walk-forward validation.

    Available methods:
      - load_model / save_model: persist and reload trained pipelines
      - apply_model: train and predict across multiple splits (S1 or S2)
      - analyze_features: compute MDI, MDA, PCA metrics for a given split
    """
    # Default directory to persist models
    DEFAULT_MODEL_DIR = "artifacts"

    def __init__(self, pipeline=None):
        self.pipeline = pipeline

    @classmethod
    def load_model(cls, path: str) -> "ModelPipeline":
        # Load a previously saved pipeline from disk and return a ModelPipeline instance.
        # This allows scoring without retraining by restoring preprocessing and model state.
        pipe = joblib.load(path)
        return cls(pipeline=pipe)

    def save_model(self, path: str = None):
        """
        Save the current scikit-learn pipeline to disk using joblib.

        If no path is provided, saves to artifacts/model_<timestamp>.joblib by default,
        preserving previous versions. To use a custom filename or directory,
        supply an explicit path argument.
        """
        if self.pipeline is None:
            raise ValueError("No pipeline loaded to save.")

        # Ensure default directory exists
        os.makedirs(self.DEFAULT_MODEL_DIR, exist_ok=True)

        if path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"model_{timestamp}.joblib"
            path = os.path.join(self.DEFAULT_MODEL_DIR, filename)

        # Persist the pipeline
        joblib.dump(self.pipeline, path)
        return path

    def apply_model(
        self,
        df_model: pd.DataFrame,
        cv_conf: dict,
        target: str,
        features: list,
        split_stage: str = 'S1',
        get_feature_analysis: bool = False
    ) -> dict:
        """
        Apply purged walk-forward CV across splits, returning predictions and scores.

        Args:
            df_model: DataFrame with MultiIndex (date, instrument)
            cv_conf: dict parameters for PurgedWalkForwardCV
            target: name of target column
            features: list of feature column names
            split_stage: 'S1' or 'S2' to choose split set
            get_feature_analysis: if True, return MDI/MDA/PCA per split

        Returns:
            dict containing:
              - 'val_preds': concatenated validation predictions
              - 'ml_superiority_mean': average ML superiority score
              - 'mae_mean': average MAE
              - optional 'mdi','mda','pca' DataFrames
        """
        all_pred = []
        feat_info = {'mdi': [], 'mda': [], 'pca': []}

        pcv = PurgedWalkForwardCV(**cv_conf)
        splits = pcv.get_splits()[split_stage]

        for split_id, split in enumerate(splits, start=1):
            train_idx, test_idx = split['train_idx'], split['test_idx']
            df_train = df_model[df_model.index.get_level_values('date').isin(train_idx)]
            df_test  = df_model[df_model.index.get_level_values('date').isin(test_idx)]

            X_train, y_train = df_train[features], df_train[target]
            X_test,  y_test  = df_test[features],  df_test[target]

            pipe = clone(self.pipeline)
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

            df_tmp = df_test.copy()
            df_tmp['y_true'] = y_test.values
            df_tmp['y_pred'] = y_pred
            all_pred.append(df_tmp)

            if get_feature_analysis:
                info = self.analyze_features(pipe, X_test, y_test, split_id, features)
                for k, df_info in info.items():
                    feat_info[k].append(df_info)

        df_preds = pd.concat(all_pred).sort_index()
        df_preds = df_preds.drop_duplicates(
            subset=['date', df_preds.index.get_level_values(1).name], keep='last'
        )

        ml_score  = compute_ml_superiority(df_preds, pred_col='y_pred')
        mae_score = compute_mae(df_preds['y_true'], df_preds['y_pred'])

        results = {
            'val_preds': df_preds,
            'ml_superiority_mean': ml_score,
            'mae_mean': mae_score
        }
        if get_feature_analysis:
            results.update({k: pd.concat(v) for k, v in feat_info.items()})
        return results

    def analyze_features(
        self,
        fitted_pipe,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        split_id: int,
        feature_names: list
    ) -> dict:
        """
        Analyze feature importances (MDI, MDA) and PCA for a single split.

        Returns a dict of DataFrames for keys 'mdi', 'mda', and 'pca'.
        """
        names = fitted_pipe.named_steps['preprocessor'].get_feature_names_out()
        info = {}

        # MDI (Mean Decrease in Impurity) for tree-based models
        est = fitted_pipe.named_steps.get('estimator', fitted_pipe)
        if hasattr(est, 'feature_importances_'):
            mdi_df = pd.DataFrame({
                'feature': names,
                'importance': est.feature_importances_,
                'split': split_id
            })
            info['mdi'] = mdi_df

        # MDA (Mean Decrease in Accuracy) via permutation importance
        perm = permutation_importance(
            fitted_pipe, X_test, y_test,
            n_repeats=10, scoring='neg_mean_absolute_error', random_state=42, n_jobs=-1
        )
        mda_df = pd.DataFrame({
            'feature': names,
            'importance': perm.importances_mean,
            'split': split_id
        })
        info['mda'] = mda_df

        # PCA explained variance per feature
        X_proc = fitted_pipe.named_steps['preprocessor'].transform(X_test)
        Z = StandardScaler().fit_transform(X_proc)
        pca = PCA(n_components=len(names)).fit(Z)
        pca_df = pd.DataFrame({
                'feature': names,
                'explained_variance': pca.explained_variance_,
                'split': split_id
        })
        info['pca'] = pca_df

        return info



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
