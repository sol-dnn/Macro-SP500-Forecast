import copy
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import norm

class Forecaster:
    def __init__(self, model, n_splits=5):
        """
        Initialize the Forecaster with a specified model and number of splits for time series cross-validation.
        
        Parameters:
        - model: Regression model
        - n_splits: Number of time series cross-validation splits
        """
        self.model = model
        self.n_splits = n_splits
        self.models = []  # Store different trained models per fold
        self.results = []
        self.best_model = None  # Store best model

    def compute_conf_interval(self, mean, std):
        if std == 0 or self.n_splits <= 1:
            return (round(mean, 4), round(mean, 4))
        return norm.interval(0.95, loc=mean, scale=std / np.sqrt(self.n_splits))


    def time_series_cross_validation(self, X, y, dates, beta=0.9):
        """
        Perform time series cross-validation and store detailed results.
        
        Parameters:
        - X: Feature matrix
        - y: Target variable
        - dates: Corresponding dates for each observation
        """
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        rmse_scores, mae_scores, da_scores = [], [], []
        fold_details = []
        best_score = float("-inf")

        dates_series = dates.to_series()

        for fold, (train_idx, validation_idx) in enumerate(tscv.split(X)):
            X_train, X_validation = X.iloc[train_idx], X.iloc[validation_idx]
            y_train, y_validation = y.iloc[train_idx], y.iloc[validation_idx]
            train_dates, validation_dates = dates_series.iloc[train_idx], dates_series.iloc[validation_idx]

            model_copy = copy.deepcopy(self.model)
            model_copy.fit(X_train, y_train)
            y_pred = model_copy.predict(X_validation)
            
            rmse = np.sqrt(mean_squared_error(y_validation, y_pred))
            mae = mean_absolute_error(y_validation, y_pred)
            da = np.mean(np.sign(y_pred) == np.sign(y_validation))
            
            rmse_scores.append(rmse)
            mae_scores.append(mae)
            da_scores.append(da)
            self.models.append(model_copy)
            

            score = (1-beta) * da - beta * rmse
            if score > best_score:
                best_score = score
                self.best_model = copy.deepcopy(model_copy)


            fold_details.append({
                'fold': fold + 1,
                'train_start': train_dates.min(),
                'train_end': train_dates.max(),
                'validation_start': validation_dates.min(),
                'validation_end': validation_dates.max(),
                'RMSE': rmse,
                'MAE': mae,
                'DA': da
            })
            
            print(f"\nFold {fold + 1} | Train: {train_dates.min()} → {train_dates.max()} | Test: {validation_dates.min()} → {validation_dates.max()}")
            print(f"----------Fold {fold + 1} Results: RMSE: {rmse:.4f} | MAE: {mae:.4f} | DA: {da:.4f}")
        
        mean_rmse, std_rmse = np.mean(rmse_scores), np.std(rmse_scores)
        mean_mae, std_mae = np.mean(mae_scores), np.std(mae_scores)
        mean_da, std_da = np.mean(da_scores), np.std(da_scores)
        rmse_conf_interval = norm.interval(0.95, loc=mean_rmse, scale=std_rmse / np.sqrt(self.n_splits))
        mae_conf_interval = norm.interval(0.95, loc=mean_mae, scale=std_mae / np.sqrt(self.n_splits))
        da_conf_interval = norm.interval(0.95, loc=mean_da, scale=std_da / np.sqrt(self.n_splits))
        
        self.results = {
            'mean_RMSE': mean_rmse,
            'RMSE_confidence_interval': self.compute_conf_interval(mean_rmse, std_rmse),
            'mean_MAE': mean_mae,
            'MAE_confidence_interval': self.compute_conf_interval(mean_mae, std_mae),
            'mean_DA': mean_da,
            'DA_confidence_interval': self.compute_conf_interval(mean_da, std_da),
            'fold_details': fold_details
        } 

        print("\nTime Series Cross-Validation Summary:")
        print(f"Mean RMSE: {mean_rmse:.4f} | 95% CI: {self.results['RMSE_confidence_interval']}")
        print(f"Mean MAE: {mean_mae:.4f}  | 95% CI: {self.results['MAE_confidence_interval']}")
        print(f"Mean DA: {mean_da:.4f}    | 95% CI: {self.results['DA_confidence_interval']}")

        return self.best_model, self.results

    
    def test_prediction(self, X_train, y_train, X_test, y_test):
        """
        Train on the full training set and make predictions on the test set.
        
        Parameters:
        - X_train: Full training feature set
        - y_train: Full training target set
        - X_test: Test feature set
        - y_test: True values for test set (for evaluation)
        
        Returns:
        - final_model: Trained model
        - y_pred: Predictions on the test set
        """
        print("\nTraining final model on full dataset...")

        final_model = copy.deepcopy(self.model)
        final_model.fit(X_train, y_train)
        y_pred = final_model.predict(X_test)
        
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        test_mae = mean_absolute_error(y_test, y_pred)
        test_da = np.mean(np.sign(y_pred) == np.sign(y_test)) 
        
        print("\nFinal Model Evaluation on Test Set:")
        print(f"Test Set RMSE: {test_rmse:.4f}")
        print(f"Test Set MAE: {test_mae:.4f}")
        print(f"Test Set DA: {test_da:.4f}")
        
        return final_model, y_pred
