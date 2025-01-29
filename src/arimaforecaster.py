import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import norm
import warnings
from statsmodels.tsa.arima.model import ARIMA


class ARIMAForecaster:
    def __init__(self, arima_order=(1, 0, 0), n_splits=5):
        """
        Initialize the ARIMA forecaster with specified order and cross-validation settings.
        
        Parameters:
        - arima_order: Tuple representing ARIMA (p, d, q) parameters.
        - n_splits: Number of time series cross-validation splits.
        """
        self.arima_order = arima_order
        self.n_splits = n_splits
        self.models = []
        self.results = []
    
    def time_series_cross_validation(self, y):
        """
        Perform time series cross-validation with ARIMA and store detailed results.
        
        Parameters:
        - y: Target variable (log returns).
        """
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=FutureWarning)
        
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        rmse_scores = []
        mae_scores = []
        da_scores = []
        fold_details = []
        
        dates_series = y.index.to_series()
        
        print("\nPerforming ARIMA with Fixed Time Series Splits...\n")
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(y)):
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            train_dates, test_dates = dates_series.iloc[train_idx], dates_series.iloc[test_idx]
            
            try:
                y_train = y_train.resample('W-FRI').last().dropna()
                
                model = ARIMA(y_train, order=self.arima_order)
                model_fit = model.fit()
                y_pred = model_fit.forecast(steps=len(y_test))
                
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                da = np.mean(np.sign(y_pred) == np.sign(y_test))
                
                rmse_scores.append(rmse)
                mae_scores.append(mae)
                da_scores.append(da)
                self.models.append(model_fit)
                
                fold_details.append({
                    'fold': fold + 1,
                    'train_start': train_dates.min(),
                    'train_end': train_dates.max(),
                    'test_start': test_dates.min(),
                    'test_end': test_dates.max(),
                    'RMSE': rmse,
                    'MAE': mae,
                    'DA': da
                })
                
                print(f"----------Fold {fold + 1} | Train: {train_dates.min()} → {train_dates.max()} | Test: {test_dates.min()} → {test_dates.max()}")
                print(f"RMSE: {rmse:.4f} | MAE: {mae:.4f} | DA: {da:.4f}")
                
            except Exception as e:
                print(f"ARIMA failed at fold {fold + 1}: {e}")
                continue
        
        mean_rmse, std_rmse = np.mean(rmse_scores), np.std(rmse_scores)
        mean_mae, std_mae = np.mean(mae_scores), np.std(mae_scores)
        mean_da, std_da = np.mean(da_scores), np.std(da_scores)
        rmse_conf_interval = norm.interval(0.95, loc=mean_rmse, scale=std_rmse / np.sqrt(self.n_splits))
        mae_conf_interval = norm.interval(0.95, loc=mean_mae, scale=std_mae / np.sqrt(self.n_splits))
        da_conf_interval = norm.interval(0.95, loc=mean_da, scale=std_da / np.sqrt(self.n_splits))
        
        self.results = {
            'mean_RMSE': mean_rmse,
            'RMSE_confidence_interval': rmse_conf_interval,
            'mean_MAE': mean_mae,
            'MAE_confidence_interval': mae_conf_interval,
            'mean_DA': mean_da,
            'DA_confidence_interval': da_conf_interval,
            'fold_details': fold_details
        }
        
        print("\nARIMA Cross-Validation Summary:")
        print(f"Mean RMSE: {mean_rmse:.4f} | 95% CI: ({rmse_conf_interval[0]:.4f}, {rmse_conf_interval[1]:.4f})")
        print(f"Mean MAE: {mean_mae:.4f}  | 95% CI: ({mae_conf_interval[0]:.4f}, {mae_conf_interval[1]:.4f})")
        print(f"Mean DA: {mean_da:.4f}    | 95% CI: ({da_conf_interval[0]:.4f}, {da_conf_interval[1]:.4f})")

                
        return self.results
    
    def train_final_model(self, y_train, y_test):
        """
        Train ARIMA on the full training set and make predictions on the test set.
        
        Parameters:
        - y_train: Full training target set.
        - y_test: True values for test set (for evaluation).
        
        Returns:
        - final_model: Trained ARIMA model.
        - y_pred: Predictions on the test set.
        """
        print("\nTraining ARIMA on Full Dataset...\n")
        
        try:
            y_train = y_train.resample('W-FRI').last().dropna()
            model = ARIMA(y_train, order=self.arima_order)
            final_model = model.fit()
            y_pred = final_model.forecast(steps=len(y_test))
            
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            test_mae = mean_absolute_error(y_test, y_pred)
            test_da = np.mean(np.sign(y_pred) == np.sign(y_test))
            
            print("\nFinal Model Evaluation on Test Set:")
            print(f"Test Set RMSE: {test_rmse:.4f}")
            print(f"Test Set MAE: {test_mae:.4f}")
            print(f"Test Set DA: {test_da:.4f}")
            
            return final_model, y_pred
        
        except Exception as e:
            print(f"ARIMA final model training failed: {e}")
            return None, None
