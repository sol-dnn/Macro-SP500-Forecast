import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats.mstats import winsorize

class Preprocessor:
    def __init__(self, cutoff_date="2007-01-01", missing_threshold=10, apply_winsorization=False):
        """
        Initializes the Preprocessor class.
        
        Parameters:
        - cutoff_date (str): Start date for train-test split.
        - missing_threshold (float): Percentage threshold for dropping columns with excessive NaNs.
        - apply_winsorization (bool): Whether to apply winsorization for outliers.
        """
        if not isinstance(cutoff_date, str):
            raise ValueError("cutoff_date must be a string representing a valid date.")
        if not (0 <= missing_threshold <= 100):
            raise ValueError("missing_threshold must be between 0 and 100.")
        if not isinstance(apply_winsorization, bool):
            raise ValueError("apply_winsorization must be a boolean.")
        
        self.cutoff_date = cutoff_date
        self.missing_threshold = missing_threshold
        self.apply_winsorization = apply_winsorization
        self.scaler = StandardScaler()
        self.train_columns = None  # Stores columns in training set

    def filter_by_cutoff(self, X, y):
        """Filters data based on the cutoff date."""
        if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
            raise TypeError("X must be a DataFrame and y must be a Series.")
        
        X_filtered = X.loc[self.cutoff_date:]
        y_filtered = y.loc[self.cutoff_date:]
        return X_filtered, y_filtered

    def train_test_split(self, X, y, split_date="2020-01-01"):
        """Splits dataset into training and testing sets based on a given date."""
        if split_date not in X.index:
            raise ValueError("split_date must be within the index range of X.")
        
        X_train, X_test = X.loc[:split_date].iloc[:-1], X.loc[split_date:]
        y_train, y_test = y.loc[:split_date].iloc[:-1], y.loc[split_date:]
        return X_train, X_test, y_train, y_test

    def handle_missing_values(self, X_train, X_test):
        """
        Handles missing values for both train and test sets simultaneously.
        - Drops high-NaN columns from training set and applies the same column removal to test set.
        - Applies forward fill (ffill), backward fill (bfill), and median imputation.
        """
        # Drop columns with > missing_threshold% NaNs (only in train, ensure same columns in test)
        missing_percentage = X_train.isna().mean() * 100
        columns_to_drop = missing_percentage[missing_percentage > self.missing_threshold].index
        X_train = X_train.drop(columns=columns_to_drop)
        X_test = X_test.drop(columns=columns_to_drop, errors='ignore')
        bfill_features = ["VVIX_Index_Zscore_250d"]  # Stable macro features
        median_features = list(set(X_train.columns) - set(bfill_features))
        
        # Apply forward fill
        X_train = X_train.fillna(method="ffill")
        X_test = X_test.fillna(method="ffill")
        
        # Apply backward fill
        X_train[bfill_features] = X_train[bfill_features].fillna(method="bfill")
        X_test[bfill_features] = X_test[bfill_features].fillna(method="bfill")
        
        # Apply median imputation (using train statistics)
        median_values = X_train[median_features].median()
        X_train[median_features] = X_train[median_features].fillna(median_values)
        X_test[median_features] = X_test[median_features].fillna(median_values)
        
        return X_train, X_test

    
    def cap_outliers(self, X):
        """Caps extreme outliers using Winsorization if enabled."""
        if not self.apply_winsorization:
            return X  # Skip if winsorization is not enabled
        
        def winsorize_series(series, lower=0.01, upper=0.99):
            return winsorize(series, limits=[lower, upper])
        
        return X.apply(winsorize_series)

    def normalize(self, X_train, X_test):
        """Normalizes training and testing datasets using StandardScaler."""
        self.scaler.fit(X_train)
        X_train_scaled = pd.DataFrame(self.scaler.transform(X_train),
                                      index=X_train.index, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(self.scaler.transform(X_test),
                                     index=X_test.index, columns=X_test.columns)
        return X_train_scaled, X_test_scaled

    def get_preprocess_data(self, X, y, split_date="2020-01-01"):
        """Full pipeline to preprocess the dataset."""
        print("Starting preprocessing pipeline...")
        
        X, y = self.filter_by_cutoff(X, y)
        X_train, X_test, y_train, y_test = self.train_test_split(X, y, split_date)
        X_train, X_test = self.handle_missing_values(X_train, X_test)
        X_train = self.cap_outliers(X_train)
        X_test = self.cap_outliers(X_test)
        X_train_scaled, X_test_scaled = self.normalize(X_train, X_test)
        
        print("Preprocessing complete!")
        return X_train_scaled, X_test_scaled, y_train, y_test
