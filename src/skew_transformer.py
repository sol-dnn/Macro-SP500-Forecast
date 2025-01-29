import numpy as np

class SkewTransformer:
    def __init__(self, skew_threshold=1):
        self.skew_threshold = skew_threshold
        self.skewed_features = None 

    def fit(self, X_train):
        """Identifies and stores skewed features from training data."""
        skewness = X_train.skew()
        self.skewed_features = skewness[abs(skewness) > self.skew_threshold].index
        print(f"Identified {len(self.skewed_features)} skewed features for transformation:")
        print(list(self.skewed_features))

    def transform(self, X):
        """Applies log transformation for positive values and signed log for negatives."""
        X_transformed = X.copy()

        for feature in self.skewed_features:
            if (X_transformed[feature] > 0).all():
                X_transformed[feature] = np.log1p(X_transformed[feature])  # Log(1+x) for positive values
            else:
                X_transformed[feature] = np.sign(X_transformed[feature]) * np.log1p(abs(X_transformed[feature]))  # Signed log

        return X_transformed

    def fit_transform(self, X_train):
        """Fits on training data and applies transformation."""
        self.fit(X_train)
        return self.transform(X_train)
