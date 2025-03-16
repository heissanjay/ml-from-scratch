import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

class NaiveCatBoost:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, smoothing=1.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.smoothing = smoothing
        self.models = []
        self.initial_prediction = None

    def _encode_categorical_features(self, X, y=None, prior=None):
        if y is None:  # For prediction phase
            return np.array([
                [self.category_stats[cat] if cat in self.category_stats else prior for cat in col]
                for col in X.T
            ]).T
        
        encoded_X = np.zeros_like(X, dtype=float)
        self.category_stats = {}
        global_mean = np.mean(y)

        for col_idx in range(X.shape[1]):
            col = X[:, col_idx]
            unique_cats = np.unique(col)
            stats = {}

            for cat in unique_cats:
                indices = np.where(col == cat)[0]
                n_cat = len(indices)
                cat_mean = np.mean(y[indices])
                stats[cat] = (cat_mean * n_cat + global_mean * self.smoothing ) / (n_cat + self.smoothing)

            self.category_stats.update(stats)
            encoded_X[:, col_idx] = np.array([stats[cat] if cat in stats else global_mean for cat in col])

        return encoded_X 

    def fit(self,X, y, categorical_features=None):
        if categorical_features is None:
            categorical_features = []

        self.initial_prediction = np.mean(y)
        F = np.full(y.shape, self.initial_prediction)

        X_encoded = X.copy()
        if categorical_features:
            X_encoded[:, categorical_features] = self._encode_categorical_features(
                    X[:, categorical_features], y, prior=self.initial_prediction
            )

        for _ in range(self.n_estimators):
            residuals = y - F 

            model = DecisionTreeRegressor(max_depth=self.max_depth)

            model.fit(X_encoded, residuals)
            self.models.append(model)

            F += self.learning_rate * model.predict(X_encoded)

    def predict(self, X, categorical_features=None):
        if categorical_features is None:
            categorical_features = []

        X_encoded = X.copy()
        if categorical_features:
            X_encoded[:, categorical_features] = self._encode_categorical_features(
                    X[:, categorical_features], prior=self.initial_prediction
                    )

        F = np.full(X.shape[0], self.initial_prediction)

        for model in self.models:
            F += self.learning_rate * model.predict(X_encoded)

        return F 

if __name__ == "__main__":
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score

    X, y = make_regression(n_samples=500, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train[:, 0] = (X_train[:, 0] > 0).astype(int)
    X_test[:, 0] = (X_test[:, 0] > 0).astype(int)


    catboost = NaiveCatBoost(n_estimators=50, learning_rate=0.1, max_depth=3, smoothing=1.0)
    catboost.fit(X_train, y_train, categorical_features=[0])

    y_pred = catboost.predict(X_test, categorical_features=[0])

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print("r2 score: ", r2)
    print("mse: ", mse)
