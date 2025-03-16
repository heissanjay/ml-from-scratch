import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error


class NaiveLightGBM:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, n_bins=16):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.n_bins=16
        self.models = []
        self.initial_prediction = None

    def _bin_data(self, X):
        binned_X = np.zeros_like(X, dtype=int)
        for col in range(X.shape[1]):
            bin_edges = np.percentile(X[:, col], np.linspace(0, 100, self.n_bins+1))
            binned_X[:, col] = np.digitize(X[:, col], bin_edges, right=True)

        return binned_X

    def fit(self, X, y):
        self.initial_prediction = np.mean(y)

        F = np.full(y.shape, self.initial_prediction)

        binned_X = self._bin_data(X)

        for _ in range(self.n_estimators):
            residuals = y - F 

            model = DecisionTreeRegressor(max_depth=self.max_depth)
            model.fit(binned_X, residuals)
            self.models.append(model)

            F += self.learning_rate * model.predict(binned_X)

    def predict(self, X):
        binned_X = self._bin_data(X)

        F = np.full(X.shape[0], self.initial_prediction)

        for model in self.models:
            F += self.learning_rate * model.predict(binned_X)

        return F

if __name__ == "__main__":
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score


    X, y = make_regression(n_samples=500, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lgbm = NaiveLightGBM(n_estimators=50, learning_rate=0.1, max_depth=3, n_bins=16)
    lgbm.fit(X_train, y_train)

    y_pred = lgbm.predict(X_test)

    r2 = r2_score(y_test, y_pred)

    mse = mean_squared_error(y_test, y_pred)

    print(f"R2 score: ", r2)
    print(f"mse: ", mse)


