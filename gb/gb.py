import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score, mean_squared_error


class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.models = []
        self.initial_prediction = None  

    def fit(self, X, y):
        self.initial_prediction = np.mean(y)
        F = np.full(y.shape, self.initial_prediction)

        for _ in range(self.n_estimators):
            residuals = y - F

            model = DecisionTreeRegressor(max_depth=self.max_depth)
            model.fit(X, residuals)
            self.models.append(model)


            F += self.learning_rate * model.predict(X)

    def predict(self, X):
        F = np.full(X.shape[0], self.initial_prediction)

        for model in self.models:
            F += self.learning_rate * model.predict(X)

        return F


if __name__ == "__main__":

    X, y = make_regression(n_samples=200, n_features=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
    gb.fit(X_train, y_train)

    y_pred = gb.predict(X_test)

    score = r2_score(y_test, y_pred)
    acc = max(1, score * 100)  
    mse = mean_squared_error(y_test, y_pred)

    print(f"Accuracy: {acc:.2f}")
    print(f"MSE: {mse:.6f}")