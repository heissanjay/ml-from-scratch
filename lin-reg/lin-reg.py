import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.1, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = 0
        self.b = 0

    def fit(self, X, y):
        n_samples = len(y)

        for _ in range(self.epochs):
            y_pred = self.w * X + self.b

            error = y_pred - y

            dw = (2 / n_samples) * np.dot(X, error)
            db = (2 / n_samples) * np.sum(error)


            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

    def predict(self, X):
        return self.w * X + self.b


X = np.array([1, 2, 3, 4])
y = np.array([2, 3, 4, 5])


model = LinearRegression(learning_rate=0.1, epochs=100)
model.fit(X, y)


print("Predicted y:", model.predict(9))
print("Learned weight:", model.w)
print("Learned bias:", model.b)