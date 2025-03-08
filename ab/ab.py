import numpy as np
from sklearn.tree import DecisionTreeClassifier



class AdaBoost:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.models = []
        self.alphas = []


    def fit(self, X, y):
        n_samples = X.shape[0]
        weights = np.ones(n_samples) / n_samples 

        for t in range(self.n_estimators):

            model = DecisionTreeClassifier(max_depth=1) # weak learner: decision tree stump 

            model.fit(X, y, sample_weight=weights)
            pred = model.predict(X)

            error = np.sum(weights * (pred != y))

            alpha = 1 / 2 * np.log((1 - error) / (error + 1e-10))

            weights *= np.exp(-alpha * y * pred)

            weights /= np.sum(weights)

            self.models.append(model)
            self.alphas.append(alpha)

    def predict(self, X):
        preds = np.zeros(X.shape[0])
        for alpha, model in zip(self.alphas, self.models):
            preds += alpha * model.predict(X)

        return np.sign(preds)
    

if __name__ == "__main__":

    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split


    X, y = make_classification(n_samples=10000, n_features=5, random_state=42)
    y = np.where(y == 0, -1, 1)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    ab_model = AdaBoost(n_estimators=10)
    ab_model.fit(X, y)


    y_pred = ab_model.predict(X_test)

    accuracy = np.mean(y_pred == y_test)

    print(f"test accuracy: {accuracy: .2f}")