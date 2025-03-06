from dt.dt import DecisionTree
import numpy as np
from collections import Counter 


class RandomForest:
    def __init__(self, n_trees=10, max_features=None, max_depth=None, min_sample_split=2):
        self.n_trees = n_trees
        # self.max_features = max_features
        self.max_depth = max_depth
        # self.min_sample_split = min_sample_split 
        self.trees = []

    def bootstrap_sample(self, X, y):
        n_samples = X.shape[0]

        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]
    
    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            X_sample, y_sample = self.bootstrap_sample(X, y)
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _majority_voting(self, predictions):
        counter = Counter(predictions)
        return counter.most_common(1)[0][0]
    
    def predict(self, X):
        trees_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.apply_along_axis(self._majority_voting, axis=0, arr=trees_preds)
    

# Example usage with dummy dataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate a random dataset
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
rf = RandomForest(n_trees=10, max_features=5, max_depth=5)
rf.fit(X_train, y_train)

# Predict
y_pred = rf.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))


