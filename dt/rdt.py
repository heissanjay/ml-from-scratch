import numpy as np
from collections import Counter 


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left  = left
        self.right = right
        self.value = value

    
def mse(y):
    return np.var(y) if len(y) > 0 else 0

def mse_reduction(y, left_y, right_y):
    parent_mse = mse(y)
    weight_left = len(left_y) / len(y)
    weight_right = len(right_y) / len(y)

    return mse(y) - (weight_left * mse(left_y) + weight_right * mse(right_y))


def find_best_split(X, y):
    n_samples, n_features = X.shape
    best_feature, best_threshold = None, None

    max_gain = -float('inf')

    for feature in range(n_features):
        thresholds = np.unique(X[:, feature])
        for threshold in thresholds:
            left_mask = X[:, feature] <= threshold
            right_mask = X[:, feature] > threshold

            left_y, right_y = y[left_mask], y[right_mask]

            if len(left_y) == 0 or len(right_y) == 0:
                continue

            gain = mse_reduction(y, left_y, right_y)
            if gain > max_gain:
                max_gain = gain
                best_feature = feature
                best_threshold = threshold

    return best_feature, best_threshold

class RegressionTree:
    def __init__(self, max_depth = 5):
        self.max_depth = max_depth
        self.root = None

    def build_tree(self, X, y, depth=0):
        if len(set(y)) == 1 or depth >= self.max_depth:
            return Node(value=np.mean(y))
        
        feature, threshold = find_best_split(X, y)
        if feature is None:
            return Node(value=np.mean(y))
        
        left_mask = X[:, feature] <= threshold
        right_mask = X[:, feature] > threshold

        left_tree = self.build_tree(X[left_mask], y[left_mask], depth = depth+1)
        right_tree = self.build_tree(X[right_mask], y[right_mask], depth = depth+1)

        return Node(feature=feature, threshold=threshold, left=left_tree, right=right_tree)
    

    def fit(self, X, y):
        self.root = self.build_tree(X, y)

    def predict_sample(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self.predict_sample(x, node.left)
        else:
            return self.predict_sample(x, node.right)
        
    def predict(self, X):
        return np.array([self.predict_sample(x, self.root) for x in X])


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
X, y = housing.data, housing.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tree = RegressionTree(max_depth=15)
tree.fit(X_train, y_train)

y_pred = tree.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 score: ", r2_score(y_test, y_pred))