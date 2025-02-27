import numpy as np
import pandas as pd
from collections import Counter


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature=feature
        self.threshold=threshold
        self.left=left
        self.right=right
        self.value=value

def entropy(y):
    class_counts = np.bincount(y)
    probabilities = class_counts / len(y)
    return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

def information_gain(y, left_y, right_y):
    parent_entropy = entropy(y)
    left_entropy = entropy(left_y)
    right_entropy = entropy(right_y)

    # weighted average entropy after split
    weight_left = len(left_y) / len(y)
    weight_right = len(right_y) / len(y)

    return parent_entropy - (weight_left * left_entropy + weight_right * right_entropy)


def best_split(X, y):
    n_samples,n_features = X.shape
    best_feature, best_threshold = None, None
    max_gain = -1

    for feature in range(n_features):
        thresholds = np.unique(X[:, feature])
        for threshold in thresholds:
            left_mask = X[:, feature] <= threshold
            right_mask = X[:, feature] > threshold
            left_y, right_y = y[left_mask], y[right_mask]
            # when threshold = threshold[0] and threshold[len(thresholds)-1]
            if len(left_y) == 0 or len(right_y) == 0:
                continue

            gain = information_gain(y, left_y, right_y)
            if gain > max_gain:
                max_gain = gain
                best_feature = feature
                best_threshold = threshold

    return best_feature, best_threshold


class DecisionTree:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.root = None

    def build_tree(self, X, y, depth=0):
        if(len(set(y))) == 1:
            return Node(value=y[0])
        
        if depth >= self.max_depth:
            return Node(value=Counter(y).most_common(1)[0][0])
        
        feature, threshold = best_split(X, y)
        if feature is None:
            return Node(value=Counter(y).most_common(1)[0][0])
        
        left_mask = X[:, feature] <= threshold
        right_mask = X[:, feature] > threshold 
        left_subtree = self.build_tree(X[left_mask], y[left_mask],depth+1)
        right_subtree = self.build_tree(X[right_mask], y[right_mask], depth+1)

        return Node(feature=feature,threshold=threshold, left=left_subtree, right=right_subtree)
    
    def fit(self, X, y):
        self.root = self.build_tree(X, y)

    def predict_samples(self, x, node):
        if node.value is not None:
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self.predict_samples(x, node.left)
        else:
            return self.predict_samples(x, node.right)
        
    def predict(self, X):
        return np.array([self.predict_samples(x, self.root) for x in X])
    

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tree = DecisionTree(max_depth=5)
tree.fit(X_train, y_train)

y_pred = tree.predict(X_test)

print("Accuracy: ", accuracy_score(y_test, y_pred))


if __name__ == "__main__":
    pass