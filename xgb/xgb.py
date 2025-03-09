import numpy as np 


class XGBoostRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, lambda_reg=1.0, gamma=0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.lambda_reg = lambda_reg
        self.gamma = gamma
        self.trees = []
        
        
    def _gradient_hession(self, y_true, y_pred):
        # first and secord order derivatives.
        gradient = 2 * (y_pred - y_true)
        hession = np.full_like(y_true, 2.0)
        
        return gradient, hession
    
    def _split_gain(self, G_L, H_L, G_R, H_R):
        # compute gain for a split using the gain formula
       return 0.5 * ((G_L ** 2 / (H_L + self.lambda_reg)) +
                    (G_R ** 2 / (H_R + self.lambda_reg)) - 
                    ((G_L + G_R)**2 / (H_L + H_R + self.lambda_reg))) - self.gamma
       
    def _best_split(self, X, y, y_pred):
        G, H = self._gradient_hession(y, y_pred)
        best_gain, best_feature, best_threshold = -np.inf, None, None 
        
        
        for feature_idx in range(X.shape[1]):
            sorted_idx = np.argsort(X[:, feature_idx])
            X_sorted, G_sorted, H_sorted = X[sorted_idx], G[sorted_idx], H[sorted_idx]
            G_L, H_L = 0, 0
            G_R, H_R = np.sum(G_sorted), np.sum(H_sorted)
            
            for i in range(1, len(X_sorted)):
                G_L += G_sorted[i - 1]
                H_L += H_sorted[i - 1]
                G_R -= G_sorted[i - 1]
                H_R -= H_sorted[i - 1]
                
                gain = self._split_gain(G_L, H_L, G_R, H_R)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = X_sorted[i]
                    
        return best_feature, best_threshold
    
    def _build_tree(self, X, y, y_pred, depth=0):
        if depth >= self.max_depth or len(X) <= 1:
            G, H = self._gradient_hession(y, y_pred)
            leaf_value = -np.sum(G) / (np.sum(H) + self.lambda_reg)
            return leaf_value
        
        feature, threshold = self._best_split(X, y, y_pred)
        
        if feature is None:
            G, H = self._gradient_hession(y, y_pred)
            return -np.sum(G) / (np.sum(H) + self.lambda_reg)
        
        left_idx = X[:, feature] <= threshold
        right_idx = X[:, feature] > threshold
        
        left_tree = self._build_tree(X[left_idx], y[left_idx], y_pred[left_idx], depth+1)
        right_tree = self._build_tree(X[right_idx], y[right_idx], y_pred[right_idx], depth+1)
        
        return (feature, threshold, left_tree, right_tree)
        
    def _predict_tree(self, tree, X):
        if isinstance(tree, float):
            return np.full(X.shape[0], tree)
        
        feature, threshold, left_tree, right_tree = tree 
        left_idx = X[:, feature] <= threshold
        right_idx = X[:, feature] > threshold

        preds = np.zeros(X.shape[0])
        preds[left_idx] = self._predict_tree(left_tree, X[left_idx])
        preds[right_idx] = self._predict_tree(right_tree, X[right_idx])
        return preds
    
    def fit(self, X, y):
        y_pred = np.zeros_like(y, dtype=np.float64)

        for _ in range(self.n_estimators):
            tree = self._build_tree(X, y, y_pred)
            self.trees.append(tree)

            preds = self._predict_tree(tree, X)
            y_pred += self.learning_rate * preds  

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for tree in self.trees:
            y_pred += self.learning_rate * self._predict_tree(tree, X)
        return y_pred
    

np.random.seed(42)
X = np.random.rand(100, 1)  
y = 3 * X.squeeze() + np.random.randn(100) * 0.1 


X_train, X_test = X[:80], X[80:]
y_train, y_test = y[:80], y[80:]

model = XGBoostRegressor(n_estimators=10, learning_rate=0.1, max_depth=3, lambda_reg=1.0)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = np.mean((y_test - y_pred) ** 2)
print(f"Test MSE: {mse}")