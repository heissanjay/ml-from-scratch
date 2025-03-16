import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.1, epochs=100):
        self.learning_rate = learning_rate
        self.epochs=epochs
        self.w = None
        self.b = 0
        
    def fit(self, X, y):
        
        n_samples, n_features = X.shape
        
        self.w = np.zeros(n_features) 
        
        for _ in range(self.epochs):
            # numpy implicitly treats w as columns vector for broadcasting
            y_pred = 1 / (1 + (np.exp(-(X @ self.w + self.b)))) 
            
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            
            db = (1 / n_samples) * np.sum((y_pred - y)) 
            
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db
            
    def predict(self, X):
        y_pred = 1 / (1 + (np.exp(-(X @ self.w + self.b))))
        return (y_pred >= 0.5 ).astype(int)
    

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_classification
    from sklearn.metrics import accuracy_score
    
    X, y = make_classification(n_samples=1000, n_classes=2, n_features=10, random_state=42)
    
    X_train, X_test,y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    lr = LogisticRegression(learning_rate=0.1, epochs=100)
    
    lr.fit(X_train, y_train)
    
    y_pred = lr.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    
    print("Accuary: ", acc)