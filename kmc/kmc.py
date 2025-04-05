# implementation of K Means Clustering from scratch
import numpy as np

class KMeans:
    def __init__(self, k=2, max_iteration=100, random_state=None):
        self.k = k
        self.max_iteration=max_iteration
        self.random_state=random_state
        self.centroids = None
        self.labels = None
        
    def fit(self,  X):
        np.random.seed(self.random_state)
        
        n_samples, n_features = X.shape
        self.centroids = X[np.random.choice(n_samples, self.k, replace=False)]
        
        for _ in range(self.max_iteration):
            dist = self._calculate_distances(X)
            self.labels = np.argmin(dist, axis=1)
            
            new_centroids = np.array([X[self.labels == i].mean(axis=0) for i in range(self.k)])
            
            if np.allclose(self.centroids, new_centroids):
                break
            
            self.centroids = new_centroids
            
            
    def predict(self, X):
        dist = self._calculate_distances(X)
        return np.argmin(dist, axis=1)
    
    def _calculate_distances(self, X):
        return np.sqrt(((X[:, np.newaxis] - self.centroids) ** 2).sum(axis=2))
    

if __name__ == "__main__":
    
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
    
    kmeans = KMeans(k=4, max_iteraton=100, random_state=42)
    kmeans.fit(X)
    
    import matplotlib.pyplot as plt
    plt.scatter(X[:, 0], X[:,1], c=kmeans.labels, s=50, cmap='viridis')
    plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c='red', s=200, alpha=0.75, label='Centroids')
    plt.legend()
    plt.savefig('output.png')