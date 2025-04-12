import numpy as np
from collections import deque


class DBSCAN:
    
    def __init__(self, eps, minpts):
        self.eps=eps
        self.minpts=minpts
        self.labels_ = None 
        
    def fit(self, X):
        
        n_points = X.shape[0]
        
        self.labels_ = np.full(n_points, -1)
        cluster_id = 0
        
        for point_idx in range(n_points):
            if self.labels_[point_idx] != -1:
                continue
            
            neighbors = self._region_query(X, point_idx)
            
            if len(neighbors) < self.minpts:
                self.labels_[point_idx] = -1
                continue
            
            
            self._expand_cluster(X, point_idx, neighbors, cluster_id)
            cluster_id += 1
            
    def _region_query(self, X, point_idx):
        distances = np.linalg.norm(X - X[point_idx], axis=1)
        return list(np.where(distances <= self.eps)[0])
    
    def _expand_cluster(self, X, point_idx, neighbors, cluster_id):
        self.labels_[point_idx] = cluster_id
        
        queue = deque(neighbors)
        
        while queue:
            
            neighbor_idx = queue.popleft()
            
            if self.labels_[neighbor_idx] == -1:
                self.labels_[neighbor_idx] = cluster_id
                
            if self.labels_[neighbor_idx] == -1:
                self.labels_[neighbor_idx] = cluster_id
                
                new_neighbors = self._region_query(X, neighbor_idx)
                
                if len(new_neighbors) >= self.minpts:
                    queue.extend(new_neighbors)
                    
                    
if __name__ == "__main__":
    from sklearn.datasets import make_moons
    import matplotlib.pyplot as plt 
    
    
    X, _ = make_moons(n_samples=300, noise=0.05, random_state=42)
    
    
    dbscan  = DBSCAN(eps=0.2, minpts=5)
    dbscan.fit(X)
    
    plt.scatter(X[:, 0], X[:, 1], c=dbscan.labels_, cmap='viridis', s=50)
    plt.title("DBSCAN Clustering")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.savefig("output.png")
    
    
    