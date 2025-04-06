import numpy as np
import matplotlib.pyplot as plt


class HierarchicalCluster:
    def __init__(self, linkage="single"):
        self.linkage = linkage
        self.clusters = None
        self.distances = None
        self.original_distances = None
        self.history = []
        self.labels = None

    def _compute_distance_matrix(self, X):
        n_samples = X.shape[0]
        distances = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                dist = np.linalg.norm(X[i] - X[j])
                distances[i, j] = dist
                distances[j, i] = dist
        return distances

    def _find_closest_clusters(self):
        min_dist = np.inf
        pair = None
        n_clusters = len(self.clusters)
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                dist = self._compute_linkage_distance(i, j)
                if dist < min_dist:
                    min_dist = dist
                    pair = (i, j)
        return min_dist, pair

    def _compute_linkage_distance(self, i, j):
        cluster_i = self.clusters[i]
        cluster_j = self.clusters[j]

        if self.linkage == "single":
            distances = [self.original_distances[x, y] for x in cluster_i for y in cluster_j]
            return min(distances) if distances else 0
        elif self.linkage == "complete":
            distances = [self.original_distances[x, y] for x in cluster_i for y in cluster_j]
            return max(distances) if distances else 0
        elif self.linkage == "average":
            distances = [self.original_distances[x, y] for x in cluster_i for y in cluster_j]
            return sum(distances) / len(distances) if distances else 0
        else:
            raise ValueError("Invalid linkage method")

    def fit(self, X):
        n_samples = X.shape[0]
        self.clusters = [[i] for i in range(n_samples)]
        self.original_distances = self._compute_distance_matrix(X)
        self.labels = np.arange(n_samples)

        while len(self.clusters) > 1:
            min_dist, pair = self._find_closest_clusters()
            ci, cj = pair
            self.history.append((min_dist, self.clusters[ci], self.clusters[cj]))
            new_cluster = self.clusters[ci] + self.clusters[cj]
            self.clusters = [self.clusters[i] for i in range(len(self.clusters)) if i != ci and i != cj]
            self.clusters.append(new_cluster)

    def _get_cluster_heights(self):
        n_samples = len(self.original_distances)
        Z = np.zeros((n_samples - 1, 4))
        
        cluster_dict = {i: [i] for i in range(n_samples)}
        current_id = n_samples
        
        for i, (height, left_cluster, right_cluster) in enumerate(self.history):
            left_id = None
            right_id = None
            
            for cluster_id, cluster in cluster_dict.items():
                if sorted(cluster) == sorted(left_cluster):
                    left_id = cluster_id
                if sorted(cluster) == sorted(right_cluster):
                    right_id = cluster_id
                
                if left_id is not None and right_id is not None:
                    break

            Z[i, 0] = left_id
            Z[i, 1] = right_id
            Z[i, 2] = height
            Z[i, 3] = len(left_cluster) + len(right_cluster)
            
            merged_cluster = left_cluster + right_cluster
            cluster_dict[current_id] = merged_cluster
            del cluster_dict[left_id]
            del cluster_dict[right_id]
            
            current_id += 1
            
        return Z

    def plot_dendrogram(self):
        from scipy.cluster.hierarchy import dendrogram
        
        Z = self._get_cluster_heights()
        
        plt.figure(figsize=(10, 7))
        dendrogram(Z)
        plt.title("Hierarchical Clustering Dendrogram")
        plt.xlabel("Sample index")
        plt.ylabel("Distance")
        plt.savefig("output.png")


if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(n_samples=30, centers=4, cluster_std=0.60, random_state=0)

    hc = HierarchicalCluster(linkage="single")
    hc.fit(X)
    hc.plot_dendrogram()