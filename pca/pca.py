# implementation of principal component analysis algorithm 

import numpy as np


def standardize(X):
    return (X - np.mean(X)) / np.std(X)

# compute co-variance matrix
def compute_covariance_matrix(X):
    n = len(X)
    
    X = (X.T @ X) / (n - 1)
    
    return X

def perform_eigen_decomposition(X):
    return np.linalg.eig(X)

def find_top_k_pc(eigenvalues, eigenvectors):
    idx = np.argsort(-eigenvalues)
    
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    return eigenvalues, eigenvectors

X = np.array([
    [23.4, 11.2, 7.5, 3.8, 19.1],
    [17.3, 8.5, 2.1, 5.6, 22.9],
    [30.8, 15.6, 9.2, 1.9, 18.5],
    [25.9, 12.8, 6.3, 4.2, 20.7],
    [19.2, 9.1, 3.9, 6.1, 24.3],
    [28.5, 14.2, 8.1, 2.6, 19.9],
    [22.1, 10.5, 5.7, 4.8, 21.5],
    [26.7, 13.4, 7.1, 3.2, 20.1],
    [20.6, 9.6, 4.3, 5.9, 23.2],
    [29.3, 16.1, 9.6, 1.4, 18.1]
])

X_std = standardize(X)

X = compute_covariance_matrix(X_std)

e_val, e_vec = perform_eigen_decomposition(X)

e_val, e_vec = find_top_k_pc(e_val, e_vec)

print(e_val)
print(e_vec)

print("cumulative explained variance:")

cum_exp_var = np.cumsum(e_val) / np.sum(e_val)


print(cum_exp_var)

k = np.argmax(cum_exp_var >= 0.95) + 1

print(f"Number of principal components needed to explain at least 95% of the variance: {k}")

pc_components = e_vec[:, :k]

transformed_data = X_std @ pc_components

print("Transformed Data:")
print(transformed_data)