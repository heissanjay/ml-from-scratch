import numpy as np
import pandas as pd

from collections import Counter




def load_data(file_name):
    data = pd.read_csv(file_name)
    return data


def normalize_values(X):
    # Z-Score Normalization (Standardization)

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_normalized = (X - mean) / std
    return X_normalized

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))


def get_neighbors(X_train, y_train, x_test, K):
    distances = []
    for i in range(len(X_train)):
        dist = euclidean_distance(x_test, X_train[i])
        distances.append((dist, y_train[i]))

    distances.sort(key=lambda x: x[0])
    neighbors = distances[:K]
    return neighbors

def predict(X_train, y_train, x_test, K):
    # get all the neighbors
    neighbors = get_neighbors(X_train, y_train, x_test, K)
    labels = [ neighbor[1] for neighbor in neighbors ]
    most_common = Counter(labels).most_common(1)
    return most_common[0][0]


def evaluate(X_train, y_train, X_test, y_test, K):
    predictions = []

    for x_test in X_test:
        pred = predict(X_train, y_train, x_test, K)
        predictions.append(pred)
    accuracy = np.sum(predictions == y_test) / len(y_test)
    return accuracy


if __name__ == "__main__":

    file_path = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    df = load_data(file_path)

    column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
    df.columns = column_names

    X = df.iloc[:,:-1].values
    y = df.iloc[:,-1].values


    X = normalize_values(X)

    train_size = int(0.8 * len(X))

    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    K = 3

    accuracy = evaluate(X_train, y_train, X_test, y_test, K)
    print(f"accuracy: {accuracy * 100:.2f}%")