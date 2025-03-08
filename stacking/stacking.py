import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error


X, y = make_regression(n_samples=1000, n_features=10, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


base_models = [
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42)),
    ('svr', SVR(kernel='rbf'))
]


meta_model = LinearRegression()

def get_meta_features(base_models, X_train, y_train, X_test, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    meta_train = np.zeros((X_train.shape[0], len(base_models)))
    meta_test = np.zeros((X_test.shape[0], len(base_models)))

    for i, (name, model) in enumerate(base_models):
        test_folds_preds = []

        for train_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            model.fit(X_tr, y_tr)

            meta_train[val_idx, i] =  model.predict(X_val)

            test_folds_preds.append(model.predict(X_test))
            
        meta_test[:, i] = np.mean(test_folds_preds, axis=0)

    return meta_train, meta_test

meta_train, meta_test = get_meta_features(base_models, X_train, y_train, X_test)
meta_model.fit(meta_train, y_train)
y_pred_meta = meta_model.predict(meta_test)

mse = mean_squared_error(y_test, y_pred_meta)
print(f"Stacking MSE: {mse:.4f}")