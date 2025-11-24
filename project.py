import numpy as np
import time
from dataclasses import dataclass


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


@dataclass
class TreeNode:
    feature_index: int = None
    threshold: float = None
    left: 'TreeNode' = None
    right: 'TreeNode' = None
    value: float = None  # leaf value

class DecisionTreeRegressorScratch:
    """
    Very simple CART-style regression tree:
    - Uses variance reduction (MSE) as split criterion
    - Only numeric features
    """
    def __init__(self, max_depth=3, min_samples_split=10):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.root = self._build_tree(X, y, depth=0)

    def predict(self, X):
        X = np.asarray(X)
        return np.array([self._predict_row(row, self.root) for row in X])

    def _build_tree(self, X, y, depth):
        num_samples, num_features = X.shape

        # Stopping conditions
        if (depth >= self.max_depth or
            num_samples < self.min_samples_split or
            np.var(y) < 1e-8):
            leaf_value = float(np.mean(y))
            return TreeNode(value=leaf_value)

        best_feature, best_threshold, best_mse = None, None, float("inf")
        best_left_idx, best_right_idx = None, None

        # Try all features and candidate thresholds
        for feature_idx in range(num_features):
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)
            if unique_values.shape[0] == 1:
                continue

            # Use midpoints between sorted unique values as thresholds
            sorted_vals = np.sort(unique_values)
            candidate_thresholds = (sorted_vals[:-1] + sorted_vals[1:]) / 2.0

            for threshold in candidate_thresholds:
                left_idx = feature_values <= threshold
                right_idx = feature_values > threshold

                if left_idx.sum() == 0 or right_idx.sum() == 0:
                    continue

                y_left, y_right = y[left_idx], y[right_idx]

                # Weighted MSE
                mse_left = np.var(y_left) * len(y_left)
                mse_right = np.var(y_right) * len(y_right)
                mse_total = (mse_left + mse_right) / num_samples

                if mse_total < best_mse:
                    best_mse = mse_total
                    best_feature = feature_idx
                    best_threshold = threshold
                    best_left_idx = left_idx
                    best_right_idx = right_idx

        # If no useful split found -> leaf
        if best_feature is None:
            leaf_value = float(np.mean(y))
            return TreeNode(value=leaf_value)

        # Recurse
        left_node = self._build_tree(X[best_left_idx], y[best_left_idx], depth + 1)
        right_node = self._build_tree(X[best_right_idx], y[best_right_idx], depth + 1)
        return TreeNode(feature_index=best_feature,
                        threshold=best_threshold,
                        left=left_node,
                        right=right_node)

    def _predict_row(self, row, node):
        if node.value is not None:
            return node.value
        if row[node.feature_index] <= node.threshold:
            return self._predict_row(row, node.left)
        else:
            return self._predict_row(row, node.right)


class GBMRegressorScratch:
    """
    Gradient Boosting for regression with squared error loss.
    Uses DecisionTreeRegressorScratch as weak learner.
    """
    def __init__(self,
                 n_estimators=100,
                 learning_rate=0.1,
                 max_depth=3,
                 min_samples_split=10):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.init_prediction_ = None
        self.trees_ = []

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        # F0(x) = mean(y)
        self.init_prediction_ = float(np.mean(y))
        self.trees_ = []

        # Current model predictions
        current_pred = np.full_like(y, self.init_prediction_, dtype=float)

        for m in range(self.n_estimators):
            # For MSE, negative gradient = (y - F_{m-1}(x))
            residuals = y - current_pred

            tree = DecisionTreeRegressorScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            tree.fit(X, residuals)

            update = tree.predict(X)

            # F_m(x) = F_{m-1}(x) + learning_rate * tree_m(x)
            current_pred = current_pred + self.learning_rate * update
            self.trees_.append(tree)

    def predict(self, X):
        X = np.asarray(X)
        n_samples = X.shape[0]

        preds = np.full(n_samples, self.init_prediction_, dtype=float)
        for tree in self.trees_:
            preds += self.learning_rate * tree.predict(X)
        return preds

def kfold_indices(n_samples, n_splits=3, random_state=42):
    rng = np.random.RandomState(random_state)
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
    fold_sizes[: n_samples % n_splits] += 1

    current = 0
    folds = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        val_idx = indices[start:stop]
        train_idx = np.concatenate([indices[:start], indices[stop:]])
        folds.append((train_idx, val_idx))
        current = stop
    return folds

def cross_val_gbm(X, y, n_estimators, learning_rate, max_depth,
                  min_samples_split=10, n_splits=3):
    X = np.asarray(X)
    y = np.asarray(y)
    folds = kfold_indices(len(y), n_splits=n_splits)

    rmse_scores = []
    r2_scores = []

    for train_idx, val_idx in folds:
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = GBMRegressorScratch(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=min_samples_split
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        rmse_scores.append(rmse(y_val, y_pred))
        r2_scores.append(r2_score(y_val, y_pred))

    return float(np.mean(rmse_scores)), float(np.mean(r2_scores))

def grid_search_gbm(X, y,
                    learning_rates,
                    n_estimators_list,
                    max_depth_list,
                    min_samples_split=10,
                    n_splits=3):
    best_params = None
    best_rmse = float("inf")
    history = []

    for lr in learning_rates:
        for n_est in n_estimators_list:
            for depth in max_depth_list:
                avg_rmse, avg_r2 = cross_val_gbm(
                    X, y,
                    n_estimators=n_est,
                    learning_rate=lr,
                    max_depth=depth,
                    min_samples_split=min_samples_split,
                    n_splits=n_splits
                )
                history.append({
                    "learning_rate": lr,
                    "n_estimators": n_est,
                    "max_depth": depth,
                    "cv_rmse": avg_rmse,
                    "cv_r2": avg_r2
                })
                print(f"[CV] lr={lr}, n_estimators={n_est}, max_depth={depth} "
                      f"-> RMSE={avg_rmse:.4f}, R2={avg_r2:.4f}")
                if avg_rmse < best_rmse:
                    best_rmse = avg_rmse
                    best_params = {
                        "learning_rate": lr,
                        "n_estimators": n_est,
                        "max_depth": depth
                    }
    return best_params, history

def main():
    # 1. Load dataset
    from sklearn.datasets import fetch_california_housing
    from sklearn.ensemble import GradientBoostingRegressor

    data = fetch_california_housing()
    X = data.data
    y = data.target

    # 2. Train-test split (80/20)
    rng = np.random.RandomState(42)
    indices = np.arange(X.shape[0])
    rng.shuffle(indices)
    train_size = int(0.8 * len(indices))
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # 3. Manual standardization (no sklearn scalers)
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train_std = (X_train - mean) / std
    X_test_std = (X_test - mean) / std

    # 4. Hyperparameter tuning for custom GBM (via CV)
    learning_rates = [0.05, 0.1, 0.2]
    n_estimators_list = [50, 100]
    max_depth_list = [2, 3]

    print("=== Hyperparameter Tuning (Custom GBM) ===")
    best_params, cv_history = grid_search_gbm(
        X_train_std, y_train,
        learning_rates=learning_rates,
        n_estimators_list=n_estimators_list,
        max_depth_list=max_depth_list,
        min_samples_split=20,
        n_splits=3
    )
    print("\nBest params (based on CV RMSE):", best_params)

    # 5. Train final custom GBM with best hyperparameters
    gbm_custom = GBMRegressorScratch(
        n_estimators=best_params["n_estimators"],
        learning_rate=best_params["learning_rate"],
        max_depth=best_params["max_depth"],
        min_samples_split=20
    )

    start_time = time.perf_counter()
    gbm_custom.fit(X_train_std, y_train)
    training_time_custom = time.perf_counter() - start_time

    start_time = time.perf_counter()
    y_pred_custom = gbm_custom.predict(X_test_std)
    prediction_time_custom = time.perf_counter() - start_time

    rmse_custom = rmse(y_test, y_pred_custom)
    r2_custom = r2_score(y_test, y_pred_custom)

    # 6. Train production-grade GBM (sklearn)
    gbm_sklearn = GradientBoostingRegressor(
        n_estimators=best_params["n_estimators"],
        learning_rate=best_params["learning_rate"],
        max_depth=best_params["max_depth"],
        random_state=42
    )

    start_time = time.perf_counter()
    gbm_sklearn.fit(X_train_std, y_train)
    training_time_sklearn = time.perf_counter() - start_time

    start_time = time.perf_counter()
    y_pred_sklearn = gbm_sklearn.predict(X_test_std)
    prediction_time_sklearn = time.perf_counter() - start_time

    rmse_sklearn = rmse(y_test, y_pred_sklearn)
    r2_sklearn = r2_score(y_test, y_pred_sklearn)

    # 7. Summary
    print("\n=== Test Set Performance ===")
    print("Custom GBM:")
    print(f"  RMSE: {rmse_custom:.4f}")
    print(f"  R2  : {r2_custom:.4f}")
    print(f"  Training time   : {training_time_custom:.4f} seconds")
    print(f"  Prediction time : {prediction_time_custom:.6f} seconds")

    print("\nSklearn GradientBoostingRegressor:")
    print(f"  RMSE: {rmse_sklearn:.4f}")
    print(f"  R2  : {r2_sklearn:.4f}")
    print(f"  Training time   : {training_time_sklearn:.4f} seconds")
    print(f"  Prediction time : {prediction_time_sklearn:.6f} seconds")

    # Small text conclusion
    improvement = rmse_custom - rmse_sklearn
    print("\nRMSE difference (custom - sklearn):", f"{improvement:.4f}")

if __name__ == "__main__":
    main()
