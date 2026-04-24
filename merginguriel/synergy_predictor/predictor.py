"""
Combined synergy predictor: correlation analysis + LOO-CV classifier.

Given feature vectors for targets with known merging effect,
finds which features predict synergy/interference and evaluates
a combined classifier.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler


def correlation_analysis(
    features_df: pd.DataFrame,
    target_col: str = "merging_effect",
) -> pd.DataFrame:
    """
    Compute Pearson and Spearman correlations between each feature
    and merging effect.

    Returns DataFrame sorted by absolute Spearman correlation.
    """
    feature_cols = [c for c in features_df.columns if c not in [target_col, "target", "is_synergy"]]
    results = []

    y = features_df[target_col].values
    for col in feature_cols:
        x = features_df[col].values
        if np.std(x) < 1e-10:
            continue
        pearson_r, pearson_p = stats.pearsonr(x, y)
        spearman_r, spearman_p = stats.spearmanr(x, y)
        results.append({
            "feature": col,
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "spearman_r": spearman_r,
            "spearman_p": spearman_p,
            "abs_spearman": abs(spearman_r),
        })

    df = pd.DataFrame(results).sort_values("abs_spearman", ascending=False)
    return df.reset_index(drop=True)


def single_feature_threshold_loo(
    features_df: pd.DataFrame,
    target_col: str = "merging_effect",
) -> pd.DataFrame:
    """
    For each feature, find the optimal threshold for binary classification
    (synergy vs interference) using leave-one-out cross-validation.

    Returns DataFrame with feature name, best direction, LOO accuracy.
    """
    feature_cols = [c for c in features_df.columns if c not in [target_col, "target", "is_synergy"]]
    y_binary = (features_df[target_col].values > 0).astype(int)
    results = []

    for col in feature_cols:
        x = features_df[col].values
        if np.std(x) < 1e-10:
            continue

        # Try both directions: higher = synergy, or lower = synergy
        for direction in ["higher", "lower"]:
            loo = LeaveOneOut()
            correct = 0
            for train_idx, test_idx in loo.split(x):
                x_train, y_train = x[train_idx], y_binary[train_idx]
                x_test = x[test_idx]

                # Find threshold that maximizes accuracy on train set
                thresholds = np.sort(np.unique(x_train))
                best_acc = 0
                best_thresh = thresholds[0]
                for t in thresholds:
                    if direction == "higher":
                        preds = (x_train > t).astype(int)
                    else:
                        preds = (x_train < t).astype(int)
                    acc = (preds == y_train).mean()
                    if acc > best_acc:
                        best_acc = acc
                        best_thresh = t

                # Predict on test
                if direction == "higher":
                    pred = int(x_test[0] > best_thresh)
                else:
                    pred = int(x_test[0] < best_thresh)
                correct += int(pred == y_binary[test_idx[0]])

            loo_acc = correct / len(y_binary)
            results.append({
                "feature": col,
                "direction": direction,
                "loo_accuracy": loo_acc,
                "loo_correct": correct,
                "loo_total": len(y_binary),
            })

    df = pd.DataFrame(results).sort_values("loo_accuracy", ascending=False)
    return df.reset_index(drop=True)


def logistic_regression_loo(
    features_df: pd.DataFrame,
    target_col: str = "merging_effect",
    feature_cols: list[str] | None = None,
    top_k: int = 3,
) -> dict[str, float]:
    """
    LOO-CV evaluation of logistic regression classifier.

    If feature_cols not specified, selects top-k by absolute Spearman correlation.

    Returns dict with accuracy, per-target predictions.
    """
    if feature_cols is None:
        corr_df = correlation_analysis(features_df, target_col)
        feature_cols = corr_df["feature"].head(top_k).tolist()

    y_binary = (features_df[target_col].values > 0).astype(int)
    X = features_df[feature_cols].values

    loo = LeaveOneOut()
    predictions = []
    probabilities = []

    for train_idx, test_idx in loo.split(X):
        X_train, y_train = X[train_idx], y_binary[train_idx]
        X_test = X[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        clf.fit(X_train_s, y_train)

        pred = clf.predict(X_test_s)[0]
        prob = clf.predict_proba(X_test_s)[0]

        predictions.append(pred)
        probabilities.append(prob)

    predictions = np.array(predictions)
    accuracy = (predictions == y_binary).mean()

    # Per-target results
    targets = features_df["target"].tolist()
    per_target = {}
    for i, target in enumerate(targets):
        per_target[target] = {
            "predicted": "SYNERGY" if predictions[i] == 1 else "INTERFERENCE",
            "actual": "SYNERGY" if y_binary[i] == 1 else "INTERFERENCE",
            "correct": predictions[i] == y_binary[i],
            "confidence": float(max(probabilities[i])),
        }

    return {
        "accuracy": float(accuracy),
        "correct": int((predictions == y_binary).sum()),
        "total": len(y_binary),
        "features_used": feature_cols,
        "per_target": per_target,
    }
