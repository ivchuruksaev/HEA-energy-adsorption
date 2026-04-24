import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from reproducibility_utils import build_lq_feature_matrix, save_json, set_global_seed


def compute_metrics(y_true, y_pred):
    tau, _ = kendalltau(y_true, y_pred)

    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "kendall_tau": float(tau),
    }


def fit_predict_lq(X_train, y_train, X_eval):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_eval_scaled = scaler.transform(X_eval)

    alphas = np.logspace(-6, 3, 20)
    model = RidgeCV(alphas=alphas, cv=5)
    model.fit(X_train_scaled, y_train)

    return model.predict(X_eval_scaled), model.alpha_


def run_learning_curve(input_path, out_dir, split_dir, train_fractions, n_splits, seed):
    set_global_seed(seed)

    input_path = Path(input_path)
    out_dir = Path(out_dir)
    split_dir = Path(split_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    if "D_teacher" in df.columns:
        target_column = "D_teacher"
    else:
        target_column = "D"

    X, _ = build_lq_feature_matrix(df)
    y = df[target_column].to_numpy(dtype=float)

    records = []
    split_records = []
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fraction in train_fractions:
        train_size = max(1, int(round(len(df) * fraction)))

        for fold, (train_pool_idx, test_idx) in enumerate(kfold.split(X), start=1):
            fold_seed = seed + int(round(fraction * 1_000_000)) + fold
            rng = np.random.default_rng(fold_seed)
            selected_size = min(train_size, len(train_pool_idx))
            train_idx = rng.choice(train_pool_idx, selected_size, replace=False)

            y_pred_all, alpha = fit_predict_lq(X[train_idx], y[train_idx], X)
            metrics = compute_metrics(y, y_pred_all)

            record = {
                "model": "LQ",
                "train_fraction": float(fraction),
                "train_size": int(selected_size),
                "fold": int(fold),
                "seed": int(fold_seed),
                "alpha": float(alpha),
                **metrics,
            }

            records.append(record)

            split_records.append({
                "train_fraction": float(fraction),
                "train_size": int(selected_size),
                "fold": int(fold),
                "seed": int(fold_seed),
                "train_indices": [int(item) for item in train_idx],
                "test_indices": [int(item) for item in test_idx],
            })

    metrics_df = pd.DataFrame(records)
    metrics_path = out_dir / "benchmark_metrics.csv"
    summary_path = out_dir / "learning_curve_summary.csv"
    splits_path = split_dir / f"learning_curve_splits_seed{seed}.json"

    metrics_df.to_csv(metrics_path, index=False)

    summary_df = metrics_df.groupby(["model", "train_fraction", "train_size"], as_index=False).agg({
        "MAE": ["mean", "std"],
        "R2": ["mean", "std"],
        "kendall_tau": ["mean", "std"],
    })

    summary_df.columns = [
        "_".join(column).strip("_") if isinstance(column, tuple) else column
        for column in summary_df.columns
    ]

    summary_df.to_csv(summary_path, index=False)
    save_json(split_records, splits_path)

    return metrics_path, summary_path, splits_path


def parse_fractions(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/descriptor_table.csv")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--split-dir", default="splits")
    parser.add_argument("--train-fractions", default="0.005,0.01,0.015,0.02,0.03,0.05,0.1")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = run_learning_curve(
        input_path=args.input,
        out_dir=args.out_dir,
        split_dir=args.split_dir,
        train_fractions=parse_fractions(args.train_fractions),
        n_splits=args.n_splits,
        seed=args.seed,
    )

    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
