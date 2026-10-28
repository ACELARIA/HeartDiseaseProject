#!/usr/bin/env python3
"""
HC-AAL heart disease experiment
Implementation based on the paper:
"Heterogeneous Committee-Based Adaptive Active Learning for Efficient Heart Disease Prediction"

Paper-described components:
- XGBoost, Random Forest, Logistic Regression (scaled LR pipeline)
- CW-QBC uncertainty score
- Select top 2q uncertain candidates, then K-Means with k=q
- Select the candidate nearest each cluster centroid
- Initial labeled set of 15; 78 iterations; q cycles through 3, 4, 5
- XGBoost is used as the final classifier, as described in the paper

Important: the paper does not fully specify the split, all model hyperparameters,
or the exact adaptive query schedule. Therefore exact reported metrics are not
guaranteed. Do not tune against the test set.
"""

import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

try:
    from xgboost import XGBClassifier
except ImportError as exc:
    raise SystemExit(
        "xgboost is not installed. Install dependencies with:\n"
        "python3 -m pip install xgboost scikit-learn pandas numpy matplotlib"
    ) from exc

warnings.filterwarnings("ignore", category=UserWarning)

# -------------------------- CONFIGURATION --------------------------
DATA_PATH = os.environ.get("HEART_CSV", "heart.csv")
TARGET_COLUMN = "target"
N_RUNS = 10
INITIAL_LABELLED = 15
N_ITERATIONS = 78
QUERY_SIZES = [3, 4, 5]  # paper uses list(range(3, 6))
TEST_SIZE = 0.20
BASE_SEED = 42
OUTPUT_DIR = Path("hc_aal_results")
EPSILON = 1e-10


def load_dataset(path):
    """Load CSV, normalize common target-column names, and validate it."""
    csv_path = Path(path)
    if not csv_path.exists():
        candidates = [
            Path("data/heart.csv"),
            Path("Heart.csv"),
            Path("heart_disease.csv"),
            Path("HeartDisease.csv"),
        ]
        found = next((p for p in candidates if p.exists()), None)
        if found is None:
            raise FileNotFoundError(
                f"Could not find '{path}'. Put heart.csv in {Path.cwd()} "
                "or set HEART_CSV=/full/path/to/heart.csv."
            )
        csv_path = found

    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]

    # Common aliases for the label column.
    target_aliases = ["target", "heart_disease", "HeartDisease", "output", "condition"]
    target = next((c for c in target_aliases if c in df.columns), None)
    if target is None:
        raise ValueError(
            f"Could not identify target column. Columns found: {list(df.columns)}"
        )

    if target != TARGET_COLUMN:
        df = df.rename(columns={target: TARGET_COLUMN})

    df = df.dropna().copy()
    y = df[TARGET_COLUMN].astype(int)
    X = df.drop(columns=[TARGET_COLUMN])

    # This paper's dataset has numeric clinical attributes.
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise ValueError(
            f"Non-numeric feature columns found: {non_numeric}. "
            "Encode these columns before running this implementation."
        )

    unique_labels = sorted(y.unique().tolist())
    if unique_labels != [0, 1]:
        raise ValueError(f"Expected binary target values 0 and 1; found {unique_labels}")

    return X.reset_index(drop=True), y.reset_index(drop=True), csv_path


def make_committee(seed):
    """Create the three heterogeneous committee members."""
    xgb = XGBClassifier(
        n_estimators=180,
        max_depth=3,
        learning_rate=0.04,
        subsample=0.90,
        colsample_bytree=0.90,
        min_child_weight=1,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        n_jobs=-1,
        verbosity=0,
    )

    rf = RandomForestClassifier(
        n_estimators=250,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight=None,
        random_state=seed,
        n_jobs=-1,
    )

    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=1.0,
            max_iter=3000,
            solver="liblinear",
            random_state=seed,
        )),
    ])

    return [("XGBoost", xgb), ("Random Forest", rf), ("Logistic Regression", lr)]


def fit_committee(X_labeled, y_labeled, seed):
    """Fit committee, handling tiny initial sets that may contain one class."""
    models = make_committee(seed)
    # Stratified initialization usually gives both classes, but guard against
    # unusual random samples where the initial 15 contain only one class.
    if y_labeled.nunique() < 2:
        raise RuntimeError(
            "The initial labeled set contains only one class. "
            "Use the stratified initialization in run_one_experiment."
        )
    for _, model in models:
        model.fit(X_labeled, y_labeled)
    return models


def cw_qbc_scores(models, X_unlabeled):
    """
    Paper's CW-QBC score:
      sum_m p_m(x) * log((p_m(x)+epsilon)/(mean_p(x)+epsilon))
    where p_m is each model's positive-class probability.
    """
    probabilities = np.vstack([
        model.predict_proba(X_unlabeled)[:, 1] for _, model in models
    ])
    mean_probability = probabilities.mean(axis=0)
    scores = np.sum(
        probabilities
        * np.log((probabilities + EPSILON) / (mean_probability + EPSILON)),
        axis=0,
    )
    return scores


def select_by_uncertainty_and_kmeans(X_unlabeled, scores, q, seed):
    """Choose top 2q uncertain samples, then one centroid-nearest sample per cluster."""
    n_pool = len(X_unlabeled)
    if n_pool == 0:
        return np.array([], dtype=int)

    q = min(q, n_pool)
    candidate_count = min(2 * q, n_pool)
    candidate_local_positions = np.argsort(scores)[-candidate_count:]
    X_candidates = X_unlabeled.iloc[candidate_local_positions].to_numpy(dtype=float)

    # K-Means is scale-sensitive, so standardize only the clustering space.
    cluster_scaler = StandardScaler()
    X_scaled = cluster_scaler.fit_transform(X_candidates)

    n_clusters = min(q, len(X_candidates))
    if n_clusters == 1:
        chosen_candidate_positions = [int(np.argmin(
            np.sum((X_scaled - X_scaled.mean(axis=0)) ** 2, axis=1)
        ))]
    else:
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=seed,
            n_init=10,
        )
        cluster_labels = kmeans.fit_predict(X_scaled)
        chosen_candidate_positions = []

        for cluster_id in range(n_clusters):
            members = np.flatnonzero(cluster_labels == cluster_id)
            if len(members) == 0:
                continue
            centroid = kmeans.cluster_centers_[cluster_id]
            distances = np.sum((X_scaled[members] - centroid) ** 2, axis=1)
            chosen_candidate_positions.append(int(members[np.argmin(distances)]))

        # Defensive fallback in case an empty cluster is returned.
        for candidate_position in np.argsort(scores[candidate_local_positions])[::-1]:
            if len(chosen_candidate_positions) >= q:
                break
            candidate_position = int(candidate_position)
            if candidate_position not in chosen_candidate_positions:
                chosen_candidate_positions.append(candidate_position)

    selected_pool_positions = candidate_local_positions[
        np.asarray(chosen_candidate_positions[:q], dtype=int)
    ]
    return np.unique(selected_pool_positions)


def calculate_metrics(y_true, predictions, probabilities):
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "auc": roc_auc_score(y_true, probabilities),
    }


def run_one_experiment(X_train, y_train, X_test, y_test, run_number):
    seed = BASE_SEED + run_number * 100
    rng = np.random.default_rng(seed)

    # Stratified initial labeled set ensures both target classes are represented.
    all_positions = np.arange(len(X_train))
    initial_positions, pool_positions = train_test_split(
        all_positions,
        train_size=min(INITIAL_LABELLED, len(X_train) - 1),
        random_state=seed,
        stratify=y_train,
    )
    labeled_positions = list(map(int, initial_positions))
    unlabeled_positions = list(map(int, pool_positions))

    history = []
    print(f"\nRunning experiment {run_number + 1}/{N_RUNS} ...")

    for iteration in range(N_ITERATIONS + 1):
        X_labeled = X_train.iloc[labeled_positions]
        y_labeled = y_train.iloc[labeled_positions]

        # Evaluation classifier follows the paper: final learner is XGBoost.
        final_xgb = make_committee(seed + iteration)[0][1]
        final_xgb.fit(X_labeled, y_labeled)
        test_prob = final_xgb.predict_proba(X_test)[:, 1]
        test_pred = (test_prob >= 0.5).astype(int)

        metrics = calculate_metrics(y_test, test_pred, test_prob)
        metrics.update({
            "run": run_number + 1,
            "iteration": iteration,
            "labeled_train_samples": len(labeled_positions),
            "labeled_pct_of_train": 100.0 * len(labeled_positions) / len(X_train),
            "labeled_pct_of_full_dataset": 100.0 * len(labeled_positions) / (
                len(X_train) + len(X_test)
            ),
        })
        history.append(metrics)

        if iteration == N_ITERATIONS or not unlabeled_positions:
            break

        committee = fit_committee(X_labeled, y_labeled, seed + iteration)
        X_pool = X_train.iloc[unlabeled_positions]
        scores = cw_qbc_scores(committee, X_pool)

        q = QUERY_SIZES[iteration % len(QUERY_SIZES)]
        selected_local = select_by_uncertainty_and_kmeans(
            X_pool, scores, q, seed + iteration
        )
        if len(selected_local) == 0:
            break

        newly_labeled = [unlabeled_positions[int(i)] for i in selected_local]
        selected_set = set(newly_labeled)
        labeled_positions.extend(newly_labeled)
        unlabeled_positions = [
            pos for pos in unlabeled_positions if pos not in selected_set
        ]

        if (iteration + 1) % 10 == 0 or iteration < 3:
            print(
                f"  Iteration {iteration + 1:02d}/{N_ITERATIONS}: "
                f"Accuracy={metrics['accuracy']:.4f} | "
                f"F1={metrics['f1']:.4f} | AUC={metrics['auc']:.4f} | "
                f"Labelled={len(labeled_positions)}"
            )

    history_df = pd.DataFrame(history)
    final = history_df.iloc[-1].to_dict()
    print(
        f"  Final: Accuracy={final['accuracy']:.4f} | "
        f"Precision={final['precision']:.4f} | "
        f"Recall={final['recall']:.4f} | F1={final['f1']:.4f} | "
        f"AUC={final['auc']:.4f} | Labelled={int(final['labeled_train_samples'])}"
    )
    return history_df, final


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    X, y, csv_path = load_dataset(DATA_PATH)

    print("=" * 68)
    print("HC-AAL HEART DISEASE EXPERIMENT")
    print("=" * 68)
    print(f"Dataset file: {csv_path}")
    print(f"Dataset shape: {X.shape[0], X.shape[1] + 1}")
    print(f"Feature count: {X.shape[1]}")
    print(f"Target column: {TARGET_COLUMN}")
    print("Target distribution:")
    print(y.value_counts().sort_index().rename("target").to_string())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=BASE_SEED,
        stratify=y,
    )
    print(f"\nTrain rows: {len(X_train)} | Held-out test rows: {len(X_test)}")
    print(
        f"Initial labelled set: {INITIAL_LABELLED}; "
        f"iterations: {N_ITERATIONS}; query sizes: {QUERY_SIZES}"
    )

    all_histories = []
    final_rows = []

    for run in range(N_RUNS):
        history, final = run_one_experiment(
            X_train, y_train, X_test, y_test, run
        )
        all_histories.append(history)
        final_rows.append(final)

    all_iterations_df = pd.concat(all_histories, ignore_index=True)
    final_results_df = pd.DataFrame(final_rows)

    metric_columns = [
        "accuracy", "precision", "recall", "f1", "auc",
        "labeled_pct_of_train", "labeled_pct_of_full_dataset",
    ]
    summary = pd.DataFrame({
        "metric": metric_columns,
        "mean": [final_results_df[c].mean() for c in metric_columns],
        "std": [final_results_df[c].std(ddof=1) for c in metric_columns],
    })

    all_iterations_df.to_csv(OUTPUT_DIR / "all_iteration_results.csv", index=False)
    final_results_df.to_csv(OUTPUT_DIR / "final_results_each_run.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "summary_mean_std.csv", index=False)

    mean_accuracy_by_iteration = (
        all_iterations_df.groupby("iteration", as_index=False)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_labeled_pct=("labeled_pct_of_full_dataset", "mean"),
        )
    )
    mean_accuracy_by_iteration.to_csv(
        OUTPUT_DIR / "mean_accuracy_by_iteration.csv", index=False
    )

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(
        mean_accuracy_by_iteration["mean_labeled_pct"],
        mean_accuracy_by_iteration["mean_accuracy"],
        marker="o",
        markersize=3,
        linewidth=1.5,
        label="Mean test accuracy",
    )
    ax1.set_xlabel("Labelled data (% of full dataset)")
    ax1.set_ylabel("Mean held-out test accuracy")
    ax1.set_title("HC-AAL: Accuracy vs. Labelled Data")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "accuracy_vs_labeled_data.png", dpi=200)
    plt.close(fig)

    print("\n" + "=" * 68)
    print("FINAL RESULTS: MEAN OVER 10 RUNS")
    print("=" * 68)
    for _, row in summary.iterrows():
        print(f"{row['metric']:32s}: {row['mean']:.4f} +/- {row['std']:.4f}")

    print("\nFiles saved in:", OUTPUT_DIR.resolve())
    print(" - all_iteration_results.csv")
    print(" - final_results_each_run.csv")
    print(" - summary_mean_std.csv")
    print(" - mean_accuracy_by_iteration.csv")
    print(" - accuracy_vs_labeled_data.png")
    print(
        "\nNote: Paper reports 96.04% accuracy and 31.9% labeled data. "
        "This implementation follows the described method, but exact metrics "
        "may differ because the paper does not fully specify every detail."
    )


if __name__ == "__main__":
    main()
