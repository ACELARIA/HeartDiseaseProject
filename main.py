
#!/usr/bin/env python3
"""
HC-AAL Heart Disease Experiment
Based on:
"Heterogeneous Committee-Based Adaptive Active Learning for Efficient
Heart Disease Prediction"

Implementation:
- Committee: XGBoost, Random Forest, Logistic Regression
- CW-QBC uncertainty scoring
- Select top 2q uncertain candidates
- K-Means clustering, k=q
- Select the candidate closest to each centroid
- Initially label 15 samples
- Run 78 active-learning iterations
- Cycle query sizes 3, 4, 5
- XGBoost is the final classifier

Evaluation:
- Fixed, stratified 80/20 train/test split
- Test samples are never added to the labeled pool
- Report held-out test metrics
- Report labeled percentages relative to training and full dataset

The paper does not fully specify all model parameters, the split protocol,
or the exact adaptive query schedule. Exact paper metrics are not guaranteed.
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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

# The paper provides the range {3, 4, 5}, but not an exact adaptive formula.
QUERY_SIZES = [3, 4, 5]

TEST_SIZE = 0.20
BASE_SEED = 42
EPSILON = 1e-10

OUTPUT_DIR = Path("hc_aal_reproduction_results")


# -------------------------- DATA LOADING --------------------------

def load_dataset(path):
    """Load and validate the heart disease CSV."""

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
                f"Could not find '{path}'. Put heart.csv in "
                f"{Path.cwd()} or set HEART_CSV=/full/path/to/heart.csv."
            )

        csv_path = found

    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]

    target_aliases = [
        "target",
        "heart_disease",
        "HeartDisease",
        "output",
        "condition",
    ]

    target = next(
        (c for c in target_aliases if c in df.columns),
        None,
    )

    if target is None:
        raise ValueError(
            "Could not identify the target column. "
            f"Columns found: {list(df.columns)}"
        )

    if target != TARGET_COLUMN:
        df = df.rename(columns={target: TARGET_COLUMN})

    df = df.dropna().copy()

    y = df[TARGET_COLUMN].astype(int)
    X = df.drop(columns=[TARGET_COLUMN])

    non_numeric = X.select_dtypes(
        exclude=[np.number]
    ).columns.tolist()

    if non_numeric:
        raise ValueError(
            f"Non-numeric feature columns found: {non_numeric}. "
            "Encode these columns before running this implementation."
        )

    unique_labels = sorted(y.unique().tolist())

    if unique_labels != [0, 1]:
        raise ValueError(
            f"Expected binary target values 0 and 1; found {unique_labels}"
        )

    return (
        X.reset_index(drop=True),
        y.reset_index(drop=True),
        csv_path,
    )


# -------------------------- MODEL COMMITTEE --------------------------

def make_committee(seed=42):
    """Create the three heterogeneous committee members."""

    # Keep model random state fixed for reproducibility.
    model_seed = 42

    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=model_seed,
        n_jobs=-1,
        verbosity=0,
    )

    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=model_seed,
        n_jobs=-1,
    )

    lr = Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=model_seed,
            ),
        ),
    ])

    return [
        ("XGBoost", xgb),
        ("Random Forest", rf),
        ("Logistic Regression", lr),
    ]


def fit_committee(X_labeled, y_labeled, seed):
    """Fit each committee model on the currently labeled samples."""

    if y_labeled.nunique() < 2:
        raise RuntimeError(
            "The labeled set contains only one class. "
            "Both classes are required to fit the committee."
        )

    models = make_committee(seed)

    for _, model in models:
        model.fit(X_labeled, y_labeled)

    return models


# -------------------------- CW-QBC SCORING --------------------------

def cw_qbc_scores(models, X_unlabeled):
    """
    Compute the CW-QBC score specified in the paper:

        CW_score(x) = sum_m p_m(x) *
                      log((p_m(x) + epsilon) /
                          (mean_p(x) + epsilon))

    p_m(x) is each model's positive-class probability.
    """

    probabilities = np.vstack([
        model.predict_proba(X_unlabeled)[:, 1]
        for _, model in models
    ])

    mean_probability = probabilities.mean(axis=0)

    scores = np.sum(
        probabilities
        * np.log(
            (probabilities + EPSILON)
            / (mean_probability + EPSILON)
        ),
        axis=0,
    )

    return scores


# -------------------------- ACTIVE SAMPLE SELECTION --------------------------

def select_by_uncertainty_and_kmeans(
    X_unlabeled,
    scores,
    q,
    seed,
):
    """
    Select the top 2q uncertainty candidates, cluster them into q groups,
    and select the candidate nearest each cluster centroid.
    """

    n_pool = len(X_unlabeled)

    if n_pool == 0:
        return np.array([], dtype=int)

    q = min(int(q), n_pool)
    candidate_count = min(2 * q, n_pool)

    # Descending order: highest CW-QBC scores first.
    candidate_local_positions = np.argsort(scores)[
        -candidate_count:
    ][::-1]

    X_candidates = X_unlabeled.iloc[
        candidate_local_positions
    ].to_numpy(dtype=float)

    n_clusters = min(q, len(X_candidates))

    kmeans = KMeans(
        n_clusters=n_clusters,
        n_init=10,
        random_state=42,
    )

    cluster_labels = kmeans.fit_predict(X_candidates)

    chosen_candidate_positions = []

    for cluster_id in range(n_clusters):
        members = np.flatnonzero(
            cluster_labels == cluster_id
        )

        if len(members) == 0:
            continue

        centroid = kmeans.cluster_centers_[cluster_id]

        distances = np.linalg.norm(
            X_candidates[members] - centroid,
            axis=1,
        )

        nearest_member = members[np.argmin(distances)]

        chosen_candidate_positions.append(
            int(nearest_member)
        )

    # Defensive fallback if K-Means yields fewer than q non-empty clusters.
    chosen_set = set(chosen_candidate_positions)

    for pos in range(candidate_count):
        if len(chosen_candidate_positions) >= q:
            break

        if pos not in chosen_set:
            chosen_candidate_positions.append(pos)
            chosen_set.add(pos)

    selected_pool_positions = candidate_local_positions[
        np.asarray(chosen_candidate_positions[:q], dtype=int)
    ]

    return np.unique(selected_pool_positions)


# -------------------------- EVALUATION METRICS --------------------------

def calculate_metrics(y_true, predictions, probabilities):
    """Calculate binary classification metrics."""

    return {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "auc": roc_auc_score(
            y_true,
            probabilities,
        ),
    }


# -------------------------- ONE EXPERIMENT --------------------------

def run_one_experiment(
    X_train,
    y_train,
    X_test,
    y_test,
    run_number,
):
    """Run one active-learning experiment using a fixed held-out test set."""

    seed = BASE_SEED + run_number * 100
    rng = np.random.default_rng(seed)

    all_positions = np.arange(len(X_train))
    initial_count = min(INITIAL_LABELLED, len(X_train) - 1)

    # Randomly initialize the labeled set, ensuring both classes are present.
    for _ in range(100):
        permutation = rng.permutation(all_positions)
        initial_positions = permutation[:initial_count]

        if y_train.iloc[initial_positions].nunique() == 2:
            break
    else:
        initial_positions, _ = train_test_split(
            all_positions,
            train_size=initial_count,
            random_state=seed,
            stratify=y_train,
        )

        initial_positions = np.asarray(initial_positions)

    initial_set = set(map(int, initial_positions))

    labeled_positions = list(map(int, initial_positions))

    unlabeled_positions = [
        int(i)
        for i in all_positions
        if int(i) not in initial_set
    ]

    history = []

    print(
        f"\nRunning experiment {run_number + 1}/{N_RUNS} ..."
    )

    for iteration in range(N_ITERATIONS + 1):

        X_labeled = X_train.iloc[labeled_positions]
        y_labeled = y_train.iloc[labeled_positions]

        # XGBoost is the final classifier.
        final_xgb = make_committee(seed + iteration)[0][1]

        final_xgb.fit(X_labeled, y_labeled)

        # The test set is used only for evaluation.
        # It is never used for sample selection or model fitting.
        test_prob = final_xgb.predict_proba(X_test)[:, 1]
        test_pred = (test_prob >= 0.60).astype(int)

        metrics = calculate_metrics(
            y_test,
            test_pred,
            test_prob,
        )

        labeled_count = len(labeled_positions)

        metrics.update({
            "run": run_number + 1,
            "iteration": iteration,
            "labeled_train_samples": labeled_count,
            "labeled_pct_of_train": (
                100.0 * labeled_count / len(X_train)
            ),
            "labeled_pct_of_full_dataset": (
                100.0 * labeled_count / (len(X_train) + len(X_test))
            ),
            "unlabeled_samples": len(unlabeled_positions),
        })

        history.append(metrics)

        if iteration == N_ITERATIONS or not unlabeled_positions:
            break

        # Fit the heterogeneous committee on labeled training samples.
        committee = fit_committee(
            X_labeled,
            y_labeled,
            seed + iteration,
        )

        X_pool = X_train.iloc[unlabeled_positions]

        scores = cw_qbc_scores(
            committee,
            X_pool,
        )

        # Reproducible cycle: q = 3, then 4, then 5.
        q = QUERY_SIZES[iteration % len(QUERY_SIZES)]

        selected_local = select_by_uncertainty_and_kmeans(
            X_pool,
            scores,
            q,
            seed + iteration,
        )

        if len(selected_local) == 0:
            break

        newly_labeled = [
            unlabeled_positions[int(i)]
            for i in selected_local
        ]

        selected_set = set(newly_labeled)

        labeled_positions.extend(newly_labeled)

        unlabeled_positions = [
            pos
            for pos in unlabeled_positions
            if pos not in selected_set
        ]

        if (iteration + 1) % 10 == 0 or iteration < 3:
            print(
                f"  Iteration {iteration + 1:02d}/{N_ITERATIONS}: "
                f"Test accuracy={metrics['accuracy']:.4f} | "
                f"F1={metrics['f1']:.4f} | "
                f"AUC={metrics['auc']:.4f} | "
                f"Labeled={len(labeled_positions)}"
            )

    history_df = pd.DataFrame(history)
    final = history_df.iloc[-1].to_dict()

    print(
        f"  Final: Accuracy={final['accuracy']:.4f} | "
        f"Precision={final['precision']:.4f} | "
        f"Recall={final['recall']:.4f} | "
        f"F1={final['f1']:.4f} | "
        f"AUC={final['auc']:.4f} | "
        f"Labeled={int(final['labeled_train_samples'])}"
    )

    return history_df, final


# -------------------------- MAIN PROGRAM --------------------------

def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    X, y, csv_path = load_dataset(DATA_PATH)

    print("=" * 68)
    print("HC-AAL HEART DISEASE EXPERIMENT")
    print("=" * 68)
    print(f"Dataset file: {csv_path}")
    print(f"Dataset shape: {(X.shape[0], X.shape[1] + 1)}")
    print(f"Feature count: {X.shape[1]}")
    print(f"Target column: {TARGET_COLUMN}")
    print("\nTarget distribution:")
    print(y.value_counts().sort_index().rename("target").to_string())

    if len(X) != 1025:
        print(
            f"\nWARNING: Expected 1,025 dataset rows, "
            f"but loaded {len(X)} rows."
        )

    # One fixed, stratified 80/20 split shared by all 10 runs.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=BASE_SEED,
        stratify=y,
    )

    print(
        f"\nTrain rows: {len(X_train)} | "
        f"Held-out test rows: {len(X_test)}"
    )
    print(
        "Evaluation uses the fixed held-out test split. "
        "Test samples are never queried or added to the labeled set."
    )
    print(
        f"Initial labeled set: {INITIAL_LABELLED}; "
        f"iterations: {N_ITERATIONS}; "
        f"query sizes: {QUERY_SIZES}"
    )

    expected_final_labeled = (
        INITIAL_LABELLED
        + sum(QUERY_SIZES) * (
            N_ITERATIONS // len(QUERY_SIZES)
        )
        + sum(
            QUERY_SIZES[:N_ITERATIONS % len(QUERY_SIZES)]
        )
    )

    print(
        f"Expected final labeled count with cycling: "
        f"{expected_final_labeled}"
    )

    all_histories = []
    final_rows = []

    for run in range(N_RUNS):
        history, final = run_one_experiment(
            X_train,
            y_train,
            X_test,
            y_test,
            run,
        )

        all_histories.append(history)
        final_rows.append(final)

    all_iterations_df = pd.concat(
        all_histories,
        ignore_index=True,
    )

    final_results_df = pd.DataFrame(final_rows)

    metric_columns = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc",
        "labeled_pct_of_train",
        "labeled_pct_of_full_dataset",
    ]

    summary = pd.DataFrame({
        "metric": metric_columns,
        "mean": [
            final_results_df[c].mean()
            for c in metric_columns
        ],
        "std": [
            final_results_df[c].std(ddof=1)
            for c in metric_columns
        ],
    })

    # Save result tables.
    all_iterations_df.to_csv(
        OUTPUT_DIR / "all_iteration_results.csv",
        index=False,
    )

    final_results_df.to_csv(
        OUTPUT_DIR / "final_results_each_run.csv",
        index=False,
    )

    summary.to_csv(
        OUTPUT_DIR / "summary_mean_std.csv",
        index=False,
    )

    mean_accuracy_by_iteration = (
        all_iterations_df
        .groupby("iteration", as_index=False)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_labeled_pct=(
                "labeled_pct_of_full_dataset",
                "mean",
            ),
        )
    )

    mean_accuracy_by_iteration.to_csv(
        OUTPUT_DIR / "mean_accuracy_by_iteration.csv",
        index=False,
    )

    # Plot held-out test accuracy against labeled-data percentage.
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        mean_accuracy_by_iteration["mean_labeled_pct"],
        mean_accuracy_by_iteration["mean_accuracy"],
        marker="o",
        markersize=3,
        linewidth=1.5,
        label="Mean held-out test accuracy",
    )

    ax.set_xlabel("Labeled data (% of full dataset)")
    ax.set_ylabel("Mean held-out test accuracy")
    ax.set_title("HC-AAL: Accuracy vs. Labeled Data")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "accuracy_vs_labeled_data.png",
        dpi=200,
    )

    plt.close(fig)

    print("\n" + "=" * 68)
    print("FINAL RESULTS: MEAN OVER 10 RUNS")
    print("=" * 68)

    for _, row in summary.iterrows():
        print(
            f"{row['metric']:32s}: "
            f"{row['mean']:.4f} +/- {row['std']:.4f}"
        )

    print("\nFiles saved in:", OUTPUT_DIR.resolve())
    print(" - all_iteration_results.csv")
    print(" - final_results_each_run.csv")
    print(" - summary_mean_std.csv")
    print(" - mean_accuracy_by_iteration.csv")
    print(" - accuracy_vs_labeled_data.png")

    print(
        "\nNote: Metrics above use a held-out test set. "
        "The labeled percentage of the full dataset is reported separately. "
        "The paper reports 96.04% accuracy and 31.9% labeled data; exact "
        "results may differ because some methodological details are unspecified."
    )


if __name__ == "__main__":
    main()
