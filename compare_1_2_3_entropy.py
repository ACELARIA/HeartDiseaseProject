
import time
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = "heart.csv"
TARGET = "target"

OUTPUT_DIR = Path("committee_1_2_3_entropy_results")

SEEDS = [42, 43, 44, 45, 46]

TEST_SIZE = 0.20
INITIAL_LABELED = 15
ACTIVE_LEARNING_ROUNDS = 78
QUERY_SCHEDULE = [3, 4, 5]
THRESHOLD = 0.50

# All configurations use the SAME predictive entropy
# acquisition function.

METHODS = {
    # Original three-model committee
    "3-model: XGBoost + RF + LR": ("xgb", "rf", "lr"),

    # Two-model committees
    "2-model: XGBoost + RF": ("xgb", "rf"),
    "2-model: XGBoost + LR": ("xgb", "lr"),
    "2-model: RF + LR": ("rf", "lr"),

    # Single-model approaches
    "1-model: XGBoost": ("xgb",),
    "1-model: Random Forest": ("rf",),
    "1-model: Logistic Regression": ("lr",),
}


# ============================================================
# MODEL CREATION
# ============================================================

def make_model(name, seed):
    """Create a fresh model for each active-learning iteration."""

    if name == "xgb":
        return XGBClassifier(
            n_estimators=120,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
        )

    if name == "rf":
        return RandomForestClassifier(
            n_estimators=120,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        )

    if name == "lr":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                random_state=seed,
            ),
        )

    raise ValueError(f"Unknown model: {name}")


# ============================================================
# POSITIVE CLASS PROBABILITY
# ============================================================

def get_positive_probability(model, X):
    """Return predicted probability for class 1."""

    probabilities = model.predict_proba(X)

    if hasattr(model, "classes_"):
        classes = model.classes_
    else:
        # Pipeline: access LogisticRegression classes
        classes = model[-1].classes_

    positions = np.where(np.asarray(classes) == 1)[0]

    if len(positions) == 0:
        return np.zeros(len(X))

    return probabilities[:, int(positions[0])]


# ============================================================
# COMMON UNCERTAINTY CALCULATION
# ============================================================

def acquisition_scores(models, X_pool):
    """
    Use predictive entropy for every configuration.

    1 model:
        Entropy of its predicted probability.

    2 models:
        Entropy of the average predicted probability.

    3 models:
        Entropy of the average predicted probability.

    Higher entropy = higher uncertainty.
    """

    probabilities = np.vstack([
        get_positive_probability(model, X_pool)
        for model in models
    ])

    # Average probability across committee members
    mean_probability = np.mean(probabilities, axis=0)

    # Prevent log(0)
    epsilon = 1e-10

    mean_probability = np.clip(
        mean_probability,
        epsilon,
        1 - epsilon,
    )

    # Binary predictive entropy
    entropy = -(
        mean_probability * np.log(mean_probability)
        +
        (1 - mean_probability)
        * np.log(1 - mean_probability)
    )

    return entropy


# ============================================================
# K-MEANS SAMPLE SELECTION
# ============================================================

def choose_samples(X_pool, scores, query_size, seed):
    """
    Select uncertain candidates and choose diverse representatives
    using K-Means.
    """

    pool_size = len(X_pool)
    q = min(query_size, pool_size)

    if q == pool_size:
        return np.arange(pool_size)

    # Select top 2q uncertain candidates
    candidate_count = min(pool_size, 2 * q)

    candidates = np.argsort(scores)[::-1][:candidate_count]

    X_candidates = X_pool[candidates]

    if len(candidates) <= q:
        return candidates

    kmeans = KMeans(
        n_clusters=q,
        random_state=seed,
        n_init=10,
    )

    cluster_ids = kmeans.fit_predict(X_candidates)

    selected = []

    # Select candidate nearest to each cluster center
    for cluster_id in range(q):

        members = np.where(cluster_ids == cluster_id)[0]

        if len(members) == 0:
            continue

        center = kmeans.cluster_centers_[cluster_id]

        distances = np.linalg.norm(
            X_candidates[members] - center,
            axis=1,
        )

        local_index = members[np.argmin(distances)]

        selected.append(int(candidates[local_index]))

    # Fill missing selections if required
    for candidate in candidates:

        if len(selected) >= q:
            break

        if int(candidate) not in selected:
            selected.append(int(candidate))

    return np.asarray(selected[:q], dtype=int)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred, y_probability):

    try:
        auc = roc_auc_score(y_true, y_probability)
    except ValueError:
        auc = np.nan

    return {
        "accuracy": accuracy_score(y_true, y_pred),

        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),

        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),

        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),

        "auc": auc,
    }


# ============================================================
# SINGLE EXPERIMENT
# ============================================================

def run_experiment(X, y, method_name, members, seed):

    # --------------------------------------------------------
    # TRAIN-TEST SPLIT
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=seed,
    )

    rng = np.random.default_rng(seed)

    initial_count = min(
        INITIAL_LABELED,
        len(X_train) - 1,
    )

    labeled = list(
        rng.choice(
            len(X_train),
            size=initial_count,
            replace=False,
        )
    )

    is_unlabeled = np.ones(len(X_train), dtype=bool)

    is_unlabeled[labeled] = False

    fit_time = 0.0
    selection_time = 0.0

    experiment_start = time.perf_counter()

    # --------------------------------------------------------
    # ACTIVE LEARNING
    # --------------------------------------------------------

    for iteration in range(ACTIVE_LEARNING_ROUNDS):

        pool_indices = np.where(is_unlabeled)[0]

        if len(pool_indices) == 0:
            break

        X_labeled = X_train[labeled]
        y_labeled = y_train[labeled]

        X_pool = X_train[pool_indices]

        # Train only models belonging to this configuration
        models = []

        start = time.perf_counter()

        for member in members:

            model = make_model(
                member,
                seed + iteration,
            )

            model.fit(
                X_labeled,
                y_labeled,
            )

            models.append(model)

        fit_time += time.perf_counter() - start

        query_size = QUERY_SCHEDULE[
            iteration % len(QUERY_SCHEDULE)
        ]

        # Calculate uncertainty using SAME entropy function
        start = time.perf_counter()

        scores = acquisition_scores(
            models,
            X_pool,
        )

        selected_relative = choose_samples(
            X_pool,
            scores,
            query_size,
            seed + iteration,
        )

        selection_time += time.perf_counter() - start

        selected_indices = pool_indices[selected_relative]

        labeled.extend(
            map(int, selected_indices)
        )

        is_unlabeled[selected_indices] = False

    # --------------------------------------------------------
    # FINAL PREDICTOR
    # --------------------------------------------------------
    # Same final XGBoost predictor for all configurations.
    # This isolates the effect of sample acquisition.

    final_model = make_model("xgb", seed)

    start = time.perf_counter()

    final_model.fit(
        X_train[labeled],
        y_train[labeled],
    )

    fit_time += time.perf_counter() - start

    y_probability = get_positive_probability(
        final_model,
        X_test,
    )

    y_prediction = (
        y_probability >= THRESHOLD
    ).astype(int)

    metrics = calculate_metrics(
        y_test,
        y_prediction,
        y_probability,
    )

    total_time = (
        time.perf_counter() - experiment_start
    )

    return {
        "method": method_name,
        "seed": seed,
        "committee_size": len(members),

        **metrics,

        "labeled_samples": len(labeled),

        "labeled_pct_train":
            100 * len(labeled) / len(X_train),

        "labeled_pct_full_dataset":
            100 * len(labeled) / len(X),

        "fit_time_seconds": fit_time,

        "selection_time_seconds": selection_time,

        "total_runtime_seconds": total_time,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not Path(DATA_FILE).exists():

        raise FileNotFoundError(
            f"{DATA_FILE} not found. "
            "Run this script from the folder containing heart.csv."
        )

    df = pd.read_csv(DATA_FILE)

    if TARGET not in df.columns:

        raise ValueError(
            f"Target column '{TARGET}' not found."
        )

    df = df.dropna(
        subset=[TARGET]
    ).copy()

    X_df = df.drop(
        columns=[TARGET]
    ).apply(
        pd.to_numeric,
        errors="coerce",
    )

    X_df = X_df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # Note: This reproduces the earlier preprocessing approach.
    # For a strict evaluation, move imputation inside a pipeline
    # fitted only on the training data.

    X_df = X_df.fillna(
        X_df.median(numeric_only=True)
    )

    if X_df.isna().any().any():

        raise ValueError(
            "Some feature values could not be imputed."
        )

    X = X_df.to_numpy(dtype=float)

    y = pd.to_numeric(
        df[TARGET],
        errors="raise",
    ).astype(int).to_numpy()

    if set(np.unique(y)) != {0, 1}:

        raise ValueError(
            "Expected binary target labels 0 and 1."
        )

    print("=" * 85)
    print("HEART DISEASE: 1 vs 2 vs 3 MODEL COMPARISON")
    print("COMMON UNCERTAINTY: PREDICTIVE ENTROPY")
    print("=" * 85)

    print(f"Rows: {len(df)}")
    print(f"Features: {X.shape[1]}")
    print(f"Runs: {len(SEEDS)}")
    print(f"Initial labeled samples: {INITIAL_LABELED}")
    print(f"Active-learning rounds: {ACTIVE_LEARNING_ROUNDS}")
    print(f"Query schedule: {QUERY_SCHEDULE}")

    all_results = []

    # --------------------------------------------------------
    # RUN ALL CONFIGURATIONS
    # --------------------------------------------------------

    for method_name, members in METHODS.items():

        print(f"\nRunning {method_name}")

        for seed in SEEDS:

            result = run_experiment(
                X,
                y,
                method_name,
                members,
                seed,
            )

            all_results.append(result)

            print(
                f"  Seed {seed}: "
                f"Accuracy={result['accuracy']:.4f}, "
                f"Precision={result['precision']:.4f}, "
                f"Recall={result['recall']:.4f}, "
                f"F1={result['f1']:.4f}, "
                f"AUC={result['auc']:.4f}, "
                f"Runtime={result['total_runtime_seconds']:.2f}s"
            )

    results = pd.DataFrame(all_results)

    results.to_csv(
        OUTPUT_DIR / "results_each_run.csv",
        index=False,
    )

    # --------------------------------------------------------
    # MEAN AND STANDARD DEVIATION
    # --------------------------------------------------------

    metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc",
        "labeled_samples",
        "labeled_pct_train",
        "labeled_pct_full_dataset",
        "fit_time_seconds",
        "selection_time_seconds",
        "total_runtime_seconds",
    ]

    summary_rows = []

    for method_name in METHODS:

        group = results[
            results["method"] == method_name
        ]

        row = {
            "method": method_name,

            "committee_size":
                len(METHODS[method_name]),
        }

        for metric in metrics:

            row[f"{metric}_mean"] = (
                group[metric].mean()
            )

            row[f"{metric}_std"] = (
                group[metric].std(ddof=1)
            )

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)

    summary.to_csv(
        OUTPUT_DIR / "metrics_mean_std.csv",
        index=False,
    )

    # --------------------------------------------------------
    # DISPLAY SUMMARY
    # --------------------------------------------------------

    display_columns = [
        "method",
        "accuracy_mean",
        "precision_mean",
        "recall_mean",
        "f1_mean",
        "auc_mean",
        "labeled_pct_train_mean",
        "total_runtime_seconds_mean",
    ]

    print("\n" + "=" * 115)
    print("FINAL SUMMARY: MEAN RESULTS")
    print("=" * 115)

    print(
        summary[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    # --------------------------------------------------------
    # GENERATE GRAPHS
    # --------------------------------------------------------

    plot_metrics = [
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1-score"),
        ("auc", "ROC-AUC"),
        ("labeled_pct_train", "Labeled training data (%)"),
        ("total_runtime_seconds", "Runtime (seconds)"),
    ]

    for metric, title in plot_metrics:

        means = summary[
            f"{metric}_mean"
        ].to_numpy()

        stds = summary[
            f"{metric}_std"
        ].fillna(0).to_numpy()

        labels = summary["method"].tolist()

        fig, ax = plt.subplots(
            figsize=(13, 6)
        )

        positions = np.arange(
            len(labels)
        )

        ax.bar(
            positions,
            means,
            yerr=stds,
            capsize=4,
        )

        ax.set_title(
            f"Committee Size Comparison: {title}"
        )

        ax.set_ylabel(title)

        ax.set_xticks(positions)

        ax.set_xticklabels(
            labels,
            rotation=25,
            ha="right",
        )

        ax.grid(
            axis="y",
            linestyle="--",
            alpha=0.35,
        )

        fig.tight_layout()

        fig.savefig(
            OUTPUT_DIR / f"comparison_{metric}.png",
            dpi=200,
        )

        plt.close(fig)

    print("\nResults saved in:")
    print(OUTPUT_DIR.resolve())

    print("\nGenerated files:")
    print("  results_each_run.csv")
    print("  metrics_mean_std.csv")
    print("  comparison_accuracy.png")
    print("  comparison_precision.png")
    print("  comparison_recall.png")
    print("  comparison_f1.png")
    print("  comparison_auc.png")
    print("  comparison_labeled_pct_train.png")
    print("  comparison_total_runtime_seconds.png")


if __name__ == "__main__":
    main()
