"""Run every experiment, write the result tables and the headline numbers."""

from __future__ import annotations

import json
import time

import pandas as pd

from mushroom import experiments as ex
from mushroom.config import PROCESSED_CSV, RESULTS_JSON, TABLES_DIR
from mushroom.data import load_processed


def _save(df: pd.DataFrame, name: str) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES_DIR / f"{name}.csv", index=False, float_format="%.6g")


def run() -> dict[str, float]:
    start = time.perf_counter()
    X, y = load_processed(PROCESSED_CSV)

    inertia = ex.inertia_table(X, y)
    axis1 = ex.axis1_categories(X, y)
    cluster_table, ari = ex.clusters(X, y)
    curve = ex.axis_curve(X, y)
    sweep = ex.tree_depth_sweep(X, y)
    depth = ex.smallest_perfect_depth(sweep)
    reference = ex.reference_models(X, y, tree_depth=depth)
    folds = ex.fold_ordering(X, y)

    for name, table in {
        "mca_inertia": inertia,
        "mca_axis1_categories": axis1,
        "clusters": cluster_table,
        "axis_curve": curve,
        "tree_depth": sweep,
        "reference_models": reference,
        "fold_ordering": folds,
    }.items():
        _save(table, name)

    results = headline(X, y, inertia, curve, reference, sweep, folds, cluster_table, ari)
    RESULTS_JSON.write_text(json.dumps(results, indent=2) + "\n")
    print(f"pipeline finished in {time.perf_counter() - start:.0f} s")
    return results


def headline(
    X: pd.DataFrame,
    y: pd.Series,
    inertia: pd.DataFrame,
    curve: pd.DataFrame,
    reference: pd.DataFrame,
    sweep: pd.DataFrame,
    folds: pd.DataFrame,
    cluster_table: pd.DataFrame,
    ari: float,
) -> dict[str, float]:
    """The numbers quoted in the README and the report, in one auditable place."""

    def ref(model: str, features: str) -> pd.Series:
        return reference[(reference["model"] == model) & (reference["features"] == features)].iloc[
            0
        ]

    def acc(model: str, k: int) -> float:
        row = curve[(curve["model"] == model) & (curve["n_axes"] == k)]
        return float(row["mean"].iloc[0])

    def axes_to_reach(model: str, target: float) -> int:
        hit = curve[(curve["model"] == model) & (curve["mean"] >= target)]
        return int(hit["n_axes"].min())

    odour = reference[reference["model"] == "Odour-only lookup"].iloc[0]
    # ddof=0 to match the "mean ± std" of cross_val_score used everywhere else.
    spread = folds.groupby(["model", "split"])["accuracy"].agg(
        mean="mean", std=lambda s: s.std(ddof=0)
    )
    eta = inertia.set_index("axis")["eta2_class"]
    linear_axes = inertia[inertia["axis"].between(2, 9)]
    depth = ex.smallest_perfect_depth(sweep)
    return {
        "n_specimens": len(X),
        "n_variables": X.shape[1],
        "n_categories": int(X.nunique().sum()),
        "n_axes": len(inertia),
        "n_poisonous": int(y.sum()),
        "poisonous_share": float(y.mean()),
        "total_inertia": float(inertia["eigenvalue"].sum()),
        "inertia_axes_1_5_pct": float(inertia["cumulative_inertia_pct"].iloc[4]),
        "inertia_axes_1_10_pct": float(inertia["cumulative_inertia_pct"].iloc[9]),
        "benzecri_axes_1_5_pct": float(inertia["benzecri_pct"].iloc[:5].sum()),
        "benzecri_axes_1_10_pct": float(inertia["benzecri_pct"].iloc[:10].sum()),
        "eta2_axis1": float(eta.loc[1]),
        "eta2_axis10": float(eta.loc[10]),
        "eta2_axes_2_9_max": float(linear_axes["eta2_class"].max()),
        "eta2_best_minor_axis": int(eta.loc[21:].idxmax()),
        "eta2_best_minor_value": float(eta.loc[21:].max()),
        "lda_1_axis": acc("LDA", 1),
        "lda_max_axes_1_9": float(
            curve[(curve["model"] == "LDA") & (curve["n_axes"] <= 9)]["mean"].max()
        ),
        "lda_5_axes": acc("LDA", 5),
        "lda_10_axes": acc("LDA", 10),
        "lda_all_axes": float(ref("LDA", f"MCA, all {len(inertia)} axes")["cv_accuracy_mean"]),
        "rf_5_axes": acc("Random forest", 5),
        "knn_5_axes": acc("k-NN (15)", 5),
        "lda_axes_for_99": axes_to_reach("LDA", 0.99),
        "rf_axes_for_99": axes_to_reach("Random forest", 0.99),
        "odour_accuracy": float(odour["cv_accuracy_mean"]),
        "odour_poisonous_called_edible": int(odour["poisonous_called_edible"]),
        "odour_edible_called_poisonous": int(odour["edible_called_poisonous"]),
        "lda_5_poisonous_called_edible": int(ref("LDA", "MCA, 5 axes")["poisonous_called_edible"]),
        "rf_5_poisonous_called_edible": int(
            ref("Random forest", "MCA, 5 axes")["poisonous_called_edible"]
        ),
        "tree_perfect_depth": depth,
        "tree_perfect_leaves": int(sweep.loc[sweep["depth"] == depth, "n_leaves"].iloc[0]),
        "tree_perfect_accuracy": float(sweep.loc[sweep["depth"] == depth, "mean"].iloc[0]),
        "unshuffled_lda_mean": float(spread.loc[("LDA, 5 axes", "unshuffled (cv=5)"), "mean"]),
        "unshuffled_lda_std": float(spread.loc[("LDA, 5 axes", "unshuffled (cv=5)"), "std"]),
        "shuffled_lda_std": float(spread.loc[("LDA, 5 axes", "shuffled"), "std"]),
        "unshuffled_rf_mean": float(
            spread.loc[("Random forest, 5 axes", "unshuffled (cv=5)"), "mean"]
        ),
        "unshuffled_rf_std": float(
            spread.loc[("Random forest, 5 axes", "unshuffled (cv=5)"), "std"]
        ),
        "shuffled_rf_mean": float(spread.loc[("Random forest, 5 axes", "shuffled"), "mean"]),
        "kmeans_ari": ari,
        "kmeans_largest_cluster_poisonous_pct": float(cluster_table["poisonous_pct"].iloc[0]),
    }
