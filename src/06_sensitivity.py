"""
06 — Analyse de sensibilite au choix de k (nombre d'axes ACM).

Etudie l'impact du nombre de composantes ACM retenues sur :
  - La performance de classification (LDA, CV 5-fold)
  - La qualite du clustering (silhouette score, K-Means k=3)
  - L'inertie cumulee

Repond a la question critique : "Pourquoi k=5 et pas k=8 ?"
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import cross_val_score

from utils import (
    load_processed_data, load_mca_coordinates,
    save_figure, save_table, print_section, print_step,
)


# ── Configuration ──────────────────────────────────────────

K_VALUES = [2, 3, 4, 5, 6, 7, 8, 9, 10]
N_CLUSTERS = 3
CV_FOLDS = 5
RANDOM_STATE = 42


# ── Pipeline ───────────────────────────────────────────────

def sensitivity_analysis() -> pd.DataFrame:
    """Analyse de sensibilite : impact de k sur LDA et clustering.

    Returns
    -------
    pd.DataFrame
        Tableau comparatif pour chaque valeur de k.
    """

    print_section("06 — Analyse de sensibilite (impact de k)")

    coords = load_mca_coordinates()
    df = load_processed_data()
    y = (df["class"] == "e").astype(int)

    # Inertie cumulee (depuis les eigenvalues)
    from pathlib import Path
    eigen_path = Path(__file__).resolve().parent.parent / "reports" / "tables" / "mca_eigenvalues.csv"
    eigen_df = pd.read_csv(eigen_path)
    cumulative_inertia = eigen_df["Cumulative_Inertia_%"].values

    results = []

    for k in K_VALUES:
        X = coords.iloc[:, :k].values
        print_step(f"k={k:2d} axes ({cumulative_inertia[k-1]:.1f}% inertie)")

        # LDA — CV 5-fold
        lda = LinearDiscriminantAnalysis()
        cv_scores = cross_val_score(lda, X, y, cv=CV_FOLDS, scoring="accuracy")
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()

        # LDA — Train accuracy
        lda.fit(X, y)
        train_acc = lda.score(X, y)

        # Clustering — Silhouette
        kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
        labels = kmeans.fit_predict(X)
        sil = silhouette_score(X, labels)

        # Clustering — Inertie K-Means
        km_inertia = kmeans.inertia_

        print(
            f"         LDA train={train_acc:.3f}  "
            f"CV={cv_mean:.3f}+/-{cv_std:.3f}  "
            f"Silhouette={sil:.3f}"
        )

        results.append({
            "k": k,
            "cumulative_inertia_%": round(cumulative_inertia[k-1], 1),
            "lda_train_accuracy": round(train_acc, 4),
            "lda_cv_mean": round(cv_mean, 4),
            "lda_cv_std": round(cv_std, 4),
            "silhouette_score": round(sil, 4),
            "kmeans_inertia": round(km_inertia, 2),
            "overfitting_gap": round(train_acc - cv_mean, 4),
        })

    results_df = pd.DataFrame(results)
    save_table(results_df, "sensitivity_k.csv")

    # ── Identifier le meilleur k ──

    best_cv_idx = results_df["lda_cv_mean"].idxmax()
    best_k = results_df.loc[best_cv_idx, "k"]
    best_cv = results_df.loc[best_cv_idx, "lda_cv_mean"]
    print()
    print_step(f"Meilleur k (CV) = {best_k} axes ({best_cv:.4f} accuracy)")

    # ── Figure 1 : LDA accuracy vs k ──

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1 : Accuracy
    axes[0].plot(
        results_df["k"], results_df["lda_train_accuracy"],
        "o-", label="Train", color="#2196F3", linewidth=2, markersize=8,
    )
    axes[0].plot(
        results_df["k"], results_df["lda_cv_mean"],
        "s-", label="CV 5-fold", color="#F44336", linewidth=2, markersize=8,
    )
    axes[0].fill_between(
        results_df["k"],
        results_df["lda_cv_mean"] - results_df["lda_cv_std"],
        results_df["lda_cv_mean"] + results_df["lda_cv_std"],
        alpha=0.15, color="#F44336",
    )
    axes[0].set_xlabel("k (nombre d'axes ACM)", fontsize=12)
    axes[0].set_ylabel("Accuracy", fontsize=12)
    axes[0].set_title("LDA — Accuracy vs k", fontsize=13, fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(K_VALUES)

    # Panel 2 : Silhouette
    colors_sil = ["#4CAF50" if s > 0.3 else "#FF9800" if s > 0.2 else "#F44336"
                   for s in results_df["silhouette_score"]]
    axes[1].bar(
        results_df["k"], results_df["silhouette_score"],
        color=colors_sil, alpha=0.85, edgecolor="black", linewidth=0.5,
    )
    axes[1].set_xlabel("k (nombre d'axes ACM)", fontsize=12)
    axes[1].set_ylabel("Silhouette Score", fontsize=12)
    axes[1].set_title("Clustering — Silhouette vs k", fontsize=13, fontweight="bold")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].set_xticks(K_VALUES)

    # Panel 3 : Overfitting gap
    axes[2].bar(
        results_df["k"], results_df["overfitting_gap"],
        color="#9C27B0", alpha=0.7, edgecolor="black", linewidth=0.5,
    )
    axes[2].axhline(y=0.1, color="r", linestyle="--", alpha=0.5, label="Seuil 10%")
    axes[2].set_xlabel("k (nombre d'axes ACM)", fontsize=12)
    axes[2].set_ylabel("Gap (Train - CV)", fontsize=12)
    axes[2].set_title("Sur-apprentissage — Gap vs k", fontsize=13, fontweight="bold")
    axes[2].legend()
    axes[2].grid(axis="y", alpha=0.3)
    axes[2].set_xticks(K_VALUES)

    fig.tight_layout()
    save_figure(fig, "sensitivity_k_analysis.png")

    # ── Figure 2 : Inertie cumulee vs performance ──

    fig, ax1 = plt.subplots(figsize=(10, 6))
    color1 = "#2196F3"
    color2 = "#F44336"

    ax1.set_xlabel("k (nombre d'axes ACM)", fontsize=12)
    ax1.set_ylabel("Inertie cumulee (%)", fontsize=12, color=color1)
    ax1.plot(
        results_df["k"], results_df["cumulative_inertia_%"],
        "o-", color=color1, linewidth=2, markersize=8, label="Inertie cumulee",
    )
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.axhline(y=90, color=color1, linestyle="--", alpha=0.3)
    ax1.set_xticks(K_VALUES)

    ax2 = ax1.twinx()
    ax2.set_ylabel("LDA CV Accuracy", fontsize=12, color=color2)
    ax2.plot(
        results_df["k"], results_df["lda_cv_mean"],
        "s-", color=color2, linewidth=2, markersize=8, label="CV Accuracy",
    )
    ax2.tick_params(axis="y", labelcolor=color2)

    fig.suptitle(
        "Inertie ACM vs Performance LDA",
        fontsize=14, fontweight="bold",
    )
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    save_figure(fig, "sensitivity_inertia_vs_accuracy.png")

    # ── Resume ──

    print()
    print_step("Analyse de sensibilite terminee.")
    print()
    print("  Outputs :")
    print("    Tables  — sensitivity_k.csv")
    print("    Figures — sensitivity_k_analysis.png, sensitivity_inertia_vs_accuracy.png")
    print()

    return results_df


if __name__ == "__main__":
    sensitivity_analysis()
