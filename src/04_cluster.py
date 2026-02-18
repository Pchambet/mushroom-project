"""
04 — Clustering sur coordonnées ACM.

Effectue une Classification Ascendante Hiérarchique (Ward) pour guider
le choix du nombre de clusters, puis consolide avec K-Means.
Profile les clusters par modalités sur/sous-représentées.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage

from utils import (
    get_project_root, load_processed_data, load_mca_coordinates,
    save_figure, save_table, print_section, print_step,
)


# ── Configuration ──────────────────────────────────────────

K_AXES = 5
N_CLUSTERS = 3
DENDRO_SAMPLE = 500
DENDRO_CUT_HEIGHT = 15
OVER_REP_THRESHOLD = 1.5
UNDER_REP_THRESHOLD = 0.5
RANDOM_STATE = 42


# ── Pipeline ───────────────────────────────────────────────

def perform_clustering() -> None:
    """Effectue le clustering sur les coordonnées ACM."""

    print_section("04 — Clustering sur coordonnées ACM")

    # ── Chargement ──

    coords = load_mca_coordinates()
    df = load_processed_data()
    X_mca = coords.iloc[:, :K_AXES].values

    print_step(f"Coordonnées ACM chargées : {X_mca.shape} (k={K_AXES} axes)")

    # ── CAH (Ward) sur échantillon ──

    print_step("Classification Ascendante Hiérarchique (Ward)")
    rng = np.random.default_rng(RANDOM_STATE)
    sample_size = min(DENDRO_SAMPLE, len(X_mca))
    sample_idx = rng.choice(len(X_mca), sample_size, replace=False)
    linkage_matrix = linkage(X_mca[sample_idx], method="ward")

    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(linkage_matrix, no_labels=True, color_threshold=None, ax=ax)
    ax.axhline(
        y=DENDRO_CUT_HEIGHT, color="r", linestyle="--", alpha=0.7,
        label=f"Coupe suggérée (h={DENDRO_CUT_HEIGHT})",
    )
    ax.set_title("Dendrogramme — CAH (Ward)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Individus", fontsize=12)
    ax.set_ylabel("Distance", fontsize=12)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, "cluster_dendrogram.png")

    # ── K-Means ──

    print_step(f"K-Means clustering (k={N_CLUSTERS})")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    labels = kmeans.fit_predict(X_mca)

    # ── Table : tailles des clusters ──

    cluster_sizes = pd.DataFrame({
        "cluster": range(N_CLUSTERS),
        "size": [int((labels == i).sum()) for i in range(N_CLUSTERS)],
        "percentage": [
            round((labels == i).sum() / len(labels) * 100, 2)
            for i in range(N_CLUSTERS)
        ],
    })
    save_table(cluster_sizes, "cluster_sizes.csv")

    for _, row in cluster_sizes.iterrows():
        print(f"    Cluster {int(row['cluster'])}: {int(row['size']):,} ({row['percentage']:.1f}%)")

    # ── Figure : clusters sur plan factoriel ──

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        coords.iloc[:, 0], coords.iloc[:, 1],
        c=labels, cmap="viridis", alpha=0.6, s=30,
    )
    ax.set_xlabel("Dim 1", fontsize=12)
    ax.set_ylabel("Dim 2", fontsize=12)
    ax.set_title(
        f"K-Means — {N_CLUSTERS} clusters (plan factoriel ACM 1-2)",
        fontsize=14, fontweight="bold",
    )
    fig.colorbar(scatter, ax=ax, label="Cluster")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_figure(fig, "cluster_on_acm12.png")

    # ── Table : clusters x variable cible ──

    df_clusters = df.copy()
    df_clusters["cluster"] = labels

    crosstab = pd.crosstab(df_clusters["cluster"], df_clusters["class"], margins=True)
    save_table(crosstab, "cluster_vs_target.csv", index=True)

    # ── Table : profils de clusters ──

    print_step("Profiling des clusters (modalités caractéristiques)")
    profiles = []

    for cid in range(N_CLUSTERS):
        mask = labels == cid
        for col in df.columns:
            if col == "class":
                continue
            cluster_dist = df[mask][col].value_counts(normalize=True)
            global_dist = df[col].value_counts(normalize=True)

            for modality in cluster_dist.index:
                if modality not in global_dist.index:
                    continue
                ratio = cluster_dist[modality] / global_dist[modality]
                if ratio > OVER_REP_THRESHOLD or ratio < UNDER_REP_THRESHOLD:
                    profiles.append({
                        "cluster": cid,
                        "variable": col,
                        "modality": modality,
                        "cluster_freq_%": round(cluster_dist[modality] * 100, 2),
                        "global_freq_%": round(global_dist[modality] * 100, 2),
                        "over_representation": round(ratio, 2),
                    })

    profile_df = pd.DataFrame(profiles).sort_values(
        ["cluster", "over_representation"], ascending=[True, False]
    )
    save_table(profile_df, "cluster_profiles.csv")
    print(f"    {len(profile_df)} modalités significatives identifiées")

    for cid in range(N_CLUSTERS):
        top = profile_df[profile_df["cluster"] == cid].head(3)
        if not top.empty:
            mods = ", ".join(f"{r['variable']}={r['modality']}" for _, r in top.iterrows())
            print(f"    Cluster {cid} (top 3) : {mods}")

    # ── Résumé ──

    print()
    print_step("Clustering terminé.")
    print()
    print("  Outputs :")
    print("    Tables  — cluster_sizes.csv, cluster_vs_target.csv, cluster_profiles.csv")
    print("    Figures — cluster_dendrogram.png, cluster_on_acm12.png")
    print()


if __name__ == "__main__":
    perform_clustering()
