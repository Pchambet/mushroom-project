"""
03 — Analyse en Composantes Multiples (Mission A3).

Effectue l'ACM sur les 22 variables catégorielles, exporte les coordonnées
factorielles (fichier interface ``mca_coords.csv`` pour Personne B), les
valeurs propres, les contributions des modalités, et les visualisations.

Librairie ACM : ``prince`` (https://github.com/MaxHalford/prince)
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import prince

from utils import (
    get_project_root, load_processed_data,
    save_figure, save_table, print_section, print_step,
)


# ── Configuration ──────────────────────────────────────────

N_COMPONENTS = 10
INERTIA_THRESHOLD = 0.90
TOP_CONTRIB = 15


# ── Pipeline ───────────────────────────────────────────────

def perform_mca() -> None:
    """Effectue l'ACM sur le dataset (Mission A3).

    Produit le fichier interface ``data/processed/mca_coords.csv``
    contenant les coordonnées de chaque individu sur les axes factoriels.
    """

    print_section("03 — Analyse en Composantes Multiples (Mission A3)")

    processed_dir = get_project_root() / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # ── Préparation ──

    df = load_processed_data()
    y = df["class"]
    X = df.drop("class", axis=1)
    X = X.fillna(X.mode().iloc[0])

    print_step(f"Données préparées : {X.shape[0]:,} individus x {X.shape[1]} variables")

    # ── ACM ──

    mca = prince.MCA(n_components=N_COMPONENTS, n_iter=3, copy=True, check_input=True)
    mca = mca.fit(X)
    print_step(f"ACM effectuée ({N_COMPONENTS} composantes)")

    # ── Fichier interface : mca_coords.csv ──

    row_coords = mca.transform(X)
    row_coords.columns = [f"Dim{i+1}" for i in range(N_COMPONENTS)]
    row_coords.to_csv(processed_dir / "mca_coords.csv", index=False)
    print_step(f"Interface exportée : mca_coords.csv ({row_coords.shape})")

    # ── Valeurs propres et inertie ──

    eigenvalues = mca.eigenvalues_
    explained = eigenvalues / eigenvalues.sum()
    cumulative = np.cumsum(explained)
    n_actual = len(explained)

    eigenvalues_df = pd.DataFrame({
        "Component": [f"Dim{i+1}" for i in range(n_actual)],
        "Eigenvalue": eigenvalues,
        "Explained_Inertia_%": explained * 100,
        "Cumulative_Inertia_%": cumulative * 100,
    })
    save_table(eigenvalues_df, "mca_eigenvalues.csv")

    # Recommandation k
    k_idx = np.where(cumulative >= INERTIA_THRESHOLD)[0]
    k_recommended = int(k_idx[0]) + 1 if len(k_idx) > 0 else N_COMPONENTS
    print_step(
        f"k recommandé = {k_recommended} axes "
        f"(>= {INERTIA_THRESHOLD*100:.0f}% inertie cumulée : "
        f"{cumulative[k_recommended-1]*100:.1f}%)"
    )

    # ── Contributions des modalités ──

    col_coords = mca.column_coordinates(X)
    contributions = mca.column_contributions_

    for axis_idx, axis_name in [(0, "axis1"), (1, "axis2")]:
        contrib_df = pd.DataFrame({
            "modality": col_coords.index,
            f"coord_dim{axis_idx+1}": col_coords.iloc[:, axis_idx],
            f"contrib_dim{axis_idx+1}_%": contributions.iloc[:, axis_idx] * 100,
        }).sort_values(f"contrib_dim{axis_idx+1}_%", ascending=False).head(TOP_CONTRIB)
        save_table(contrib_df, f"mca_modalities_contrib_{axis_name}.csv")

    # ── Figure : scree plot ──

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(
        range(1, n_actual + 1), explained * 100,
        "o-", linewidth=2, markersize=8, color="#2196F3",
    )
    axes[0].axhline(
        y=100 / n_actual, color="r", linestyle="--", alpha=0.5, label="Moyenne"
    )
    axes[0].set_xlabel("Composante", fontsize=12)
    axes[0].set_ylabel("Inertie expliquée (%)", fontsize=12)
    axes[0].set_title("Scree plot — Inertie par composante", fontsize=13, fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        range(1, n_actual + 1), cumulative * 100,
        "o-", linewidth=2, markersize=8, color="#4CAF50",
    )
    axes[1].axhline(y=90, color="r", linestyle="--", alpha=0.5, label="Seuil 90%")
    axes[1].set_xlabel("Nombre de composantes", fontsize=12)
    axes[1].set_ylabel("Inertie cumulée (%)", fontsize=12)
    axes[1].set_title("Inertie cumulée", fontsize=13, fontweight="bold")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    save_figure(fig, "acm_scree.png")

    # ── Figure : plan factoriel des modalités ──

    fig, ax = plt.subplots(figsize=(12, 10))
    for idx, modality in enumerate(col_coords.index):
        x, y_val = col_coords.iloc[idx, 0], col_coords.iloc[idx, 1]
        ax.scatter(x, y_val, alpha=0.7, s=100)
        ax.annotate(modality, (x, y_val), fontsize=8, alpha=0.8)

    ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.3)
    ax.axvline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.3)
    ax.set_xlabel(f"Dim 1 ({explained[0]*100:.2f}%)", fontsize=12)
    ax.set_ylabel(f"Dim 2 ({explained[1]*100:.2f}%)", fontsize=12)
    ax.set_title(
        "ACM — Plan factoriel des modalités (Dim 1-2)",
        fontsize=14, fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_figure(fig, "acm_modalities_12.png")

    # ── Figure : individus colorés par classe ──

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = np.array([1 if c == "e" else 0 for c in df["class"]])
    scatter = ax.scatter(
        row_coords.iloc[:, 0].values, row_coords.iloc[:, 1].values,
        c=colors, cmap="RdYlGn", alpha=0.6, s=20,
    )
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.3)
    ax.axvline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.3)
    ax.set_xlabel(f"Dim 1 ({explained[0]*100:.2f}%)", fontsize=12)
    ax.set_ylabel(f"Dim 2 ({explained[1]*100:.2f}%)", fontsize=12)
    ax.set_title(
        "ACM — Individus colorés par classe (Dim 1-2)",
        fontsize=14, fontweight="bold",
    )
    cbar = fig.colorbar(scatter, ax=ax, label="Classe")
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Vénéneux (p)", "Comestible (e)"])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_figure(fig, "acm_individuals_12_color_target.png")

    # ── Résumé ──

    print()
    print_step("Mission A3 terminée.")
    print()
    print("  Livraison pour Personne B :")
    print(f"    mca_coords.csv  ({row_coords.shape[0]:,} individus x {row_coords.shape[1]} axes)")
    print(f"    k recommandé    = {k_recommended} axes")
    print(f"    Inertie (k={k_recommended})  = {cumulative[k_recommended-1]*100:.1f}%")
    print()


if __name__ == "__main__":
    perform_mca()
