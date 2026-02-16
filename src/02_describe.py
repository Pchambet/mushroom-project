"""
02 — Statistiques descriptives (Mission A2).

Génère les statistiques univariées, la distribution de la variable cible,
et les visualisations des variables clés. Exporte les résultats dans
``reports/tables/`` et ``reports/figures/``.
"""

from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from utils import (
    get_project_root, load_processed_data,
    save_figure, save_table, print_section, print_step,
)


# ── Configuration ──────────────────────────────────────────

KEY_VARIABLES = ["odor", "gill-color", "spore-print-color"]
TARGET_COLORS = {"e": "#2ecc71", "p": "#e74c3c"}


# ── Pipeline ───────────────────────────────────────────────

def describe_data() -> None:
    """Génère les statistiques descriptives du dataset (Mission A2)."""

    print_section("02 — Statistiques descriptives (Mission A2)")

    df = load_processed_data()
    print_step(f"Dataset chargé : {df.shape[0]:,} x {df.shape[1]}")

    # ── Table : résumé univarié ──

    rows = []
    for col in df.columns:
        n_mod = df[col].nunique()
        top = df[col].mode().iloc[0] if not df[col].mode().empty else None
        top_freq = (df[col] == top).sum() / len(df) * 100 if top else 0
        pct_na = df[col].isna().sum() / len(df) * 100
        rows.append({
            "variable": col,
            "n_modalities": n_mod,
            "top_modality": top,
            "top_freq_pct": round(top_freq, 2),
            "missing_pct": round(pct_na, 2),
        })

    univariate_df = pd.DataFrame(rows)
    save_table(univariate_df, "univariate_summary.csv")

    # ── Table : distribution cible ──

    target_dist = df["class"].value_counts().reset_index()
    target_dist.columns = ["class", "count"]
    target_dist["percentage"] = round(
        target_dist["count"] / target_dist["count"].sum() * 100, 2
    )
    save_table(target_dist, "target_distribution.csv")

    # ── Figure : bar chart cible ──

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = [TARGET_COLORS.get(c, "#888") for c in target_dist["class"]]
    bars = ax.bar(
        target_dist["class"], target_dist["count"],
        color=colors, alpha=0.85, edgecolor="black", linewidth=0.5,
    )
    ax.set_title(
        "Distribution de la classe (Edible vs Poisonous)",
        fontsize=14, fontweight="bold",
    )
    ax.set_xlabel("Classe", fontsize=12)
    ax.set_ylabel("Fréquence", fontsize=12)
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, h,
            f"{int(h):,}", ha="center", va="bottom", fontsize=11,
        )
    fig.tight_layout()
    save_figure(fig, "desc_target_bar.png")

    # ── Figures : variables clés ──

    for var in KEY_VARIABLES:
        if var not in df.columns:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        vc = df[var].value_counts().head(10)
        palette = plt.cm.Set3(range(len(vc)))

        bars = ax.bar(
            range(len(vc)), vc.values,
            color=palette, alpha=0.85, edgecolor="black", linewidth=0.5,
        )
        ax.set_xticks(range(len(vc)))
        ax.set_xticklabels(vc.index, rotation=45, ha="right")
        ax.set_title(
            f"Distribution des modalités : {var}",
            fontsize=14, fontweight="bold",
        )
        ax.set_xlabel("Modalité", fontsize=12)
        ax.set_ylabel("Fréquence", fontsize=12)
        ax.grid(axis="y", alpha=0.3)

        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, h,
                f"{int(h):,}", ha="center", va="bottom", fontsize=9,
            )
        fig.tight_layout()
        save_figure(fig, f"desc_top_modalities_{var}.png")

    # ── Résumé ──

    print()
    print_step("Mission A2 terminée.")
    print()
    print("  Outputs :")
    print("    Tables  — univariate_summary.csv, target_distribution.csv")
    print("    Figures — desc_target_bar.png" + "".join(
        f", desc_top_modalities_{v}.png" for v in KEY_VARIABLES if v in df.columns
    ))
    print()


if __name__ == "__main__":
    describe_data()
