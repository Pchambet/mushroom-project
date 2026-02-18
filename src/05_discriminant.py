"""
05 — Analyse discriminante linéaire (LDA).

Effectue une LDA sur les coordonnées ACM pour la classification binaire
comestible/vénéneux. Évalue le modèle par matrice de confusion et
validation croisée 5-fold.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report

from utils import (
    get_project_root, load_processed_data, load_mca_coordinates,
    save_figure, save_table, print_section, print_step,
)


# ── Configuration ──────────────────────────────────────────

K_AXES = 5
CV_FOLDS = 5
TARGET_NAMES = ["Poisonous", "Edible"]


# ── Pipeline ───────────────────────────────────────────────

def perform_discriminant_analysis() -> None:
    """Effectue l'analyse discriminante linéaire."""

    print_section("05 — Analyse discriminante (LDA)")

    # ── Chargement ──

    coords = load_mca_coordinates()
    df = load_processed_data()

    X = coords.iloc[:, :K_AXES].values
    y = (df["class"] == "e").astype(int)  # 1 = edible, 0 = poisonous

    print_step(f"Données préparées : {X.shape} (k={K_AXES} axes)")
    print(f"    Poisonous: {(y == 0).sum():,}  |  Edible: {(y == 1).sum():,}")

    # ── LDA ──

    print_step("Analyse Discriminante Linéaire (LDA)")
    lda = LinearDiscriminantAnalysis()
    lda.fit(X, y)
    y_pred = lda.predict(X)

    # Coefficients
    coef_df = pd.DataFrame({
        "axis": [f"Dim{i+1}" for i in range(K_AXES)],
        "coefficient": lda.coef_[0],
        "abs_coefficient": np.abs(lda.coef_[0]),
    }).sort_values("abs_coefficient", ascending=False)
    print("    Coefficients LDA :")
    for _, r in coef_df.iterrows():
        print(f"      {r['axis']:>5s}: {r['coefficient']:+.4f}")

    # ── Matrice de confusion (train) ──

    cm = confusion_matrix(y, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["Actual_Poisonous", "Actual_Edible"],
        columns=["Pred_Poisonous", "Pred_Edible"],
    )
    save_table(cm_df, "da_confusion.csv", index=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Vénéneux (0)", "Comestible (1)"],
        yticklabels=["Vénéneux (0)", "Comestible (1)"],
        cbar_kws={"label": "Fréquence"}, ax=ax,
    )
    ax.set_title(
        "Matrice de confusion — Analyse discriminante",
        fontsize=14, fontweight="bold",
    )
    ax.set_ylabel("Vraie classe", fontsize=12)
    ax.set_xlabel("Classe prédite", fontsize=12)
    fig.tight_layout()
    save_figure(fig, "da_confusion.png")

    # ── Métriques ──

    report = classification_report(y, y_pred, target_names=TARGET_NAMES, output_dict=True)
    metrics_rows = [
        {"metric": "Accuracy", "value": report["accuracy"]},
        {"metric": "Precision_Poisonous", "value": report["Poisonous"]["precision"]},
        {"metric": "Recall_Poisonous", "value": report["Poisonous"]["recall"]},
        {"metric": "F1_Poisonous", "value": report["Poisonous"]["f1-score"]},
        {"metric": "Precision_Edible", "value": report["Edible"]["precision"]},
        {"metric": "Recall_Edible", "value": report["Edible"]["recall"]},
        {"metric": "F1_Edible", "value": report["Edible"]["f1-score"]},
    ]

    # ── Validation croisée ──

    print_step(f"Validation croisée ({CV_FOLDS}-fold)")
    cv_scores = cross_val_score(lda, X, y, cv=CV_FOLDS, scoring="accuracy")
    print(f"    Scores : {', '.join(f'{s:.4f}' for s in cv_scores)}")
    print(f"    Moyenne : {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    metrics_rows.extend([
        {"metric": "CV_Mean_Accuracy", "value": cv_scores.mean()},
        {"metric": "CV_Std_Accuracy", "value": cv_scores.std()},
    ])
    metrics_df = pd.DataFrame(metrics_rows)
    save_table(metrics_df, "da_metrics.csv")

    # ── Figure : scores CV ──

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        range(1, CV_FOLDS + 1), cv_scores,
        color="steelblue", alpha=0.85, edgecolor="black", linewidth=0.5,
    )
    ax.axhline(
        cv_scores.mean(), color="red", linestyle="--", linewidth=2,
        label=f"Moyenne = {cv_scores.mean():.4f}",
    )
    ax.set_xlabel("Fold", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(
        "Validation croisée — Scores par fold",
        fontsize=14, fontweight="bold",
    )
    ax.set_ylim([min(cv_scores) - 0.02, 1.0])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for i, bar in enumerate(bars):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, h,
            f"{cv_scores[i]:.4f}", ha="center", va="bottom", fontsize=10,
        )
    fig.tight_layout()
    save_figure(fig, "da_cv_scores.png")

    # ── Figure : confusion CV ──

    y_pred_cv = cross_val_predict(lda, X, y, cv=CV_FOLDS)
    cm_cv = confusion_matrix(y, y_pred_cv)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_cv, annot=True, fmt="d", cmap="Greens",
        xticklabels=["Vénéneux (0)", "Comestible (1)"],
        yticklabels=["Vénéneux (0)", "Comestible (1)"],
        cbar_kws={"label": "Fréquence"}, ax=ax,
    )
    ax.set_title(
        f"Matrice de confusion — Validation croisée ({CV_FOLDS}-fold)",
        fontsize=14, fontweight="bold",
    )
    ax.set_ylabel("Vraie classe", fontsize=12)
    ax.set_xlabel("Classe prédite", fontsize=12)
    fig.tight_layout()
    save_figure(fig, "da_confusion_cv.png")

    # ── Résumé ──

    print()
    print_step("Analyse discriminante terminée.")
    print()
    print("  Outputs :")
    print("    Tables  — da_metrics.csv, da_confusion.csv")
    print("    Figures — da_confusion.png, da_cv_scores.png, da_confusion_cv.png")
    print()


if __name__ == "__main__":
    perform_discriminant_analysis()
