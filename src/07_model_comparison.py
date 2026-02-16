"""
07 — Comparaison de modeles de classification.

Compare 4 classifieurs sur les coordonnees ACM (k=5) avec validation
croisee 5-fold : LDA, Logistic Regression, Random Forest, SVM.

Repond a la question : "Est-ce que 88.7% c'est bien ? Peut-on faire mieux ?"
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_score, recall_score, f1_score,
)
import warnings

from utils import (
    load_processed_data, load_mca_coordinates,
    save_figure, save_table, print_section, print_step,
)


# ── Configuration ──────────────────────────────────────────

K_AXES = 5
CV_FOLDS = 5
RANDOM_STATE = 42

MODELS = {
    "LDA": LinearDiscriminantAnalysis(),
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=RANDOM_STATE,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=RANDOM_STATE,
    ),
    "SVM (RBF)": SVC(
        kernel="rbf", random_state=RANDOM_STATE,
    ),
}


# ── Pipeline ───────────────────────────────────────────────

def model_comparison() -> pd.DataFrame:
    """Compare plusieurs classifieurs sur les coordonnees ACM.

    Returns
    -------
    pd.DataFrame
        Tableau comparatif de tous les modeles.
    """

    print_section("07 — Comparaison de modeles de classification")

    coords = load_mca_coordinates()
    df = load_processed_data()

    X = coords.iloc[:, :K_AXES].values
    y = (df["class"] == "e").astype(int)

    print_step(f"Donnees : {X.shape} (k={K_AXES} axes ACM)")
    print(f"    Poisonous: {(y == 0).sum():,}  |  Edible: {(y == 1).sum():,}")

    all_results = []
    cv_scores_dict = {}

    for name, model in MODELS.items():
        print()
        print_step(f"{name}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # CV 5-fold
            scores = cross_val_score(model, X, y, cv=CV_FOLDS, scoring="accuracy")
            cv_scores_dict[name] = scores

            # CV predictions pour metriques detaillees
            y_pred_cv = cross_val_predict(model, X, y, cv=CV_FOLDS)

            # Train
            model.fit(X, y)
            train_acc = model.score(X, y)

        # Metriques
        cv_mean = scores.mean()
        cv_std = scores.std()
        prec_p = precision_score(y, y_pred_cv, pos_label=0)
        rec_p = recall_score(y, y_pred_cv, pos_label=0)
        prec_e = precision_score(y, y_pred_cv, pos_label=1)
        rec_e = recall_score(y, y_pred_cv, pos_label=1)
        f1_macro = f1_score(y, y_pred_cv, average="macro")

        print(
            f"    Train={train_acc:.3f}  "
            f"CV={cv_mean:.3f}+/-{cv_std:.3f}  "
            f"F1(macro)={f1_macro:.3f}"
        )

        all_results.append({
            "model": name,
            "train_accuracy": round(train_acc, 4),
            "cv_accuracy_mean": round(cv_mean, 4),
            "cv_accuracy_std": round(cv_std, 4),
            "precision_poisonous": round(prec_p, 4),
            "recall_poisonous": round(rec_p, 4),
            "precision_edible": round(prec_e, 4),
            "recall_edible": round(rec_e, 4),
            "f1_macro": round(f1_macro, 4),
            "overfitting_gap": round(train_acc - cv_mean, 4),
        })

    results_df = pd.DataFrame(all_results).sort_values("cv_accuracy_mean", ascending=False)
    save_table(results_df, "model_comparison.csv")

    best = results_df.iloc[0]
    print()
    print_step(
        f"Meilleur modele (CV) : {best['model']} "
        f"({best['cv_accuracy_mean']:.4f} accuracy)"
    )

    # ── Figure 1 : Comparaison globale ──

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1 : Accuracy (train vs CV)
    model_names = results_df["model"].tolist()
    x_pos = np.arange(len(model_names))
    width = 0.35

    bars1 = axes[0].bar(
        x_pos - width / 2, results_df["train_accuracy"],
        width, label="Train", color="#2196F3", alpha=0.85, edgecolor="black", linewidth=0.5,
    )
    bars2 = axes[0].bar(
        x_pos + width / 2, results_df["cv_accuracy_mean"],
        width, label="CV 5-fold", color="#F44336", alpha=0.85, edgecolor="black", linewidth=0.5,
    )
    axes[0].errorbar(
        x_pos + width / 2, results_df["cv_accuracy_mean"],
        yerr=results_df["cv_accuracy_std"],
        fmt="none", ecolor="black", capsize=4,
    )
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(model_names, rotation=15, ha="right")
    axes[0].set_ylabel("Accuracy", fontsize=12)
    axes[0].set_title("Accuracy — Train vs CV", fontsize=13, fontweight="bold")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].set_ylim([0.6, 1.05])

    # Panel 2 : F1 macro + Recall poisonous
    axes[1].bar(
        x_pos - width / 2, results_df["f1_macro"],
        width, label="F1 (macro)", color="#4CAF50", alpha=0.85, edgecolor="black", linewidth=0.5,
    )
    axes[1].bar(
        x_pos + width / 2, results_df["recall_poisonous"],
        width, label="Recall venéneux", color="#FF9800", alpha=0.85, edgecolor="black", linewidth=0.5,
    )
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(model_names, rotation=15, ha="right")
    axes[1].set_ylabel("Score", fontsize=12)
    axes[1].set_title("F1 macro + Recall venéneux", fontsize=13, fontweight="bold")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].set_ylim([0.5, 1.05])

    fig.tight_layout()
    save_figure(fig, "model_comparison.png")

    # ── Figure 2 : Box plots CV scores ──

    fig, ax = plt.subplots(figsize=(10, 6))
    box_data = [cv_scores_dict[name] for name in model_names]
    bp = ax.boxplot(
        box_data, tick_labels=model_names, patch_artist=True,
        boxprops=dict(facecolor="#E3F2FD", edgecolor="black"),
        medianprops=dict(color="#F44336", linewidth=2),
    )
    colors_box = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
    for patch, color in zip(bp["boxes"], colors_box[:len(bp["boxes"])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.3)

    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Distribution des scores CV (5-fold)", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, "model_comparison_boxplot.png")

    # ── Resume ──

    print()
    print_step("Comparaison de modeles terminee.")
    print()
    print("  Outputs :")
    print("    Tables  — model_comparison.csv")
    print("    Figures — model_comparison.png, model_comparison_boxplot.png")
    print()

    return results_df


if __name__ == "__main__":
    model_comparison()
