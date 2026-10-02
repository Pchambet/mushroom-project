"""Static figures for the README, drawn from the saved result tables."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mushroom.config import (
    AMBER,
    CLASS_COLORS,
    FIGURES_DIR,
    GRID,
    INK,
    PROCESSED_CSV,
    RESULTS_JSON,
    SLATE,
    TABLES_DIR,
    TEAL,
)
from mushroom.data import load_processed
from mushroom.mca import MCA

MODEL_COLORS = {"LDA": SLATE, "k-NN (15)": AMBER, "Random forest": TEAL}

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": SLATE,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.titlesize": 12.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": SLATE,
        "ytick.color": SLATE,
        "font.size": 10.5,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)


def _table(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLES_DIR / f"{name}.csv")


def _save(fig: plt.Figure, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png")
    plt.close(fig)


def hero(results: dict) -> None:
    """Accuracy against number of axes: what compression keeps, and who can read it."""
    curve = _table("axis_curve")
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for model, color in MODEL_COLORS.items():
        sub = curve[curve["model"] == model]
        ax.plot(sub["n_axes"], 100 * sub["mean"], color=color, lw=2.2, marker="o", ms=3.5)
    odour = 100 * results["odour_accuracy"]
    ax.axhline(odour, color=INK, lw=1, ls="--")
    ax.text(1, odour + 0.25, f"odour alone: {odour:.1f}%", color=INK, fontsize=9.5)
    ax.axvline(5, color=SLATE, lw=1, ls=":")
    labels = [
        ("Random forest", TEAL, f"{100 * results['rf_5_axes']:.1f}% with 5 axes"),
        ("k-NN (15)", AMBER, f"{100 * results['knn_5_axes']:.1f}% with 5 axes"),
        (
            "LDA (linear)",
            SLATE,
            (
                f"{100 * results['lda_5_axes']:.1f}% with 5 axes, "
                f"99% only at {results['lda_axes_for_99']}"
            ),
        ),
    ]
    for i, (name, color, text) in enumerate(labels):
        y = 86.2 - 1.6 * i
        ax.text(8.2, y, f"{name}: {text}", color=color, fontsize=10, fontweight="bold")
    ax.set_xscale("log")
    ticks = [1, 2, 3, 5, 10, 20, 40, results["n_axes"]]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_ylim(80, 100.8)
    ax.set_xlim(0.9, results["n_axes"] * 1.25)
    ax.set_xlabel("MCA axes kept (log scale)")
    ax.set_ylabel("Out-of-fold accuracy (%)")
    ax.set_title(
        "Five MCA axes keep the toxicity signal, but a linear rule cannot read it",
        pad=12,
    )
    _save(fig, "hero")


def axis_signal(results: dict) -> None:
    inertia = _table("mca_inertia")
    fig, ax = plt.subplots(figsize=(9, 4.2))
    strong = inertia["eta2_class"] >= 0.05
    ax.bar(
        inertia["axis"],
        inertia["eta2_class"],
        color=np.where(strong, TEAL, SLATE),
        width=0.8,
    )
    for _, row in inertia[strong].iterrows():
        ax.text(
            row["axis"],
            row["eta2_class"] + 0.012,
            f"{int(row['axis'])}",
            ha="center",
            fontsize=8.5,
            color=TEAL,
        )
    ax.set_xlabel("MCA axis (ordered by inertia, largest first)")
    ax.set_ylabel("η² with the edible/poisonous label")
    ax.set_xlim(0, len(inertia) + 1)
    ax.set_title(
        "The class signal sits on axis 1 and on low-inertia axes a variance cut would drop",
        pad=10,
    )
    _save(fig, "axis_signal")


def fold_ordering(results: dict) -> None:
    folds = _table("fold_ordering")
    floor = 100 * folds["accuracy"].min() - 7
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, (model, sub) in zip(axes, folds.groupby("model", sort=False), strict=True):
        for i, (split, part) in enumerate(sub.groupby("split", sort=False)):
            color = AMBER if split.startswith("unshuffled") else TEAL
            x = i + np.linspace(-0.15, 0.15, len(part))
            ax.scatter(x, 100 * part["accuracy"], color=color, s=36, zorder=3)
            mean, std = 100 * part["accuracy"].mean(), 100 * part["accuracy"].std(ddof=0)
            ax.hlines(mean, i - 0.28, i + 0.28, color=color, lw=2)
            ax.text(
                i, floor + 1.5, f"{mean:.1f} ± {std:.1f}", ha="center", color=color, fontsize=9.5
            )
        ax.set_xticks([0, 1], ["unshuffled\n(sklearn cv=5)", "shuffled\nstratified"])
        ax.set_xlim(-0.6, 1.6)
        ax.set_title(model, fontsize=11)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Accuracy per fold (%)")
    axes[0].set_ylim(floor, 101)
    fig.suptitle(
        "The 'unstable model' was the file order: shuffling the folds removes the spread",
        x=0.01,
        ha="left",
        fontweight="bold",
        color=INK,
        fontsize=12.5,
    )
    fig.tight_layout()
    _save(fig, "fold_ordering")


def decision_errors(results: dict) -> None:
    ref = _table("reference_models")
    ref = ref[ref["model"] != "Majority class"].iloc[::-1]
    labels = [f"{m}  ·  {f}" for m, f in zip(ref["model"], ref["features"], strict=True)]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    bars = ax.barh(
        labels,
        ref["poisonous_called_edible"],
        color=[AMBER if v > 0 else TEAL for v in ref["poisonous_called_edible"]],
    )
    for bar, n, acc in zip(
        bars, ref["poisonous_called_edible"], ref["cv_accuracy_mean"], strict=True
    ):
        ax.text(
            bar.get_width() + 8,
            bar.get_y() + bar.get_height() / 2,
            f"{n}   (accuracy {100 * acc:.2f}%)",
            va="center",
            fontsize=9,
            color=INK,
        )
    ax.set_xlabel(f"Poisonous specimens predicted edible (out of {int(results['n_poisonous'])})")
    ax.set_xlim(0, ref["poisonous_called_edible"].max() * 1.45)
    ax.grid(axis="y", visible=False)
    ax.set_title(
        f"Odour alone is {100 * results['odour_accuracy']:.1f}% accurate, and every one of "
        f"its {results['odour_poisonous_called_edible']} errors is a poisonous mushroom",
        pad=10,
    )
    _save(fig, "decision_errors")


def mca_map() -> None:
    X, y = load_processed(PROCESSED_CSV)
    mca = MCA(n_components=2).fit(X)
    coords = mca.transform(X)
    fig, ax = plt.subplots(figsize=(7.5, 5.6))
    for label, name in [(0, "edible"), (1, "poisonous")]:
        pts = coords[y.to_numpy() == label]
        ax.scatter(
            pts[:, 0], pts[:, 1], s=6, alpha=0.35, color=CLASS_COLORS[name], label=name, lw=0
        )
    pct = 100 * mca.explained_inertia_
    ax.set_xlabel(f"Axis 1 ({pct[0]:.1f}% of total inertia)")
    ax.set_ylabel(f"Axis 2 ({pct[1]:.1f}% of total inertia)")
    ax.legend(frameon=False, markerscale=3)
    ax.set_title("Axis 1 splits most edible from most poisonous specimens", pad=10)
    _save(fig, "mca_map")


def build() -> None:
    results = json.loads(RESULTS_JSON.read_text())
    hero(results)
    axis_signal(results)
    fold_ordering(results)
    decision_errors(results)
    mca_map()
