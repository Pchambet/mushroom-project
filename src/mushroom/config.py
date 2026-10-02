"""Paths, data source and the shared visual identity."""

from __future__ import annotations

from pathlib import Path

from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_CSV = ROOT / "data" / "processed" / "mushroom.csv"
TABLES_DIR = ROOT / "reports" / "tables"
RESULTS_JSON = ROOT / "reports" / "results.json"
FIGURES_DIR = ROOT / "docs" / "figures"
SITE_HTML = ROOT / "site" / "index.html"

# UCI Machine Learning Repository, "Mushroom" (id 73), CC BY 4.0.
DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
)
DATA_SHA256 = "e65d082030501a3ebcbcd7c9f7c71aa9d28fdfff463bf4cf4716a3fe13ac360e"

SEED = 42
N_FOLDS = 5


def cv_splitter() -> StratifiedKFold:
    """The one splitter used everywhere.

    The UCI file is sorted in long same-class runs, so an unshuffled split puts
    very different class mixes in each fold; shuffling is not optional here.
    """
    return StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)


INK = "#0f172a"
TEAL = "#0d9488"
AMBER = "#d97706"
SLATE = "#64748b"
GRID = "#e2e8f0"
CLASS_COLORS = {"edible": TEAL, "poisonous": AMBER}
