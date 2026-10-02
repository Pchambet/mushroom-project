"""Every number in the README comes from one of these functions.

All supervised scores are out-of-fold: the MCA, the encoder and the classifier
are refitted inside each training fold, so the test fold never shapes the axes.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from mushroom.config import SEED, cv_splitter
from mushroom.mca import MCA

N_JOBS = 3
AXIS_GRID = [*range(1, 21), 25, 30, 40, 60, 70, 80]
CLUSTER_AXES = 5
NO_SHRINKAGE = "LDA, no shrinkage (sklearn default)"
N_CLUSTERS = 3


def shrinkage_lda() -> LinearDiscriminantAnalysis:
    """LDA with Ledoit-Wolf shrinkage of the within-class covariance.

    On UCI Mushroom the label is an exact linear function of the indicator
    columns, so with every MCA axis kept the within-class covariance is singular
    precisely along the discriminant direction. The default SVD solver discards
    that direction and collapses (see ``reference_models``); shrinkage keeps it.
    With few axes the two agree to within a few specimens.
    """
    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")


def classifiers() -> dict[str, Callable[[], ClassifierMixin]]:
    """A linear reader, a local reader and a flexible reader of the MCA space."""
    return {
        "LDA": shrinkage_lda,
        "k-NN (15)": lambda: KNeighborsClassifier(n_neighbors=15),
        "Random forest": lambda: RandomForestClassifier(
            n_estimators=200, random_state=SEED, n_jobs=N_JOBS
        ),
    }


def mca_model(n_axes: int | None, clf: ClassifierMixin) -> Pipeline:
    return make_pipeline(MCA(n_components=n_axes), clf)


def onehot_model(clf: ClassifierMixin, columns: list[str] | None = None) -> Pipeline:
    encoder = OneHotEncoder(handle_unknown="ignore")
    if columns is None:
        return make_pipeline(encoder, clf)
    return make_pipeline(ColumnTransformer([("onehot", encoder, columns)]), clf)


def n_mca_axes(X: pd.DataFrame) -> int:
    """Axes with non-zero inertia; below the J - Q bound when descriptors are redundant."""
    return MCA().fit(X).rank_


def label_linear_residual(X: pd.DataFrame, y: pd.Series) -> float:
    """Largest residual of a least-squares fit of the 0/1 label on the indicator columns.

    Zero (up to rounding) means the label is an exact linear function of the
    categories, which is what makes the all-axes within-class covariance singular.
    """
    Z = OneHotEncoder(sparse_output=False).fit_transform(X)
    w, *_ = np.linalg.lstsq(Z, y.to_numpy(dtype=float), rcond=None)
    return float(np.abs(Z @ w - y.to_numpy()).max())


# ── Unsupervised description ──────────────────────────────────────────────


def correlation_ratio(coord: np.ndarray, groups: np.ndarray) -> float:
    """eta^2: share of an axis' variance explained by group membership."""
    grand = coord.mean()
    between = sum(
        (groups == g).sum() * (coord[groups == g].mean() - grand) ** 2 for g in np.unique(groups)
    )
    total = ((coord - grand) ** 2).sum()
    return float(between / total)


def inertia_table(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    mca = MCA().fit(X)
    coords = mca.transform(X)
    return pd.DataFrame(
        {
            "axis": np.arange(1, len(mca.eigenvalues_) + 1),
            "eigenvalue": mca.eigenvalues_,
            "inertia_pct": 100 * mca.explained_inertia_,
            "cumulative_inertia_pct": 100 * np.cumsum(mca.explained_inertia_),
            "benzecri_pct": 100 * mca.benzecri_explained_inertia(),
            "eta2_class": [correlation_ratio(c, y.to_numpy()) for c in coords.T],
        }
    )


def axis1_categories(X: pd.DataFrame, y: pd.Series, top: int = 12) -> pd.DataFrame:
    """Categories that build axis 1, with how poisonous their specimens are."""
    mca = MCA(n_components=1).fit(X)
    onehot = mca.encoder_.transform(X)
    mass = onehot.sum(axis=0) / onehot.sum()
    coord = mca.column_coordinates_["Dim1"].to_numpy()
    contribution = mass * coord**2 / mca.eigenvalues_[0]
    poisonous_share = (onehot * y.to_numpy()[:, None]).sum(axis=0) / onehot.sum(axis=0)
    table = pd.DataFrame(
        {
            "category": [n.replace("_", " = ", 1) for n in mca.column_coordinates_.index],
            "coordinate": coord,
            "contribution_pct": 100 * contribution,
            "poisonous_pct": 100 * poisonous_share,
            "n_specimens": onehot.sum(axis=0).astype(int),
        }
    )
    return table.sort_values("contribution_pct", ascending=False).head(top)


def clusters(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, float]:
    """K-means on the first MCA axes, compared with the labels it never saw."""
    coords = MCA(n_components=CLUSTER_AXES).fit(X).transform(X)
    labels = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=SEED).fit_predict(coords)
    # Name clusters by size so the table is stable across library versions.
    order = {c: i for i, c in enumerate(pd.Series(labels).value_counts().index)}
    labels = np.vectorize(order.get)(labels)
    table = pd.crosstab(labels, y.map({0: "edible", 1: "poisonous"}))
    table.index.name = "cluster"
    table = table.reset_index()
    table["size"] = table["edible"] + table["poisonous"]
    table["poisonous_pct"] = 100 * table["poisonous"] / table["size"]
    return table, float(adjusted_rand_score(y, labels))


# ── Supervised, cross-validated ───────────────────────────────────────────


def axis_curve(X: pd.DataFrame, y: pd.Series, grid: list[int] = AXIS_GRID) -> pd.DataFrame:
    """Out-of-fold accuracy as a function of the number of MCA axes kept.

    The default-solver LDA is tracked too, to show where it departs from the
    shrinkage LDA used everywhere else.
    """
    all_axes = n_mca_axes(X)
    rows = []
    for name, make in (classifiers() | {NO_SHRINKAGE: LinearDiscriminantAnalysis}).items():
        for k in [*[k for k in grid if k < all_axes], all_axes]:
            # The last point keeps every axis of each training fold (None), which
            # stays valid even if a rare category is absent from a fold.
            n_components = None if k == all_axes else k
            model = mca_model(n_components, make())
            scores = cross_val_score(model, X, y, cv=cv_splitter())
            rows.append({"model": name, "n_axes": k, "mean": scores.mean(), "std": scores.std()})
    return pd.DataFrame(rows)


def evaluate(model: BaseEstimator, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Accuracy plus the two error counts that matter for a forager."""
    scores = cross_val_score(clone(model), X, y, cv=cv_splitter())
    pred = cross_val_predict(clone(model), X, y, cv=cv_splitter())
    _, fp, fn, _ = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "cv_accuracy_mean": scores.mean(),
        "cv_accuracy_std": scores.std(),
        "poisonous_called_edible": int(fn),
        "edible_called_poisonous": int(fp),
    }


def reference_models(X: pd.DataFrame, y: pd.Series, tree_depth: int) -> pd.DataFrame:
    all_axes = n_mca_axes(X)
    n_categories = int(X.nunique().sum())
    onehot = f"one-hot ({n_categories} columns)"
    candidates = [
        ("Majority class", "none", DummyClassifier(), 0),
        (
            "Odour-only lookup",
            "odor",
            onehot_model(DecisionTreeClassifier(), ["odor"]),
            X["odor"].nunique(),
        ),
        ("LDA", "MCA, 1 axis", mca_model(1, shrinkage_lda()), 1),
        ("LDA", "MCA, 5 axes", mca_model(5, shrinkage_lda()), 5),
        ("k-NN (15)", "MCA, 5 axes", mca_model(5, KNeighborsClassifier(15)), 5),
        ("Random forest", "MCA, 5 axes", mca_model(5, classifiers()["Random forest"]()), 5),
        ("LDA", f"MCA, all {all_axes} axes", mca_model(None, shrinkage_lda()), all_axes),
        (
            NO_SHRINKAGE,
            f"MCA, all {all_axes} axes",
            mca_model(None, LinearDiscriminantAnalysis()),
            all_axes,
        ),
        (
            "Logistic regression",
            onehot,
            onehot_model(LogisticRegression(max_iter=2000)),
            n_categories,
        ),
        (
            f"Decision tree (depth {tree_depth})",
            onehot,
            onehot_model(DecisionTreeClassifier(max_depth=tree_depth, random_state=SEED)),
            n_categories,
        ),
    ]
    rows = []
    for name, features, model, n_features in candidates:
        rows.append(
            {"model": name, "features": features, "n_features": n_features} | evaluate(model, X, y)
        )
    return pd.DataFrame(rows)


def tree_depth_sweep(X: pd.DataFrame, y: pd.Series, max_depth: int = 8) -> pd.DataFrame:
    rows = []
    for depth in range(1, max_depth + 1):
        model = onehot_model(DecisionTreeClassifier(max_depth=depth, random_state=SEED))
        scores = cross_val_score(model, X, y, cv=cv_splitter())
        n_leaves = model.fit(X, y)[-1].get_n_leaves()
        rows.append(
            {"depth": depth, "n_leaves": n_leaves, "mean": scores.mean(), "std": scores.std()}
        )
    return pd.DataFrame(rows)


def unseen_in_training(X: pd.DataFrame, train: np.ndarray, test: np.ndarray) -> tuple[int, int]:
    """Categories in the test rows that the training rows never show, and specimens carrying one.

    The MCA cannot place a category it has never seen (it contributes nothing to the
    row's coordinates), so these counts say how far a test fold lies outside training.
    """
    unseen = np.zeros(len(test), dtype=bool)
    n_categories = 0
    for column in X.columns:
        test_values = X[column].iloc[test]
        missing = ~test_values.isin(set(X[column].iloc[train])).to_numpy()
        n_categories += test_values[missing].nunique()
        unseen |= missing
    return n_categories, int(unseen.sum())


def fold_ordering(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Per-fold accuracy with sklearn's default ``cv=5`` (no shuffle) versus shuffled folds.

    ``cv=5`` on a classifier is stratified, so every test fold has the same class mix.
    But it does not shuffle, and the UCI file is ordered in blocks of similar descriptor
    patterns (roughly by species), so each unshuffled test fold is a contiguous block
    whose categories may never appear in its training folds. The unshuffled spread is
    therefore a rough proxy for performance on species missing from training, not a
    property of the model on the population that random folds sample.
    """
    splitters = {
        "unshuffled (cv=5)": StratifiedKFold(n_splits=5),
        "shuffled": cv_splitter(),
    }
    fold_facts = {
        split_name: [
            (len(test), 100 * y.iloc[test].mean(), *unseen_in_training(X, train, test))
            for train, test in splitter.split(X, y)
        ]
        for split_name, splitter in splitters.items()
    }
    models = {
        "LDA, 5 axes": mca_model(5, shrinkage_lda()),
        "Random forest, 5 axes": mca_model(5, classifiers()["Random forest"]()),
    }
    rows = []
    for model_name, model in models.items():
        for split_name, splitter in splitters.items():
            scores = cross_val_score(clone(model), X, y, cv=splitter)
            rows += [
                {
                    "model": model_name,
                    "split": split_name,
                    "fold": i + 1,
                    "accuracy": score,
                    "test_size": size,
                    "test_poisonous_pct": poisonous_pct,
                    "unseen_categories": n_unseen,
                    "specimens_with_unseen": n_specimens,
                }
                for i, (score, (size, poisonous_pct, n_unseen, n_specimens)) in enumerate(
                    zip(scores, fold_facts[split_name], strict=True)
                )
            ]
    return pd.DataFrame(rows)


def smallest_perfect_depth(sweep: pd.DataFrame) -> int | None:
    """Shallowest depth with 100% out-of-fold accuracy, or None if no depth gets there."""
    perfect = sweep[sweep["mean"] >= 1.0]
    return int(perfect["depth"].min()) if not perfect.empty else None


def reference_tree_depth(sweep: pd.DataFrame) -> int:
    """The tree quoted as a reference: the shallowest perfect one, else the best-scoring."""
    depth = smallest_perfect_depth(sweep)
    return depth if depth is not None else int(sweep.loc[sweep["mean"].idxmax(), "depth"])
