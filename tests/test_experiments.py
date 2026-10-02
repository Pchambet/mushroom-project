import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from mushroom import experiments as ex
from mushroom.config import cv_splitter
from mushroom.pipeline import describe_tree


def test_correlation_ratio_hand_cases():
    groups = np.array([0, 0, 1, 1])
    assert ex.correlation_ratio(np.array([1.0, 1.0, 3.0, 3.0]), groups) == pytest.approx(1.0)
    assert ex.correlation_ratio(np.array([1.0, 3.0, 1.0, 3.0]), groups) == pytest.approx(0.0)
    # group means 1 and 3, grand mean 2: between = 4 * 1 = 4, total = 4 + 0 + 0 + 4 = 8
    assert ex.correlation_ratio(np.array([0.0, 2.0, 2.0, 4.0]), groups) == pytest.approx(0.5)


def test_splitter_is_shuffled_and_stratified():
    y = np.array([0] * 50 + [1] * 50)  # sorted, like the UCI file
    splitter = cv_splitter()
    assert isinstance(splitter, StratifiedKFold) and splitter.shuffle
    for _, test in splitter.split(np.zeros(len(y)), y):
        assert y[test].mean() == pytest.approx(0.5)
        assert np.diff(test).max() > 1  # not a contiguous block


def test_inertia_table_is_consistent(sample):
    X, y = sample
    table = ex.inertia_table(X, y)
    assert len(table) == ex.n_mca_axes(X) < X.nunique().sum() - X.shape[1]
    assert table["eigenvalue"].min() > 1e-12
    assert table["cumulative_inertia_pct"].iloc[-1] == pytest.approx(100)
    assert table["eta2_class"].between(0, 1).all()


def test_evaluate_counts_errors_against_the_poisonous_class(sample):
    X, y = sample
    scores = ex.evaluate(ex.onehot_model(ex.DecisionTreeClassifier(), ["odor"]), X, y)
    n_errors = scores["poisonous_called_edible"] + scores["edible_called_poisonous"]
    assert n_errors == pytest.approx(len(y) * (1 - scores["cv_accuracy_mean"]), abs=1)


def test_mca_pipeline_reaches_high_accuracy_on_real_sample(sample):
    X, y = sample
    scores = ex.evaluate(ex.mca_model(5, ex.KNeighborsClassifier(5)), X, y)
    assert scores["cv_accuracy_mean"] > 0.9


def test_clusters_table_sums_to_sample(sample):
    X, y = sample
    table, ari = ex.clusters(X, y)
    assert table["size"].sum() == len(y)
    assert -1 <= ari <= 1


def test_smallest_perfect_depth():
    sweep = pd.DataFrame({"depth": [1, 2, 3, 4], "mean": [0.8, 0.99, 1.0, 1.0]})
    assert ex.smallest_perfect_depth(sweep) == 3
    assert ex.reference_tree_depth(sweep) == 3


def test_no_perfect_depth_is_not_reported_as_perfect():
    sweep = pd.DataFrame({"depth": [1, 2, 3, 4], "mean": [0.8, 0.995, 0.999, 0.998]})
    assert ex.smallest_perfect_depth(sweep) is None
    assert ex.reference_tree_depth(sweep) == 3  # best score, not the deepest tree


@pytest.mark.parametrize("seed", [0, 13, 42])
def test_axis_curve_all_axes_point_survives_folds_with_fewer_axes(sample, monkeypatch, seed):
    # On the 301-row fixture every training fold has fewer MCA axes than the full
    # sample (rare categories fall out), so a fixed "all axes" count would raise.
    X, y = sample
    monkeypatch.setattr(
        ex, "cv_splitter", lambda: StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    )
    monkeypatch.setattr(ex, "classifiers", lambda: {"LDA": ex.shrinkage_lda})
    curve = ex.axis_curve(X, y, grid=[2])
    for _, sub in curve.groupby("model"):
        assert sub["n_axes"].tolist() == [2, ex.n_mca_axes(X)]
    assert curve["mean"].between(0, 1).all()


def test_label_is_an_exact_linear_function_of_the_indicators(sample):
    X, y = sample
    assert ex.label_linear_residual(X, y) < 1e-8
    # Without odour and spore print colour, the strongest predictors, it no longer is.
    assert ex.label_linear_residual(X.drop(columns=["odor", "spore-print-color"]), y) > 0.1


def test_describe_tree_only_claims_perfection_when_reached():
    perfect = {"tree_depth": 7, "tree_leaves": 14, "tree_accuracy": 1.0, "tree_is_perfect": True}
    assert describe_tree(perfect) == "a depth-7 tree (14 leaves) is perfect"
    short = perfect | {"tree_depth": 6, "tree_accuracy": 0.99963, "tree_is_perfect": False}
    assert describe_tree(short) == "a depth-6 tree (14 leaves) reaches 99.96%"
