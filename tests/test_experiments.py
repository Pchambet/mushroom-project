import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from mushroom import experiments as ex
from mushroom.config import cv_splitter


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
    assert len(table) == ex.n_nontrivial_axes(X)
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
