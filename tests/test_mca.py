import numpy as np
import pandas as pd
import pytest

from mushroom.mca import MCA


def test_total_inertia_matches_closed_form(sample):
    X, _ = sample
    mca = MCA().fit(X)
    n_categories, n_variables = X.nunique().sum(), X.shape[1]
    assert mca.total_inertia_ == pytest.approx((n_categories - n_variables) / n_variables)
    # Every non-trivial axis kept: the eigenvalues exhaust the total inertia.
    assert mca.eigenvalues_.sum() == pytest.approx(mca.total_inertia_)
    assert mca.explained_inertia_.sum() == pytest.approx(1.0)


def test_row_coordinates_are_centred_with_variance_equal_to_eigenvalue(sample):
    X, _ = sample
    mca = MCA(n_components=4).fit(X)
    coords = mca.transform(X)
    np.testing.assert_allclose(coords.mean(axis=0), 0, atol=1e-10)
    np.testing.assert_allclose(coords.var(axis=0), mca.eigenvalues_, rtol=1e-8)


def test_two_variable_case_matches_hand_computation():
    # Two binary variables that always agree: one non-trivial axis with
    # eigenvalue 1 (perfect association), rows at +1 / -1.
    X = pd.DataFrame({"a": list("xxyy"), "b": list("uuvv")})
    mca = MCA().fit(X)
    assert mca.eigenvalues_ == pytest.approx([1.0, 0.0], abs=1e-12)
    np.testing.assert_allclose(np.abs(mca.transform(X)[:, 0]), 1.0)


def test_recovers_planted_groups():
    # Ground truth: two latent groups that pick their categories from disjoint
    # halves of each variable's levels, plus noise variables. Axis 1 must
    # separate the groups perfectly.
    rng = np.random.default_rng(0)
    n = 400
    group = rng.integers(0, 2, n)
    informative = {
        f"v{j}": np.where(group == 0, rng.choice(["a", "b"], n), rng.choice(["c", "d"], n))
        for j in range(4)
    }
    noise = {f"noise{j}": rng.choice(list("pqrs"), n) for j in range(4)}
    X = pd.DataFrame(informative | noise)
    axis1 = MCA(n_components=2).fit(X).transform(X)[:, 0]
    assert (axis1[group == 0].max() < axis1[group == 1].min()) or (
        axis1[group == 1].max() < axis1[group == 0].min()
    )


def test_transform_of_unseen_category_is_finite(sample):
    X, _ = sample
    mca = MCA(n_components=3).fit(X)
    new = X.head(2).copy()
    new.iloc[0, 0] = "never-seen"
    assert np.isfinite(mca.transform(new)).all()


def test_benzecri_shares_sum_to_one_when_all_axes_kept(sample):
    X, _ = sample
    mca = MCA().fit(X)
    shares = mca.benzecri_explained_inertia()
    assert shares.sum() == pytest.approx(1.0)
    assert (np.diff(shares) <= 1e-12).all()


def test_rejects_impossible_number_of_axes(sample):
    X, _ = sample
    with pytest.raises(ValueError):
        MCA(n_components=10_000).fit(X)


def test_matches_prince_reference_implementation(sample):
    prince = pytest.importorskip("prince")
    X, _ = sample
    ours = MCA(n_components=5).fit(X)
    ref = prince.MCA(n_components=5).fit(X)
    np.testing.assert_allclose(ours.eigenvalues_, np.asarray(ref.eigenvalues_), rtol=1e-6)
    coords_ref = ref.transform(X).to_numpy()
    for i in range(5):
        corr = np.corrcoef(ours.transform(X)[:, i], coords_ref[:, i])[0, 1]
        assert abs(corr) == pytest.approx(1.0, abs=1e-8)
