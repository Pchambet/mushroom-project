"""Multiple correspondence analysis as a scikit-learn transformer.

MCA is correspondence analysis of the one-hot (indicator) matrix. Implementing it
in ~60 lines of numpy, instead of calling a library, buys two things this project
needs: it can sit inside a cross-validation pipeline (fit on the training fold
only, project the test fold with the transition formula), and its inertia
bookkeeping is explicit. Explained inertia is reported against the *total*
inertia (J - Q) / Q, not against the axes that happen to be kept.

J - Q is only an upper bound on the number of axes. When some descriptors are
exactly determined by others (as on UCI Mushroom), the indicator matrix has a
lower rank and the remaining singular values are rounding error. Those
null axes are dropped: their coordinates are noise, and a model that rescales
features (LDA, k-NN after scaling) would happily learn from it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder


class MCA(TransformerMixin, BaseEstimator):
    """Multiple correspondence analysis on categorical columns.

    Parameters
    ----------
    n_components:
        Number of principal axes to keep. ``None`` keeps every axis with
        non-zero inertia.

    Attributes
    ----------
    eigenvalues_:
        Principal inertias of the kept axes (squared singular values).
    rank_:
        Number of axes with non-zero inertia (numerical rank of the centred
        indicator matrix), at most ``J - Q``.
    all_eigenvalues_:
        Principal inertias of those ``rank_`` axes.
    total_inertia_:
        Total inertia of the indicator matrix, ``(J - Q) / Q``.
    explained_inertia_:
        Share of ``total_inertia_`` carried by each kept axis.
    column_coordinates_:
        Principal coordinates of each category (one row per category).
    """

    def __init__(self, n_components: int | None = None) -> None:
        self.n_components = n_components

    def fit(self, X: pd.DataFrame, y: object = None) -> MCA:
        self.encoder_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        Z = self.encoder_.fit_transform(_as_frame(X))
        n_cats = Z.shape[1]
        self.n_variables_ = X.shape[1]

        # Correspondence matrix, row and column masses.
        P = Z / Z.sum()
        r = P.sum(axis=1)
        c = P.sum(axis=0)
        S = (P - np.outer(r, c)) / np.sqrt(np.outer(r, c))
        _, sigma, Vt = np.linalg.svd(S, full_matrices=False)

        # Same tolerance as numpy.linalg.matrix_rank: anything below is rounding error.
        tol = sigma.max(initial=0.0) * max(S.shape) * np.finfo(S.dtype).eps
        self.rank_ = int(min((sigma > tol).sum(), n_cats - self.n_variables_))
        k = self.rank_ if self.n_components is None else self.n_components
        if not 1 <= k <= self.rank_:
            raise ValueError(f"n_components must be in [1, {self.rank_}], got {k}")

        # SVD signs are arbitrary; fix them so the largest loading is positive.
        V = Vt[:k].T
        signs = np.sign(V[np.abs(V).argmax(axis=0), np.arange(k)])
        V = V * signs

        self.all_eigenvalues_ = sigma[: self.rank_] ** 2
        self.eigenvalues_ = self.all_eigenvalues_[:k]
        self.total_inertia_ = (n_cats - self.n_variables_) / self.n_variables_
        self.explained_inertia_ = self.eigenvalues_ / self.total_inertia_
        self._projection = V / np.sqrt(c)[:, None]
        self.column_coordinates_ = pd.DataFrame(
            self._projection * sigma[:k],
            index=self.encoder_.get_feature_names_out(),
            columns=[f"Dim{i + 1}" for i in range(k)],
        )
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Row principal coordinates via the transition formula ``(Z / Q) D_c^{-1/2} V``.

        Each row is divided by its number of *known* categories rather than by Q:
        a category unseen at fit time is encoded as an all-zero block, and
        dividing by Q would pull that row toward the origin. A row with no known
        category lands at the origin (the barycentre).
        """
        Z = self.encoder_.transform(_as_frame(X))
        n_known = np.maximum(Z.sum(axis=1, keepdims=True), 1)
        return Z @ self._projection / n_known

    def benzecri_explained_inertia(self) -> np.ndarray:
        """Benzécri-corrected share of inertia for axes with eigenvalue above 1/Q.

        The raw MCA percentages are pessimistic because the indicator coding
        inflates total inertia; Benzécri's correction is the usual remedy.
        """
        q = self.n_variables_
        lam = self.all_eigenvalues_
        adjusted = np.where(lam > 1 / q, (q / (q - 1)) ** 2 * (lam - 1 / q) ** 2, 0.0)
        return (adjusted / adjusted.sum())[: len(self.eigenvalues_)]


def _as_frame(X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    return X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
