"""Interactive companion to the study: explore the MCA space and re-run the models.

Run locally with ``make app`` (or ``uv run streamlit run app.py``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, confusion_matrix
from sklearn.model_selection import cross_val_predict

# Streamlit Community Cloud installs requirements.txt but not the project itself.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from mushroom import experiments as ex
from mushroom.config import (
    CLASS_COLORS,
    PROCESSED_CSV,
    RESULTS_JSON,
    TABLES_DIR,
    cv_splitter,
)
from mushroom.data import load_processed
from mushroom.mca import MCA

REPO = "https://github.com/Pchambet/mushroom-project"
REPORT = "https://pchambet.github.io/mushroom-project/"
LABELS = {0: "edible", 1: "poisonous"}

st.set_page_config(page_title="Mushroom MCA study", layout="wide")


@st.cache_data
def load() -> tuple[pd.DataFrame, pd.Series]:
    return load_processed(PROCESSED_CSV)


@st.cache_data
def mca_space(n_axes: int = 20) -> tuple[pd.DataFrame, list[float]]:
    X, _ = load()
    mca = MCA(n_components=n_axes).fit(X)
    coords = pd.DataFrame(mca.transform(X), columns=[f"Axis {i + 1}" for i in range(n_axes)])
    return coords, list(100 * mca.explained_inertia_)


@st.cache_data
def oof_predictions(model_name: str, n_axes: int) -> pd.Series:
    X, y = load()
    model = ex.mca_model(n_axes, ex.classifiers()[model_name]())
    return pd.Series(cross_val_predict(clone(model), X, y, cv=cv_splitter()))


X, y = load()
results = json.loads(RESULTS_JSON.read_text())
classes = y.map(LABELS)

st.sidebar.title("Mushroom MCA study")
page = st.sidebar.radio(
    "Page", ["Overview", "MCA space", "Classification", "Clustering"], label_visibility="collapsed"
)
st.sidebar.markdown(f"[Code and README]({REPO})  \n[Static report]({REPORT})")
st.sidebar.caption(
    f"{len(X):,} specimens · {X.shape[1]} variables · data: UCI Mushroom (CC BY 4.0)"
)

if page == "Overview":
    st.title("What does an MCA keep, and lose, on the UCI Mushroom data?")
    st.markdown(
        f"The data are almost perfectly separable: odour alone is "
        f"{results['odour_accuracy']:.1%} accurate and a depth-{results['tree_perfect_depth']} "
        "decision tree is perfect. The question here is what a low-dimensional "
        "multiple correspondence analysis preserves, and which models can read it. "
        "All scores are out-of-fold (5 shuffled stratified folds, MCA refitted per fold)."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Specimens", f"{len(X):,}")
    c2.metric(f"Edible ({(y == 0).mean():.1%})", f"{(y == 0).sum():,}")
    c3.metric(f"Poisonous ({(y == 1).mean():.1%})", f"{(y == 1).sum():,}")
    ref = pd.read_csv(TABLES_DIR / "reference_models.csv")
    ref["cv_accuracy_mean"] = (100 * ref["cv_accuracy_mean"]).round(2)
    st.subheader("Reference models (precomputed by the pipeline)")
    st.dataframe(
        ref[
            [
                "model",
                "features",
                "cv_accuracy_mean",
                "poisonous_called_edible",
                "edible_called_poisonous",
            ]
        ].rename(
            columns={
                "cv_accuracy_mean": "accuracy (%)",
                "poisonous_called_edible": "poisonous → edible",
                "edible_called_poisonous": "edible → poisonous",
            }
        ),
        hide_index=True,
        width="stretch",
    )

elif page == "MCA space":
    st.title("The MCA space")
    coords, inertia = mca_space()
    axes = list(coords.columns)
    c1, c2, c3 = st.columns(3)
    ax_x = c1.selectbox("Horizontal axis", axes, index=0)
    ax_y = c2.selectbox("Vertical axis", axes, index=1)
    colour = c3.selectbox("Colour by", ["class", *X.columns])
    plot = coords.assign(**{"class": classes}, **{c: X[c] for c in X.columns})
    fig = px.scatter(
        plot,
        x=ax_x,
        y=ax_y,
        color=colour,
        color_discrete_map=CLASS_COLORS if colour == "class" else None,
        opacity=0.45,
        hover_data=["odor", "spore-print-color"],
        render_mode="webgl",
    )
    fig.update_traces(marker_size=4)
    fig.update_layout(height=580)
    st.plotly_chart(fig, width="stretch")
    st.caption(
        f"{ax_x}: {inertia[axes.index(ax_x)]:.1f}% of total inertia · "
        f"{ax_y}: {inertia[axes.index(ax_y)]:.1f}%. Percentages are of the total inertia "
        "(J − Q) / Q, so they look small; that is normal for an MCA."
    )

elif page == "Classification":
    st.title("Classification on MCA axes")
    c1, c2 = st.columns(2)
    model_name = c1.selectbox("Model", list(ex.classifiers()))
    n_axes = c2.slider("MCA axes kept", 1, 30, 5)
    with st.spinner("Cross-validating…"):
        pred = oof_predictions(model_name, n_axes)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    m1, m2, m3 = st.columns(3)
    m1.metric("Out-of-fold accuracy", f"{(tn + tp) / len(y):.2%}")
    m2.metric("Poisonous called edible", f"{fn:,}")
    m3.metric("Edible called poisonous", f"{fp:,}")
    cm = pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=["true edible", "true poisonous"],
        columns=["predicted edible", "predicted poisonous"],
    )
    st.plotly_chart(
        px.imshow(cm, text_auto=True, color_continuous_scale="Teal").update_layout(height=360),
        width="stretch",
    )
    st.caption(
        f"With five axes, LDA scores {results['lda_5_axes']:.1%} while random forest scores "
        f"{results['rf_5_axes']:.1%}: the axes keep the information, a linear rule cannot "
        "read it."
    )

else:
    st.title("Clustering without labels")
    c1, c2 = st.columns(2)
    n_axes = c1.slider("MCA axes used", 2, 20, 5)
    k = c2.slider("Number of clusters", 2, 8, 3)
    coords, _ = mca_space()
    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(
        coords.iloc[:, :n_axes].to_numpy()
    )
    st.metric(
        "Agreement with the label (adjusted Rand index)", f"{adjusted_rand_score(y, labels):.2f}"
    )
    table = pd.crosstab(pd.Series(labels, name="cluster"), classes.rename("class"))
    table["poisonous %"] = (100 * table["poisonous"] / table.sum(axis=1)).round(1)
    st.dataframe(table, width="stretch")
    fig = px.scatter(
        coords.assign(cluster=labels.astype(str)),
        x="Axis 1",
        y="Axis 2",
        color="cluster",
        opacity=0.45,
        render_mode="webgl",
    )
    fig.update_traces(marker_size=4)
    st.plotly_chart(fig, width="stretch")
