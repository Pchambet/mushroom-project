"""
The Mushroom Project — Dashboard interactif.

Streamlit app pour explorer visuellement l'analyse ACM,
le clustering et la classification des champignons.

Lancer : streamlit run app.py
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ── Configuration page ─────────────────────────────────────

st.set_page_config(
    page_title="The Mushroom Project",
    page_icon="🍄",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed"
TABLES = ROOT / "reports" / "tables"


# ── Chargement des donnees ─────────────────────────────────

@st.cache_data
def load_data():
    df = pd.read_csv(DATA / "mushroom_processed.csv")
    coords = pd.read_csv(DATA / "mca_coords.csv")
    return df, coords


@st.cache_data
def load_tables():
    tables = {}
    for f in TABLES.glob("*.csv"):
        tables[f.stem] = pd.read_csv(f)
    return tables


df, coords = load_data()
tables = load_tables()


# ── Sidebar ────────────────────────────────────────────────

st.sidebar.title("🍄 The Mushroom Project")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "Vue d'ensemble",
        "Espace ACM",
        "Clustering",
        "Classification",
        "Sensibilite",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**8 124** specimens  \n"
    "**22** variables  \n"
    "**10** axes ACM"
)


# ── Page : Vue d'ensemble ──────────────────────────────────

if page == "Vue d'ensemble":
    st.title("The Mushroom Project")
    st.markdown(
        "> Pipeline d'analyse statistique sur donnees categorielles — "
        "De l'ACM au clustering et a la classification."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Specimens", f"{len(df):,}")
    col2.metric("Variables", f"{df.shape[1] - 1}")
    col3.metric("Comestibles", f"{(df['class'] == 'e').sum():,}", f"{(df['class'] == 'e').mean()*100:.1f}%")
    col4.metric("Venéneux", f"{(df['class'] == 'p').sum():,}", f"{(df['class'] == 'p').mean()*100:.1f}%")

    st.markdown("---")

    st.subheader("Distribution de la variable cible")
    fig = px.bar(
        x=["Comestible (e)", "Venéneux (p)"],
        y=[(df["class"] == "e").sum(), (df["class"] == "p").sum()],
        color=["Comestible", "Venéneux"],
        color_discrete_map={"Comestible": "#2ecc71", "Venéneux": "#e74c3c"},
    )
    fig.update_layout(
        showlegend=False, xaxis_title="", yaxis_title="Nombre de specimens",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Apercu du dataset")
    st.dataframe(df.head(20), use_container_width=True)

    if "univariate_summary" in tables:
        st.subheader("Resume univarie")
        st.dataframe(tables["univariate_summary"], use_container_width=True)


# ── Page : Espace ACM ─────────────────────────────────────

elif page == "Espace ACM":
    st.title("Espace factoriel ACM")
    st.markdown(
        "L'ACM transforme 22 variables categoriques en un espace euclidien continu. "
        "Explorez la projection des individus sur les axes factoriels."
    )

    col1, col2 = st.columns(2)
    dim_x = col1.selectbox("Axe X", [f"Dim{i+1}" for i in range(10)], index=0)
    dim_y = col2.selectbox("Axe Y", [f"Dim{i+1}" for i in range(10)], index=1)

    plot_df = coords.copy()
    plot_df["class"] = df["class"].map({"e": "Comestible", "p": "Venéneux"})

    color_var = st.selectbox(
        "Colorer par",
        ["Classe (edible/poisonous)"] + [c for c in df.columns if c != "class"],
    )

    if color_var == "Classe (edible/poisonous)":
        fig = px.scatter(
            plot_df, x=dim_x, y=dim_y, color="class",
            color_discrete_map={"Comestible": "#2ecc71", "Venéneux": "#e74c3c"},
            opacity=0.5, hover_data={"class": True},
        )
    else:
        plot_df[color_var] = df[color_var].astype(str)
        fig = px.scatter(
            plot_df, x=dim_x, y=dim_y, color=color_var,
            opacity=0.5,
        )

    fig.update_layout(height=600)
    fig.update_traces(marker=dict(size=4))
    st.plotly_chart(fig, use_container_width=True)

    # Inertie
    if "mca_eigenvalues" in tables:
        st.subheader("Inertie expliquee")
        eigen = tables["mca_eigenvalues"]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=eigen["Component"], y=eigen["Explained_Inertia_%"],
            name="Inertie par axe", marker_color="#2196F3",
        ))
        fig.add_trace(go.Scatter(
            x=eigen["Component"], y=eigen["Cumulative_Inertia_%"],
            name="Cumulative", mode="lines+markers",
            marker_color="#F44336", yaxis="y2",
        ))
        fig.update_layout(
            yaxis=dict(title="Inertie (%)"),
            yaxis2=dict(title="Cumul (%)", overlaying="y", side="right"),
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)


# ── Page : Clustering ──────────────────────────────────────

elif page == "Clustering":
    st.title("Clustering (K-Means sur espace ACM)")

    k_axes = st.slider("Nombre d'axes ACM", 2, 10, 5)

    from sklearn.cluster import KMeans

    X_clust = coords.iloc[:, :k_axes].values
    n_clusters = st.slider("Nombre de clusters", 2, 8, 3)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_clust)

    from sklearn.metrics import silhouette_score
    sil = silhouette_score(X_clust, labels)

    col1, col2, col3 = st.columns(3)
    col1.metric("Clusters", n_clusters)
    col2.metric("Silhouette", f"{sil:.3f}")
    col3.metric("Axes ACM", k_axes)

    # Scatter
    plot_df = coords.copy()
    plot_df["cluster"] = labels.astype(str)
    plot_df["class"] = df["class"].map({"e": "Comestible", "p": "Venéneux"})

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Clusters sur plan ACM")
        fig = px.scatter(
            plot_df, x="Dim1", y="Dim2", color="cluster",
            opacity=0.5,
        )
        fig.update_traces(marker=dict(size=4))
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("Classes reelles")
        fig = px.scatter(
            plot_df, x="Dim1", y="Dim2", color="class",
            color_discrete_map={"Comestible": "#2ecc71", "Venéneux": "#e74c3c"},
            opacity=0.5,
        )
        fig.update_traces(marker=dict(size=4))
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    # Crosstab
    st.subheader("Clusters vs Classe reelle")
    ct = pd.crosstab(
        pd.Series(labels, name="Cluster"),
        df["class"].map({"e": "Comestible", "p": "Venéneux"}),
        margins=True,
    )
    st.dataframe(ct, use_container_width=True)

    # Purete
    purity_rows = []
    for cid in range(n_clusters):
        mask = labels == cid
        size = mask.sum()
        n_e = (df.loc[mask, "class"] == "e").sum()
        n_p = size - n_e
        dominant = "Comestible" if n_e > n_p else "Venéneux"
        purity = max(n_e, n_p) / size * 100
        purity_rows.append({
            "Cluster": cid,
            "Taille": size,
            "Comestibles": n_e,
            "Venéneux": n_p,
            "Dominant": dominant,
            "Purete (%)": round(purity, 1),
        })
    st.dataframe(pd.DataFrame(purity_rows), use_container_width=True)


# ── Page : Classification ──────────────────────────────────

elif page == "Classification":
    st.title("Classification (LDA sur espace ACM)")

    k_axes = st.slider("Nombre d'axes ACM (k)", 2, 10, 5)

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import cross_val_score, cross_val_predict
    from sklearn.metrics import confusion_matrix, classification_report

    X = coords.iloc[:, :k_axes].values
    y = (df["class"] == "e").astype(int)

    lda = LinearDiscriminantAnalysis()
    cv_scores = cross_val_score(lda, X, y, cv=5, scoring="accuracy")
    y_pred_cv = cross_val_predict(lda, X, y, cv=5)
    lda.fit(X, y)
    train_acc = lda.score(X, y)

    col1, col2, col3 = st.columns(3)
    col1.metric("Train Accuracy", f"{train_acc:.1%}")
    col2.metric("CV Accuracy", f"{cv_scores.mean():.1%}", f"+/- {cv_scores.std():.1%}")
    col3.metric("Axes ACM", k_axes)

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Matrice de confusion (CV)")
        cm = confusion_matrix(y, y_pred_cv)
        fig = px.imshow(
            cm,
            labels=dict(x="Predit", y="Reel", color="Nombre"),
            x=["Venéneux", "Comestible"],
            y=["Venéneux", "Comestible"],
            color_continuous_scale="Blues",
            text_auto=True,
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("Scores par fold")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[f"Fold {i+1}" for i in range(5)],
            y=cv_scores,
            marker_color="steelblue",
        ))
        fig.add_hline(y=cv_scores.mean(), line_dash="dash", line_color="red")
        fig.update_layout(
            yaxis_title="Accuracy",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Rapport de classification")
    report = classification_report(
        y, y_pred_cv,
        target_names=["Venéneux", "Comestible"],
        output_dict=True,
    )
    report_df = pd.DataFrame(report).T
    st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)

    # Comparaison modeles
    if "model_comparison" in tables:
        st.subheader("Comparaison de modeles")
        st.dataframe(
            tables["model_comparison"].style.highlight_max(
                subset=["cv_accuracy_mean", "f1_macro"],
                color="#c8e6c9",
            ),
            use_container_width=True,
        )


# ── Page : Sensibilite ────────────────────────────────────

elif page == "Sensibilite":
    st.title("Analyse de sensibilite — Impact de k")
    st.markdown(
        "Comment le nombre d'axes ACM retenus affecte-t-il la performance "
        "de la classification et la qualite du clustering ?"
    )

    if "sensitivity_k" in tables:
        sens = tables["sensitivity_k"]

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("LDA Accuracy vs k")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=sens["k"], y=sens["lda_train_accuracy"],
                mode="lines+markers", name="Train",
                line=dict(color="#2196F3"),
            ))
            fig.add_trace(go.Scatter(
                x=sens["k"], y=sens["lda_cv_mean"],
                mode="lines+markers", name="CV 5-fold",
                line=dict(color="#F44336"),
                error_y=dict(type="data", array=sens["lda_cv_std"], visible=True),
            ))
            fig.update_layout(
                xaxis_title="k (axes ACM)",
                yaxis_title="Accuracy",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Silhouette Score vs k")
            fig = px.bar(
                sens, x="k", y="silhouette_score",
                color="silhouette_score",
                color_continuous_scale="RdYlGn",
            )
            fig.update_layout(
                xaxis_title="k (axes ACM)",
                yaxis_title="Silhouette",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Tableau complet")
        st.dataframe(
            sens.style.highlight_max(
                subset=["lda_cv_mean", "silhouette_score"],
                color="#c8e6c9",
            ).highlight_min(
                subset=["overfitting_gap"],
                color="#c8e6c9",
            ),
            use_container_width=True,
        )

        best_k = sens.loc[sens["lda_cv_mean"].idxmax(), "k"]
        best_acc = sens.loc[sens["lda_cv_mean"].idxmax(), "lda_cv_mean"]
        st.info(f"Meilleur k par CV accuracy : **k={int(best_k)}** ({best_acc:.4f})")
    else:
        st.warning(
            "Executez d'abord : `python src/06_sensitivity.py` "
            "pour generer les resultats."
        )
