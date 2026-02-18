"""
The Mushroom Project — Dashboard interactif.

Application Streamlit pour explorer visuellement l'analyse ACM,
le clustering et la classification des champignons.

Lancer : streamlit run app.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score
from sklearn.model_selection import cross_val_predict, cross_val_score

# ── Configuration ─────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed"
TABLES = ROOT / "reports" / "tables"
GITHUB_URL = "https://github.com/Pchambet/mushroom-project"

# ── Configuration page ───────────────────────────────────────

st.set_page_config(
    page_title="The Mushroom Project",
    page_icon="🍄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Génération des données si absentes (Streamlit Cloud) ─────────

def run_pipeline() -> None:
    """Exécute le pipeline complet. Utilisé quand data/ et reports/ sont absents."""
    scripts = [
        "src/00_download.py",
        "src/01_prepare.py",
        "src/02_describe.py",
        "src/03_mca.py",
        "src/04_cluster.py",
        "src/05_discriminant.py",
        "src/06_sensitivity.py",
        "src/07_model_comparison.py",
    ]
    for script in scripts:
        subprocess.run(
            [sys.executable, script],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )


# ── Chargement des données ───────────────────────────────────

@st.cache_data
def load_data():
    """Charge le dataset et les coordonnées ACM."""
    df = pd.read_csv(DATA / "mushroom_processed.csv")
    coords = pd.read_csv(DATA / "mca_coords.csv")
    return df, coords


@st.cache_data
def load_tables():
    """Charge toutes les tables CSV du projet."""
    tables = {}
    for f in TABLES.glob("*.csv"):
        tables[f.stem] = pd.read_csv(f)
    return tables


# ── Chargement sécurisé ──────────────────────────────────────

# Si les données n'existent pas (ex. Streamlit Cloud), exécuter le pipeline
if not (DATA / "mca_coords.csv").exists():
    with st.spinner("Première exécution : génération des données (2 à 5 min)…"):
        run_pipeline()
    st.rerun()

try:
    df, coords = load_data()
    tables = load_tables()
    data_ready = True
except FileNotFoundError as e:
    data_ready = False
    load_error = str(e)
except Exception as e:
    data_ready = False
    load_error = str(e)


# ── Sidebar ──────────────────────────────────────────────────

st.sidebar.title("🍄 The Mushroom Project")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "Vue d'ensemble",
        "Espace ACM",
        "Clustering",
        "Classification",
        "Sensibilité",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**8 124** specimens  \n"
    "**22** variables  \n"
    "**10** axes ACM"
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    f"[📂 Code source]({GITHUB_URL})  \n"
    f"[📖 README]({GITHUB_URL}#readme)"
)


# ── Blocage si données manquantes ────────────────────────────

if not data_ready:
    st.error(
        "**Données introuvables.** Le pipeline n'a pas été exécuté ou les fichiers "
        "sont manquants. Pour générer les données :\n\n"
        "```bash\n"
        "git clone https://github.com/Pchambet/mushroom-project.git\n"
        "cd mushroom-project\n"
        "make install && make run-all\n"
        "```\n\n"
        f"Voir le [README]({GITHUB_URL}) pour plus de détails."
    )
    st.stop()


# ── Page : Vue d'ensemble ───────────────────────────────────

if page == "Vue d'ensemble":
    st.title("The Mushroom Project")
    st.markdown(
        "> Pipeline d'analyse statistique sur **données catégorielles** — "
        "de l'ACM au clustering et à la classification."
    )
    st.markdown(
        "Ce dashboard explore les résultats du projet. Pour le contexte complet, "
        "la méthodologie et les explications détaillées, consultez le "
        f"[README sur GitHub]({GITHUB_URL})."
    )
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Specimens", f"{len(df):,}")
    col2.metric("Variables", f"{df.shape[1] - 1}")
    col3.metric("Comestibles", f"{(df['class'] == 'e').sum():,}", f"{(df['class'] == 'e').mean()*100:.1f} %")
    col4.metric("Vénéneux", f"{(df['class'] == 'p').sum():,}", f"{(df['class'] == 'p').mean()*100:.1f} %")

    st.markdown("---")

    st.subheader("Distribution de la variable cible")
    class_counts = df["class"].value_counts()
    labels = [f"Comestible ({c})" if c == "e" else f"Vénéneux ({c})" for c in class_counts.index]
    fig = px.bar(
        x=labels,
        y=class_counts.values,
        color=class_counts.index.map({"e": "Comestible", "p": "Vénéneux"}),
        color_discrete_map={"Comestible": "#2ecc71", "Vénéneux": "#e74c3c"},
    )
    fig.update_layout(
        showlegend=False,
        xaxis_title="",
        yaxis_title="Nombre de specimens",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Aperçu du dataset")
    st.caption("Les 20 premières lignes — 22 variables morphologiques (forme, odeur, couleur…) et la classe.")
    st.dataframe(df.head(20), use_container_width=True)

    if "univariate_summary" in tables:
        st.subheader("Résumé univarié")
        st.caption(
            "Pour chaque variable : nombre de modalités distinctes, modalité la plus fréquente, "
            "fréquence (%) et part des valeurs manquantes."
        )
        univ = tables["univariate_summary"].copy()
        univ.columns = ["Variable", "Modalités", "Plus fréquent", "Fréq. %", "Manquants %"]
        st.dataframe(univ, use_container_width=True)


# ── Page : Espace ACM ────────────────────────────────────────

elif page == "Espace ACM":
    st.title("Espace factoriel ACM")
    st.markdown(
        "L'ACM transforme 22 variables **catégorielles** en un espace euclidien continu. "
        "Explorez la projection des individus sur les axes factoriels."
    )

    col1, col2, col3 = st.columns([1, 1, 2])
    dim_x = col1.selectbox("Axe X", [f"Dim{i+1}" for i in range(10)], index=0)
    dim_y = col2.selectbox("Axe Y", [f"Dim{i+1}" for i in range(10)], index=1)
    point_size = col3.slider("Taille des points", 2, 12, 4, help="Ajustez la lisibilité du nuage")

    plot_df = coords.copy()
    plot_df["class"] = df["class"].map({"e": "Comestible", "p": "Vénéneux"})

    color_var = st.selectbox(
        "Colorer par",
        ["Classe (comestible/vénéneux)"] + [c for c in df.columns if c != "class"],
    )

    if color_var == "Classe (comestible/vénéneux)":
        fig = px.scatter(
            plot_df, x=dim_x, y=dim_y, color="class",
            color_discrete_map={"Comestible": "#2ecc71", "Vénéneux": "#e74c3c"},
            opacity=0.5, hover_data={"class": True},
        )
    else:
        plot_df[color_var] = df[color_var].fillna("Manquant").astype(str)
        fig = px.scatter(
            plot_df, x=dim_x, y=dim_y, color=color_var,
            opacity=0.5,
        )

    fig.update_layout(height=600)
    fig.update_traces(marker=dict(size=point_size))
    st.plotly_chart(fig, use_container_width=True)

    if "mca_eigenvalues" in tables:
        st.subheader("Inertie expliquée")
        eigen = tables["mca_eigenvalues"]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=eigen["Component"], y=eigen["Explained_Inertia_%"],
            name="Inertie par axe", marker_color="#2196F3",
        ))
        fig.add_trace(go.Scatter(
            x=eigen["Component"], y=eigen["Cumulative_Inertia_%"],
            name="Cumul", mode="lines+markers",
            marker_color="#F44336", yaxis="y2",
        ))
        fig.update_layout(
            yaxis=dict(title="Inertie (%)"),
            yaxis2=dict(title="Cumul (%)", overlaying="y", side="right"),
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)


# ── Page : Clustering ────────────────────────────────────────

elif page == "Clustering":
    st.title("Clustering (K-Means sur espace ACM)")
    st.markdown(
        "Régroupez les champignons par similarité morphologique. "
        "Les graphiques affichent une **projection sur les axes 1 et 2** — le clustering est calculé sur l'ensemble des axes sélectionnés."
    )

    col_params, _ = st.columns([2, 3])
    with col_params:
        k_axes = st.slider("Nombre d'axes ACM", 2, 10, 5)
        n_clusters = st.slider("Nombre de clusters", 2, 8, 3)

    X_clust = coords.iloc[:, :k_axes].values

    with st.spinner("Calcul du clustering…"):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_clust)
        sil = silhouette_score(X_clust, labels)

    col1, col2, col3 = st.columns(3)
    col1.metric("Clusters", n_clusters)
    col2.metric("Silhouette", f"{sil:.3f}")
    col3.metric("Axes ACM", k_axes)

    plot_df = coords.copy()
    plot_df["cluster"] = labels.astype(str)
    plot_df["class"] = df["class"].map({"e": "Comestible", "p": "Vénéneux"})

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
        st.subheader("Classes réelles")
        fig = px.scatter(
            plot_df, x="Dim1", y="Dim2", color="class",
            color_discrete_map={"Comestible": "#2ecc71", "Vénéneux": "#e74c3c"},
            opacity=0.5,
        )
        fig.update_traces(marker=dict(size=4))
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Clusters vs classe réelle")
    ct = pd.crosstab(
        pd.Series(labels, name="Cluster"),
        df["class"].map({"e": "Comestible", "p": "Vénéneux"}),
        margins=True,
    )
    st.dataframe(ct, use_container_width=True)

    purity_rows = []
    for cid in range(n_clusters):
        mask = labels == cid
        size = mask.sum()
        n_e = (df.loc[mask, "class"] == "e").sum()
        n_p = size - n_e
        dominant = "Comestible" if n_e > n_p else "Vénéneux"
        purity = max(n_e, n_p) / size * 100
        purity_rows.append({
            "Cluster": cid,
            "Taille": size,
            "Comestibles": n_e,
            "Vénéneux": n_p,
            "Dominant": dominant,
            "Pureté (%)": round(purity, 1),
        })
    st.dataframe(pd.DataFrame(purity_rows), use_container_width=True)


# ── Page : Classification ────────────────────────────────────

elif page == "Classification":
    st.title("Classification (LDA sur espace ACM)")
    st.markdown(
        "L'analyse discriminante linéaire trace la meilleure frontière de séparation "
        "entre comestibles et vénéneux dans l'espace ACM."
    )

    k_axes = st.slider("Nombre d'axes ACM (k)", 2, 10, 5)

    X = coords.iloc[:, :k_axes].values
    y = (df["class"] == "e").astype(int)

    with st.spinner("Calcul de la validation croisée…"):
        lda = LinearDiscriminantAnalysis()
        cv_scores = cross_val_score(lda, X, y, cv=5, scoring="accuracy")
        y_pred_cv = cross_val_predict(lda, X, y, cv=5)
        lda.fit(X, y)
        train_acc = lda.score(X, y)

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy entraînement", f"{train_acc:.1%}")
    col2.metric("Accuracy CV (5-fold)", f"{cv_scores.mean():.1%}", f"± {cv_scores.std():.1%}")
    col3.metric("Axes ACM", k_axes)

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Matrice de confusion (validation croisée)")
        cm = confusion_matrix(y, y_pred_cv)
        fig = px.imshow(
            cm,
            labels=dict(x="Prédit", y="Réel", color="Effectif"),
            x=["Vénéneux", "Comestible"],
            y=["Vénéneux", "Comestible"],
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
        target_names=["Vénéneux", "Comestible"],
        output_dict=True,
    )
    report_df = pd.DataFrame(report).T
    st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)

    if "model_comparison" in tables:
        st.subheader("Comparaison de modèles")
        st.caption("Résultats pré-calculés par le pipeline (LDA, Random Forest, SVM, Régression logistique).")
        mc = tables["model_comparison"].copy()
        mc.columns = [
            "Modèle", "Accuracy train", "Accuracy CV", "Écart-type CV",
            "Préc. vénéneux", "Recall vénéneux", "Préc. comestible", "Recall comestible",
            "F1 macro", "Écart overfitting"
        ]
        mc_display = mc[["Modèle", "Accuracy train", "Accuracy CV", "F1 macro"]]
        st.dataframe(
            mc_display.style.highlight_max(subset=["Accuracy CV", "F1 macro"], color="#c8e6c9"),
            use_container_width=True,
        )


# ── Page : Sensibilité ───────────────────────────────────────

elif page == "Sensibilité":
    st.title("Analyse de sensibilité — Impact de k")
    st.markdown(
        "Comment le nombre d'axes ACM retenus affecte-t-il la performance "
        "de la classification et la qualité du clustering ?"
    )

    if "sensitivity_k" in tables:
        sens = tables["sensitivity_k"]

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Accuracy LDA vs k")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=sens["k"], y=sens["lda_train_accuracy"],
                mode="lines+markers", name="Entraînement",
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
            st.subheader("Score Silhouette vs k")
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

        st.subheader("Inertie ACM vs performance")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sens["k"], y=sens["cumulative_inertia_%"],
            mode="lines+markers", name="Inertie cumulée (%)",
            line=dict(color="#2196F3"), yaxis="y",
        ))
        fig.add_trace(go.Scatter(
            x=sens["k"], y=sens["lda_cv_mean"],
            mode="lines+markers", name="Accuracy CV",
            line=dict(color="#F44336"), yaxis="y2",
        ))
        fig.update_layout(
            xaxis_title="k (axes ACM)",
            yaxis=dict(title="Inertie cumulée (%)", side="left"),
            yaxis2=dict(title="Accuracy CV", overlaying="y", side="right"),
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Tableau complet")
        sens_display = sens.copy()
        sens_display.columns = [
            "k", "Inertie cum. %", "Accuracy train", "Accuracy CV", "Écart-type CV",
            "Silhouette", "Inertie K-Means", "Écart overfitting"
        ]
        st.dataframe(
            sens_display.style.highlight_max(
                subset=["Accuracy CV", "Silhouette"],
                color="#c8e6c9",
            ).highlight_min(
                subset=["Écart overfitting"],
                color="#c8e6c9",
            ),
            use_container_width=True,
        )

        best_k = sens.loc[sens["lda_cv_mean"].idxmax(), "k"]
        best_acc = sens.loc[sens["lda_cv_mean"].idxmax(), "lda_cv_mean"]
        st.info(f"**Meilleur k** (par accuracy CV) : **k = {int(best_k)}** ({best_acc:.4f})")
    else:
        st.warning(
            "Les résultats de sensibilité ne sont pas disponibles. "
            "Exécutez le pipeline complet (`make run-all`) pour les générer — "
            f"voir le [README]({GITHUB_URL}) pour les instructions."
        )
