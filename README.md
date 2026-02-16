<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn">
  <a href="https://pchambet-mushroom-project.streamlit.app/"><img src="https://img.shields.io/badge/Streamlit-Live_Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Dashboard"></a>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
</p>

<h1 align="center">The Mushroom Project</h1>

<p align="center">
  <em>8 124 champignons. 22 descripteurs. Une question de vie ou de mort.</em>
</p>

---

Un champignon devant vous. Chapeau convexe, odeur anisee, lamelles libres, anneau pendant.

**Comestible ou mortel ?**

Un mycologue experimente croise ces indices mentalement, en une seconde. Mais comment faire comprendre ca a une machine ? Parce que le vrai probleme, ce n'est pas la classification. C'est que **toutes ces informations sont des mots**. Pas un seul chiffre. Et les algorithmes de machine learning ne parlent que le langage des nombres.

Alors comment transformer des mots en geometrie ?

---

## Le pont

L'**Analyse en Composantes Multiples** (ACM) est la reponse.

Elle prend 22 variables categorielles — des descriptions textuelles comme "odeur = anis" ou "chapeau = convexe" — et les projette dans un **espace euclidien continu**. Soudain, chaque champignon a des coordonnees. Des distances. Une position dans un espace ou les mathematiques peuvent operer.

```
22 variables categorielles    →    ACM    →    Espace continu    →    Modeles
  (forme, odeur, couleur...)       10 axes      Coordonnees reelles     K-Means, LDA, RF, SVM
```

C'est ce pont qui rend tout le reste possible.

Et ce qui emerge dans cet espace est remarquable.

---

## Ce qui se passe quand on projette

Des qu'on place les 8 124 champignons sur les deux premiers axes de l'ACM, quelque chose saute aux yeux :

<p align="center">
  <img src="reports/figures/acm_individuals_12_color_target.png" width="700" alt="ACM - Individus colores par classe">
</p>

Les comestibles (vert) et les veneneux (rouge) **se separent naturellement**. Personne n'a demande a l'algorithme de les trier — l'ACM ne connait meme pas les etiquettes. Elle a juste trouve la structure geometrique cachee dans les descriptions morphologiques.

Le scree plot montre comment l'information se repartit sur les 10 axes. Pas de coude franc — c'est typique des donnees categoriques. 90% d'inertie cumulee a 8 axes.

<p align="center">
  <img src="reports/figures/acm_scree.png" width="700" alt="Scree plot ACM">
</p>

La question qui suit naturellement : est-ce que cette structure est assez forte pour faire emerger des groupes ?

---

## Des groupes apparaissent sans qu'on les demande

Le clustering (CAH Ward + K-Means) dans l'espace ACM revele **3 groupes naturels** :

<p align="center">
  <img src="reports/figures/cluster_on_acm12.png" width="48%" alt="Clusters sur plan ACM">
  <img src="reports/figures/cluster_dendrogram.png" width="48%" alt="Dendrogramme">
</p>

| Cluster | Taille | Composition | Purete |
|---|---|---|---|
| **0** | 4 824 (59.4%) | 82% comestible | Mixte |
| **1** | 192 (2.4%) | 100% comestible | **Pur** |
| **2** | 3 108 (38.3%) | 98.2% veneneux | **Quasi pur** |

Un cluster entierement comestible. Un autre presque entierement mortel. **Sans supervision.** Juste la geometrie des descriptions.

Alors si la structure est la... peut-on reellement predire ?

---

## La reponse a la question de depart

L'analyse discriminante lineaire (LDA) sur les 5 premiers axes ACM :

<p align="center">
  <img src="reports/figures/da_confusion_cv.png" width="45%" alt="Matrice de confusion CV">
  <img src="reports/figures/da_cv_scores.png" width="50%" alt="Scores CV par fold">
</p>

| Metrique | Valeur |
|---|---|
| Accuracy (train) | **88.7%** |
| Precision veneneux | **96.8%** |
| Recall comestible | **97.6%** |
| CV 5-fold | **77.1% +/- 14.0%** |

Quand le modele dit "mange-le", il a raison 97.6% du temps.

Mais deux questions restent ouvertes. **Combien d'axes ACM faut-il vraiment ?** Et **LDA est-il le meilleur modele sur cet espace ?**

---

## Plus de dimensions = meilleur ?

Non.

L'analyse de sensibilite teste k = 2 a 10 axes ACM. Resultat contre-intuitif : **k=4 donne la meilleure accuracy CV (88.9%)**, mieux que k=5 (77.1%) ou k=8 (80.0%). Ajouter du bruit dimensionnel degrade les performances.

<p align="center">
  <img src="reports/figures/sensitivity_k_analysis.png" width="700" alt="Analyse de sensibilite">
</p>

---

## Et si on changeait de modele ?

Quatre classifieurs sur les memes coordonnees ACM. Random Forest domine en accuracy brute — mais regardez l'ecart train/CV :

<p align="center">
  <img src="reports/figures/model_comparison.png" width="700" alt="Comparaison de modeles">
</p>

<p align="center">
  <img src="reports/figures/model_comparison_boxplot.png" width="500" alt="Box plots CV">
</p>

| Modele | Train | CV 5-fold | F1 (macro) |
|---|---|---|---|
| **Random Forest** | 99.9% | **85.3%** | **0.853** |
| SVM (RBF) | 96.3% | 82.6% | 0.826 |
| LDA | 88.7% | 77.1% | 0.771 |
| Logistic Regression | 88.6% | 74.7% | 0.747 |

99.9% en train, 85.3% en CV — c'est de l'**overfitting**. LDA, avec son ecart train/CV plus contenu, reste le modele le plus honnete. Le choix entre performance brute et fiabilite est un vrai sujet.

---

## Explorer par vous-meme

Tout ce qui precede est interactif. Le dashboard permet de manipuler les axes, ajuster les clusters, comparer les modeles — en temps reel.

> **[Ouvrir le dashboard](https://pchambet-mushroom-project.streamlit.app/)** — aucune installation requise.

<p align="center">
  <img src="reports/figures/dashboard_overview.png" width="100%" alt="Dashboard">
</p>

<details>
<summary><strong>Apercu des pages</strong></summary>

**Espace ACM** — choisir les axes, colorer par classe ou par cluster :

<p align="center">
  <img src="reports/figures/dashboard_acm.png" width="100%" alt="Dashboard - Espace ACM">
</p>

**Clustering** — ajuster le nombre de clusters et d'axes en temps reel :

<p align="center">
  <img src="reports/figures/dashboard_clustering.png" width="100%" alt="Dashboard - Clustering">
</p>

</details>

---

## Reproduire

```bash
git clone https://github.com/Pchambet/mushroom-project.git
cd mushroom-project

make install       # Environnement virtuel + dependances
make run-all       # Pipeline complet (00 → 07)
make dashboard     # Dashboard en local
```

<details>
<summary><strong>Toutes les commandes</strong></summary>

```bash
make install       # Creer l'environnement + dependances
make run-all       # Pipeline complet (scripts 00 a 07)
make run-A         # Download → Prepare → Describe → ACM
make run-B         # Clustering → Discriminante
make run-extended  # Sensibilite + Comparaison modeles
make dashboard     # Lancer le dashboard Streamlit
make clean         # Supprimer les outputs
make distclean     # Nettoyage complet (outputs + venv)
make help          # Aide
```

</details>

---

## Architecture

```
mushroom-project/
├── src/                              # Pipeline (9 scripts)
│   ├── 00_download.py                #   Acquisition UCI
│   ├── 01_prepare.py                 #   Nettoyage
│   ├── 02_describe.py                #   Stats descriptives
│   ├── 03_mca.py                     #   ACM
│   ├── 04_cluster.py                 #   CAH + K-Means
│   ├── 05_discriminant.py            #   LDA
│   ├── 06_sensitivity.py             #   Sensibilite (impact de k)
│   ├── 07_model_comparison.py        #   LDA vs RF vs SVM vs LogReg
│   └── utils.py                      #   Helpers
├── app.py                            # Dashboard Streamlit
├── reports/figures/                   # Figures (300 DPI)
├── reports/tables/                    # Tables CSV
├── data/                              # Raw + processed
├── docs/                              # Audit, dictionnaire
├── Makefile                           # Orchestration
└── requirements.txt                   # Dependances
```

---

## En profondeur

| Document | Description |
|---|---|
| [`docs/AUDIT_COMPLET.md`](docs/AUDIT_COMPLET.md) | **Audit detaille** — synergie, architecture, resultats, qualite |
| [`docs/data_dictionary.md`](docs/data_dictionary.md) | Dictionnaire des 23 variables et modalites |

---

## Stack

| Composant | Librairie |
|---|---|
| ACM | [`prince`](https://github.com/MaxHalford/prince) |
| Clustering | `scikit-learn`, `scipy` |
| Classification | `scikit-learn` (LDA, LogReg, RF, SVM) |
| Dashboard | `streamlit`, `plotly` |
| Visualisation | `matplotlib`, `seaborn` |

---

MIT — [`LICENSE`](LICENSE)
