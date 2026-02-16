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

Un mycologue experimente croise ces indices mentalement, en une seconde. Mais comment faire comprendre ca a une machine ?

Parce que le vrai probleme, ce n'est pas la classification. C'est que **toutes ces informations sont des mots**. "Convexe", "anis", "libre", "pendant" — pas un seul chiffre. Or les algorithmes de machine learning ont besoin de nombres pour calculer des distances, tracer des frontieres, regrouper des points. Face a des mots, ils sont aveugles.

Alors comment transformer des descriptions en quelque chose de calculable ?

---

## Le pont

Il existe une technique pensee exactement pour ca : l'**Analyse en Composantes Multiples** (ACM).

L'idee est simple. Chaque champignon est decrit par 22 caracteristiques textuelles — forme du chapeau, odeur, couleur des lamelles, type d'anneau. L'ACM prend ces 22 descriptions et les transforme en **coordonnees numeriques**, comme si elle placait chaque champignon sur une carte. Soudain, chaque specimen a une position. Des distances. Un voisinage. Et la ou il y a des coordonnees, les mathematiques peuvent operer.

```
22 descriptions textuelles    →    ACM    →    Coordonnees numeriques    →    Modeles
  (forme, odeur, couleur...)       10 axes      Chaque champignon a           Regroupement, prediction,
                                                une position sur la carte     comparaison
```

C'est ce pont — du texte vers la geometrie — qui rend tout le reste possible.

Et ce qui emerge de l'autre cote est remarquable.

---

## Ce qui se passe quand on projette

Des qu'on place les 8 124 champignons sur cette carte — ici, les deux axes principaux de l'ACM, ceux qui capturent le plus d'information — quelque chose saute aux yeux :

<p align="center">
  <img src="reports/figures/acm_individuals_12_color_target.png" width="700" alt="ACM - Individus colores par classe">
</p>

Les comestibles (vert) et les veneneux (rouge) **se separent naturellement**. Personne n'a demande a l'algorithme de les trier — il ne connait meme pas les etiquettes "comestible" ou "veneneux". Il a juste lu les descriptions morphologiques et trouve, tout seul, que les champignons dangereux ne ressemblent pas aux autres.

Le graphique ci-dessous montre combien d'information chaque axe capture. Plus un axe est haut, plus il est utile. Il n'y a pas de coupure nette — c'est typique quand les donnees sont des mots plutot que des mesures — mais les 8 premiers axes concentrent 90% de l'information.

<p align="center">
  <img src="reports/figures/acm_scree.png" width="700" alt="Repartition de l'information par axe">
</p>

La question qui suit naturellement : est-ce que cette structure est assez forte pour faire emerger des groupes ?

---

## Des groupes apparaissent sans qu'on les demande

On peut demander a un algorithme de **regrouper** les champignons qui se ressemblent — c'est ce qu'on appelle le *clustering*. Ici, deux methodes complementaires : l'une (CAH Ward) construit un arbre hierarchique pour voir a quelle echelle les groupes se forment ; l'autre (K-Means) decoupe l'espace en un nombre fixe de groupes.

Les deux convergent vers **3 groupes naturels** :

<p align="center">
  <img src="reports/figures/cluster_on_acm12.png" width="48%" alt="Clusters sur plan ACM">
  <img src="reports/figures/cluster_dendrogram.png" width="48%" alt="Arbre hierarchique">
</p>

| Cluster | Taille | Composition | Purete |
|---|---|---|---|
| **0** | 4 824 (59.4%) | 82% comestible | Mixte |
| **1** | 192 (2.4%) | 100% comestible | **Pur** |
| **2** | 3 108 (38.3%) | 98.2% veneneux | **Quasi pur** |

Un cluster entierement comestible. Un autre presque entierement mortel. Et rappelons-le : l'algorithme n'a **jamais vu l'etiquette** "comestible" ou "veneneux". Il a juste mesure des proximites dans l'espace des descriptions.

Alors si la structure est la... peut-on reellement predire si un champignon est dangereux ?

---

## La reponse a la question de depart

Pour predire, il faut un modele qui **apprend la frontiere** entre comestibles et veneneux. L'analyse discriminante lineaire (LDA) fait exactement ca : elle trace la meilleure ligne de separation possible dans l'espace ACM, a partir d'exemples etiquetes.

On la teste sur les 5 premiers axes, et on verifie sa fiabilite par **validation croisee** — on decoupe les donnees en 5 parts, on entraine sur 4, on teste sur la 5eme, et on tourne. Ca empeche le modele de "tricher" en memorisant les reponses.

<p align="center">
  <img src="reports/figures/da_confusion_cv.png" width="45%" alt="Matrice de confusion">
  <img src="reports/figures/da_cv_scores.png" width="50%" alt="Scores par fold">
</p>

| Metrique | Ce que ca veut dire | Valeur |
|---|---|---|
| Accuracy | Le modele a bon dans X% des cas | **88.7%** |
| Precision veneneux | Quand il dit "dangereux", il a raison a | **96.8%** |
| Recall comestible | Quand un champignon est comestible, il le detecte a | **97.6%** |
| CV 5-fold | Performance sur des donnees jamais vues | **77.1% +/- 14.0%** |

Quand le modele dit "mange-le", il a raison 97.6% du temps. Pas mal pour un algorithme qui partait de mots.

Mais deux questions restent ouvertes. **Combien d'axes faut-il vraiment garder ?** Et **existe-t-il un meilleur modele ?**

---

## Plus de dimensions = meilleur ?

Intuitivement, on se dit : plus on donne d'information au modele, mieux il predit. Plus d'axes ACM = plus de details = meilleure performance. Non ?

**Non.**

On teste systematiquement de 2 a 10 axes. Resultat contre-intuitif : **4 axes donnent la meilleure performance (88.9%)**, mieux que 5 (77.1%) ou 8 (80.0%). Au-dela d'un certain point, les axes supplementaires apportent plus de bruit que de signal — le modele se perd dans des details non pertinents.

<p align="center">
  <img src="reports/figures/sensitivity_k_analysis.png" width="700" alt="Analyse de sensibilite">
</p>

C'est un resultat important : en data science, **savoir quoi enlever compte autant que savoir quoi garder.**

---

## Et si on changeait de modele ?

La LDA trace une frontiere lineaire — une ligne droite, en quelque sorte. Mais la realite est rarement lineaire. D'autres modeles savent tracer des frontieres plus complexes : le **Random Forest** (qui combine des centaines d'arbres de decision), le **SVM** (qui cherche la frontiere optimale dans un espace deforme), ou la **Regression Logistique** (cousine de la LDA, mais avec des hypotheses differentes).

Quatre classifieurs, les memes coordonnees ACM. Qui gagne ?

<p align="center">
  <img src="reports/figures/model_comparison.png" width="700" alt="Comparaison de modeles">
</p>

<p align="center">
  <img src="reports/figures/model_comparison_boxplot.png" width="500" alt="Dispersion des scores">
</p>

| Modele | Entrainement | Donnees jamais vues | F1 |
|---|---|---|---|
| **Random Forest** | 99.9% | **85.3%** | **0.853** |
| SVM | 96.3% | 82.6% | 0.826 |
| LDA | 88.7% | 77.1% | 0.771 |
| Logistic Regression | 88.6% | 74.7% | 0.747 |

Random Forest gagne en performance brute. Mais regardez la colonne "Entrainement" : 99.9%. Quasi parfait. Et sur des donnees jamais vues : 85.3%. Cet ecart a un nom — **l'overfitting**. Le modele a memorise les exemples au lieu d'apprendre la regle. C'est comme un etudiant qui retient les reponses du QCM sans comprendre le cours : brillant a l'examen blanc, fragile le jour J.

LDA, avec un ecart beaucoup plus faible entre entrainement et test, reste **le modele le plus honnete**. Le choix entre performance brute et fiabilite est un vrai sujet — et il n'y a pas de reponse universelle.

---

## Explorer par vous-meme

Tout ce qui precede est interactif. Le dashboard permet de changer les axes, ajuster les clusters, comparer les modeles — et voir les resultats bouger en temps reel.

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

**Clustering** — ajuster les parametres et voir les groupes se former :

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
