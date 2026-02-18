# Audit Complet — The Mushroom Project

> Huit experts. Un projet. Chaque section est ecrite depuis la perspective d'un specialiste de son domaine — pour comprendre en profondeur ce que ce projet fait, pourquoi il le fait, et ce qu'il revele.

---

## Table des matieres

1. [Le scientifique — Demarche et questionnement](#1-le-scientifique--demarche-et-questionnement)
2. [Le mathematicien — L'ACM en profondeur](#2-le-mathematicien--lacm-en-profondeur)
3. [Le specialiste clustering — Structures non supervisees](#3-le-specialiste-clustering--structures-non-supervisees)
4. [Le specialiste classification — Modeles et evaluation](#4-le-specialiste-classification--modeles-et-evaluation)
5. [Le data engineer — Pipeline et architecture](#5-le-data-engineer--pipeline-et-architecture)
6. [Le visualiste — Communication des resultats](#6-le-visualiste--communication-des-resultats)
7. [Le developpeur — Qualite de code](#7-le-developpeur--qualite-de-code)
8. [Le product owner — Ce que ce projet apporte](#8-le-product-owner--ce-que-ce-projet-apporte)

---

## 1. Le scientifique — Demarche et questionnement

### Le probleme de depart

On a 8 124 champignons. Chacun est decrit par 22 caracteristiques : la forme de son chapeau, son odeur, la couleur de ses lamelles, son type d'anneau, son habitat. Toutes ces informations sont des **mots**. Aucun chiffre.

La question est simple : comestible ou mortel ?

Mais la vraie question — celle que le projet pose en creux — est plus profonde : **comment extraire du sens calculable a partir de descriptions qualitatives ?**

### La demarche

Ce projet ne saute pas aux conclusions. Il suit une demarche progressive ou chaque etape construit sur la precedente :

1. **Observer** (scripts 00-02) — Telecharger, nettoyer, decrire. Avant de modeliser, on regarde. On compte les modalites, on identifie les valeurs manquantes (la variable `stalk-root` en a 30.5%), on visualise les distributions des variables cles (odeur, couleur des lamelles, sporee).

2. **Transformer** (script 03) — L'ACM convertit les 22 descriptions textuelles en coordonnees numeriques. Ce n'est pas un choix arbitraire — c'est la seule technique de reduction dimensionnelle conçue specifiquement pour des variables categorielles. C'est le moment ou le projet passe du qualitatif au quantitatif.

3. **Decouvrir** (script 04) — Avant de predire, on explore. Le clustering (CAH + K-Means) cherche des groupes naturels dans l'espace ACM. On ne lui donne pas les etiquettes "comestible" ou "veneneux". On lui demande juste : "est-ce qu'il y a des groupes ?"

4. **Predire** (script 05) — La LDA trace la meilleure frontiere de separation dans cet espace. On mesure sa performance. On la challenge avec une validation croisee.

5. **Interroger** (scripts 06-07) — Les deux scripts les plus matures du projet. On remet en question les choix : combien d'axes ? Quel modele ? On compare, on mesure, on conclut.

### Ce que cette demarche revele

Le fait de decouvrir des clusters quasi-purs **avant** d'appliquer la classification supervisee n'est pas anecdotique. Ca montre que la structure "comestible vs veneneux" est profondement ancree dans la morphologie des champignons — au point qu'un algorithme aveugle aux etiquettes la retrouve tout seul. C'est un resultat qui valide toute la chaine en amont.

---

## 2. Le mathematicien — L'ACM en profondeur

### Qu'est-ce que l'ACM fait reellement ?

L'Analyse en Composantes Multiples est l'equivalent, pour des donnees categorielles, de ce que l'ACP (Analyse en Composantes Principales) fait pour des donnees numeriques. Mais le mecanisme est different.

L'ACM travaille sur le **tableau disjonctif complet** — une matrice binaire (0/1) ou chaque colonne correspond a une modalite. Si un champignon a `odor=foul`, il a un 1 dans la colonne "odor_foul" et un 0 dans toutes les autres colonnes d'odeur. Ce tableau binaire, pour 22 variables et environ 100 modalites, fait 8 124 lignes x ~100 colonnes.

L'ACM diagonalise la matrice d'inertie de ce tableau (analogue a la matrice de covariance en ACP) pour trouver les **axes de plus grande variabilite**. Les coordonnees de chaque individu sur ces axes sont les "nouvelles variables" continues.

### Les resultats concrets

| Axe | Inertie (%) | Cumul (%) | Interpretation |
|---|---|---|---|
| Dim 1 | 15.9% | 15.9% | Separation par profil de toxicite (odeur, surface du pied) |
| Dim 2 | 14.4% | 30.3% | Diversite morphologique (couleurs, habitat) |
| Dim 3 | 13.2% | 43.5% | Variabilite secondaire |
| Dim 4 | 11.9% | 55.4% | |
| Dim 5 | 9.9% | 65.3% | ← k=5 utilise pour clustering et LDA |
| ... | ... | ... | |
| **Dim 8** | **7.1%** | **90.3%** | ← seuil 90% |
| Dim 10 | 4.7% | 100% | |

### Le diagnostic mathematique

**Pas de coude sur le scree plot.** En ACP sur des donnees continues, on cherche un "coude" — un point ou l'inertie chute brusquement — pour choisir le nombre de composantes. Ici, l'inertie decroit quasi-lineairement. C'est **normal** pour des donnees categoriques : chaque variable contribue de maniere diffuse parce que les modalites sont nombreuses et distribuees.

**L'axe 1 est un axe de "securite alimentaire".** Les modalites qui contribuent le plus au premier axe sont `odor=foul`, `odor=none`, `stalk-surface-above-ring=silky`, `ring-type=pendant`. Ce sont exactement les criteres qu'un mycologue utilise pour evaluer la toxicite. L'ACM a retrouve, de maniere purement mathematique, l'expertise humaine.

**Choix de k : le dilemme biais-variance.** Le projet utilise k=5 (65.3% d'inertie) pour le clustering et la classification. L'analyse de sensibilite (script 06) montre que k=4 donne la meilleure CV accuracy (88.9%), contre 77.1% pour k=5 et 80.0% pour k=8. Plus de dimensions = plus d'information, mais aussi plus de bruit. C'est un resultat classique en statistique — la **malediction de la dimensionnalite** a petite echelle.

### Le pont

```
22 variables categorielles    →    Tableau disjonctif    →    Diagonalisation    →    k axes continus
   (mots)                           (~100 colonnes 0/1)      (valeurs propres)       (coordonnees reelles)
```

C'est ce pont — du categoriel au continu — qui rend tout le reste possible. Sans l'ACM, ni le K-Means (qui a besoin de distances euclidiennes), ni la LDA (qui a besoin de moyennes et de variances) ne pourraient operer.

---

## 3. Le specialiste clustering — Structures non supervisees

### Le protocole

Le clustering est effectue en deux temps :

1. **CAH Ward sur echantillon** (n=500) — La Classification Ascendante Hierarchique construit un arbre (dendrogramme) qui montre comment les individus se regroupent a differentes echelles. La methode Ward minimise l'inertie intra-cluster a chaque fusion. L'echantillonnage a 500 est un compromis entre lisibilite du dendrogramme et representativite.

2. **K-Means sur population complete** (n=8 124, k=3) — Le K-Means partitionne l'espace en 3 groupes en minimisant la somme des distances au centroïde. Le choix de k=3 vient de l'observation du dendrogramme (coupe a hauteur 15).

### Les resultats

| Cluster | Taille | Comestibles | Veneneux | Purete |
|---|---|---|---|---|
| **0** | 4 824 (59.4%) | 3 961 (82.1%) | 863 (17.9%) | Mixte |
| **1** | 192 (2.4%) | 192 (100%) | 0 (0%) | **100% comestible** |
| **2** | 3 108 (38.3%) | 55 (1.8%) | 3 053 (98.2%) | **98.2% veneneux** |

### L'analyse en profondeur

**Le Cluster 1 est remarquable.** 192 champignons, 100% comestibles. Zero erreur. Ce sont des specimens aux caracteristiques tellement distinctives qu'ils forment un ilot isole dans l'espace ACM. Le profiling (script 04) identifie les modalites sur-representees dans ce cluster — ce sont les criteres morphologiques qui rendent ces champignons sans ambiguïte.

**Le Cluster 2 est le "signal d'alarme".** 3 108 specimens, 98.2% veneneux. Seulement 55 comestibles s'y retrouvent — des champignons qui, morphologiquement, ressemblent a des venéneux. Dans un contexte reel, ce cluster dirait : "attention, zone dangereuse".

**Le Cluster 0 est la zone grise.** 59.4% du dataset, melange 82/18 comestible/veneneux. C'est la que la classification supervisee prend tout son sens — le clustering seul ne suffit pas pour departager.

**Le silhouette score (0.41 pour k=3, k_axes=5)** est dans la zone "structure raisonnable" (>0.25) mais pas excellente (<0.5). C'est coherent avec la nature des donnees : les champignons ne forment pas des clusters parfaitement spheriques dans l'espace ACM. L'analyse de sensibilite montre que le silhouette est meilleur a k=2 axes (clusters plus compacts en basse dimension) et se degrade a mesure qu'on ajoute des dimensions.

### L'asymetrie fondamentale

Le clustering revele quelque chose d'important pour la securite alimentaire : **il est plus facile d'identifier un groupe "certainement dangereux" que de garantir qu'un champignon est comestible**. Le Cluster 2 (98.2% veneneux) est presque pur. Le Cluster 0 (82% comestible) ne l'est pas. C'est une asymetrie qui a des implications pratiques.

---

## 4. Le specialiste classification — Modeles et evaluation

### LDA : le modele de base

L'Analyse Discriminante Lineaire cherche la combinaison lineaire des k axes ACM qui maximise la separation entre comestibles et venéneux. C'est le prolongement naturel de l'ACM — si l'ACM trouve les axes de plus grande variabilite globale, la LDA trouve l'axe de plus grande variabilite **entre les classes**.

**Resultats (k=5 axes) :**

| Metrique | Entrainement | Validation croisee |
|---|---|---|
| Accuracy | 88.7% | 77.1% +/- 14.0% |
| Precision veneneux | 96.8% | — |
| Recall comestible | 97.6% | — |

**L'ecart-type de 14.0% est un signal fort.** Sur 5 folds, les scores varient entre ~60% et ~90%. Ca signifie que le modele est sensible a la composition des donnees d'entrainement. Certaines partitions contiennent probablement des sous-populations mieux separees que d'autres.

### La comparaison de modeles (script 07)

Quatre classifieurs sur les memes 5 coordonnees ACM :

| Modele | Train | CV 5-fold | F1 (macro) | Ecart (overfitting) |
|---|---|---|---|---|
| **Random Forest** | 99.9% | **85.3%** | 0.853 | **14.6%** |
| SVM (RBF) | 96.3% | 82.6% | 0.826 | 13.7% |
| LDA | 88.7% | 77.1% | 0.771 | 11.6% |
| Logistic Regression | 88.6% | 74.7% | 0.747 | 13.9% |

### L'analyse critique

**Random Forest gagne mais triche.** 99.9% en entrainement, 85.3% en CV. L'ecart de 14.6% est le plus grand de tous les modeles. Avec `n_estimators=200` et `max_depth=10`, le modele a assez de capacite pour memoriser une grande partie des donnees d'entrainement. Le bon score en CV montre qu'il generalise quand meme mieux que les autres — mais au prix d'une complexite qui masque les mecanismes sous-jacents.

**LDA est le modele le plus honnete.** Son ecart train/CV (11.6%) est le plus faible. C'est aussi le modele le plus interpretable — ses coefficients montrent directement quels axes ACM comptent pour la decision. C'est un modele qu'on peut expliquer. Random Forest est une boite noire.

**SVM se place au milieu.** Le kernel RBF lui permet de tracer des frontieres non-lineaires, ce qui donne de meilleurs resultats que la LDA et la LogReg. L'overfitting est contenu.

**Logistic Regression et LDA sont quasi-identiques** (88.6% vs 88.7% en train). C'est attendu — les deux sont des modeles lineaires. La difference en CV (74.7% vs 77.1%) vient des hypotheses distributionnelles differentes.

### L'analyse de sensibilite (script 06)

**k=4 est optimal pour la LDA.** C'est le resultat le plus contre-intuitif du projet :

| k | Inertie cumulee | LDA CV Accuracy | Silhouette |
|---|---|---|---|
| 2 | 30.3% | 82.3% | 0.45 |
| 3 | 43.5% | 82.3% | 0.42 |
| **4** | **55.4%** | **88.9%** | 0.41 |
| 5 | 65.3% | 77.1% | 0.41 |
| 8 | 90.3% | 80.0% | 0.23 |

L'accuracy **chute** de k=4 a k=5, puis **remonte partiellement** a k=8. Explication probable : l'axe 5 apporte du bruit qui degrade la frontiere lineaire de la LDA. Avec plus d'axes (k=8), d'autres signaux utiles compensent, mais sans retrouver le pic de k=4.

C'est un resultat pedagogiquement riche : **plus de donnees ne signifie pas automatiquement un meilleur modele.** Le choix de k est un acte de jugement, pas un automatisme.

---

## 5. Le data engineer — Pipeline et architecture

### Architecture des donnees

```
data/
├── raw/
│   ├── agaricus-lepiota.data     ← Source brute UCI (8 124 lignes, pas de header)
│   └── agaricus-lepiota.names    ← Documentation UCI
└── processed/
    ├── mushroom_processed.csv    ← Dataset nettoye (8 124 x 23)
    └── mca_coords.csv            ← Coordonnees factorielles (8 124 x 10) — PIVOT
```

Le fichier `mca_coords.csv` est le **point pivot** du pipeline. Tout ce qui est en amont (scripts 00-03) produit ce fichier. Tout ce qui est en aval (scripts 04-07 + dashboard) le consomme. C'est le contrat d'interface qui decouple l'ACM de l'exploitation.

### Idempotence

Chaque script est re-executable sans effet de bord. Le script 00 verifie si les fichiers existent avant de telecharger (`if filepath.exists(): skip`). Les scripts de production ecrasent les outputs precedents. C'est un patron essentiel pour la reproductibilite.

### Flux de donnees complet

```
00_download.py        → data/raw/*.{data,names}
01_prepare.py         → data/processed/mushroom_processed.csv
02_describe.py        → reports/tables/2 + reports/figures/4
03_mca.py             → data/processed/mca_coords.csv + reports/tables/3 + reports/figures/3
04_cluster.py         → reports/tables/3 + reports/figures/2
05_discriminant.py    → reports/tables/2 + reports/figures/3
06_sensitivity.py     → reports/tables/1 + reports/figures/2
07_model_comparison.py → reports/tables/1 + reports/figures/2
                        ─────────────────────────────────
                        Total : 12 tables, 16 figures
```

### Orchestration

Le **Makefile** centralise toutes les operations :

- `make install` — Cree l'environnement virtuel, met a jour pip, installe les dependances
- `make run-all` — Execute le pipeline complet dans l'ordre
- `make run-extended` — Execute uniquement les analyses de sensibilite et comparaison
- `make dashboard` — Lance le Streamlit
- `make clean` / `make distclean` — Nettoyage partiel ou complet

Le target `help` est auto-documente (pattern `##` dans le Makefile).

### CI/CD

Le workflow GitHub Actions (`pipeline.yml`) execute le pipeline complet sur chaque push/PR :

- **Matrice** : Python 3.10 et 3.11
- **Cache pip** : Accelere les builds suivants
- **Verification** : Liste les outputs generes pour confirmer le succes

C'est la garantie que le pipeline reste fonctionnel a chaque modification.

---

## 6. Le visualiste — Communication des resultats

### Inventaire des figures

Le projet produit **19 figures** au total (16 pipeline + 3 screenshots dashboard), toutes en 300 DPI.

**Figures d'exploration (script 02) :**
- `desc_target_bar.png` — Distribution comestible/veneneux. Couleurs vert/rouge intuitives. Les chiffres sont affiches sur les barres.
- `desc_top_modalities_*.png` — 3 figures pour les variables cles (odeur, couleur des lamelles, sporee). Palette Set3 pour distinguer les modalites.

**Figures ACM (script 03) :**
- `acm_scree.png` — Deux panneaux : inertie par axe + inertie cumulee. Ligne de seuil a 90%. C'est la figure de diagnostic standard en analyse factorielle.
- `acm_modalities_12.png` — Plan factoriel des modalites. Chaque point est une modalite annotee. Permet de comprendre ce que chaque axe "signifie".
- `acm_individuals_12_color_target.png` — La figure la plus importante du projet. 8 124 points colores par classe. La separation visuelle entre vert (comestible) et rouge (veneneux) est le moment "aha" du README.

**Figures clustering (script 04) :**
- `cluster_dendrogram.png` — Dendrogramme CAH Ward avec ligne de coupe suggeree. Echantillon de 500 pour la lisibilite.
- `cluster_on_acm12.png` — Clusters projetes sur le plan ACM 1-2. Colormap viridis.

**Figures classification (script 05) :**
- `da_confusion.png` / `da_confusion_cv.png` — Matrices de confusion en heatmap (train et CV). Colormaps Blues et Greens pour les distinguer.
- `da_cv_scores.png` — Accuracy par fold avec ligne de moyenne.

**Figures etendues (scripts 06-07) :**
- `sensitivity_k_analysis.png` — Triple panneau (accuracy vs k, silhouette vs k, overfitting gap vs k). C'est la figure la plus dense du projet — trois diagnostics en un coup d'oeil.
- `sensitivity_inertia_vs_accuracy.png` — Double axe Y (inertie et accuracy) sur le meme graphe.
- `model_comparison.png` — Double panneau (train vs CV accuracy, F1 + recall veneneux). Error bars sur les scores CV.
- `model_comparison_boxplot.png` — Box plots des scores CV par modele. Montre la dispersion, pas juste la moyenne.

### Choix de design

- **Palette coherente** : Bleu (#2196F3) pour train/reference, rouge (#F44336) pour CV/test, vert (#4CAF50) pour positif/cumul, violet (#9C27B0) pour overfitting. Ces couleurs sont constantes a travers tout le projet.
- **Taille des figures** : Adaptee au contenu (14x5 pour les multi-panneaux, 10x8 pour les scatter plots, 8x6 pour les heatmaps).
- **Annotations** : Valeurs affichees sur les barres, lignes de reference (seuil 90%, moyenne CV, seuil overfitting 10%).

### Le dashboard Streamlit

Le dashboard (`app.py`, 400 lignes) traduit les figures statiques en **explorations interactives** via Plotly :

- **Vue d'ensemble** — Metriques cles, distribution cible, apercu du dataset
- **Espace ACM** — Choix des axes (Dim1 a Dim10), coloration par classe ou par n'importe quelle variable
- **Clustering** — Sliders pour k (clusters) et k (axes ACM), calcul en temps reel du silhouette, affichage cote-a-cote clusters vs classes reelles
- **Classification** — Slider pour k axes, LDA recalculee a la volee, matrice de confusion et rapport de classification
- **Sensibilite** — Graphes interactifs accuracy vs k et silhouette vs k

Le dashboard ne duplique pas les scripts — il recalcule a la volee pour permettre l'exploration parametrique. C'est un outil de comprehension, pas de production.

---

## 7. Le developpeur — Qualite de code

### Ce qui est bien fait

| Aspect | Implementation | Evaluation |
|---|---|---|
| **Type hints** | `from __future__ import annotations` + annotations sur toutes les fonctions | Excellent |
| **Docstrings** | Format NumPy (`Parameters`, `Returns`, `Raises`) | Excellent |
| **Constantes** | Centralisees en haut de chaque script (`K_AXES`, `N_CLUSTERS`, `CV_FOLDS`) | Excellent |
| **Utils** | Module partage (`utils.py`) pour I/O, chemins, affichage | Excellent |
| **Gestion d'erreurs** | `FileNotFoundError` avec messages explicites | Bien |
| **Reproductibilite** | `RANDOM_STATE = 42` sur tous les generateurs aleatoires | Excellent |
| **Structure** | Numerotation des scripts (00-07), separation data/src/reports/docs | Excellent |
| **Console** | Output structure avec sections, etapes, et resume en fin de script | Bien |

### Ce qui pourrait etre ameliore

| Zone | Etat actuel | Suggestion | Impact |
|---|---|---|---|
| **Tests** | Aucun test unitaire | Ajouter `pytest` pour `utils.py` et les fonctions de chargement | Robustesse |
| **Config** | Constantes dupliquees entre scripts (ex: `K_AXES=5` dans 04, 05, 07) | Centraliser dans un `config.py` ou `config.yaml` | Maintenabilite |
| **Logging** | `print()` partout | Migrer vers `logging` avec niveaux (DEBUG, INFO, WARNING) | Debuggabilite |
| **Linting** | Pas de config linter | Ajouter `ruff` ou `flake8` dans le CI | Coherence |
| **Residus "Personne A/B"** | Quelques mentions dans les docstrings (ex: "Mission A3", "Personne B") | Nettoyer pour le rendu public | Presentation |
| **Imports dans fonctions** | `from sklearn.cluster import KMeans` a l'interieur de `app.py` pages | Deplacer en haut du fichier | Convention |
| **Dependances notebook** | `jupyter`, `ipykernel`, `Pillow` dans requirements.txt | Supprimer si les notebooks ont ete retires | Legerete |

### Metriques de code

| Metrique | Valeur |
|---|---|
| Fichiers Python | 10 (9 scripts + app.py) |
| Lignes de code totales | ~1 400 |
| Lignes par script (moyenne) | ~120 |
| Fonctions documentees | 100% |
| Type hints | 100% des signatures publiques |
| Scripts avec constantes centralisees | 10/10 |
| Points d'entree `if __name__ == "__main__"` | 8/8 scripts |

---

## 8. Le product owner — Ce que ce projet apporte

### La valeur demonstree

Ce projet n'est pas un exercice technique. C'est la demonstration d'une **competence rare** : savoir choisir et articuler les bons outils statistiques face a un probleme specifique.

**Le probleme** : des donnees purement qualitatives. Pas de mesures, pas de capteurs, juste des descriptions humaines. 90% des tutoriels de machine learning commencent avec des donnees numeriques. Ce projet commence la ou la plupart s'arretent.

**La solution** : une chaine methodologique coherente ou chaque maillon a une raison d'etre.

- L'ACM n'est pas la parce que c'est "une technique cool". Elle est la parce que c'est la **seule maniere mathematiquement fondee** de rendre ces donnees exploitables par des algorithmes classiques.
- Le clustering n'est pas la pour "ajouter une analyse". Il est la pour **valider** que la structure ACM est significative avant d'appliquer un modele supervise.
- La comparaison de modeles n'est pas la pour "montrer qu'on connait Random Forest". Elle est la pour montrer qu'on sait **poser les bonnes questions** — est-ce que 88.7% c'est bien ? Par rapport a quoi ?
- L'analyse de sensibilite n'est pas la pour "etre exhaustif". Elle est la pour montrer qu'on comprend que **chaque choix a des consequences** et qu'on les mesure.

### Ce que ca dit de l'auteur

1. **Comprend la chaine statistique** — De la donnee brute au resultat interprete, en passant par la bonne transformation.
2. **Pose les bonnes questions** — "Pourquoi k=5 et pas k=8 ?" est la question que trop de projets ne posent pas.
3. **Sait evaluer honnetement** — L'overfitting est explicitement mesure et discute. Le modele le plus performant n'est pas automatiquement le "meilleur".
4. **Communique clairement** — Le README raconte une histoire. Le dashboard rend les resultats tangibles. Les figures sont lisibles.
5. **Ingenierie solide** — Pipeline reproductible, CI, code type-hinte et documente.

### Les resultats en une vue

| Dimension | Resultat cle |
|---|---|
| **ACM** | 10 axes, 90% d'inertie a k=8, axe 1 = profil de toxicite |
| **Clustering** | 3 clusters, dont 1 pur comestible (100%) et 1 quasi-pur veneneux (98.2%) |
| **LDA** | 88.7% accuracy, 97.6% recall comestible, 96.8% precision veneneux |
| **Sensibilite** | k=4 optimal (88.9% CV), plus de dimensions ≠ meilleur |
| **Comparaison** | Random Forest meilleur en brut (85.3% CV) mais overfitting massif |
| **Dashboard** | 5 pages interactives, deploye sur Streamlit Cloud |
| **Pipeline** | 9 scripts, 12 tables, 19 figures, CI GitHub Actions |

### Le message final

> Un projet de data science, ce n'est pas un modele qui donne un bon score. C'est une demarche qui pose les bonnes questions, choisit les bons outils, et sait dire honnetement ce qui marche, ce qui ne marche pas, et pourquoi.

The Mushroom Project fait exactement ca.

---

*Audit realise le 18 fevrier 2026 — The Mushroom Project*
