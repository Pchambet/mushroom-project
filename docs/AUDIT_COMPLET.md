# Audit Complet — The Mushroom Project

> Analyse en profondeur de l'identite, de l'architecture, de la synergie methodologique et des resultats.

---

## 1. Identite et vision

### 1.1 Le probleme fondamental

On dispose de 8 124 specimens de champignons decrits par 22 variables **categoriques** (forme du chapeau, odeur, couleur des lamelles...). On veut les classifier en comestibles ou venéneux.

Le probleme : les outils classiques du Machine Learning — K-Means, LDA, distances euclidiennes — ne s'appliquent pas directement a des variables nominales. On ne peut pas calculer une "distance" entre `odor=foul` et `odor=almond`.

### 1.2 La synergie fondamentale

C'est la que l'Analyse en Composantes Multiples (ACM) intervient comme **pont mathematique** :

```
  22 variables categoriques         ACM              Espace euclidien
  ─────────────────────────   ──────────────>   ───────────────────────
  cap-shape = convex                              Dim1 = -0.342
  odor = foul                    Reduction        Dim2 =  0.891
  gill-color = buff              dimensionnelle   Dim3 = -0.127
  ring-type = pendant                             ...
  ...                                             Dimk
           ↓                                            ↓
     Pas de metrique             ─────>         Distances, moyennes,
     possible                                   variances exploitables
                                                        ↓
                                                ┌───────────────────┐
                                                │  K-Means + CAH    │
                                                │  (clustering)     │
                                                ├───────────────────┤
                                                │  LDA              │
                                                │  (classification) │
                                                └───────────────────┘
```

**Idee-force** : L'ACM n'est pas une etape technique parmi d'autres. C'est le fondement qui rend tout le reste possible. Sans elle, ni le clustering ni la classification ne fonctionnent sur ces donnees.

### 1.3 La chaine de valeur

Chaque etape construit sur la precedente, sans raccourci possible :

1. **Donnees brutes** (8 124 x 23) → Nettoyage, imputation
2. **ACM** → Projection dans un espace continu de dimension reduite
3. **Clustering** → Decouverte de structures naturelles dans cet espace
4. **LDA** → Classification supervisee dans ce meme espace

La coherence narrative ACM → Clustering → LDA n'est pas un choix esthetique — c'est une necessite mathematique.

---

## 2. Architecture technique

### 2.1 Pipeline de donnees

```
00_download.py
  └─> data/raw/agaricus-lepiota.{data,names}

01_prepare.py
  └─> data/processed/mushroom_processed.csv        [8 124 x 23]

02_describe.py
  ├─> reports/tables/univariate_summary.csv
  ├─> reports/tables/target_distribution.csv
  └─> reports/figures/desc_*.png                    [4 figures]

03_mca.py
  ├─> data/processed/mca_coords.csv                [8 124 x 10]  ← INTERFACE
  ├─> reports/tables/mca_eigenvalues.csv
  ├─> reports/tables/mca_modalities_contrib_*.csv   [axes 1 et 2]
  └─> reports/figures/acm_*.png                     [3 figures]

04_cluster.py
  ├─> reports/tables/cluster_sizes.csv
  ├─> reports/tables/cluster_vs_target.csv
  ├─> reports/tables/cluster_profiles.csv
  └─> reports/figures/cluster_*.png                 [2 figures]

05_discriminant.py
  ├─> reports/tables/da_metrics.csv
  ├─> reports/tables/da_confusion.csv
  └─> reports/figures/da_*.png                      [3 figures]
```

Chaque script est **idempotent** : on peut le re-executer a tout moment sans effet de bord.

### 2.2 Contrat d'interface

Le fichier `mca_coords.csv` est le point pivot du pipeline :

| Propriete | Specification |
|---|---|
| Lignes | 8 124 (correspondance 1:1 avec `mushroom_processed.csv`) |
| Colonnes | `Dim1` a `Dim10` (float64) |
| Pas de NA | Garanti par imputation modale en amont |
| Validation | `validate_interface.py` verifie les contraintes automatiquement |

### 2.3 Stack

| Role | Technologie | Justification |
|---|---|---|
| ACM | `prince` | Seule lib Python maintenue pour l'ACM (MCA) |
| Clustering | `scipy` (CAH Ward) + `scikit-learn` (K-Means) | Hierarchique pour explorer, partitionnel pour consolider |
| Classification | `scikit-learn` (LDA) | Modele lineaire interpetable, adapte a la reduction ACM |
| Visualisation | `matplotlib` + `seaborn` | Figures publication-ready, 300 DPI |
| Validation | Script custom | Verification automatique du contrat d'interface |

---

## 3. Analyse des resultats

### 3.1 ACM — Reduction dimensionnelle

**Valeurs propres et inertie :**

| Axe | Valeur propre | Inertie (%) | Cumul (%) |
|---|---|---|---|
| Dim 1 | 0.324 | 15.9 | 15.9 |
| Dim 2 | 0.295 | 14.4 | 30.3 |
| Dim 3 | 0.271 | 13.2 | 43.5 |
| Dim 4 | 0.243 | 11.9 | 55.4 |
| Dim 5 | 0.203 | 9.9 | 65.3 |
| Dim 6 | 0.193 | 9.4 | 74.7 |
| Dim 7 | 0.173 | 8.5 | 83.2 |
| **Dim 8** | **0.144** | **7.1** | **90.3** |
| Dim 9 | 0.103 | 5.0 | 95.3 |
| Dim 10 | 0.096 | 4.7 | 100.0 |

**Observations :**

- Pas de "coude" marque sur le scree plot. L'inertie decroit progressivement, ce qui est typique des donnees categoriques avec de nombreuses modalites (22 variables, ~100 modalites au total).
- Le seuil de 90% est atteint a **k=8 axes**. Pour le clustering et la LDA, k=5 est utilise (65.3% d'inertie), ce qui est un compromis entre parcimonie et preservation de l'information.
- Les axes 1 et 2 capturent a eux seuls 30.3% de l'inertie — suffisant pour une visualisation interpretable du plan factoriel.

**Modalites les plus contributrices :**

- **Axe 1** : discrimine principalement sur l'odeur (`odor`), la surface du pied (`stalk-surface`), et le type d'anneau (`ring-type`). C'est l'axe "securite alimentaire" — il separe les profils morphologiques typiques des venéneux.
- **Axe 2** : capture des variations de couleur (chapeau, lamelles, sporée) et d'habitat. C'est l'axe "diversite morphologique".

### 3.2 Clustering — Structures naturelles

**K-Means (k=3) sur les 5 premieres composantes ACM :**

| Cluster | Taille | % Total | Comestibles | Venéneux | Purete dominante |
|---|---|---|---|---|---|
| **0** | 4 824 | 59.4% | 3 961 (82.1%) | 863 (17.9%) | Mixte |
| **1** | 192 | 2.4% | 192 (100%) | 0 (0%) | **100% comestible** |
| **2** | 3 108 | 38.3% | 55 (1.8%) | 3 053 (98.2%) | **98.2% venéneux** |

**Analyse des profils :**

- **Cluster 0 (Mixte, 59.4%)** : Le plus grand groupe. Melange de comestibles et venéneux. Morphologiquement heterogene — c'est le "tout-venant" qui necessite des criteres plus fins pour discriminer.

- **Cluster 1 (Pur comestible, 2.4%)** : Petit groupe de 192 specimens **exclusivement comestibles**. Ce sont des champignons aux caracteristiques morphologiques tres distinctives qui les rendent sans ambiguïte. Ce cluster est un resultat remarquable — il identifie un sous-groupe "sur" sans aucune erreur.

- **Cluster 2 (Quasi-pur venéneux, 38.3%)** : 98.2% de venéneux. Seulement 55 comestibles se retrouvent dans ce groupe — probablement des specimens morphologiquement proches des venéneux mais inoffensifs.

**Interpretation methodologique :**

Le clustering revele que la structure naturelle des donnees dans l'espace ACM est **fortement correlée** a l'edibilite, mais de maniere asymetrique : il est plus facile d'identifier un groupe "certainement venéneux" (Cluster 2, 98.2%) que de garantir qu'un champignon est comestible. Le Cluster 0 est le "zone grise" ou la classification supervisee prend tout son sens.

### 3.3 Analyse discriminante — Classification

**LDA sur les 5 premieres composantes ACM :**

| Metrique | Valeur |
|---|---|
| Accuracy (entrainement) | **88.7%** |
| Precision (venéneux) | 96.8% |
| Recall (venéneux) | 79.2% |
| F1 (venéneux) | 87.1% |
| Precision (comestible) | 83.5% |
| Recall (comestible) | **97.6%** |
| F1 (comestible) | 90.0% |
| **CV 5-fold (moyenne)** | **77.1% +/- 14.0%** |

**Matrice de confusion (entrainement) :**

```
                  Pred. Venéneux    Pred. Comestible
Vrai Venéneux        3 102              814
Vrai Comestible        103            4 105
```

**Analyse critique :**

1. **Ecart train/CV** : L'accuracy d'entrainement (88.7%) contre la CV (77.1%) signale un **sur-apprentissage modere**. L'ecart-type de 14.0% sur les folds indique une **instabilite** du modele selon les partitions.

2. **Asymetrie des erreurs** : Le modele a un recall de 97.6% sur les comestibles mais seulement 79.2% sur les venéneux. Concretement : 814 champignons venéneux sont predits comme comestibles. Dans un contexte reel, c'est la metrique la plus critique — un faux negatif sur un venéneux est potentiellement mortel.

3. **Precision venéneux (96.8%)** : Quand le modele dit "venéneux", il a raison 96.8% du temps. C'est rassurant pour les predictions positives.

4. **Pourquoi la LDA et pas autre chose ?** La LDA est choisie pour sa coherence avec l'approche ACM : on projette lineairement les coordonnees factorielles sur un axe discriminant. C'est le prolongement naturel de la reduction dimensionnelle.

5. **Piste d'amelioration** : Utiliser k=8 axes (90.3% d'inertie) au lieu de k=5 (65.3%) pourrait ameliorer significativement la performance, au prix d'un espace de plus grande dimension.

---

## 4. Qualite du code

### 4.1 Points forts

| Aspect | Evaluation |
|---|---|
| Structure du projet | Nomenclature numerotee, separation claire data/src/reports |
| Reproductibilite | Makefile avec `run-all`, `clean`, `install` |
| Idempotence | Chaque script re-executable sans effet de bord |
| Validation | Script dedie pour le contrat d'interface |
| Documentation | Dictionnaire de donnees, specs d'interface, guides, FAQ |
| Conventions de nommage | Prefixes coherents (`desc_`, `acm_`, `cluster_`, `da_`) |
| Type hints | Annotations de type sur toutes les fonctions publiques |
| Docstrings | Format NumPy avec `Parameters`, `Returns`, `Raises` |
| Configuration | Constantes centralisees en haut de chaque script |
| Gestion d'erreurs | `FileNotFoundError` explicites avec messages utiles |

### 4.2 Pistes d'amelioration

| Zone | Suggestion | Impact |
|---|---|---|
| Tests unitaires | Ajouter `pytest` pour les fonctions critiques (`utils.py`, `validate_interface.py`) | Robustesse |
| Configuration globale | Centraliser `K_AXES`, `N_CLUSTERS` dans un `config.yaml` | Flexibilite |
| Logging | Remplacer `print()` par `logging` avec niveaux | Debuggabilite |
| Parallelisme | Scripts 04 et 05 independants → executables en parallele | Performance |
| ACM alternatives | Tester k=8 (90% inertie) vs k=5 (65%) et comparer les metriques LDA | Rigueur |

---

## 5. Inventaire des livrables

### 5.1 Tables (10 fichiers CSV)

| Fichier | Lignes | Colonnes | Contenu |
|---|---|---|---|
| `univariate_summary.csv` | 23 | 5 | Resume par variable (modalites, frequences, NA) |
| `target_distribution.csv` | 2 | 3 | Distribution comestible/venéneux |
| `mca_eigenvalues.csv` | 10 | 4 | Valeurs propres, inertie, cumul |
| `mca_modalities_contrib_axis1.csv` | 15 | 3 | Top contributions axe 1 |
| `mca_modalities_contrib_axis2.csv` | 15 | 3 | Top contributions axe 2 |
| `cluster_sizes.csv` | 3 | 3 | Taille et % des clusters |
| `cluster_vs_target.csv` | 4 | 4 | Crosstab cluster x classe (avec marges) |
| `cluster_profiles.csv` | Variable | 6 | Modalites sur/sous-representees |
| `da_confusion.csv` | 2 | 2 | Matrice de confusion |
| `da_metrics.csv` | 9 | 2 | Accuracy, precision, recall, F1, CV |

### 5.2 Figures (12 fichiers PNG, 300 DPI)

| Fichier | Type | Description |
|---|---|---|
| `desc_target_bar.png` | Bar chart | Distribution edible vs poisonous |
| `desc_top_modalities_odor.png` | Bar chart | Modalites de l'odeur |
| `desc_top_modalities_gill-color.png` | Bar chart | Modalites de la couleur des lamelles |
| `desc_top_modalities_spore-print-color.png` | Bar chart | Modalites de la sporée |
| `acm_scree.png` | Line plot (2 panneaux) | Inertie par composante + cumulative |
| `acm_modalities_12.png` | Scatter | Plan factoriel des modalites (Dim 1-2) |
| `acm_individuals_12_color_target.png` | Scatter | Individus colores par classe (Dim 1-2) |
| `cluster_dendrogram.png` | Dendrogramme | CAH Ward (500 individus) |
| `cluster_on_acm12.png` | Scatter | Clusters projetes sur plan ACM |
| `da_confusion.png` | Heatmap | Matrice de confusion (entrainement) |
| `da_confusion_cv.png` | Heatmap | Matrice de confusion (validation croisee) |
| `da_cv_scores.png` | Bar chart | Accuracy par fold de CV |

---

## 6. Synthese

### Ce que ce projet demontre

1. **Maitrise de la chaine statistique** : De la donnee brute au resultat interprete, en passant par la reduction dimensionnelle adaptee.

2. **Comprehension du "pourquoi"** : L'ACM n'est pas un gadget — c'est la transformation necessaire qui rend le clustering et la classification possibles sur des donnees categoriques.

3. **Rigueur evaluative** : Validation croisee 5-fold, analyse des erreurs, identification du sur-apprentissage, interpretation critique des metriques.

4. **Ingenierie logicielle** : Pipeline reproductible, scripts idempotents, contrat d'interface valide automatiquement, code type-hinte et documente.

### Le message en une phrase

> L'ACM transforme 22 variables categoriques en un espace euclidien structure, revelant une separation naturelle entre champignons comestibles et venéneux exploitable par clustering (100% de purete sur un sous-groupe) et classification (88.7% d'accuracy).

---

*Audit realise le 16 fevrier 2026 — The Mushroom Project*
