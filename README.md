<p align="center">
  <img src="reports/figures/banner.jpg" width="100%" alt="The Mushroom Project">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn">
  <a href="https://pchambet-mushroom-project.streamlit.app/"><img src="https://img.shields.io/badge/Streamlit-Live_Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Dashboard"></a>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
</p>

---

Un champignon devant vous. Chapeau convexe, odeur anisee, lamelles libres, anneau pendant.

**Comestible ou mortel ?**

Un mycologue experimente croise ces indices mentalement, en une seconde. Mais comment faire comprendre ca a une machine ?

Parce que le vrai probleme, ce n'est pas la classification. C'est que **toutes ces informations sont des mots**. "Convexe", "anis", "libre", "pendant" — pas un seul chiffre. Or les algorithmes de machine learning ont besoin de nombres pour calculer des distances, tracer des frontieres, regrouper des points. Face a des mots, ils sont aveugles.

La vraie question de ce projet :

> **Comment extraire du sens calculable a partir de descriptions qualitatives ?**

Pour y repondre, on va suivre une demarche en 5 temps. Chacun construit sur le precedent. Aucun ne peut etre saute.

```
Observer  →  Transformer  →  Decouvrir  →  Predire  →  Interroger
```

---

## 1. Observer

8 124 champignons. 22 caracteristiques chacun. Toutes categorielles — c'est-a-dire des etiquettes, pas des chiffres. La forme du chapeau peut etre "convexe", "plate" ou "en cloche". L'odeur peut etre "anis", "foul" ou "aucune". La couleur des lamelles, le type d'anneau, l'habitat — tout est du texte.

La variable cible est binaire : **comestible (e)** ou **veneneux (p)**. La distribution est quasi-equilibree — 51.8% comestibles, 48.2% veneneux. Pas de desequilibre majeur a corriger.

Une seule variable a des donnees manquantes : `stalk-root` (la racine du pied), avec 30.5% de valeurs inconnues. On les impute par la modalite la plus frequente — c'est le choix standard quand on ne veut pas perdre d'individus.

A ce stade, on a un tableau de mots. On ne peut rien calculer dessus. Pas de moyenne, pas de distance, pas de modele. Il faut **transformer**.

---

## 2. Transformer — ce que fait reellement l'ACM

C'est l'etape cle de tout le projet. L'**Analyse en Composantes Multiples** (ACM) transforme un tableau de mots en un tableau de nombres. Voila comment.

### L'idee

Imaginons un champignon decrit par 3 caracteristiques : `odeur=anis`, `chapeau=convexe`, `habitat=foret`. L'ACM commence par "eclater" chaque caracteristique en autant de colonnes qu'il y a de modalites possibles, avec des 0 et des 1 :

```
                 odeur_anis  odeur_foul  odeur_none  chapeau_convexe  chapeau_plat  habitat_foret  habitat_prairie
Champignon #1:      1           0           0             1               0              1              0
Champignon #2:      0           1           0             0               1              0              1
...
```

Ce tableau s'appelle le **tableau disjonctif complet**. Pour 22 variables et environ 100 modalites au total, il fait 8 124 lignes x ~100 colonnes de 0 et de 1.

### La magie

Ce tableau de 0/1 est maintenant numerique. On peut calculer des distances, des moyennes, des variances. L'ACM cherche les **axes de plus grande variabilite** dans ce tableau — les directions le long desquelles les champignons different le plus les uns des autres. C'est une technique de diagonalisation matricielle — on decompose la structure de ce gros tableau en composantes ordonnees, de la plus informative a la moins informative.

Les coordonnees de chaque champignon sur ces axes deviennent ses **nouvelles variables** : des nombres reels, continus, exploitables.

### Le resultat

```
22 variables textuelles                    ACM                     10 axes numeriques
┌─────────────────────┐      ┌──────────────────────┐      ┌────────────────────┐
│ odeur = anis        │      │ Tableau disjonctif    │      │ Dim1 = -0.342      │
│ chapeau = convexe   │  →   │ (8124 x ~100, en 0/1)│  →   │ Dim2 =  0.891      │
│ lamelles = libres   │      │ Diagonalisation       │      │ Dim3 = -0.127      │
│ anneau = pendant    │      │ (axes de variabilite) │      │ ...                │
│ ...22 variables     │      └──────────────────────┘      │ Dim10 = 0.045      │
└─────────────────────┘                                     └────────────────────┘
```

Chaque champignon a maintenant une position sur une carte a 10 dimensions. On peut calculer des distances entre eux. On peut les regrouper. On peut tracer des frontieres. **C'est ce pont — du texte vers la geometrie — qui rend tout le reste possible.**

### Ce que chaque axe capture

Tous les axes ne se valent pas. Le premier axe capture 15.9% de l'information totale, le deuxieme 14.4%, et ainsi de suite. Ensemble, les 8 premiers axes capturent 90% de l'information.

<p align="center">
  <img src="reports/figures/acm_scree.png" width="700" alt="Repartition de l'information par axe">
</p>

Pas de coupure nette entre les axes — c'est **normal** pour des donnees categorielles. En ACP (son equivalent pour des donnees numeriques), on cherche un "coude" dans ce graphique pour choisir combien d'axes garder. Ici, l'inertie decroit progressivement parce que les ~100 modalites contribuent chacune un peu.

Et le plus remarquable : **l'axe 1 est un "axe de securite alimentaire"**. Les modalites qui y contribuent le plus sont `odeur=foul`, `surface-du-pied=soyeuse`, `type-d'anneau=pendant` — exactement les criteres qu'un mycologue utilise pour evaluer la toxicite. L'ACM a retrouve, de maniere purement mathematique, l'expertise humaine.

### Le moment "aha"

Des qu'on place les 8 124 champignons sur les deux premiers axes :

<p align="center">
  <img src="reports/figures/acm_individuals_12_color_target.png" width="700" alt="ACM - Individus colores par classe">
</p>

Les comestibles (vert) et les veneneux (rouge) **se separent naturellement**. L'algorithme ne connait meme pas les etiquettes. Il a juste lu les descriptions et trouve, tout seul, que les champignons dangereux ne ressemblent pas aux autres.

Ca veut dire que l'information de toxicite est **encodee dans la morphologie**. La transformation a fonctionne. On peut passer a l'etape suivante.

---

## 3. Decouvrir — ce que le clustering revele

Avant de predire, on explore. On demande a un algorithme : **"est-ce qu'il y a des groupes naturels dans cet espace ?"** — sans lui dire lesquels chercher.

C'est ca, le **clustering** : regrouper des points qui se ressemblent, sans etiquettes, sans supervision. L'algorithme mesure des distances dans l'espace ACM et decoupe les champignons en groupes homogenes.

### Deux approches complementaires

**La CAH (Classification Ascendante Hierarchique)** construit un arbre — le dendrogramme. En bas, chaque champignon est seul. On fusionne progressivement les deux points les plus proches, puis les deux groupes les plus proches, et on remonte. L'arbre montre a quelle echelle les groupes se forment. La methode "Ward" minimise la variabilite a l'interieur de chaque groupe a chaque fusion.

**Le K-Means** prend une approche directe : on fixe un nombre de groupes (k=3 ici, guide par le dendrogramme), et l'algorithme place 3 centres dans l'espace, puis assigne chaque champignon au centre le plus proche. Il ajuste iterativement les centres jusqu'a convergence.

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

### Ce que ca signifie

**Le Cluster 1** : 192 champignons, **100% comestibles**. Zero erreur. Ce sont des specimens aux caracteristiques tellement distinctives qu'ils forment un ilot isole dans l'espace. Si un champignon tombe dans ce cluster, il est comestible — garanti.

**Le Cluster 2** : 3 108 champignons, **98.2% veneneux**. C'est le "signal d'alarme". Si un champignon tombe ici, il est quasi-certainement dangereux.

**Le Cluster 0** : la **zone grise**. 59.4% du dataset, melange 82/18. Le clustering seul ne suffit pas pour departager ces champignons-la. C'est exactement la ou un modele de prediction va prendre le relais.

Et rappelons-le : l'algorithme n'a **jamais vu l'etiquette** "comestible" ou "veneneux". Il a juste mesure des proximites. Le fait qu'il retrouve des groupes quasi-purs **valide toute la chaine en amont** : les donnees sont bonnes, la transformation ACM est pertinente, la structure est reelle.

Alors si la structure est la... peut-on reellement predire ?

---

## 4. Predire — tracer la frontiere

Pour predire, il faut un modele qui **apprend la frontiere** entre comestibles et veneneux a partir d'exemples etiquetes. Contrairement au clustering (qui ne regarde pas les etiquettes), ici on montre au modele des champignons dont on connait la classe, et il apprend la regle de separation.

### L'Analyse Discriminante Lineaire (LDA)

La LDA cherche la **meilleure droite de separation** dans l'espace ACM. Imagine les champignons comme des points sur un plan. La LDA trace la ligne qui maximise l'ecart entre les comestibles d'un cote et les veneneux de l'autre, tout en minimisant la dispersion a l'interieur de chaque groupe.

C'est le prolongement naturel de l'ACM : l'ACM trouve les axes de plus grande variabilite globale, la LDA trouve l'axe de plus grande variabilite **entre les classes**.

### Evaluer honnetement

Un modele qui obtient 88.7% de bonnes reponses sur les donnees d'entrainement, c'est bien. Mais est-ce qu'il sait generaliser ? Est-ce qu'il a vraiment appris la regle, ou est-ce qu'il a juste memorise les exemples ?

Pour le savoir, on utilise la **validation croisee** : on decoupe les 8 124 champignons en 5 parts egales. On entraine le modele sur 4 parts, on le teste sur la 5eme. Puis on tourne — 5 fois, chaque part servant de test une fois. C'est la seule maniere de mesurer la performance reelle.

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

Mais 77.1% en validation croisee, avec un ecart-type de 14.0% — ca veut dire que sur certains decoupages, le modele descend a ~60%. **Il n'est pas stable.** Ca souleve des questions. Et c'est exactement le sujet de l'etape suivante.

---

## 5. Interroger — remettre en question ses propres choix

C'est l'etape que la plupart des projets de data science sautent. On a un resultat — 88.7% d'accuracy — et on s'arrete la. Ici, on pose deux questions supplementaires.

### Combien d'axes garder ?

On a utilise 5 axes ACM. Pourquoi pas 3 ? Pourquoi pas 8 ? Est-ce qu'ajouter de l'information ameliore toujours la prediction ?

On teste systematiquement de 2 a 10 axes :

<p align="center">
  <img src="reports/figures/sensitivity_k_analysis.png" width="700" alt="Analyse de sensibilite">
</p>

Resultat contre-intuitif : **4 axes donnent la meilleure performance (88.9%)**, mieux que 5 (77.1%) ou 8 (80.0%).

Pourquoi ? Parce que les axes supplementaires n'apportent pas que de l'information utile — ils apportent aussi du **bruit**. L'axe 5, par exemple, capture des variations qui ne sont pas liees a la toxicite. Le modele lineaire, en essayant d'integrer ce bruit, perd en precision. C'est un phenomene classique en statistique : la **malediction de la dimensionnalite** — plus n'est pas toujours mieux.

En data science, **savoir quoi enlever compte autant que savoir quoi garder.**

### Quel modele choisir ?

La LDA trace une frontiere lineaire — une droite. Mais d'autres modeles savent tracer des frontieres plus complexes :

- Le **Random Forest** combine des centaines d'arbres de decision. Chaque arbre pose des questions simples ("odeur = foul ?"), et on fait voter l'ensemble.
- Le **SVM** (Support Vector Machine) deforme mathematiquement l'espace pour qu'une frontiere lineaire dans cet espace deforme corresponde a une frontiere courbe dans l'espace original.
- La **Regression Logistique** estime la probabilite d'appartenance a chaque classe via une fonction sigmoïde.

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

Random Forest gagne en performance brute. Mais regardez la colonne "Entrainement" : **99.9%**. Quasi parfait. Et sur des donnees jamais vues : 85.3%. Cet ecart a un nom — **l'overfitting**. Le modele a memorise les exemples au lieu d'apprendre la regle.

C'est comme un etudiant qui retient les reponses du QCM sans comprendre le cours : brillant a l'examen blanc, fragile le jour J.

LDA, avec un ecart beaucoup plus faible entre entrainement et test (11.6% contre 14.6% pour Random Forest), reste **le modele le plus honnete**. Moins performant, mais plus fiable. Le choix entre performance brute et fiabilite n'a pas de reponse universelle — et le reconnaitre fait partie de la rigueur.

---

## Ce que cette demarche montre

```
Observer  →  Transformer  →  Decouvrir  →  Predire  →  Interroger
   |              |              |             |              |
 8124 x 22     ACM →         Clustering    LDA →         Sensibilite
 variables     coordonnees   3 groupes     88.7%         k=4 optimal
 textuelles    numeriques    quasi-purs    accuracy      RF overfitte
```

Chaque etape repond a la question soulevee par la precedente. Et l'ensemble montre quelque chose de plus grand : que la toxicite d'un champignon est **encodee dans sa morphologie** de maniere suffisamment forte pour qu'un algorithme, partant de simples descriptions textuelles, puisse la retrouver — a condition d'utiliser la bonne transformation.

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
| [`docs/AUDIT_COMPLET.md`](docs/AUDIT_COMPLET.md) | **Audit multi-expert** — 8 perspectives (mathematicien, data engineer, classification, clustering, visualisation, code, product) |
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
