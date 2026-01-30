# Plan du rapport - Analyse Mushroom Dataset

## 1. Introduction
- Contexte et motivation
- Objectifs de l'analyse
- Présentation du dataset UCI Mushroom

## 2. Données

### 2.1 Description du dataset
- Source : UCI Machine Learning Repository
- Dimensions : 8124 observations × 23 variables qualitatives
- Variable cible : `class` (edible vs. poisonous)
- Distribution équilibrée : 51,8% comestible, 48,2% vénéneux

### 2.2 Statistiques descriptives univariées

**Variables clés identifiées** :
- `odor` : 43,4% sans odeur (none), 26,6% odeur forte (foul)
- `gill-color` : 12 modalités, distribution fragmentée (21,3% buff)
- `spore-print-color` : dominée par blanc (29,4%), marron (24%), noir (19%)

### 2.3 Gestion des valeurs manquantes

**Variable `stalk-root`** :
- 2480 valeurs "?" (30,53% du dataset)
- **Stratégie retenue** : Imputation modale (modalité "b" = bulbous)
- **Justification** : Préserve la distribution majoritaire, évite la perte de 30% des données, compatible avec ACM

**Alternative rejetée** : Suppression des lignes (perte excessive) ou création d'une modalité "missing" (augmente artificiellement K)

### 2.4 Analyse bivariée : odeur × comestibilité

Tableau croisé révélant une **association quasi-parfaite** :

| Odeur | Comestible | Vénéneux | Total |
|-------|------------|----------|-------|
| none (n) | 3408 | 120 | 3528 |
| **foul (f)** | **0** | **2160** | **2160** |
| **almond (a)** | **400** | **0** | **400** |
| **anise (l)** | **400** | **0** | **400** |
| pungent (p) | 0 | 256 | 256 |
| *Autres* | 0 | 1380 | 1380 |

**Conclusion** :
- Les odeurs agréables (almond, anise) sont **100% associées aux champignons comestibles**
- L'odeur fétide (foul) et piquante (pungent) sont **100% associées aux champignons vénéneux**
- `odor` est donc une **variable hautement discriminante** (sera reflétée dans l'ACM, axe 1)

## 3. Analyse en Composantes Multiples (ACM)

### 3.1 Méthodologie ACM appliquée

- **Tableau Disjonctif Complet (TDC)** : 22 variables → K = 111 modalités
- **Inertie totale** : I_tot ≈ 4.27
- **Diagonalisation** : extraction de 10 axes factoriels
- **Choix de k** : conservation de **k = 5 axes** (31,27% d'inertie cumulée)

**Justification du choix k=5** :
- Scree plot : coude visible après axe 5
- Compromis interprétabilité / conservation d'information
- Évite la sur-complexité (au-delà, gain marginal faible)

### 3.2 Interprétation Axe 1 (7,59%) - "Caractéristiques de surface et odeur"

**Contributions principales** (Top 5) :

| Modalité | Coordonnée | Contribution (%) |
|----------|------------|------------------|
| ring-type__l (large ring) | +1,73 | 6,68 |
| stalk-surface-below-ring__k (silky) | +1,27 | 6,41 |
| stalk-surface-above-ring__k (silky) | +1,22 | 6,08 |
| odor__f (foul) | +1,21 | 5,49 |
| spore-print-color__h (chocolate) | +1,33 | 5,01 |
| ring-type__p (pendant) | -0,67 | 3,05 |
| bruises__t (bruises present) | -0,65 | 2,49 |
| odor__n (no odor) | -0,62 | 2,36 |

**Interprétation pôle positif** (coord. > 0) :
- Champignons avec **anneau large** (ring-type__l)
- Surface du pied **soyeuse/lisse** (stalk-surface-*__k)
- **Odeur forte et désagréable** (odor__f = foul)
- Empreinte de spores **chocolat** (spore-print-color__h)

**Profil typique** : Champignons à texture lisse, odeur marquée, anneau prononcé

**Interprétation pôle négatif** (coord. < 0) :
- Champignons avec **anneau pendant** (ring-type__p)
- Présence de **bleus** (bruises__t)
- **Absence d'odeur** (odor__n)

**Profil typique** : Champignons sans caractéristiques olfactives marquantes

**Lien avec la comestibilité** :
Les modalités du pôle positif (`odor__f`, `spore-print-color__h`) sont **majoritairement vénéneuses** :
- odor__f → 100% poisonous (cf. tableau croisé section 2.4)
- L'axe 1 oppose donc les champignons **"à risque"** (pôle +) aux champignons **"neutres"** (pôle -)

**Conclusion** : Cet axe capture les **caractéristiques de surface et d'odeur**, avec un pouvoir discriminant fort pour la classe edible/poisonous.

### 3.3 Interprétation Axe 2 (6,91%) - "Modalités rares et effet de taille"

**Contributions principales** (Top 5) :

| Modalité | Coordonnée | Contribution (%) | Effectif réel |
|----------|------------|------------------|---------------|
| gill-attachment__a (attached) | +4,67 | 8,70 | **3%** |
| stalk-color-*__o (orange) | +4,43 | 7,16 | **<1%** |
| habitat__l (leaves) | +1,69 | 4,52 | 10% |
| population__c (clustered) | +2,63 | 4,44 | 8% |
| gill-color__y (yellow) | +5,18 | 4,37 | **<1%** |

**Observation clé** : Toutes les modalités dominantes de l'axe 2 sont **rares** (effectifs < 3%)

**Interprétation** :
- L'axe 2 reflète principalement des **effets de taille** (modalités rares vs. fréquentes)
- Les modalités rares sont éloignées du barycentre par construction de l'ACM
- Cet axe oppose les champignons à **caractéristiques atypiques** (lamelles attachées, couleurs orange/jaune, habitat spécifique) aux champignons "moyens"

**Limite** : Moins d'opposition sémantique forte, plus un artefact statistique

**Utilité** : Capture une partie de la variabilité intra-espèces, utile pour identifier des sous-groupes spécifiques dans le clustering

### 3.4 Axes 3 à 5 : compléments d'information

Brève caractérisation (non détaillée exhaustivement) :

- **Axe 3 (6,33%)** : Variations de couleur du chapeau et de la tige
- **Axe 4 (5,68%)** : Opposition entre habitats (bois vs. prairies)
- **Axe 5 (4,76%)** : Forme du pied (élargissement vs. effilé)

**Total inertie conservée** : 31,27% sur k=5 axes

Ces axes seront utilisés dans le clustering (Section 4) et l'analyse discriminante (Section 5) mais ne sont pas interprétés en détail dans cette section.

### 3.5 Export et livrables

**Fichiers générés** :
- `mca_coords.csv` : 8124 lignes × 10 colonnes (coordonnées factorielles)
- `mca_eigenvalues.csv` : valeurs propres + inerties pour 10 axes
- `mca_modalities_contrib_axis1.csv` et `axis2.csv` : contributions complètes
- `acm_scree.png`, `acm_modalities_12.png`, `acm_individuals_12_color_target.png` : visualisations

**Recommandation pour Personne B** : Utiliser **k=5 premiers axes** pour clustering et analyse discriminante (compromis optimal).

## 4. Clustering
- Méthodes utilisées (CAH, K-means)
- Choix du nombre de clusters
- Profiling des clusters
- Interprétation des groupes

*[Cette section sera rédigée par Personne B]*

## 5. Analyse discriminante
- Modèle discriminant sur axes ACM
- Matrice de confusion
- Validation croisée
- Performance du modèle

*[Cette section sera rédigée par Personne B]*

## 6. Résultats et Discussion
- Synthèse des résultats
- Limites de l'analyse
- Perspectives

## 7. Conclusion

## Références
