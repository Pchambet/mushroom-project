# Dictionnaire de données - UCI Mushroom Dataset

## Source
- **Dataset** : UCI Machine Learning Repository - Mushroom Dataset
- **URL** : https://archive.ics.uci.edu/ml/datasets/mushroom

## Description générale
Ce dataset contient des descriptions de champignons hypothétiques correspondant à 23 espèces de champignons à lamelles de la famille des Agaricus et Lepiota.

## Variables

### Variable cible
- **class** : Comestible (e) ou Vénéneux (p)

### Variables descriptives

| Variable | Description | Valeurs possibles |
|----------|-------------|-------------------|
| cap-shape | Forme du chapeau | bell=b, conical=c, convex=x, flat=f, knobbed=k, sunken=s |
| cap-surface | Surface du chapeau | fibrous=f, grooves=g, scaly=y, smooth=s |
| cap-color | Couleur du chapeau | brown=n, buff=b, cinnamon=c, gray=g, green=r, pink=p, purple=u, red=e, white=w, yellow=y |
| bruises | Présence de bleus | bruises=t, no=f |
| odor | Odeur | almond=a, anise=l, creosote=c, fishy=y, foul=f, musty=m, none=n, pungent=p, spicy=s |
| gill-attachment | Attachement des lamelles | attached=a, descending=d, free=f, notched=n |
| gill-spacing | Espacement des lamelles | close=c, crowded=w, distant=d |
| gill-size | Taille des lamelles | broad=b, narrow=n |
| gill-color | Couleur des lamelles | black=k, brown=n, buff=b, chocolate=h, gray=g, green=r, orange=o, pink=p, purple=u, red=e, white=w, yellow=y |
| stalk-shape | Forme du pied | enlarging=e, tapering=t |
| stalk-root | Racine du pied | bulbous=b, club=c, cup=u, equal=e, rhizomorphs=z, rooted=r, missing=? |
| stalk-surface-above-ring | Surface pied au-dessus anneau | fibrous=f, scaly=y, silky=k, smooth=s |
| stalk-surface-below-ring | Surface pied en-dessous anneau | fibrous=f, scaly=y, silky=k, smooth=s |
| stalk-color-above-ring | Couleur pied au-dessus anneau | brown=n, buff=b, cinnamon=c, gray=g, orange=o, pink=p, red=e, white=w, yellow=y |
| stalk-color-below-ring | Couleur pied en-dessous anneau | brown=n, buff=b, cinnamon=c, gray=g, orange=o, pink=p, red=e, white=w, yellow=y |
| veil-type | Type de voile | partial=p, universal=u |
| veil-color | Couleur du voile | brown=n, orange=o, white=w, yellow=y |
| ring-number | Nombre d'anneaux | none=n, one=o, two=t |
| ring-type | Type d'anneau | cobwebby=c, evanescent=e, flaring=f, large=l, none=n, pendant=p, sheathing=s, zone=z |
| spore-print-color | Couleur empreinte spores | black=k, brown=n, buff=b, chocolate=h, green=r, orange=o, purple=u, white=w, yellow=y |
| population | Population | abundant=a, clustered=c, numerous=n, scattered=s, several=v, solitary=y |
| habitat | Habitat | grasses=g, leaves=l, meadows=m, paths=p, urban=u, waste=w, woods=d |

## Valeurs manquantes
- La variable `stalk-root` contient des valeurs manquantes encodées par "?"
- **Effectif** : 2480 valeurs manquantes (30,53% du dataset)

## Preprocessing appliqué

### Stratégie de gestion des NA

**Variable concernée** : `stalk-root`

**Approche retenue** : **Imputation modale**
- Remplacement des valeurs "?" par la modalité la plus fréquente : **`b` (bulbous)**
- Effectué dans le script `01_prepare.py`

**Justification** :
1. **Préservation de la distribution** : Maintient la tendance majoritaire de la variable
2. **Évite la perte de données** : Conserver 30% du dataset (2480 observations) plutôt que les supprimer
3. **Compatible avec ACM** : L'ACM gère mal les NA, cette approche évite les problèmes de calcul
4. **Alternative rejetée** : Création d'une modalité "missing" augmenterait artificiellement K (111 → 112 modalités)

**Impact** : Distribution finale de `stalk-root` biaisée vers la modalité "b", mais préférable à la perte d'information.

## Statistiques descriptives clés

### Distribution des principales variables

| Variable | N modalités | Top modalité | Fréquence | 2e modalité | Fréquence |
|----------|-------------|--------------|-----------|-------------|-----------|
| **class** | 2 | e (edible) | 51,8% | p (poisonous) | 48,2% |
| **odor** | 9 | n (none) | 43,4% | f (foul) | 26,6% |
| **gill-color** | 12 | b (buff) | 21,3% | w (white) | 16,0% |
| **spore-print-color** | 9 | w (white) | 29,4% | n (brown) | 24,0% |
| **cap-color** | 10 | n (brown) | 28,1% | g (gray) | 18,3% |
| **gill-attachment** | 2 | f (free) | 97,4% | a (attached) | 2,6% |

**Observations** :
- Distribution de la classe **quasi-équilibrée** (évite les problèmes de classes déséquilibrées)
- Variable `gill-attachment` très déséquilibrée (97,4% vs. 2,6%)
- Variable `odor` : distribution fragmentée mais modalités fortement associées à la classe

## Variables discriminantes identifiées

### Analyse exploratoire : lien avec la comestibilité

**Variables hautement discriminantes** :
1. **`odor`** : Association quasi-parfaite avec la classe
   - Odeurs agréables (almond, anise) → 100% comestibles
   - Odeurs désagréables (foul, pungent) → 100% vénéneuses
   - Voir tableau croisé dans `rapport_plan.md` section 2.4

2. **`spore-print-color`** : Modalités spécifiques liées au poison
   - `h` (chocolate) et `r` (green) fortement associées aux champignons vénéneux
   - `w` (white) majoritairement comestible

3. **`gill-color`** : Tendances notables
   - `b` (buff) majoritairement comestible
   - Autres couleurs rares associées au poison

**Conséquence pour l'ACM** : Ces variables contribueront fortement à l'axe 1 (vérifiée dans `mca_modalities_contrib_axis1.csv`).

## Notes
- Toutes les variables sont catégorielles
- Le dataset ne contient pas de valeurs numériques
- Après nettoyage : aucune valeur manquante résiduelle

