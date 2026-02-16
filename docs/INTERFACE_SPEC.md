# Interface Specification - Handshake A ↔ B

Ce document définit le **contrat d'interface** entre Personne A et Personne B.

## 📦 Fichiers livrés par A → B

### 1. `data/processed/mushroom_processed.csv`

**Description** : Dataset nettoyé et prêt pour analyse

**Caractéristiques** :
- 8124 lignes (individus)
- 23 colonnes (1 target + 22 variables qualitatives)
- Pas de valeurs "?" (remplacées par modalité fréquente ou NA gérés)

---

### 2. `data/processed/mca_coords.csv`

**Description** : Coordonnées des individus sur les axes ACM (fichier CRITIQUE)

**Caractéristiques** :
- 8124 lignes (même ordre que `mushroom_processed.csv`)
- k colonnes (k = nombre d'axes conservés, typiquement 5-10)
- Types : float64
- **IMPORTANT** : L'index doit correspondre ligne à ligne avec `mushroom_processed.csv`

---

## 📊 Métadonnées à communiquer

### Valeur k (nombre d'axes)

**Personne A doit communiquer** :
- Valeur k choisie (ex: k=5)
- Inertie cumulée à k axes (ex: 92.4%)
- Justification (scree plot, coude)

**Personne B doit utiliser** :
- Les k premières colonnes de `mca_coords.csv`
- Même valeur k pour cohérence rapport

---

## 🗂️ Conventions de nommage

| Responsable | Préfixe | Exemples |
|-------------|---------|----------|
| A - Descriptif | `desc_` | `desc_target_bar.png` |
| A - ACM | `acm_` | `acm_scree.png`, `acm_modalities_12.png` |
| B - Clustering | `cluster_` | `cluster_dendrogram.png` |
| B - Discriminante | `da_` | `da_confusion.png` |

**Important** : Respecter strictement ces préfixes pour éviter les collisions.
