# Interface Specification - Handshake A ↔ B

Ce document définit le **contrat d'interface** entre Personne A et Personne B.

## 📦 Fichiers livrés par A → B

### 1. `data/processed/mushroom_processed.csv`

**Description** : Dataset nettoyé et prêt pour analyse

**Format** :
```csv
class,cap-shape,cap-surface,...,habitat
e,x,s,...,d
p,f,y,...,w
...
```

**Caractéristiques** :
- 8124 lignes (individus)
- 23 colonnes (1 target + 22 variables qualitatives)
- Pas de valeurs "?" (remplacées par modalité fréquente ou NA gérés)
- Types : object ou categorical

---

### 2. `data/processed/mca_coords.csv`

**Description** : Coordonnées des individus sur les axes ACM (fichier CRITIQUE)

**Format** :
```csv
Dim1,Dim2,Dim3,...,Dimk
-0.234,0.567,-0.123,...,0.089
0.456,-0.234,0.345,...,-0.123
...
```

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

## 🔍 Checks de validation

### A vérifie avant livraison :

```python
import pandas as pd

# 1. Vérifier mushroom_processed.csv
df = pd.read_csv('data/processed/mushroom_processed.csv')
assert df.shape[0] == 8124, "Nombre lignes incorrect"
assert df.shape[1] == 23, "Nombre colonnes incorrect"
assert df['class'].isin(['e', 'p']).all(), "Classe invalide"

# 2. Vérifier mca_coords.csv
coords = pd.read_csv('data/processed/mca_coords.csv')
assert coords.shape[0] == 8124, "Nombre lignes incorrect"
assert coords.shape[1] >= 5, "Pas assez d'axes"
assert coords.dtypes.apply(lambda x: x.kind == 'f').all(), "Types non float"

print("✅ Interface valide!")
```

### B vérifie après réception :

```python
import pandas as pd

# Charger les deux fichiers
df = pd.read_csv('data/processed/mushroom_processed.csv')
coords = pd.read_csv('data/processed/mca_coords.csv')

# Vérifier correspondance ligne à ligne
assert len(df) == len(coords), "Longueurs différentes!"

# Tester clustering sur k=5 axes
X = coords.iloc[:, :5].values
print(f"Shape coords pour clustering : {X.shape}")

print("✅ Interface OK, prêt pour Mission B!")
```

---

## 🗂️ Conventions de nommage

| Responsable | Préfixe | Exemples |
|-------------|---------|----------|
| A - Descriptif | `desc_` | `desc_target_bar.png` |
| A - ACM | `acm_` | `acm_scree.png`, `acm_modalities_12.png` |
| B - Clustering | `cluster_` | `cluster_dendrogram.png` |
| B - Discriminante | `da_` | `da_confusion.png` |

**Important** : Respecter strictement ces préfixes pour éviter les collisions.

---

## 📅 Timeline de livraison

```
J0 (A) : mushroom_processed.csv prêt
J2 (A) : mca_coords.csv prêt → LIVRAISON À B
J3 (B) : peut commencer clustering
J4 (B) : peut commencer discriminante
```

---

## ⚠️ Erreurs courantes à éviter

### Erreur 1 : Index décalés

```python
# ❌ FAUX
coords = mca.transform(X).reset_index(drop=False)  # Ajoute une colonne index

# ✅ CORRECT
coords = mca.transform(X)  # Pas d'index explicite
coords.to_csv(..., index=False)
```

### Erreur 2 : k non communiqué

```python
# ❌ B devine k arbitrairement
X = coords.iloc[:, :3].values  # Pourquoi 3?

# ✅ B utilise k communiqué par A
k = 5  # Valeur reçue de A
X = coords.iloc[:, :k].values
```

### Erreur 3 : Ordre individus changé

```python
# ❌ A trie le dataset avant ACM
df_sorted = df.sort_values('class')
mca.fit(df_sorted.drop('class', axis=1))

# ✅ A garde l'ordre original
mca.fit(df.drop('class', axis=1))
```

---

## 📞 Template message livraison

```
De : Personne A
À : Personne B
Objet : ACM terminée - Livraison fichiers interface

Salut !

✅ Mission A3 terminée.

Fichiers disponibles (repo à jour) :
- data/processed/mushroom_processed.csv (8124 × 23)
- data/processed/mca_coords.csv (8124 × 5)

Paramètres ACM :
- k = 5 axes conservés
- Inertie cumulée = 92.4%
- Top contrib Dim1 : odor, spore-print-color
- Top contrib Dim2 : gill-color, cap-shape

Figures principales :
- reports/figures/acm_scree.png
- reports/figures/acm_modalities_12.png

Tu peux lancer tes missions B1 et B2 !

Bon courage,
A
```

---

**Version** : 1.0 (J0)  
**Dernière mise à jour** : Création projet
