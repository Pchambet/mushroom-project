# Mission A - Actions immédiates (Personne A)

Ce guide décrit les actions concrètes pour **Personne A**.

## 🎯 Première action (J0 - Aujourd'hui)

### 1. Vérifier l'installation Python

```bash
cd mushroom-project

# Créer environnement virtuel
python3 -m venv venv
source venv/bin/activate  # Sur Mac/Linux

# Installer dépendances
pip install -r requirements.txt
```

**⚠️ Important** : Vérifie que `prince` s'installe correctement (requis pour ACM)

### 2. Exécuter Mission A1 (Data acquisition)

```bash
# Télécharger dataset UCI
python src/00_download.py

# Préparer et nettoyer → produit mushroom_processed.csv
python src/01_prepare.py
```

**✅ Livrable A1** : `data/processed/mushroom_processed.csv`

### 3. Vérifier les sanity checks

Ouvre `notebooks/00_sanity_checks.ipynb` et vérifie :
- ≥150 individus ✓
- ≥10 variables qualitatives ✓
- Pas de NA inattendus
- Types object/categorical

---

## 📅 J1 (Demain)

### Mission A2 : Statistiques descriptives

```bash
python src/02_describe.py
```

**Outputs** :
- `reports/tables/univariate_summary.csv`
- `reports/tables/target_distribution.csv`
- `reports/figures/desc_target_bar.png`
- `reports/figures/desc_top_modalities_*.png` (3 variables clés)

---

## 📅 J2 (Après-demain)

### Mission A3 : ACM (CRITIQUE - fichier interface)

```bash
python src/03_mca.py
```

**Outputs** :
- **`data/processed/mca_coords.csv`** ← Fichier interface pour B (CRUCIAL)
- `reports/tables/mca_eigenvalues.csv`
- `reports/tables/mca_modalities_contrib_axis1.csv`
- `reports/tables/mca_modalities_contrib_axis2.csv`
- `reports/figures/acm_scree.png`
- `reports/figures/acm_modalities_12.png`
- `reports/figures/acm_individuals_12_color_target.png`

**Décision à figer** :
Le script recommandera une valeur **k** (nombre d'axes à conserver). Note cette valeur et communique-la à Personne B.

---

## 📦 Livraison à Personne B

Une fois J2 terminé, partage :

1. `data/processed/mushroom_processed.csv`
2. `data/processed/mca_coords.csv`
3. Valeur **k** recommandée (affichée par `03_mca.py`)
4. `reports/figures/acm_scree.png`
5. `reports/figures/acm_modalities_12.png`

---

## ⚠️ Problèmes courants

**Erreur "prince not found"** :
```bash
pip install --upgrade prince
```

**Python 3.13 issues** :
Si `prince` ne s'installe pas, essaie Python 3.11.

**Valeurs manquantes "?"** :
Le script `01_prepare.py` les remplace automatiquement par NA puis par la modalité la plus fréquente dans `03_mca.py`.
