# Mission A - Actions immédiates (Personne A)

Ce guide décrit les actions concrètes pour **Personne A** (toi).

## 🎯 Première action (J0 - Aujourd'hui)

### 1. Vérifier l'installation Python

```bash
cd /Users/pierre/Desktop/Cours/TP-max-pierre/mushroom-project

# Créer environnement virtuel
python3 -m venv venv
source venv/bin/activate

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

**Action** : Ajoute 6-10 lignes d'interprétation dans `docs/rapport_plan.md`

---

## 📅 J2 (Après-demain)

### Mission A3 : ACM (CRITIQUE - fichier interface)

```bash
python src/03_mca.py
```

**Outputs** :
- **`data/processed/mca_coords.csv`** ← Fichier interface pour B (CRUCIAL)
- `data/processed/mca_eigenvalues.csv`
- `data/processed/mca_modalities_contrib_axis1.csv`
- `data/processed/mca_modalities_contrib_axis2.csv`
- `reports/figures/acm_scree.png`
- `reports/figures/acm_modalities_12.png`
- `reports/figures/acm_individuals_12_color_target.png`

**Décision à figer** : 
Le script recommandera une valeur **k** (nombre d'axes à conserver). Note cette valeur et communique-la à Personne B.

**Action** : Écris ½-1 page d'interprétation des axes 1-2 dans le rapport.

---

## 📦 Livraison à Personne B

Une fois J2 terminé, partage :

1. `data/processed/mushroom_processed.csv`
2. `data/processed/mca_coords.csv`
3. Valeur **k** recommandée (affichée par `03_mca.py`)
4. `reports/figures/acm_scree.png`
5. `reports/figures/acm_modalities_12.png`

**Communication** : "ACM terminée, k=X axes recommandés, coords prêtes dans `mca_coords.csv`"

---

## 🔍 Comment vérifier que tout est OK

```bash
# Après chaque mission, vérifie :
ls -lh data/processed/
ls -lh reports/figures/
ls -lh reports/tables/

# Les fichiers doivent exister avec tailles > 0
```

---

## ⚠️ Problèmes courants

**Erreur "prince not found"** :
```bash
pip install --upgrade prince
```

**Python 3.13 issues** :
Si `prince` ne s'installe pas, essaie Python 3.11 :
```bash
conda create -n mushroom python=3.11
conda activate mushroom
pip install -r requirements.txt
```

**Valeurs manquantes "?"** :
Le script `01_prepare.py` les remplace automatiquement par NA puis par la modalité la plus fréquente dans `03_mca.py`.

---

## 📝 Notes pour le rapport (à préparer en parallèle)

### Section 2 (Données) - Personne A
- Origine UCI, 8124 champignons, 23 variables qualitatives
- Gestion "?" dans `stalk-root`
- Distribution edible/poisonous (voir `desc_target_bar.png`)

### Section 3 (Descriptif) - Personne A
- Résumé `univariate_summary.csv`
- Variables clés : odor, gill-color, spore-print-color
- Modalités rares vs dominantes (justifie ACM)

### Section 4 (ACM) - Personne A
- Choix k axes (scree plot)
- Interprétation axes 1-2 (contributions modalités)
- Séparation edible/poisonous visible ?

---

**🎯 Objectif J2 soir** : Avoir livré `mca_coords.csv` à Personne B et commencé rédaction sections 2-4.
