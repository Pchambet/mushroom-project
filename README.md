# Mushroom Project - Analyse ACM et Clustering

Projet d'analyse de données sur le dataset UCI Mushroom (classification champignons comestibles/vénéneux).

## 📋 Répartition des tâches (Binôme A + B)

### 🔵 **Personne A** : Socle Data + ACM
- Mission A1 : Data acquisition + preprocessing  
- Mission A2 : Statistiques descriptives  
- Mission A3 : ACM + export coords (interface pour B)

### 🟢 **Personne B** : Clustering + Discriminante
- Mission B1 : Clustering sur composantes ACM  
- Mission B2 : Analyse discriminante  
- Mission B3 : Packaging résultats rapport

## 🚀 Installation

```bash
# Cloner le repo
cd mushroom-project

# Créer environnement virtuel (recommandé)
python3 -m venv venv
source venv/bin/activate  # Sur Mac/Linux
# ou
venv\Scripts\activate  # Sur Windows

# Installer les dépendances
pip install -r requirements.txt
```

## 📦 Exécution du pipeline

### **Personne A** doit exécuter dans l'ordre :

```bash
# Mission A1 : Téléchargement et préparation
python src/00_download.py
python src/01_prepare.py

# Mission A2 : Statistiques descriptives
python src/02_describe.py

# Mission A3 : ACM (produit mca_coords.csv pour B)
python src/03_mca.py
```

**✅ Livraison A → B** : `data/processed/mushroom_processed.csv` + `data/processed/mca_coords.csv`

---

### **Personne B** exécute ensuite (après avoir reçu les fichiers de A) :

```bash
# Mission B1 : Clustering
python src/04_cluster.py

# Mission B2 : Analyse discriminante
python src/05_discriminant.py
```

---

## 📁 Structure des outputs

```
reports/
├── figures/
│   ├── desc_*.png              # Personne A (stats desc)
│   ├── acm_*.png               # Personne A (ACM)
│   ├── cluster_*.png           # Personne B (clustering)
│   └── da_*.png                # Personne B (discriminante)
└── tables/
    ├── univariate_summary.csv  # A
    ├── target_distribution.csv # A
    ├── mca_eigenvalues.csv     # A
    ├── cluster_profiles.csv    # B
    └── da_metrics.csv          # B
```

## 🔄 Workflow Git (recommandé)

```bash
# Personne A
git checkout -b dev-A
# ... travail A1, A2, A3 ...
git add .
git commit -m "Mission A3 terminée - livraison coords ACM"
git push origin dev-A

# Personne B (après merge de dev-A)
git checkout -b dev-B
# ... travail B1, B2, B3 ...
git add .
git commit -m "Mission B2 terminée - discriminante finalisée"
git push origin dev-B
```

## ✅ Checklist "Fini" (avant rendu)

- [ ] Dataset + descriptif fournis (A1)
- [ ] ACM interprétée avec axes 1-2 + contributions (A3)
- [ ] Clustering sur coords ACM + profils (B1)
- [ ] Discriminante sur coords ACM + confusion (B2)
- [ ] Rapport 10-15 pages, pas de listing brut
- [ ] Validation croisée (5-fold) documentée (B2)
- [ ] Cohérence narrative : ACM → clustering → discriminante

## 📊 Specifications du dataset

- **Source** : UCI Machine Learning Repository - Mushroom Dataset
- **Individus** : 8124 champignons
- **Variables** : 23 variables qualitatives (cap-shape, odor, gill-color, etc.)
- **Cible** : class (e=edible, p=poisonous)
- **Manquants** : Variable `stalk-root` contient "?"

## 📖 Documentation

- `docs/data_dictionary.md` : Descriptif complet du dataset
- `docs/rapport_plan.md` : Plan du rapport final

## 🛠️ Dépannage

**Erreur `prince` not found** :
```bash
pip install prince
```

**Erreur Python version** :
Le projet nécessite Python ≥ 3.8

**Fichier `mca_coords.csv` manquant** :
Personne B doit attendre que Personne A exécute `03_mca.py`

---

**Contact** : Pour questions sur missions A → Personne A | Pour missions B → Personne B
