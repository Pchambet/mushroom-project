# Questions fréquentes - Collaboration A/B

## 🔵 Questions Personne A

### Q1 : Combien d'axes ACM dois-je garder (valeur k) ?

Le script `03_mca.py` recommande automatiquement k pour atteindre ≥90% inertie cumulée.

**Règle pratique** : Entre 5 et 10 axes suffisent généralement. Regarde le "coude" sur le scree plot.

**Communication à B** : Note la valeur k recommandée et confirme-la avec Personne B (ils utiliseront le même k).

---

### Q2 : Que faire des valeurs manquantes "?" ?

- `01_prepare.py` : les convertit en `NaN`
- `03_mca.py` : les remplace par la **modalité la plus fréquente** de chaque variable

**Alternative** (si temps) : Créer une modalité "missing" explicite. À discuter en binôme.

---

### Q3 : Quelles variables montrer dans `desc_top_modalities_*.png` ?

Le script sélectionne automatiquement :
- `odor`
- `gill-color`
- `spore-print-color`

**Justification** : Variables avec beaucoup de modalités ET discriminantes pour la classe.

Tu peux changer dans `02_describe.py` ligne 72 si tu préfères d'autres variables.

---

## 🟢 Questions Personne B

### Q1 : Combien de clusters choisir ?

**Recommandation** : 3-5 clusters selon le dendrogramme (`cluster_dendrogram.png`).

**Méthode** :
1. Regarde la "coupe" suggérée (ligne rouge sur dendrogramme)
2. Teste K-means avec k=3, 4, 5
3. Choisis celle avec profils les plus distincts

---

### Q2 : Comment interpréter les clusters ?

Utilise `cluster_profiles.csv` :
- Filtre `over_representation > 1.5` → modalités **sur-représentées**
- Filtre `over_representation < 0.5` → modalités **sous-représentées**

**Exemple profil** :
```
Cluster 0 : odor=foul (over_rep=3.2), gill-color=buff (over_rep=2.1)
→ "Cluster des champignons à odeur forte, lamelles beige"
```

---

### Q3 : Pourquoi l'analyse discriminante sur coords ACM et pas variables brutes ?

**Réponse attendue par prof** :
1. ACM réduit dimensionnalité (23 vars → k axes)
2. Variables qualitatives → coords numériques continues
3. LDA fonctionne mieux sur espace réduit (moins de colinéarité)
4. Permet d'interpréter quels **axes factoriels** discriminent le mieux

**À écrire dans rapport section 6**.

---

## 🔄 Questions d'intégration A ↔ B

### Q1 : Quand B peut-il commencer ?

**Minimum** : Après livraison `mca_coords.csv` (fin J2 de A)

**Optimal** : A peut stubber un `mca_coords.csv` fictif dès J0 pour que B teste ses scripts en parallèle.

---

### Q2 : Comment merger nos branches Git ?

**Workflow recommandé** :
```bash
# A merge d'abord dans main
git checkout main
git merge dev-A
git push origin main

# B récupère et merge
git checkout dev-B
git pull origin main
# ... continue travail B ...
git checkout main
git merge dev-B
git push origin main
```

---

### Q3 : Qui rédige quelle partie du rapport ?

| Section | Responsable | Contenu |
|---------|-------------|---------|
| 1. Introduction | **A** | Contexte, objectifs |
| 2. Données | **A** | Dataset, preprocessing |
| 3. Descriptif | **A** | Stats desc, justification ACM |
| 4. ACM | **A** | Métho, scree, interprétation axes |
| 5. Clustering | **B** | CAH, K-means, profils |
| 6. Discriminante | **B** | LDA, CV, performances |
| 7. Discussion | **A** | Limites, perspectives |
| 8. Conclusion | **A** | Synthèse finale |

---

## ⚠️ Pièges à éviter

### Pour A :
- ❌ Ne pas livrer `mca_coords.csv` trop tard (bloque B)
- ❌ Oublier de documenter la valeur k choisie
- ❌ Laisser des NaN dans les coords ACM

### Pour B :
- ❌ Changer k sans en parler à A (incohérence report)
- ❌ Oublier la validation croisée (perd des points bonus)
- ❌ Profiler les clusters uniquement sur la classe (pas assez riche)

### Pour les deux :
- ❌ Rapport > 15 pages (pénalité "bavardage")
- ❌ Coller du code brut (utiliser figures/tables exportées)
- ❌ Ne pas relier ACM → clustering → discriminante (cohérence narrative)
