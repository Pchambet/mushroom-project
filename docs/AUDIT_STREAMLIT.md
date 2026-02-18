# Audit Streamlit — The Mushroom Project

> Audit technique et fonctionnel du dashboard interactif, basé sur l'analyse du code source. Sans concession, avec pistes d'amélioration concrètes.

---

## 1. Vue d'ensemble

| Critère | Note | Commentaire |
|---|---|---|
| **Structure** | 8/10 | Cinq pages claires, sidebar lisible |
| **Performance** | 7/10 | Cache correct, mais recalculs à chaque interaction |
| **UX / Accessibilité** | 6/10 | Fonctionnel mais frustrant sur certains points |
| **Cohérence** | 5/10 | Texte français incohérent avec le reste du projet |
| **Robustesse** | 6/10 | Pas de gestion d'erreur si fichiers manquants |

**Verdict global** : Dashboard solide et utile, mais perfectible. Il répond à son objectif (explorer les résultats) sans être au niveau de qualité du README.

---

## 2. Audit page par page

### Page 1 : Vue d'ensemble

**Points forts :**
- Métriques en colonnes (specimens, variables, comestibles, vénéneux) — lecture immédiate
- Bar chart Plotly pour la distribution cible — interactif, hover OK
- Aperçu du dataset (20 premières lignes) — utile pour comprendre les données

**Problèmes identifiés :**

1. **Texte sans accents** (ligne 83-84) :
   ```python
   "> Pipeline d'analyse statistique sur donnees categorielles — "
   "De l'ACM au clustering et a la classification."
   ```
   → Incohérent avec le README. À corriger : « données », « à ».

2. **Bar chart** : Les libellés x sont codés en dur `["Comestible (e)", "Venéneux (p)"]`. Si la colonne `class` change de format, le graphique casse.

3. **Résumé univarié** : Affiché en bas si disponible. Pas d'explication de ce que c'est — un visiteur non initié ne comprend pas « modalités », « top_freq_pct », etc.

4. **Pas de lien vers le README ou le repo** : Un visiteur qui atterrit directement sur le dashboard (via le lien Streamlit) n'a aucun contexte. Une phrase d'introduction + lien GitHub manquent.

---

### Page 2 : Espace ACM

**Points forts :**
- Selectbox pour choisir les axes X et Y — excellente idée, permet d'explorer Dim3 vs Dim7 par exemple
- Colorer par variable : choix entre classe ou n'importe quelle variable morphologique — très pédagogique
- Graphique d'inertie expliquée (bar + cumul) — bien pensé

**Problèmes identifiés :**

1. **Axes fixes sur Dim1/Dim2 pour le scatter** : Les selectbox `dim_x` et `dim_y` sont bien utilisés dans `px.scatter(plot_df, x=dim_x, y=dim_y, ...)` — c'est correct. Pas d'erreur ici.

2. **Texte sans accents** (ligne 121) : « categoriques » → « catégorielles ».

3. **« Colorer par »** : L'option « Classe (edible/poisonous) » est en anglais. Pour cohérence : « Classe (comestible/vénéneux) ».

4. **Gestion des NaN** : Si une variable choisie pour la coloration a des valeurs manquantes, `df[var].astype(str)` donne `"nan"` — ça peut créer une catégorie parasite dans la légende. Un `.fillna("Manquant")` serait plus propre.

5. **Taille des points** : `marker=dict(size=4)` — avec 8 124 points, ça peut être illisible. Un slider pour la taille ou une opacité ajustable aiderait.

---

### Page 3 : Clustering

**Points forts :**
- Sliders pour k (axes) et n (clusters) — l'utilisateur peut explorer en temps réel
- Métriques : Clusters, Silhouette, Axes — feedback immédiat
- Vue côte à côte : clusters vs classes réelles — excellente idée pour comparer
- Tableau de pureté par cluster — très informatif

**Problèmes identifiés :**

1. **Recalcul à chaque interaction** : Dès qu'on bouge un slider, K-Means se relance sur 8 124 points. Pas catastrophique (quelques centaines de ms), mais ça peut laguer sur un hébergement faible. Un `@st.cache_data` sur une grille de paramètres (k, n) pourrait aider — ou accepter le trade-off pour garder la simplicité.

2. **Graphiques clusters** : Toujours en Dim1/Dim2. Si l'utilisateur choisit k_axes=6, les clusters sont calculés en 6D mais affichés en 2D. C'est logique (on ne peut pas afficher 6D), mais une note explicative (« Projection sur les deux premiers axes ») éviterait la confusion.

3. **Texte** (ligne 229) : « Classe reelle » → « Classe réelle ». « Purete » → « Pureté ».

4. **Couleurs des clusters** : `color="cluster"` utilise la palette Plotly par défaut. Pas de mapping explicite vers des couleurs qui rappellent comestible/vénéneux. Ce n'est pas une erreur, mais on pourrait documenter que les numéros de cluster ne correspondent pas à une interprétation fixe.

---

### Page 4 : Classification

**Points forts :**
- Slider k_axes — cohérent avec les autres pages
- Matrice de confusion + scores par fold — complet
- Rapport de classification (precision, recall, f1) — utile pour les initiés
- Comparaison de modèles (si table disponible) — bonne idée de réutiliser les outputs du pipeline

**Problèmes identifiés :**

1. **Recalcul LDA + CV à chaque changement de k** : `cross_val_score` et `cross_val_predict` sur 8 124 lignes, 5 folds. Ça peut prendre 1 à 2 secondes. Sur Streamlit Cloud, c'est acceptable, mais l'utilisateur peut croire que l'app a planté. Un `st.spinner("Calcul en cours…")` autour du bloc de calcul améliorerait la perception.

2. **Labels** (ligne 289-290) : « Predit » → « Prédit », « Reel » → « Réel ».

3. **Comparaison de modèles** : La table `model_comparison` a des colonnes en anglais (`cv_accuracy_mean`, `f1_macro`). Un renommage pour l'affichage (« Accuracy CV », « F1 macro ») ou une légende rendrait la lecture plus accessible.

4. **Pas de choix de modèle** : La page ne montre que la LDA. La comparaison est en lecture seule. On pourrait imaginer un selectbox « Modèle : LDA / RF / SVM / LogReg » et recalculer les métriques à la volée — mais ça complexifierait beaucoup. Pour l'instant, rester sur LDA + tableau comparatif est un bon compromis.

---

### Page 5 : Sensibilité

**Points forts :**
- Deux graphiques côte à côte (Accuracy vs k, Silhouette vs k)
- Tableau avec highlight des meilleurs k
- Message `st.info` pour le meilleur k — bonne UX

**Problèmes identifiés :**

1. **Dépendance aux outputs** : Si `sensitivity_k.csv` n'existe pas, la page affiche un warning. C'est correct, mais le message (ligne 399-401) est en mode « développeur » : `Executez d'abord : python src/06_sensitivity.py`. Un visiteur qui utilise uniquement le dashboard ne peut pas exécuter de script. Reformuler : « Les résultats de sensibilité ne sont pas disponibles. Exécutez le pipeline complet (voir le README) pour les générer. »

2. **Texte** (ligne 338-340) : « sensibilite » → « sensibilité », « qualite » → « qualité ».

3. **Colonnes du tableau** : `lda_train_accuracy`, `lda_cv_mean`, etc. — noms techniques. Des en-têtes en français (« Accuracy train », « Accuracy CV », « Silhouette », « Écart overfitting ») rendraient le tableau plus lisible.

4. **Pas de graphique « inertie vs accuracy »** : Le script 06 génère aussi `sensitivity_inertia_vs_accuracy.png`. Ce graphique à double axe Y (inertie cumulée + accuracy CV) n'est pas dans le dashboard. Ce serait un ajout pertinent.

---

## 3. Problèmes transversaux

### Imports

Les imports `sklearn` sont faits à l'intérieur des blocs `elif` (lignes 184, 192, 264-266). Ça fonctionne, mais :
- Moins lisible
- Violation des conventions PEP 8 (imports en tête de fichier)

**Recommandation** : Déplacer tous les imports en haut du fichier.

### Gestion d'erreurs

Aucun `try/except` autour de `load_data()` ou `load_tables()`. Si `mushroom_processed.csv` ou `mca_coords.csv` n'existe pas, l'app plante dès le chargement avec une `FileNotFoundError` brutale.

**Recommandation** : Encapsuler le chargement dans un try/except et afficher un `st.error()` explicite avec un lien vers le README pour lancer le pipeline.

### Cache

`@st.cache_data` sur `load_data()` et `load_tables()` — correct. Les données ne sont pas rechargées à chaque rerun.

En revanche, les calculs (K-Means, LDA, CV) ne sont pas cachés. Pour un dashboard exploratoire, c'est acceptable : les paramètres changent à chaque interaction, donc le cache ne servirait que pour des combinaisons (k, n) déjà visitées. La complexité du cache (clé = (k_axes, n_clusters)) peut ne pas valoir le coup. À évaluer au cas par cas.

### Thème

Le `config.toml` définit une couleur primaire `#E91E63` (rose/magenta). Le dashboard n'utilise pas cette couleur dans les graphiques — Plotly a ses propres palettes. La cohérence visuelle entre le thème Streamlit et les graphiques est partielle.

---

## 4. Synthèse des corrections prioritaires

| Priorité | Action | Effort |
|---|---|---|
| **P1** | Corriger tous les accents et fautes de français dans les textes | Faible |
| **P1** | Ajouter gestion d'erreur au chargement (fichiers manquants) | Faible |
| **P2** | Déplacer les imports en tête de fichier | Faible |
| **P2** | Ajouter un court paragraphe d'intro + lien GitHub sur la Vue d'ensemble | Faible |
| **P2** | Renommer « Classe (edible/poisonous) » en « Classe (comestible/vénéneux) » | Faible |
| **P2** | Reformuler le message d'erreur Sensibilité (visiteur ≠ développeur) | Faible |
| **P3** | Ajouter `st.spinner` autour des calculs lourds (Classification, Clustering) | Faible |
| **P3** | Traduire les en-têtes des tableaux (sensitivity, model_comparison) pour l'affichage | Moyen |
| **P4** | Ajouter le graphique inertie vs accuracy sur la page Sensibilité | Moyen |
| **P4** | Gérer les NaN dans la coloration (Espace ACM) | Faible |

---

## 5. Ce qui est bien fait (et à conserver)

- **Structure en 5 pages** : Claire, logique, alignée avec la démarche du README
- **Plotly** : Graphiques interactifs (zoom, pan, hover) — bien meilleur que des images statiques
- **Sliders** : Feedback immédiat, exploration libre
- **Vue clusters vs classes réelles** : Pédagogiquement très forte
- **Métriques en cartes** : `st.metric` — lisible et professionnel
- **Cache sur les données** : Pas de rechargement inutile
- **Layout wide** : Bonne utilisation de l'espace

---

## 6. Verdict final

Le dashboard est **fonctionnel, pertinent et utile**. Il permet d'explorer les résultats du projet sans toucher au code. Les problèmes identifiés sont principalement :
- **Linguistiques** (accents, cohérence français)
- **UX** (feedback pendant les calculs, messages d'erreur)
- **Conventions de code** (imports)

Aucun problème bloquant. Avec les corrections P1 et P2, le dashboard serait déjà aligné avec le niveau de qualité du reste du projet. Les P3 et P4 sont des améliorations de confort.

**Note globale : 7/10** — Solide, avec une marge évidente de progression vers le 9/10.

---

*Audit réalisé le 18 février 2026 — The Mushroom Project*
