# Compilation du rapport LaTeX sur Overleaf

## 📁 Fichiers à uploader sur Overleaf

1. **Fichier principal** : `rapport_latex.tex`
2. **Dossier figures** : `reports/figures/` (avec toutes les images PNG)

## 🚀 Instructions de compilation

### 1. Créer un nouveau projet sur Overleaf

1. Aller sur [www.overleaf.com](https://www.overleaf.com)
2. Cliquer sur "New Project" → "Blank Project"
3. Nommer le projet : `Mushroom-Analysis-Report`

### 2. Upload des fichiers

**Option A : Upload manuel**
1. Cliquer sur "Upload" dans le menu de gauche
2. Upload `rapport_latex.tex`
3. Créer un dossier `reports/figures/` :
   - Clic droit → "New Folder" → `reports`
   - Dans `reports/`, créer un sous-dossier `figures`
4. Upload toutes les images PNG dans `reports/figures/`

**Option B : Upload du ZIP**
1. Compresser le dossier `mushroom-project/` localement
2. Sur Overleaf : "New Project" → "Upload Project" → sélectionner le ZIP

### 3. Structure des fichiers sur Overleaf

```
mushroom-project/
├── rapport_latex.tex          (fichier principal)
└── reports/
    └── figures/
        ├── desc_target_bar.png
        ├── desc_top_modalities_odor.png
        ├── desc_top_modalities_gill-color.png
        ├── desc_top_modalities_spore-print-color.png
        ├── acm_scree.png
        ├── acm_modalities_12.png
        └── acm_individuals_12_color_target.png
```

### 4. Compilation

1. S'assurer que le compilateur est réglé sur **pdfLaTeX**
2. Cliquer sur "Recompile"
3. Le PDF devrait compiler sans erreur

### 5. Si erreur de compilation

**Problème courant** : Chemin des images non trouvé

**Solution** : Modifier les chemins dans le .tex
- Chercher `\includegraphics`
- Remplacer `reports/figures/` par le chemin relatif correct sur Overleaf

**Exemple** :
```latex
% Si l'erreur persiste, essayer :
\includegraphics[width=0.6\textwidth]{desc_target_bar.png}
% Au lieu de :
\includegraphics[width=0.6\textwidth]{reports/figures/desc_target_bar.png}
```

## 📊 Images nécessaires

| Fichier | Section | Description |
|---------|---------|-------------|
| `desc_target_bar.png` | 3.1 | Distribution classe |
| `desc_top_modalities_odor.png` | 3.4 | Distribution odeur |
| `desc_top_modalities_gill-color.png` | 3.4 | Distribution couleur lamelles |
| `desc_top_modalities_spore-print-color.png` | 3.4 | Distribution empreinte spores |
| `acm_scree.png` | 4.2 | Scree plot (valeurs propres) |
| `acm_modalities_12.png` | 4.4 | Plan factoriel modalités |
| `acm_individuals_12_color_target.png` | 4.4 | Plan factoriel individus |

## ✅ Checklist avant rendu

- [ ] Toutes les images compilent correctement
- [ ] Remplacer `[Nom Personne A]`, `[Nom Personne B]`, `[Nom Personne C]` par vos vrais noms
- [ ] Remplacer `[Université]`, `[Master]`, `[Prof]` par les bonnes informations
- [ ] Vérifier la table des matières (TOC)
- [ ] Vérifier la bibliographie
- [ ] Relire les sections 1-4 pour cohérence
- [ ] Attendre les sections 5-6 de Personne B avant finalisation

## 🎨 Personnalisation (optionnel)

### Changer les couleurs

Dans le préambule du .tex :

```latex
% Ajouter après \usepackage{xcolor}
\definecolor{primary}{RGB}{0,102,204}
\definecolor{secondary}{RGB}{204,0,0}
```

### Ajouter un logo

```latex
% Dans la page de garde
\includegraphics[width=3cm]{logo_universite.png}
```

## 📤 Export final

1. Sur Overleaf : Menu → "Download" → "PDF"
2. Renommer le PDF : `Nom1_Nom2_Nom3_Mushroom_Report.pdf`
3. Préparer le dossier de rendu :
   ```
   rendu/
   ├── Nom1_Nom2_Nom3_Mushroom_Report.pdf
   ├── mushroom_processed.csv
   └── data_dictionary.md
   ```

## 🔧 Dépannage

**Erreur : "File not found"**
→ Vérifier que toutes les images sont dans le bon dossier

**Erreur : "Undefined control sequence"**
→ Vérifier que tous les packages sont chargés (normalement OK avec le préambule fourni)

**Erreur : "! LaTeX Error: Environment thebibliography undefined"**
→ Remplacer `\begin{thebibliography}` par `\begin{thebibliography}{9}` (déjà fait dans le .tex)

**PDF trop lourd**
→ Compresser les PNG avant upload (utiliser tinypng.com ou ImageOptim)

## 📞 Support

En cas de problème spécifique à Overleaf :
- Documentation : https://www.overleaf.com/learn
- Forum : https://www.overleaf.com/contact

Bon courage ! 🚀
