# mushroom-project (version française)

**Que garde, et que perd, une analyse des correspondances multiples (ACM) quand elle résume 21 descripteurs qualitatifs en quelques axes ? Une étude en validation croisée honnête sur les données UCI Mushroom.**

*English version (référence) : [README.md](README.md). Rapport interactif : [pchambet.github.io/mushroom-project](https://pchambet.github.io/mushroom-project/).*

![Précision hors échantillon selon le nombre d'axes ACM conservés](docs/figures/hero.png)

## En bref

- **Les données sont presque parfaitement séparables.** L'odeur seule donne 98,52 % de précision hors échantillon ; un arbre de décision de profondeur 7 (14 feuilles) sur le tableau disjonctif atteint 100 %. Classer n'est pas la difficulté.
- **Cinq axes ACM gardent l'information, mais pas sous une forme linéaire.** Sur les mêmes cinq axes, la LDA obtient 88,2 %, une forêt aléatoire 99,8 % et un vote des 15 plus proches voisins 99,3 %. La LDA reste à 88,8 % au plus jusqu'à neuf axes et doit en garder 19 pour atteindre 99 %.
- **Inertie n'est pas pertinence.** L'axe 1 porte la moitié de l'étiquette (η² = 0,51) ; les axes 2 à 9 au plus 0,09, alors que l'axe 90, l'un des plus petits, atteint 0,34.
- **L'erreur qui compte est asymétrique.** Les 120 erreurs de la règle « odeur » sont toutes des champignons vénéneux déclarés comestibles (espèces vénéneuses sans odeur) ; la LDA sur cinq axes en commet 816, la forêt aléatoire 10.
- **La première version de ce projet tirait ses conclusions d'un artefact de validation croisée.** Des plis non mélangés sur un fichier UCI trié produisaient un écart de ±15,6 points, lu comme un « modèle instable » et une forêt aléatoire « en surapprentissage ». Avec des plis stratifiés mélangés, l'écart tombe à ±1,0 point et la forêt aléatoire est le meilleur modèle sur les axes ACM.

## Démarche

1. **Données.** UCI Mushroom (8 124 spécimens, 22 descripteurs), téléchargées une fois et vérifiées par somme de contrôle. Les codes sont décodés avec le dictionnaire UCI ; les 2 480 valeurs inconnues de `stalk-root` deviennent une modalité `missing` plutôt qu'une imputation ; `veil-type`, constante, est retirée. Résultat : 21 variables, 116 modalités, 95 axes non triviaux.
2. **ACM.** Une implémentation numpy courte, sous forme de transformeur scikit-learn : elle est réajustée dans chaque pli d'apprentissage et le pli de test est projeté par la formule de transition. Ses valeurs propres et coordonnées coïncident avec celles de `prince` (testé).
3. **Évaluation.** Cinq plis stratifiés mélangés (`random_state=42`) pour tous les scores ; l'ACM, l'encodage et le classifieur sont dans un même pipeline. Chaque modèle rapporte aussi le nombre de vénéneux déclarés comestibles.

## Ce qui a été corrigé par rapport à la première version

| Affirmation initiale | Ce que montrent les données |
|---|---|
| « LDA instable : 77,1 % ± 14,0 % » | Artefact de plis non mélangés ; avec des plis mélangés, 88,2 % ± 1,0 % |
| « La forêt aléatoire surapprend » | Elle obtient 99,8 % ± 0,1 % hors échantillon sur cinq axes |
| « k = 4 axes est optimal » | Le choix venait du bruit des plis ; la LDA plafonne vers 88 % jusqu'à 9 axes |
| « 8 axes concentrent 90 % de l'information » | 90 % des *dix premiers* axes, qui portent 48,3 % de l'inertie totale |
| « Quand le modèle dit comestible, il a raison à 97,6 % » | 97,6 % était le rappel des comestibles ; la précision était de 83,5 % |

## Reproduire

```bash
make setup && make data && make run && make report   # environ 4,5 min de calcul (5 à 10 min réelles)
make test lint
```

## Limites

Les enregistrements sont des spécimens hypothétiques générés à partir d'un guide de terrain pour 23 espèces, pas des observations ; les plis aléatoires mesurent la reconnaissance d'espèces connues, pas d'une espèce nouvelle. Rien ici n'est un conseil de cueillette.

---

Réalisé par [Pierre Chambet](https://github.com/Pchambet) — decision science for operations under uncertainty.
