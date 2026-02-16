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

## Notes
- Toutes les variables sont catégorielles
- Le dataset ne contient pas de valeurs numériques
