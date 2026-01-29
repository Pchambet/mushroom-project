"""
Script de téléchargement du dataset UCI Mushroom
Télécharge et fige le dataset brut dans data/raw/
"""

import urllib.request
import os
from pathlib import Path


def download_mushroom_dataset():
    """Télécharge le dataset mushroom depuis UCI ML Repository"""
    
    # Définir les chemins
    raw_dir = Path(__file__).parent.parent / 'data' / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # URLs du dataset
    data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data'
    names_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.names'
    
    # Fichiers de destination
    data_file = raw_dir / 'agaricus-lepiota.data'
    names_file = raw_dir / 'agaricus-lepiota.names'
    
    print("Téléchargement du dataset mushroom...")
    
    # Télécharger le fichier de données
    if not data_file.exists():
        print(f"Téléchargement de {data_url}")
        urllib.request.urlretrieve(data_url, data_file)
        print(f"✓ Données sauvegardées : {data_file}")
    else:
        print(f"✓ Fichier déjà existant : {data_file}")
    
    # Télécharger le fichier de description
    if not names_file.exists():
        print(f"Téléchargement de {names_url}")
        urllib.request.urlretrieve(names_url, names_file)
        print(f"✓ Description sauvegardée : {names_file}")
    else:
        print(f"✓ Fichier déjà existant : {names_file}")
    
    print("\n✅ Téléchargement terminé!")
    print(f"Fichiers disponibles dans : {raw_dir}")


if __name__ == "__main__":
    download_mushroom_dataset()
