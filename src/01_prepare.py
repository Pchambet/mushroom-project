"""
Script de préparation des données
Parse le dataset brut, ajoute les headers, gère les valeurs manquantes "?"
Sauvegarde le dataset nettoyé dans data/processed/
"""

import pandas as pd
from pathlib import Path


def prepare_data():
    """Prépare et nettoie le dataset mushroom"""
    
    # Définir les chemins
    raw_file = Path(__file__).parent.parent / 'data' / 'raw' / 'agaricus-lepiota.data'
    processed_dir = Path(__file__).parent.parent / 'data' / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_file = processed_dir / 'mushroom_processed.csv'
    
    print("Préparation du dataset...")
    
    # Noms des colonnes
    column_names = [
        'class',
        'cap-shape', 'cap-surface', 'cap-color',
        'bruises', 'odor',
        'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
        'stalk-shape', 'stalk-root',
        'stalk-surface-above-ring', 'stalk-surface-below-ring',
        'stalk-color-above-ring', 'stalk-color-below-ring',
        'veil-type', 'veil-color',
        'ring-number', 'ring-type',
        'spore-print-color',
        'population', 'habitat'
    ]
    
    # Lire le dataset
    df = pd.read_csv(raw_file, header=None, names=column_names)
    
    print(f"✓ Dataset chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    
    # Gestion des valeurs manquantes ("?")
    missing_before = (df == '?').sum().sum()
    print(f"✓ Valeurs manquantes ('?') détectées : {missing_before}")
    
    # Remplacer "?" par NaN
    df = df.replace('?', pd.NA)
    
    # Sauvegarder le dataset nettoyé
    df.to_csv(processed_file, index=False)
    print(f"✓ Dataset nettoyé sauvegardé : {processed_file}")
    
    # Afficher un résumé
    print("\n--- Résumé du dataset ---")
    print(f"Dimensions : {df.shape}")
    print(f"\nValeurs manquantes par colonne :")
    print(df.isna().sum()[df.isna().sum() > 0])
    print(f"\nDistribution de la variable cible :")
    print(df['class'].value_counts())
    
    print("\n✅ Préparation terminée!")


if __name__ == "__main__":
    prepare_data()
