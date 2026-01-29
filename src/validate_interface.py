"""
Script de vérification de l'interface A→B
Valide que les fichiers livrés par A sont conformes au contrat
"""

import pandas as pd
from pathlib import Path
import sys


def validate_interface():
    """Valide les fichiers d'interface A→B"""
    
    project_root = Path(__file__).parent.parent
    errors = []
    warnings = []
    
    print("="*60)
    print("  VALIDATION INTERFACE A → B")
    print("="*60)
    
    # ========== CHECK 1 : mushroom_processed.csv ==========
    print("\n[1] Vérification mushroom_processed.csv...")
    processed_file = project_root / 'data' / 'processed' / 'mushroom_processed.csv'
    
    if not processed_file.exists():
        errors.append("❌ mushroom_processed.csv manquant!")
    else:
        df = pd.read_csv(processed_file)
        print(f"  ✓ Fichier trouvé : {df.shape}")
        
        # Vérifications
        if df.shape[0] != 8124:
            warnings.append(f"⚠️  Attendu 8124 lignes, trouvé {df.shape[0]}")
        
        if df.shape[1] != 23:
            errors.append(f"❌ Attendu 23 colonnes, trouvé {df.shape[1]}")
        
        if 'class' not in df.columns:
            errors.append("❌ Colonne 'class' manquante!")
        elif not df['class'].isin(['e', 'p']).all():
            errors.append("❌ Classe contient valeurs autres que e/p!")
        
        # Vérifier valeurs manquantes
        na_count = df.isna().sum().sum()
        if na_count > 0:
            warnings.append(f"⚠️  {na_count} valeurs manquantes trouvées")
    
    # ========== CHECK 2 : mca_coords.csv ==========
    print("\n[2] Vérification mca_coords.csv...")
    coords_file = project_root / 'data' / 'processed' / 'mca_coords.csv'
    
    if not coords_file.exists():
        errors.append("❌ mca_coords.csv manquant! (fichier CRITIQUE)")
    else:
        coords = pd.read_csv(coords_file)
        print(f"  ✓ Fichier trouvé : {coords.shape}")
        
        # Vérifications
        if coords.shape[0] != 8124:
            errors.append(f"❌ Attendu 8124 lignes, trouvé {coords.shape[0]}")
        
        if coords.shape[1] < 5:
            errors.append(f"❌ Attendu ≥5 axes, trouvé {coords.shape[1]}")
        
        # Vérifier types float
        non_float_cols = [col for col in coords.columns if coords[col].dtype.kind != 'f']
        if non_float_cols:
            errors.append(f"❌ Colonnes non-float : {non_float_cols}")
        
        # Vérifier noms colonnes
        expected_pattern = all(col.startswith('Dim') for col in coords.columns)
        if not expected_pattern:
            warnings.append("⚠️  Colonnes ne suivent pas le pattern 'Dim1', 'Dim2', etc.")
    
    # ========== CHECK 3 : Correspondance ligne à ligne ==========
    if processed_file.exists() and coords_file.exists():
        print("\n[3] Vérification correspondance ligne à ligne...")
        if len(df) == len(coords):
            print("  ✓ Nombre de lignes identique")
        else:
            errors.append(f"❌ Longueurs différentes: df={len(df)}, coords={len(coords)}")
    
    # ========== CHECK 4 : Fichiers figures ACM ==========
    print("\n[4] Vérification figures ACM...")
    figures_dir = project_root / 'reports' / 'figures'
    
    required_figures = ['acm_scree.png', 'acm_modalities_12.png']
    for fig in required_figures:
        fig_path = figures_dir / fig
        if fig_path.exists():
            print(f"  ✓ {fig}")
        else:
            warnings.append(f"⚠️  Figure manquante : {fig}")
    
    # ========== RÉSULTAT ==========
    print("\n" + "="*60)
    print("  RÉSULTAT")
    print("="*60)
    
    if errors:
        print("\n❌ ERREURS BLOQUANTES :")
        for err in errors:
            print(f"  {err}")
    
    if warnings:
        print("\n⚠️  AVERTISSEMENTS :")
        for warn in warnings:
            print(f"  {warn}")
    
    if not errors and not warnings:
        print("\n✅ INTERFACE VALIDE !")
        print("\nPersonne B peut commencer Missions B1 et B2.")
        return True
    elif not errors:
        print("\n✅ INTERFACE ACCEPTABLE (avec warnings)")
        print("\nPersonne B peut commencer, mais vérifier les warnings.")
        return True
    else:
        print("\n❌ INTERFACE INVALIDE")
        print("\nPersonne A doit corriger les erreurs avant livraison.")
        return False


if __name__ == "__main__":
    success = validate_interface()
    sys.exit(0 if success else 1)
