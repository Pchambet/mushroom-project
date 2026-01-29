"""
Script de statistiques descriptives (Mission A2)
Génère les statistiques descriptives et exporte les figures/tables selon plan
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def describe_data():
    """Génère les statistiques descriptives du dataset (Mission A2)"""
    
    # Définir les chemins
    processed_file = Path(__file__).parent.parent / 'data' / 'processed' / 'mushroom_processed.csv'
    figures_dir = Path(__file__).parent.parent / 'reports' / 'figures'
    tables_dir = Path(__file__).parent.parent / 'reports' / 'tables'
    
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    print("=== Mission A2 : Analyse descriptive ===\n")
    
    # Charger les données
    df = pd.read_csv(processed_file)
    print(f"✓ Dataset chargé : {df.shape}")
    
    # ========== TABLE 1 : univariate_summary.csv ==========
    print("\n--- Génération univariate_summary.csv ---")
    univariate_summary = []
    
    for col in df.columns:
        n_modalities = df[col].nunique()
        top_modality = df[col].mode()[0] if len(df[col].mode()) > 0 else None
        top_freq = (df[col] == top_modality).sum() / len(df) * 100 if top_modality else 0
        pct_missing = df[col].isna().sum() / len(df) * 100
        
        univariate_summary.append({
            'variable': col,
            'n_modalities': n_modalities,
            'top_modality': top_modality,
            'top_freq_pct': round(top_freq, 2),
            'missing_pct': round(pct_missing, 2)
        })
    
    univariate_df = pd.DataFrame(univariate_summary)
    univariate_df.to_csv(tables_dir / 'univariate_summary.csv', index=False)
    print(f"✓ Sauvegardé : {tables_dir / 'univariate_summary.csv'}")
    print(univariate_df.to_string(index=False))
    
    # ========== TABLE 2 : target_distribution.csv ==========
    print("\n--- Génération target_distribution.csv ---")
    target_dist = df['class'].value_counts().reset_index()
    target_dist.columns = ['class', 'count']
    target_dist['percentage'] = round(target_dist['count'] / target_dist['count'].sum() * 100, 2)
    target_dist.to_csv(tables_dir / 'target_distribution.csv', index=False)
    print(f"✓ Sauvegardé : {tables_dir / 'target_distribution.csv'}")
    print(target_dist.to_string(index=False))
    
    # ========== FIGURE 1 : desc_target_bar.png ==========
    print("\n--- Génération desc_target_bar.png ---")
    plt.figure(figsize=(8, 6))
    colors = ['#2ecc71', '#e74c3c']  # vert pour edible, rouge pour poisonous
    bars = plt.bar(target_dist['class'], target_dist['count'], color=colors, alpha=0.8, edgecolor='black')
    plt.title('Distribution de la classe (Edible vs Poisonous)', fontsize=14, fontweight='bold')
    plt.xlabel('Classe', fontsize=12)
    plt.ylabel('Fréquence', fontsize=12)
    plt.xticks(rotation=0)
    
    # Ajouter les valeurs sur les barres
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'desc_target_bar.png', dpi=300)
    plt.close()
    print(f"✓ Sauvegardé : {figures_dir / 'desc_target_bar.png'}")
    
    # ========== FIGURES 2-4 : desc_top_modalities_<var>.png (max 3 variables) ==========
    # Choisir 3 variables clés avec le plus de modalités ou plus intéressantes
    key_variables = ['odor', 'gill-color', 'spore-print-color']
    
    for var in key_variables:
        if var in df.columns:
            print(f"\n--- Génération desc_top_modalities_{var}.png ---")
            plt.figure(figsize=(10, 6))
            
            value_counts = df[var].value_counts().head(10)  # Top 10 modalités
            colors_palette = plt.cm.Set3(range(len(value_counts)))
            
            bars = plt.bar(range(len(value_counts)), value_counts.values, 
                          color=colors_palette, alpha=0.8, edgecolor='black')
            plt.xticks(range(len(value_counts)), value_counts.index, rotation=45, ha='right')
            plt.title(f'Distribution des modalités : {var}', fontsize=14, fontweight='bold')
            plt.xlabel('Modalité', fontsize=12)
            plt.ylabel('Fréquence', fontsize=12)
            plt.grid(axis='y', alpha=0.3)
            
            # Ajouter les valeurs sur les barres
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            plt.savefig(figures_dir / f'desc_top_modalities_{var}.png', dpi=300)
            plt.close()
            print(f"✓ Sauvegardé : {figures_dir / 'desc_top_modalities_{var}.png'}")
    
    print("\n" + "="*60)
    print("✅ Mission A2 terminée!")
    print("="*60)
    print("\nFichiers générés :")
    print("  Tables:")
    print("    - univariate_summary.csv")
    print("    - target_distribution.csv")
    print("  Figures:")
    print("    - desc_target_bar.png")
    for var in key_variables:
        if var in df.columns:
            print(f"    - desc_top_modalities_{var}.png")


if __name__ == "__main__":
    describe_data()
