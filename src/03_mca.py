"""
Script d'Analyse en Composantes Multiples - ACM (Mission A3)
Calcule les coordonnées, contributions, cos² et exporte les résultats
Produit le fichier interface mca_coords.csv pour Personne B
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import prince


def perform_mca():
    """Effectue l'ACM sur le dataset (Mission A3)"""
    
    # Définir les chemins
    processed_file = Path(__file__).parent.parent / 'data' / 'processed' / 'mushroom_processed.csv'
    processed_dir = Path(__file__).parent.parent / 'data' / 'processed'
    figures_dir = Path(__file__).parent.parent / 'reports' / 'figures'
    tables_dir = Path(__file__).parent.parent / 'reports' / 'tables'
    
    print("=== Mission A3 : Analyse en Composantes Multiples (ACM) ===\n")
    
    # Charger les données
    df = pd.read_csv(processed_file)
    
    # Séparer la variable cible
    y = df['class']
    X = df.drop('class', axis=1)
    
    # Remplacer les valeurs manquantes par la modalité la plus fréquente
    X = X.fillna(X.mode().iloc[0])
    
    print(f"✓ Données préparées : {X.shape}")
    print(f"  Variables : {X.shape[1]}")
    print(f"  Individus : {X.shape[0]}")
    
    # Effectuer l'ACM
    n_components = 10
    mca = prince.MCA(n_components=n_components, n_iter=3, copy=True, check_input=True)
    mca = mca.fit(X)
    
    print(f"\n✓ ACM effectuée avec {n_components} composantes")
    
    # ========== FICHIER INTERFACE 1 : mca_coords.csv (CRITIQUE pour B) ==========
    print("\n--- Génération mca_coords.csv (fichier interface pour B) ---")
    row_coords = mca.transform(X)
    row_coords.columns = [f'Dim{i+1}' for i in range(n_components)]
    row_coords.to_csv(processed_dir / 'mca_coords.csv', index=False)
    print(f"✓ Sauvegardé : {processed_dir / 'mca_coords.csv'}")
    print(f"  Shape: {row_coords.shape}")
    
    # ========== FICHIER 2 : mca_eigenvalues.csv ==========
    print("\n--- Génération mca_eigenvalues.csv ---")
    eigenvalues = mca.eigenvalues_
    total_inertia = mca.total_inertia_
    explained_inertia = eigenvalues / total_inertia
    
    eigenvalues_df = pd.DataFrame({
        'Component': [f'Dim{i+1}' for i in range(n_components)],
        'Eigenvalue': eigenvalues,
        'Explained_Inertia_%': explained_inertia * 100,
        'Cumulative_Inertia_%': np.cumsum(explained_inertia) * 100
    })
    eigenvalues_df.to_csv(processed_dir / 'mca_eigenvalues.csv', index=False)
    print(f"✓ Sauvegardé : {processed_dir / 'mca_eigenvalues.csv'}")
    print(eigenvalues_df.to_string(index=False))
    
    # Choix de k (nombre d'axes à conserver)
    cumulative_90 = np.where(np.cumsum(explained_inertia) >= 0.90)[0]
    k_recommended = cumulative_90[0] + 1 if len(cumulative_90) > 0 else 5
    print(f"\n💡 Recommandation k : {k_recommended} axes (pour ≥90% inertie cumulée)")
    
    # ========== FICHIER 3 : mca_modalities_contrib_axis1.csv ==========
    print("\n--- Génération mca_modalities_contrib_axis1.csv ---")
    # Contributions des modalités à l'axe 1
    col_coords = mca.column_coordinates(X)
    contrib_axis1 = pd.DataFrame({
        'modality': col_coords.index,
        'coord_dim1': col_coords[0],
        'contrib_dim1_%': mca.column_contributions_.iloc[:, 0] * 100
    }).sort_values('contrib_dim1_%', ascending=False).head(15)
    
    contrib_axis1.to_csv(processed_dir / 'mca_modalities_contrib_axis1.csv', index=False)
    print(f"✓ Sauvegardé : {processed_dir / 'mca_modalities_contrib_axis1.csv'}")
    print(contrib_axis1.to_string(index=False))
    
    # ========== FICHIER 4 : mca_modalities_contrib_axis2.csv ==========
    print("\n--- Génération mca_modalities_contrib_axis2.csv ---")
    contrib_axis2 = pd.DataFrame({
        'modality': col_coords.index,
        'coord_dim2': col_coords[1],
        'contrib_dim2_%': mca.column_contributions_.iloc[:, 1] * 100
    }).sort_values('contrib_dim2_%', ascending=False).head(15)
    
    contrib_axis2.to_csv(processed_dir / 'mca_modalities_contrib_axis2.csv', index=False)
    print(f"✓ Sauvegardé : {processed_dir / 'mca_modalities_contrib_axis2.csv'}")
    print(contrib_axis2.to_string(index=False))
    
    # ========== FIGURE 1 : acm_scree.png ==========
    print("\n--- Génération acm_scree.png ---")
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scree plot - valeurs propres
    ax[0].plot(range(1, n_components + 1), explained_inertia * 100, 'o-', linewidth=2, markersize=8)
    ax[0].set_xlabel('Composante', fontsize=12)
    ax[0].set_ylabel('Inertie expliquée (%)', fontsize=12)
    ax[0].set_title('Scree plot - Inertie par composante', fontsize=13, fontweight='bold')
    ax[0].grid(True, alpha=0.3)
    ax[0].axhline(y=100/n_components, color='r', linestyle='--', alpha=0.5, label='Moyenne')
    ax[0].legend()
    
    # Inertie cumulée
    ax[1].plot(range(1, n_components + 1), np.cumsum(explained_inertia) * 100, 'o-', 
              linewidth=2, markersize=8, color='green')
    ax[1].axhline(y=90, color='r', linestyle='--', alpha=0.5, label='90%')
    ax[1].set_xlabel('Nombre de composantes', fontsize=12)
    ax[1].set_ylabel('Inertie cumulée (%)', fontsize=12)
    ax[1].set_title('Inertie cumulée', fontsize=13, fontweight='bold')
    ax[1].grid(True, alpha=0.3)
    ax[1].legend()
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'acm_scree.png', dpi=300)
    plt.close()
    print(f"✓ Sauvegardé : {figures_dir / 'acm_scree.png'}")
    
    # ========== FIGURE 2 : acm_modalities_12.png ==========
    print("\n--- Génération acm_modalities_12.png ---")
    plt.figure(figsize=(12, 10))
    
    # Afficher les modalités sur le plan 1-2
    for idx, modality in enumerate(col_coords.index):
        x = col_coords.iloc[idx, 0]
        y = col_coords.iloc[idx, 1]
        plt.scatter(x, y, alpha=0.7, s=100)
        plt.annotate(modality, (x, y), fontsize=8, alpha=0.8)
    
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.3)
    plt.axvline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.3)
    plt.xlabel(f'Dim 1 ({explained_inertia[0]*100:.2f}%)', fontsize=12)
    plt.ylabel(f'Dim 2 ({explained_inertia[1]*100:.2f}%)', fontsize=12)
    plt.title('ACM - Plan factoriel des modalités (Dim 1-2)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / 'acm_modalities_12.png', dpi=300)
    plt.close()
    print(f"✓ Sauvegardé : {figures_dir / 'acm_modalities_12.png'}")
    
    # ========== FIGURE 3 : acm_individuals_12_color_target.png ==========
    print("\n--- Génération acm_individuals_12_color_target.png ---")
    
    # Reload y from the original dataframe to ensure it's fresh
    df_colors = pd.read_csv(processed_file)
    y_colors = df_colors['class']
    
    plt.figure(figsize=(10, 8))
    
    # Couleur selon la classe (convert to numpy array directly)
    colors = np.array((y_colors == 'e').astype(int))
    scatter = plt.scatter(row_coords.iloc[:, 0], row_coords.iloc[:, 1], 
                         c=colors, cmap='RdYlGn', alpha=0.6, s=20)
    
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.3)
    plt.axvline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.3)
    plt.xlabel(f'Dim 1 ({explained_inertia[0]*100:.2f}%)', fontsize=12)
    plt.ylabel(f'Dim 2 ({explained_inertia[1]*100:.2f}%)', fontsize=12)
    plt.title('ACM - Plan factoriel des individus (coloré par classe)', fontsize=14, fontweight='bold')
    cbar = plt.colorbar(scatter, label='Classe')
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Vénéneux (p)', 'Comestible (e)'])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / 'acm_individuals_12_color_target.png', dpi=300)
    plt.close()
    print(f"✓ Sauvegardé : {figures_dir / 'acm_individuals_12_color_target.png'}")
    
    print("\n" + "="*60)
    print("✅ Mission A3 terminée!")
    print("="*60)
    print("\n📦 LIVRAISON POUR PERSONNE B :")
    print(f"  ✓ mca_coords.csv ({row_coords.shape[0]} individus × {row_coords.shape[1]} axes)")
    print(f"  ✓ k recommandé = {k_recommended} axes")
    print(f"  ✓ Inertie cumulée (k={k_recommended}) = {np.cumsum(explained_inertia)[k_recommended-1]*100:.2f}%")
    print("\nFichiers data :")
    print("  - mca_coords.csv (INTERFACE)")
    print("  - mca_eigenvalues.csv")
    print("  - mca_modalities_contrib_axis1.csv")
    print("  - mca_modalities_contrib_axis2.csv")
    print("\nFigures :")
    print("  - acm_scree.png")
    print("  - acm_modalities_12.png")
    print("  - acm_individuals_12_color_target.png")


if __name__ == "__main__":
    perform_mca()
