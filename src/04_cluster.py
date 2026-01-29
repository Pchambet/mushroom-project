"""
Script de clustering (Mission B1 - pour Personne B)
Effectue CAH et K-means sur les coordonnées ACM
Profiling des clusters
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import chi2_contingency


def perform_clustering():
    """Effectue le clustering sur les coordonnées ACM (Mission B1)"""
    
    # Définir les chemins
    coords_file = Path(__file__).parent.parent / 'data' / 'processed' / 'mca_coords.csv'
    processed_file = Path(__file__).parent.parent / 'data' / 'processed' / 'mushroom_processed.csv'
    figures_dir = Path(__file__).parent.parent / 'reports' / 'figures'
    tables_dir = Path(__file__).parent.parent / 'reports' / 'tables'
    
    print("=== Mission B1 : Clustering sur coordonnées ACM ===\n")
    
    # Charger les coordonnées ACM et dataset original
    coords = pd.read_csv(coords_file)
    df = pd.read_csv(processed_file)
    
    # Utiliser les k premières composantes (à ajuster selon recommandation A)
    k = 5
    X_mca = coords.iloc[:, :k].values
    
    print(f"✓ Coordonnées ACM chargées : {X_mca.shape}")
    print(f"  Utilisation de k={k} premières composantes")
    
    # ========== CAH (sur échantillon pour dendrogramme) ==========
    print("\n--- Classification Ascendante Hiérarchique (Ward) ---")
    sample_size = min(500, len(X_mca))
    sample_idx = np.random.choice(len(X_mca), sample_size, replace=False)
    X_sample = X_mca[sample_idx]
    
    linkage_matrix = linkage(X_sample, method='ward')
    
    # FIGURE 1 : cluster_dendrogram.png
    plt.figure(figsize=(12, 6))
    dendrogram(linkage_matrix, no_labels=True, color_threshold=None)
    plt.title('Dendrogramme - CAH (Ward)', fontsize=14, fontweight='bold')
    plt.xlabel('Individus', fontsize=12)
    plt.ylabel('Distance', fontsize=12)
    plt.axhline(y=15, color='r', linestyle='--', alpha=0.7, label='Coupe suggérée')
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / 'cluster_dendrogram.png', dpi=300)
    plt.close()
    print(f"✓ Sauvegardé : {figures_dir / 'cluster_dendrogram.png'}")
    
    # ========== K-means ==========
    print("\n--- K-means clustering ---")
    n_clusters = 3  # À ajuster selon dendrogramme
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_mca)
    
    print(f"✓ K-means effectué avec {n_clusters} clusters")
    
    # TABLE 1 : cluster_sizes.csv
    cluster_sizes = pd.DataFrame({
        'cluster': range(n_clusters),
        'size': [sum(cluster_labels == i) for i in range(n_clusters)],
        'percentage': [sum(cluster_labels == i) / len(cluster_labels) * 100 for i in range(n_clusters)]
    })
    cluster_sizes.to_csv(tables_dir / 'cluster_sizes.csv', index=False)
    print(f"\n✓ Sauvegardé : {tables_dir / 'cluster_sizes.csv'}")
    print(cluster_sizes.to_string(index=False))
    
    # FIGURE 2 : cluster_on_acm12.png
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(coords.iloc[:, 0], coords.iloc[:, 1], 
                         c=cluster_labels, cmap='viridis', alpha=0.6, s=30)
    plt.xlabel('Dim 1', fontsize=12)
    plt.ylabel('Dim 2', fontsize=12)
    plt.title(f'K-means - {n_clusters} clusters (plan factoriel ACM 1-2)', 
             fontsize=14, fontweight='bold')
    plt.colorbar(scatter, label='Cluster')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / 'cluster_on_acm12.png', dpi=300)
    plt.close()
    print(f"✓ Sauvegardé : {figures_dir / 'cluster_on_acm12.png'}")
    
    # ========== Profiling des clusters ==========
    print("\n--- Profiling des clusters ---")
    df_clusters = df.copy()
    df_clusters['cluster'] = cluster_labels
    
    # TABLE 2 : cluster_vs_target.csv
    cluster_vs_target = pd.crosstab(df_clusters['cluster'], df_clusters['class'], margins=True)
    cluster_vs_target.to_csv(tables_dir / 'cluster_vs_target.csv')
    print(f"\n✓ Sauvegardé : {tables_dir / 'cluster_vs_target.csv'}")
    print(cluster_vs_target)
    
    # TABLE 3 : cluster_profiles.csv (modalités sur/sous-représentées)
    print("\n--- Analyse modalités sur/sous-représentées ---")
    profiles = []
    
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        
        for col in df.columns:
            if col == 'class':
                continue
                
            # Comparer distributions
            cluster_dist = df[cluster_mask][col].value_counts(normalize=True)
            global_dist = df[col].value_counts(normalize=True)
            
            for modality in cluster_dist.index:
                if modality in global_dist.index:
                    over_rep = cluster_dist[modality] / global_dist[modality]
                    
                    # Garder seulement les modalités significativement sur/sous-rep
                    if over_rep > 1.5 or over_rep < 0.5:
                        profiles.append({
                            'cluster': cluster_id,
                            'variable': col,
                            'modality': modality,
                            'cluster_freq_%': round(cluster_dist[modality] * 100, 2),
                            'global_freq_%': round(global_dist[modality] * 100, 2),
                            'over_representation': round(over_rep, 2)
                        })
    
    profile_df = pd.DataFrame(profiles).sort_values(['cluster', 'over_representation'], 
                                                    ascending=[True, False])
    profile_df.to_csv(tables_dir / 'cluster_profiles.csv', index=False)
    print(f"✓ Sauvegardé : {tables_dir / 'cluster_profiles.csv'}")
    print(f"  Total de {len(profile_df)} modalités significatives trouvées")
    
    # Afficher top 5 par cluster
    for cluster_id in range(n_clusters):
        print(f"\n  Cluster {cluster_id} (top 5 modalités caractéristiques) :")
        top_cluster = profile_df[profile_df['cluster'] == cluster_id].head(5)
        print(top_cluster[['variable', 'modality', 'over_representation']].to_string(index=False))
    
    print("\n" + "="*60)
    print("✅ Mission B1 terminée!")
    print("="*60)
    print("\nFichiers générés :")
    print("  Tables:")
    print("    - cluster_sizes.csv")
    print("    - cluster_vs_target.csv")
    print("    - cluster_profiles.csv")
    print("  Figures:")
    print("    - cluster_dendrogram.png")
    print("    - cluster_on_acm12.png")


if __name__ == "__main__":
    perform_clustering()
