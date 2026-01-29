"""
Script d'analyse discriminante (Mission B2 - pour Personne B)
Effectue une analyse discriminante sur les axes ACM
Matrice de confusion et validation croisée
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


def perform_discriminant_analysis():
    """Effectue l'analyse discriminante (Mission B2)"""
    
    # Définir les chemins
    coords_file = Path(__file__).parent.parent / 'data' / 'processed' / 'mca_coords.csv'
    processed_file = Path(__file__).parent.parent / 'data' / 'processed' / 'mushroom_processed.csv'
    figures_dir = Path(__file__).parent.parent / 'reports' / 'figures'
    tables_dir = Path(__file__).parent.parent / 'reports' / 'tables'
    
    print("=== Mission B2 : Analyse discriminante sur axes ACM ===\n")
    
    # Charger les données
    coords = pd.read_csv(coords_file)
    df = pd.read_csv(processed_file)
    
    # Utiliser les k premières composantes (même k que clustering)
    k = 5
    X = coords.iloc[:, :k].values
    y = (df['class'] == 'e').astype(int)  # 1 = edible, 0 = poisonous
    
    print(f"✓ Données préparées : {X.shape}")
    print(f"  Classes : {dict(zip(['Poisonous', 'Edible'], [sum(y==0), sum(y==1)]))}")
    
    # ========== Analyse discriminante linéaire ==========
    print("\n--- Analyse Discriminante Linéaire (LDA) ---")
    lda = LinearDiscriminantAnalysis()
    lda.fit(X, y)
    
    # Prédictions
    y_pred = lda.predict(X)
    
    # Coefficients (importance des axes)
    coef_df = pd.DataFrame({
        'axis': [f'Dim{i+1}' for i in range(k)],
        'coefficient': lda.coef_[0],
        'abs_coefficient': np.abs(lda.coef_[0])
    }).sort_values('abs_coefficient', ascending=False)
    
    print("\n✓ Coefficients LDA (importance des axes) :")
    print(coef_df.to_string(index=False))
    
    # ========== TABLE 1 : da_confusion.csv ==========
    cm = confusion_matrix(y, y_pred)
    cm_df = pd.DataFrame(cm, 
                        index=['Actual_Poisonous', 'Actual_Edible'],
                        columns=['Pred_Poisonous', 'Pred_Edible'])
    cm_df.to_csv(tables_dir / 'da_confusion.csv')
    print(f"\n✓ Sauvegardé : {tables_dir / 'da_confusion.csv'}")
    
    # FIGURE 1 : da_confusion.png
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Vénéneux (0)', 'Comestible (1)'],
                yticklabels=['Vénéneux (0)', 'Comestible (1)'],
                cbar_kws={'label': 'Fréquence'})
    plt.title('Matrice de confusion - Analyse discriminante', fontsize=14, fontweight='bold')
    plt.ylabel('Vraie classe', fontsize=12)
    plt.xlabel('Classe prédite', fontsize=12)
    plt.tight_layout()
    plt.savefig(figures_dir / 'da_confusion.png', dpi=300)
    plt.close()
    print(f"✓ Sauvegardé : {figures_dir / 'da_confusion.png'}")
    
    # ========== TABLE 2 : da_metrics.csv ==========
    report_dict = classification_report(y, y_pred, 
                                       target_names=['Poisonous', 'Edible'],
                                       output_dict=True)
    
    metrics_df = pd.DataFrame({
        'metric': ['Accuracy', 'Precision_Poisonous', 'Recall_Poisonous', 'F1_Poisonous',
                  'Precision_Edible', 'Recall_Edible', 'F1_Edible'],
        'value': [
            report_dict['accuracy'],
            report_dict['Poisonous']['precision'],
            report_dict['Poisonous']['recall'],
            report_dict['Poisonous']['f1-score'],
            report_dict['Edible']['precision'],
            report_dict['Edible']['recall'],
            report_dict['Edible']['f1-score']
        ]
    })
    
    print("\n--- Rapport de classification ---")
    print(classification_report(y, y_pred, target_names=['Poisonous', 'Edible']))
    
    # ========== Validation croisée (5-fold) ==========
    print("\n--- Validation croisée (5-fold) ---")
    cv_scores = cross_val_score(lda, X, y, cv=5, scoring='accuracy')
    
    cv_results = pd.DataFrame({
        'Fold': [f'Fold {i+1}' for i in range(5)] + ['Mean', 'Std'],
        'Accuracy': list(cv_scores) + [cv_scores.mean(), cv_scores.std()]
    })
    
    print(f"Scores CV : {cv_scores}")
    print(f"Accuracy moyenne : {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Ajouter les métriques CV au tableau
    metrics_df = pd.concat([
        metrics_df,
        pd.DataFrame({
            'metric': ['CV_Mean_Accuracy', 'CV_Std_Accuracy'],
            'value': [cv_scores.mean(), cv_scores.std()]
        })
    ], ignore_index=True)
    
    metrics_df.to_csv(tables_dir / 'da_metrics.csv', index=False)
    print(f"\n✓ Sauvegardé : {tables_dir / 'da_metrics.csv'}")
    print(metrics_df.to_string(index=False))
    
    # FIGURE 2 : da_cv_scores.png (bonus)
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(1, 6), cv_scores, color='steelblue', alpha=0.8, edgecolor='black')
    plt.axhline(cv_scores.mean(), color='red', linestyle='--', linewidth=2, 
               label=f'Moyenne = {cv_scores.mean():.4f}')
    plt.xlabel('Fold', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Validation croisée - Scores par fold', fontsize=14, fontweight='bold')
    plt.ylim([min(cv_scores) - 0.01, 1.0])
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Ajouter valeurs sur barres
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{cv_scores[i]:.4f}',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'da_cv_scores.png', dpi=300)
    plt.close()
    print(f"✓ Sauvegardé : {figures_dir / 'da_cv_scores.png'}")
    
    # Prédictions avec validation croisée pour matrice confusion CV
    y_pred_cv = cross_val_predict(lda, X, y, cv=5)
    cm_cv = confusion_matrix(y, y_pred_cv)
    
    # FIGURE 3 : da_confusion_cv.png
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_cv, annot=True, fmt='d', cmap='Greens',
                xticklabels=['Vénéneux (0)', 'Comestible (1)'],
                yticklabels=['Vénéneux (0)', 'Comestible (1)'],
                cbar_kws={'label': 'Fréquence'})
    plt.title('Matrice de confusion - Validation croisée (5-fold)', 
             fontsize=14, fontweight='bold')
    plt.ylabel('Vraie classe', fontsize=12)
    plt.xlabel('Classe prédite', fontsize=12)
    plt.tight_layout()
    plt.savefig(figures_dir / 'da_confusion_cv.png', dpi=300)
    plt.close()
    print(f"✓ Sauvegardé : {figures_dir / 'da_confusion_cv.png'}")
    
    print("\n" + "="*60)
    print("✅ Mission B2 terminée!")
    print("="*60)
    print("\nFichiers générés :")
    print("  Tables:")
    print("    - da_metrics.csv")
    print("    - da_confusion.csv")
    print("  Figures:")
    print("    - da_confusion.png")
    print("    - da_cv_scores.png (bonus)")
    print("    - da_confusion_cv.png")


if __name__ == "__main__":
    perform_discriminant_analysis()
