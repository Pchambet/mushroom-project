"""
Fonctions utilitaires pour le projet mushroom
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def get_project_root():
    """Retourne le chemin racine du projet"""
    return Path(__file__).parent.parent


def load_processed_data():
    """Charge le dataset nettoyé"""
    processed_file = get_project_root() / 'data' / 'processed' / 'mushroom_processed.csv'
    return pd.read_csv(processed_file)


def load_mca_coordinates():
    """Charge les coordonnées ACM"""
    coords_file = get_project_root() / 'reports' / 'tables' / 'mca_row_coordinates.csv'
    return pd.read_csv(coords_file, index_col=0)


def save_figure(fig, filename, dpi=300):
    """Sauvegarde une figure dans reports/figures/"""
    figures_dir = get_project_root() / 'reports' / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    filepath = figures_dir / filename
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Figure sauvegardée : {filepath}")


def save_table(df, filename):
    """Sauvegarde un DataFrame dans reports/tables/"""
    tables_dir = get_project_root() / 'reports' / 'tables'
    tables_dir.mkdir(parents=True, exist_ok=True)
    filepath = tables_dir / filename
    df.to_csv(filepath)
    print(f"✓ Table sauvegardée : {filepath}")


def print_section(title):
    """Affiche un titre de section"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)
