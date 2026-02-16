"""
Utilitaires partagés — The Mushroom Project.

Fonctions d'I/O, chemins, et helpers réutilisés par l'ensemble du pipeline.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional


# ── Chemins ────────────────────────────────────────────────

def get_project_root() -> Path:
    """Retourne le chemin racine du projet."""
    return Path(__file__).resolve().parent.parent


def _ensure_dir(path: Path) -> Path:
    """Crée le répertoire s'il n'existe pas et le retourne."""
    path.mkdir(parents=True, exist_ok=True)
    return path


# ── Chargement des données ─────────────────────────────────

def load_processed_data() -> pd.DataFrame:
    """Charge le dataset nettoyé depuis ``data/processed/mushroom_processed.csv``.

    Returns
    -------
    pd.DataFrame
        Dataset avec 23 colonnes (``class`` + 22 variables morphologiques).

    Raises
    ------
    FileNotFoundError
        Si le fichier n'existe pas (exécuter ``01_prepare.py`` d'abord).
    """
    path = get_project_root() / "data" / "processed" / "mushroom_processed.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {path}\n"
            "Exécuter d'abord : python src/01_prepare.py"
        )
    return pd.read_csv(path)


def load_mca_coordinates() -> pd.DataFrame:
    """Charge les coordonnées ACM depuis ``data/processed/mca_coords.csv``.

    Returns
    -------
    pd.DataFrame
        Coordonnées factorielles (8 124 lignes x k dimensions).

    Raises
    ------
    FileNotFoundError
        Si le fichier n'existe pas (exécuter ``03_mca.py`` d'abord).
    """
    path = get_project_root() / "data" / "processed" / "mca_coords.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Coordonnées ACM introuvables : {path}\n"
            "Exécuter d'abord : python src/03_mca.py"
        )
    return pd.read_csv(path)


# ── Sauvegarde ─────────────────────────────────────────────

def save_figure(fig: plt.Figure, filename: str, dpi: int = 300) -> Path:
    """Sauvegarde une figure matplotlib dans ``reports/figures/``.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure à sauvegarder.
    filename : str
        Nom du fichier (ex: ``acm_scree.png``).
    dpi : int
        Résolution (défaut : 300).

    Returns
    -------
    Path
        Chemin absolu du fichier sauvegardé.
    """
    figures_dir = _ensure_dir(get_project_root() / "reports" / "figures")
    filepath = figures_dir / filename
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [figure] {filepath.relative_to(get_project_root())}")
    return filepath


def save_table(df: pd.DataFrame, filename: str, index: bool = False) -> Path:
    """Sauvegarde un DataFrame dans ``reports/tables/``.

    Parameters
    ----------
    df : pd.DataFrame
        Données à sauvegarder.
    filename : str
        Nom du fichier (ex: ``mca_eigenvalues.csv``).
    index : bool
        Inclure l'index dans le CSV (défaut : False).

    Returns
    -------
    Path
        Chemin absolu du fichier sauvegardé.
    """
    tables_dir = _ensure_dir(get_project_root() / "reports" / "tables")
    filepath = tables_dir / filename
    df.to_csv(filepath, index=index)
    print(f"  [table]  {filepath.relative_to(get_project_root())}")
    return filepath


# ── Affichage ──────────────────────────────────────────────

def print_section(title: str) -> None:
    """Affiche un titre de section formaté."""
    width = 60
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def print_step(message: str) -> None:
    """Affiche un message d'étape avec indicateur visuel."""
    print(f"  -> {message}")
