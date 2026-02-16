"""
01 — Préparation des données.

Parse le dataset brut, ajoute les en-têtes de colonnes, remplace les
valeurs manquantes ``"?"`` par ``pd.NA``, et sauvegarde le dataset
nettoyé dans ``data/processed/mushroom_processed.csv``.
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path


# ── Configuration ──────────────────────────────────────────

COLUMN_NAMES = [
    "class",
    "cap-shape", "cap-surface", "cap-color",
    "bruises", "odor",
    "gill-attachment", "gill-spacing", "gill-size", "gill-color",
    "stalk-shape", "stalk-root",
    "stalk-surface-above-ring", "stalk-surface-below-ring",
    "stalk-color-above-ring", "stalk-color-below-ring",
    "veil-type", "veil-color",
    "ring-number", "ring-type",
    "spore-print-color",
    "population", "habitat",
]

MISSING_SENTINEL = "?"


# ── Pipeline ───────────────────────────────────────────────

def prepare_data() -> pd.DataFrame:
    """Prépare et nettoie le dataset mushroom.

    Returns
    -------
    pd.DataFrame
        Dataset nettoyé (8 124 lignes x 23 colonnes).
    """
    project_root = Path(__file__).resolve().parent.parent
    raw_file = project_root / "data" / "raw" / "agaricus-lepiota.data"
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_file = processed_dir / "mushroom_processed.csv"

    print("=" * 60)
    print("  01 — Préparation des données")
    print("=" * 60)
    print()

    # Chargement
    df = pd.read_csv(raw_file, header=None, names=COLUMN_NAMES)
    print(f"  [load] Dataset chargé : {df.shape[0]:,} lignes x {df.shape[1]} colonnes")

    # Gestion des valeurs manquantes
    n_missing = (df == MISSING_SENTINEL).sum().sum()
    print(f"  [clean] Valeurs manquantes ('{MISSING_SENTINEL}') détectées : {n_missing:,}")
    df = df.replace(MISSING_SENTINEL, pd.NA)

    # Sauvegarde
    df.to_csv(processed_file, index=False)
    print(f"  [save] Dataset nettoyé : {processed_file}")

    # Résumé
    print()
    print("  --- Résumé ---")
    print(f"  Dimensions : {df.shape}")

    missing_cols = df.isna().sum()
    missing_cols = missing_cols[missing_cols > 0]
    if not missing_cols.empty:
        print(f"  Valeurs manquantes :")
        for col, count in missing_cols.items():
            pct = count / len(df) * 100
            print(f"    {col}: {count:,} ({pct:.1f}%)")

    print(f"  Distribution cible :")
    for cls, count in df["class"].value_counts().items():
        pct = count / len(df) * 100
        print(f"    {cls}: {count:,} ({pct:.1f}%)")

    print()
    return df


if __name__ == "__main__":
    prepare_data()
