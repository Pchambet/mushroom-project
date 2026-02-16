"""
Validation du contrat d'interface A -> B.

Vérifie que les fichiers livrés par Personne A sont conformes aux
spécifications définies dans ``docs/INTERFACE_SPEC.md`` :
  - ``mushroom_processed.csv`` : 8 124 lignes, 23 colonnes, colonne ``class``
  - ``mca_coords.csv`` : 8 124 lignes, >= 5 colonnes flottantes (Dim*)
  - Correspondance ligne-à-ligne entre les deux fichiers
  - Présence des figures ACM requises
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path
import sys

from utils import get_project_root, print_section


# ── Configuration ──────────────────────────────────────────

EXPECTED_ROWS = 8124
EXPECTED_COLS = 23
MIN_AXES = 5
REQUIRED_FIGURES = ["acm_scree.png", "acm_modalities_12.png"]


# ── Validation ─────────────────────────────────────────────

def validate_interface() -> bool:
    """Valide les fichiers d'interface A -> B.

    Returns
    -------
    bool
        ``True`` si l'interface est valide (éventuellement avec warnings).
    """
    root = get_project_root()
    errors: list[str] = []
    warnings: list[str] = []

    print_section("Validation de l'interface A -> B")

    # ── Check 1 : mushroom_processed.csv ──

    print("\n  [1] mushroom_processed.csv")
    processed_path = root / "data" / "processed" / "mushroom_processed.csv"
    df = None

    if not processed_path.exists():
        errors.append("mushroom_processed.csv manquant")
    else:
        df = pd.read_csv(processed_path)
        print(f"      Trouvé : {df.shape}")

        if df.shape[0] != EXPECTED_ROWS:
            warnings.append(f"Attendu {EXPECTED_ROWS:,} lignes, trouvé {df.shape[0]:,}")
        if df.shape[1] != EXPECTED_COLS:
            errors.append(f"Attendu {EXPECTED_COLS} colonnes, trouvé {df.shape[1]}")
        if "class" not in df.columns:
            errors.append("Colonne 'class' manquante")
        elif not df["class"].isin(["e", "p"]).all():
            errors.append("Colonne 'class' contient des valeurs hors {{e, p}}")

        na_count = df.isna().sum().sum()
        if na_count > 0:
            warnings.append(f"{na_count:,} valeurs manquantes trouvées")

    # ── Check 2 : mca_coords.csv ──

    print("  [2] mca_coords.csv")
    coords_path = root / "data" / "processed" / "mca_coords.csv"
    coords = None

    if not coords_path.exists():
        errors.append("mca_coords.csv manquant (fichier CRITIQUE)")
    else:
        coords = pd.read_csv(coords_path)
        print(f"      Trouvé : {coords.shape}")

        if coords.shape[0] != EXPECTED_ROWS:
            errors.append(f"mca_coords: attendu {EXPECTED_ROWS:,} lignes, trouvé {coords.shape[0]:,}")
        if coords.shape[1] < MIN_AXES:
            errors.append(f"mca_coords: attendu >= {MIN_AXES} axes, trouvé {coords.shape[1]}")

        non_float = [c for c in coords.columns if coords[c].dtype.kind != "f"]
        if non_float:
            errors.append(f"mca_coords: colonnes non-float : {non_float}")

        if not all(c.startswith("Dim") for c in coords.columns):
            warnings.append("mca_coords: colonnes ne suivent pas le pattern Dim*")

    # ── Check 3 : correspondance ligne-à-ligne ──

    if df is not None and coords is not None:
        print("  [3] Correspondance ligne-à-ligne")
        if len(df) == len(coords):
            print("      OK")
        else:
            errors.append(f"Longueurs différentes : df={len(df):,}, coords={len(coords):,}")

    # ── Check 4 : figures ACM ──

    print("  [4] Figures ACM")
    figures_dir = root / "reports" / "figures"
    for fig_name in REQUIRED_FIGURES:
        path = figures_dir / fig_name
        if path.exists():
            print(f"      {fig_name} OK")
        else:
            warnings.append(f"Figure manquante : {fig_name}")

    # ── Résultat ──

    print()
    print("  " + "-" * 40)

    if errors:
        print("  ERREURS :")
        for e in errors:
            print(f"    [x] {e}")

    if warnings:
        print("  WARNINGS :")
        for w in warnings:
            print(f"    [!] {w}")

    if not errors and not warnings:
        print("  RESULTAT : Interface valide.")
        print("  Personne B peut commencer les missions B1 et B2.")
        return True
    elif not errors:
        print("  RESULTAT : Interface acceptable (avec warnings).")
        return True
    else:
        print("  RESULTAT : Interface invalide.")
        print("  Personne A doit corriger les erreurs avant livraison.")
        return False


if __name__ == "__main__":
    success = validate_interface()
    sys.exit(0 if success else 1)
