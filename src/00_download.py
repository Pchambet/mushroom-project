"""
00 — Téléchargement du dataset UCI Mushroom.

Télécharge les fichiers bruts depuis le UCI Machine Learning Repository
et les stocke dans ``data/raw/``. Idempotent : ne re-télécharge pas
si les fichiers existent déjà.

Source : https://archive.ics.uci.edu/ml/datasets/Mushroom
"""

from __future__ import annotations

import urllib.request
from pathlib import Path


# ── Configuration ──────────────────────────────────────────

BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom"

FILES = {
    "agaricus-lepiota.data": "Données (8 124 spécimens, 23 champs)",
    "agaricus-lepiota.names": "Documentation officielle UCI",
}


# ── Pipeline ───────────────────────────────────────────────

def download_mushroom_dataset() -> None:
    """Télécharge le dataset mushroom depuis le UCI ML Repository."""
    raw_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  00 — Téléchargement du dataset UCI Mushroom")
    print("=" * 60)
    print()

    for filename, description in FILES.items():
        filepath = raw_dir / filename
        if filepath.exists():
            print(f"  [skip] {filename} (déjà présent)")
        else:
            url = f"{BASE_URL}/{filename}"
            print(f"  [download] {filename} — {description}")
            urllib.request.urlretrieve(url, filepath)
            print(f"  [ok]   Sauvegardé : {filepath}")

    print()
    print(f"  Fichiers disponibles dans : {raw_dir}")
    print()


if __name__ == "__main__":
    download_mushroom_dataset()
