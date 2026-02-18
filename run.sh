#!/bin/bash
# Lance le pipeline complet puis le dashboard Streamlit.
# Utilisé par Streamlit Cloud pour générer data/ et reports/ avant de lancer l'app.
set -e
echo "→ Installation des dépendances..."
pip install -q -r requirements.txt
echo "→ Exécution du pipeline (00 → 07)..."
python src/00_download.py
python src/01_prepare.py
python src/02_describe.py
python src/03_mca.py
python src/04_cluster.py
python src/05_discriminant.py
python src/06_sensitivity.py
python src/07_model_comparison.py
echo "→ Lancement du dashboard..."
exec streamlit run app.py --server.headless true
