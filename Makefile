# ──────────────────────────────────────────────────────────────
# The Mushroom Project — Makefile
# Pipeline d'analyse statistique multivariée
# ──────────────────────────────────────────────────────────────

PYTHON  := ./venv/bin/python
PIP     := ./venv/bin/pip
SRC     := src
VENV    := venv

.PHONY: help install run-all run-extended dashboard clean distclean

# ── Aide ────────────────────────────────────────────────────

help: ## Afficher cette aide
	@echo ""
	@echo "  The Mushroom Project — Commandes disponibles"
	@echo "  ─────────────────────────────────────────────"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*##"}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'
	@echo ""

# ── Installation ────────────────────────────────────────────

install: ## Créer l'environnement virtuel et installer les dépendances
	@echo "→ Création de l'environnement virtuel..."
	python3 -m venv $(VENV)
	@echo "→ Installation des dépendances..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "✓ Installation terminée."

# ── Exécution du pipeline ──────────────────────────────────

run-all: ## Exécuter l'intégralité du pipeline (00 → 07)
	@echo "═══ Pipeline complet ═══"
	$(PYTHON) $(SRC)/00_download.py
	$(PYTHON) $(SRC)/01_prepare.py
	$(PYTHON) $(SRC)/02_describe.py
	$(PYTHON) $(SRC)/03_mca.py
	$(PYTHON) $(SRC)/04_cluster.py
	$(PYTHON) $(SRC)/05_discriminant.py
	$(PYTHON) $(SRC)/06_sensitivity.py
	$(PYTHON) $(SRC)/07_model_comparison.py
	@echo ""
	@echo "✓ Pipeline complet terminé."

run-extended: ## Exécuter les analyses étendues (06 → 07 : Sensibilité, Comparaison)
	@echo "═══ Analyses étendues ═══"
	$(PYTHON) $(SRC)/06_sensitivity.py
	$(PYTHON) $(SRC)/07_model_comparison.py
	@echo ""
	@echo "✓ Analyses étendues terminées."

# ── Dashboard ─────────────────────────────────────────────

dashboard: ## Lancer le dashboard Streamlit interactif
	$(VENV)/bin/streamlit run app.py

# ── Nettoyage ──────────────────────────────────────────────

clean: ## Supprimer les outputs générés (figures, tables, données processées)
	@echo "→ Suppression des outputs..."
	rm -f reports/figures/*.png
	rm -f reports/tables/*.csv
	rm -f data/processed/*.csv
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Nettoyage terminé."

distclean: clean ## Nettoyage complet (outputs + environnement virtuel)
	@echo "→ Suppression de l'environnement virtuel..."
	rm -rf $(VENV)
	@echo "✓ Nettoyage complet terminé."
