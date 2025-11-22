# Guide Qualité Code ATMR

Ce document décrit les outils et pratiques de qualité de code utilisés dans le projet ATMR.

## Outils de Qualité

### Ruff

**Ruff** est un linter et formateur Python rapide qui remplace plusieurs outils (flake8, isort, black, etc.).

**Installation** :

```bash
pip install ruff
```

**Utilisation** :

```bash
# Formater le code
cd backend
ruff format .

# Vérifier le formatage (sans modifier)
ruff format --check .

# Linter le code
ruff check .

# Linter avec format GitHub Actions
ruff check . --output-format=github
```

**Configuration** : Voir `pyproject.toml` ou `.ruff.toml` si présent.

### Flake8

**Flake8** est utilisé pour la vérification de conformité PEP8.

**Installation** :

```bash
pip install flake8
```

**Utilisation** :

```bash
cd backend
flake8 .
```

**Règles principales** :

- E501 : Ligne trop longue (>120 caractères)
- E402 : Import pas en haut de fichier
- F401 : Import inutilisé
- W291/W293 : Whitespace trailing/blank lines
- F841 : Variable assignée mais inutilisée

### Autoflake

**Autoflake** supprime automatiquement les imports et variables inutilisés.

**Installation** :

```bash
pip install autoflake
```

**Utilisation** :

```bash
cd backend
# Prévisualiser les changements
autoflake --remove-all-unused-imports --recursive --check backend/

# Appliquer les changements
autoflake --in-place --remove-all-unused-imports --recursive backend/
```

## Pre-commit Hooks

### Installation

```bash
# Installer pre-commit
pip install pre-commit

# Installer les hooks
pre-commit install

# Tester les hooks sur tous les fichiers
pre-commit run --all-files
```

### Configuration

Le fichier `.pre-commit-config.yaml` configure les hooks suivants :

- **ruff** : Linting et formatage
- **autoflake** : Suppression imports/variables inutilisés
- **isort** : Tri des imports
- **trailing-whitespace** : Suppression whitespace trailing
- **end-of-file-fixer** : Fix ligne de fin de fichier
- **check-yaml** : Vérification syntaxe YAML
- **check-added-large-files** : Vérification taille fichiers

### Utilisation

Les hooks s'exécutent automatiquement avant chaque commit. Pour exécuter manuellement :

```bash
# Sur les fichiers modifiés (staged)
pre-commit run

# Sur tous les fichiers
pre-commit run --all-files

# Sur un hook spécifique
pre-commit run ruff --all-files
```

## Workflow CI/CD

Le workflow GitHub Actions (`.github/workflows/backend-tests.yml`) exécute les outils dans l'ordre suivant :

1. **Ruff format** : Formate automatiquement le code
2. **Ruff lint** : Vérifie les erreurs de linting
3. **Flake8 check** : Vérifie la conformité PEP8
4. **MyPy** : Vérification de types
5. **Vulture** : Détection de code mort
6. **Tests pytest** : Exécution des tests

## Règles de Formatage

### Longueur de ligne

- **Limite** : 120 caractères
- **Outils** : Ruff format, Flake8 E501

### Imports

- **Ordre** : Imports standard → imports tiers → imports locaux
- **Outils** : isort (via ruff)
- **Exception E402** : Imports après `sys.path.insert()` ou configuration environnement (ajouter `# noqa: E402`)

### Whitespace

- **Pas de whitespace trailing** : Supprimé automatiquement par ruff format
- **Lignes vides** : Pas de whitespace dans les lignes vides

## Patterns SQLAlchemy

### Utilisation correcte de Flask-SQLAlchemy

**Pattern correct** :

```python
# db est l'instance Flask-SQLAlchemy
db.session.add(obj)
db.session.commit()
db.session.query(Model).filter_by(...).first()
```

**Pattern incorrect** :

```python
# ❌ ERREUR : db n'a pas de méthode add() directe
db.add(obj)  # AttributeError: add
```

**Dans persisted_fixture()** :

```python
# ✅ CORRECT : Passer l'instance Flask-SQLAlchemy
company = persisted_fixture(db, CompanyFactory(), Company)

# ❌ INCORRECT : Ne pas passer db.session
# company = persisted_fixture(db.session, CompanyFactory(), Company)
```

Voir `backend/tests/README_FIXTURES.md` pour plus de détails.

## Exceptions et NoQA

### Quand utiliser `# noqa`

Utilisez `# noqa` avec justification uniquement quand nécessaire :

```python
# noqa: E402 - Imports après sys.path.insert (nécessaire pour résoudre les imports)
from app import create_app

# noqa: F841 - Variable de debug (timing)
_start_time = time.time()

# noqa: F824 - counter est utilisé via counter[0] (assignation indirecte)
nonlocal counter
```

### Règles

- Toujours justifier l'utilisation de `# noqa`
- Préférer corriger le code plutôt que d'ajouter `# noqa`
- Documenter les exceptions dans les commentaires

## Bonnes Pratiques

### Avant de commiter

1. Exécuter `ruff format .` pour formater le code
2. Exécuter `ruff check .` pour vérifier les erreurs
3. Exécuter `flake8 .` pour vérifier la conformité PEP8
4. Exécuter les tests : `pytest backend/tests -v`

### Dans la CI/CD

- Le formatage est automatique (ruff format)
- Les erreurs de linting bloquent le pipeline
- Les tests doivent tous passer

### Maintenance

- Nettoyer régulièrement les imports inutilisés avec autoflake
- Vérifier les violations flake8 après chaque PR
- Maintenir la documentation à jour

## Ressources

- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [Flake8 Rules](https://www.flake8rules.com/)
- [PEP 8 Style Guide](https://pep8.org/)
- [Pre-commit Hooks](https://pre-commit.com/)

## Support

Pour toute question sur la qualité du code, consulter :

- `backend/tests/README_FIXTURES.md` : Patterns SQLAlchemy et fixtures
- `.pre-commit-config.yaml` : Configuration pre-commit
- `.github/workflows/backend-tests.yml` : Workflow CI/CD
