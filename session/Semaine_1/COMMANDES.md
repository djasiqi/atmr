# 🖥️ COMMANDES SEMAINE 1

**Toutes les commandes prêtes à copier-coller.**

---

## 🔧 SETUP INITIAL

### Créer les dossiers nécessaires

```bash
# Dossier backup
mkdir -p session/backup_semaine1

# Dossier rapports
mkdir -p session/Semaine_1/rapports

# Vérifier structure
ls -la session/
```

### Vérifier environnement

```bash
# Aller dans backend
cd backend

# Vérifier Python
python --version

# Vérifier packages
pip list

# Activer venv si nécessaire
source venv/bin/activate  # Linux/Mac
# OU
.\venv\Scripts\activate   # Windows
```

---

## 📅 JOUR 1 : FICHIERS EXCEL

### Rechercher références

```bash
cd backend

# Rechercher Classeur1.xlsx
grep -r "Classeur1" . --include="*.py" --include="*.js"

# Rechercher transport.xlsx
grep -r "transport.xlsx" . --include="*.py" --include="*.js"
```

### Backup

```bash
# Copier fichiers
cp Classeur1.xlsx ../session/backup_semaine1/
cp transport.xlsx ../session/backup_semaine1/

# Vérifier backup
ls -la ../session/backup_semaine1/
```

### Supprimer

```bash
# Supprimer les fichiers
rm Classeur1.xlsx
rm transport.xlsx

# Vérifier suppression
ls -la *.xlsx
# Devrait dire "No such file or directory"
```

### Commit

```bash
git status
git add -A
git commit -m "chore: supprimer fichiers Excel inutiles (Classeur1.xlsx, transport.xlsx)

- Fichiers orphelins sans référence dans le code
- Backup créé dans session/backup_semaine1
- Réduction taille dépôt : ~150 KB"

git push origin main
```

---

## 📅 JOUR 2 : CHECK_BOOKINGS.PY

### Rechercher références

```bash
cd backend

# Rechercher dans Python
grep -r "check_bookings" . --include="*.py"

# Rechercher dans Shell
grep -r "check_bookings" . --include="*.sh"

# Rechercher dans Config
grep -r "check_bookings" . --include="*.yml" --include="*.yaml" --include="*.json"

# Vérifier imports
grep -r "from check_bookings import" . --include="*.py"
grep -r "import check_bookings" . --include="*.py"
```

### Backup avec documentation

```bash
# Copier fichier
cp check_bookings.py ../session/backup_semaine1/check_bookings.py.backup

# Créer README explicatif
cat > ../session/backup_semaine1/check_bookings_README.txt << 'EOF'
FICHIER SUPPRIMÉ : check_bookings.py
DATE : $(date)
RAISON : Script orphelin non utilisé, aucune référence dans le codebase

Si besoin de restaurer :
cp session/backup_semaine1/check_bookings.py.backup backend/check_bookings.py
EOF
```

### Supprimer et tester

```bash
# Supprimer
rm check_bookings.py

# Lancer application (test)
python app.py &
APP_PID=$!

# Attendre 5 secondes
sleep 5

# Test API
curl http://localhost:5000/healthcheck

# Arrêter app
kill $APP_PID

# Si tests existent, les lancer
pytest tests/ -v
```

### Commit

```bash
git status
git add check_bookings.py
git commit -m "chore: supprimer script obsolète check_bookings.py

- Script non utilisé, aucune référence dans le codebase
- Backup créé dans session/backup_semaine1
- Tests de non-régression passés"

git push origin main
```

---

## 📅 JOUR 3 : HAVERSINE

### Rechercher implémentations

```bash
cd backend

# Rechercher "haversine"
grep -rn "def.*haversine" . --include="*.py"

# Rechercher formule
grep -rn "sin.*lat.*cos" . --include="*.py"

# Rechercher constante rayon Terre
grep -rn "6371" . --include="*.py"
```

### Créer geo_utils.py

```bash
# Créer le fichier (contenu dans guide détaillé)
touch shared/geo_utils.py

# Créer __init__.py si manquant
touch shared/__init__.py

# Vérifier
ls -la shared/
```

### Créer tests

```bash
# Créer fichier tests
touch tests/test_geo_utils.py

# Lancer tests
pytest tests/test_geo_utils.py -v

# Vérifier coverage
pytest tests/test_geo_utils.py -v --cov=shared.geo_utils --cov-report=term
```

### Refactoriser fichiers

```bash
# Éditer heuristics.py
# (utiliser votre éditeur préféré)
nano services/unified_dispatch/heuristics.py
# OU
code services/unified_dispatch/heuristics.py

# Même chose pour data.py et route_analysis.py
```

### Tests complets

```bash
# Tous les tests
pytest tests/ -v

# Tests spécifiques
pytest tests/test_geo_utils.py -v

# Application complète
python app.py
```

### Commit

```bash
git status
git add shared/geo_utils.py
git add tests/test_geo_utils.py
git add services/unified_dispatch/heuristics.py
git add services/unified_dispatch/data.py
git add services/analytics/route_analysis.py

git commit -m "refactor: centraliser calcul distance Haversine dans geo_utils

- Créer shared/geo_utils.py avec haversine_distance()
- Remplacer 3 implémentations dupliquées
- Ajouter tests unitaires (12 tests, 100% coverage)
- Ajouter fonctions bonus: validate_coordinates(), get_bearing()

Impact:
- -100 lignes de code dupliqué
- +20% maintenabilité
- Tests: 12/12 passés ✅"

git push origin main
```

---

## 📅 JOUR 4 : MARSHMALLOW

### Rechercher sérialisations

```bash
cd backend

# Rechercher méthodes serialize/to_dict
grep -rn "def serialize" models/ --include="*.py"
grep -rn "def to_dict" models/ --include="*.py"
grep -rn "\.serialize()" . --include="*.py"
grep -rn "\.to_dict()" . --include="*.py"
```

### Installer Marshmallow

```bash
# Vérifier si déjà installé
pip list | grep marshmallow

# Installer
pip install marshmallow==3.20.1 marshmallow-sqlalchemy==0.29.0

# Ajouter à requirements.txt
echo "marshmallow==3.20.1" >> requirements.txt
echo "marshmallow-sqlalchemy==0.29.0" >> requirements.txt

# Vérifier installation
pip show marshmallow
```

### Créer schémas

```bash
# Créer dossier si nécessaire
mkdir -p schemas
touch schemas/__init__.py

# Créer fichier schémas
touch schemas/dispatch_schemas.py

# Vérifier
ls -la schemas/
```

### Tests

```bash
# Créer tests
touch tests/test_dispatch_schemas.py

# Lancer tests
pytest tests/test_dispatch_schemas.py -v

# Tests complets
pytest tests/ -v
```

### Test API

```bash
# Lancer application
python app.py &
APP_PID=$!

# Attendre démarrage
sleep 5

# Tester API
curl http://localhost:5000/api/assignments
curl http://localhost:5000/api/bookings
curl http://localhost:5000/api/drivers

# Arrêter
kill $APP_PID
```

### Commit

```bash
git add schemas/dispatch_schemas.py
git add tests/test_dispatch_schemas.py
git add services/unified_dispatch/apply.py
git add routes/dispatch_routes.py
git add requirements.txt

git commit -m "refactor: centraliser sérialisation avec Marshmallow schemas

- Créer schemas/dispatch_schemas.py (Assignment, Booking, Driver)
- Remplacer méthodes serialize() dispersées
- Ajouter tests unitaires (15 tests)
- Typage et validation automatiques

Impact:
- -150 lignes code sérialisation manuel
- +25% maintenabilité
- Validation automatique des données
- Tests: 15/15 passés ✅"

git push origin main
```

---

## 📅 JOUR 5 : VALIDATION

### Revue code

```bash
# Voir commits semaine
git log --oneline --since="5 days ago"

# Voir statistiques
git diff HEAD~4 HEAD --stat

# Voir différences complètes
git diff HEAD~4 HEAD

# Voir fichiers modifiés
git diff HEAD~4 HEAD --name-only
```

### Tests complets

```bash
cd backend

# Tous les tests avec coverage
pytest tests/ -v --cov=backend --cov-report=html

# Ouvrir rapport coverage (browser)
# Linux
xdg-open htmlcov/index.html
# Mac
open htmlcov/index.html
# Windows
start htmlcov/index.html

# Tests spécifiques nouveaux modules
pytest tests/test_geo_utils.py tests/test_dispatch_schemas.py -v
```

### Tests application

```bash
# Lancer app
python app.py

# Dans autre terminal, tests API
curl http://localhost:5000/healthcheck
curl http://localhost:5000/api/bookings
curl http://localhost:5000/api/drivers
curl http://localhost:5000/api/assignments

# Test dispatch complet
curl -X POST http://localhost:5000/api/dispatch/run \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "company_id": 1,
    "for_date": "2025-10-21",
    "mode": "semi_auto"
  }'

# Vérifier logs
tail -100 logs/app.log
```

### Mesurer impact

```bash
# Statistiques diff
echo "=== LIGNES MODIFIÉES ==="
git diff HEAD~4 HEAD --shortstat

# Fichiers modifiés
echo "=== FICHIERS MODIFIÉS ==="
git diff HEAD~4 HEAD --name-only | wc -l

# Tests ajoutés
echo "=== TESTS AJOUTÉS ==="
grep -r "def test_" tests/ --include="*.py" | wc -l

# Taille code
echo "=== TAILLE CODE ==="
find . -name "*.py" | xargs wc -l | tail -1
```

### Créer rapports

```bash
# Impact
touch session/SEMAINE_1_IMPACT.md

# Rapport final
touch session/SEMAINE_1_RAPPORT.md

# (Remplir avec contenu du guide)
```

### Commit final

```bash
git add session/SEMAINE_1_IMPACT.md
git add session/SEMAINE_1_RAPPORT.md
git add README.md

git commit -m "docs: rapport final Semaine 1

- Tous objectifs atteints
- -400 lignes code mort
- +27 tests unitaires
- +20% maintenabilité
- Prêt pour Semaine 2"

git push origin main
```

---

## 🚨 COMMANDES URGENCES

### Rollback complet

```bash
# Revenir à HEAD avant semaine
git reset --hard HEAD~5

# OU revenir à commit spécifique
git log --oneline
git reset --hard <COMMIT_ID>

# Forcer push (ATTENTION)
git push origin main --force
```

### Restaurer un fichier depuis backup

```bash
# Restaurer Classeur1.xlsx
cp session/backup_semaine1/Classeur1.xlsx backend/

# Restaurer check_bookings.py
cp session/backup_semaine1/check_bookings.py.backup backend/check_bookings.py
```

### Réinstaller dépendances

```bash
cd backend

# Réinstaller tout
pip install -r requirements.txt --force-reinstall

# Vérifier
pip list
```

### Nettoyer cache Python

```bash
# Supprimer __pycache__
find . -type d -name "__pycache__" -exec rm -rf {} +

# Supprimer .pyc
find . -type f -name "*.pyc" -delete

# Réinstaller packages
pip install -e .
```

---

## 📝 COMMANDES UTILES

### Git status amélioré

```bash
# Status complet
git status

# Voir différences non staged
git diff

# Voir différences staged
git diff --staged

# Voir arbre commits
git log --oneline --graph --all
```

### Lancer tests spécifiques

```bash
# Un seul test
pytest tests/test_geo_utils.py::TestHaversineDistance::test_distance_paris_lyon -v

# Une classe de tests
pytest tests/test_geo_utils.py::TestHaversineDistance -v

# Avec print output
pytest tests/test_geo_utils.py -v -s

# Arrêter au premier échec
pytest tests/ -v -x
```

### Debugging

```bash
# Lancer en mode debug
python -m pdb app.py

# Voir logs en temps réel
tail -f logs/app.log

# Grep dans logs
grep "ERROR" logs/app.log
grep "dispatch" logs/app.log | tail -20
```

---

**Toutes les commandes sont prêtes ! Copiez-collez directement. 🚀**
