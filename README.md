# 🚗 ATMR - Système de Transport Médical

Application complète de gestion de transport médical avec dispatch automatique, planification et suivi en temps réel.

## 📋 Stack Technique

**Backend:**

- Flask (API REST)
- SQLAlchemy (ORM)
- Celery (tâches asynchrones)
- PostgreSQL (base de données)
- Redis (cache & broker)
- Socket.IO (temps réel)

**Frontend:**

- React 18
- Socket.IO Client
- Sentry (monitoring)

**Infrastructure:**

- Docker & Docker Compose
- OSRM (routing & optimisation)
- Nginx (reverse proxy - production)

## 🚀 Démarrage Rapide

### Prérequis

- Docker & Docker Compose
- Node.js 18+ (pour développement frontend)
- Python 3.11+ (pour développement backend)

### Installation

```bash
# 1. Cloner le projet
git clone <repo-url>
cd atmr

# 2. Configurer les variables d'environnement
# Éditer backend/.env et frontend/.env
# (Générer SECRET_KEY et JWT_SECRET_KEY si besoin)

# 3. Lancer avec Docker
docker-compose up -d

# 4. Vérifier que tout fonctionne
curl http://localhost:5000/health/detailed
```

### URLs de l'application

- **Frontend**: http://localhost:3000
- **API Backend**: http://localhost:5000
- **Flower (Celery)**: http://localhost:5555
- **OSRM**: http://localhost:5000/route/...

## 📁 Structure du Projet

```
atmr/
├── backend/               # API Flask + Celery
│   ├── models/           # Modèles SQLAlchemy
│   ├── routes/           # Endpoints API
│   ├── services/         # Logique métier
│   │   └── unified_dispatch/  # Système de dispatch
│   ├── tasks/            # Tâches Celery
│   └── migrations/       # Alembic
│
├── frontend/             # Application React
│   ├── src/
│   │   ├── components/   # Composants réutilisables
│   │   ├── pages/        # Pages de l'app
│   │   ├── services/     # Services API
│   │   └── utils/        # Utilitaires (logger, etc.)
│
├── mobile/               # Applications mobiles (React Native)
│   ├── client-app/       # App patient
│   └── driver-app/       # App chauffeur
│
├── scripts/              # Scripts utilitaires
│   ├── backup_db.sh      # Backup PostgreSQL
│   ├── restore_db.sh     # Restauration PostgreSQL
│   ├── test_backup_restore.sh  # Test backup/restore
│   └── smoke_api.sh      # Tests de smoke
│
└── docker-compose.yml    # Orchestration Docker
```

## 🔧 Développement

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

### Frontend

```bash
cd frontend
npm install
npm start
```

## 📊 Fonctionnalités Principales

### ✅ Dispatch Automatique

- Algorithme d'optimisation de routes
- Priorisation des chauffeurs réguliers
- Gestion des urgences
- Recalcul en temps réel

### 📅 Planification

- Gestion des réservations
- Courses aller-retour
- Retours avec heure à confirmer
- Validation médicale

### 🚨 Monitoring en Temps Réel

- Détection automatique des retards
- Suggestions de réassignation
- Notifications WebSocket
- Dashboard de suivi

### 📄 Facturation

- Génération automatique de factures
- QR-Bills Swiss (ISO 20022)
- Export PDF
- Suivi des paiements

## 🔐 Sécurité

- JWT pour l'authentification
- RBAC (Role-Based Access Control)
- Masquage des données sensibles (PII)
- HTTPS en production
- Rate limiting
- CORS configuré

## 📈 Monitoring & Observabilité

**Backend:**

- Healthcheck: `/health` et `/health/detailed`
- Logs structurés avec masquage PII
- Sentry pour tracking d'erreurs (optionnel)

**Frontend:**

- Sentry intégré (erreurs JS + performance)
- Web Vitals tracking
- Error Boundary avec fallback UI

**Configuration Sentry:**

```bash
# Backend
SENTRY_DSN=https://your-dsn@sentry.io/project

# Frontend
REACT_APP_SENTRY_DSN=https://your-dsn@sentry.io/project
```

## ⚠️ Chaos Engineering (Tests de Résilience)

**✅ D3: Système de tests de catastrophe pour valider la résilience.**

Le système inclut des injecteurs de chaos pour simuler des pannes (OSRM down, DB read-only, réseau flaky) et valider que le système reste opérationnel.

### ⚠️ ATTENTION : Ne JAMAIS activer en production !

Les variables d'environnement suivantes contrôlent le chaos :

```bash
# Désactivé par défaut (sécurité)
CHAOS_ENABLED=false          # Activer/désactiver chaos (défaut: false)
CHAOS_OSRM_DOWN=false        # Simuler OSRM down (défaut: false)
CHAOS_DB_READ_ONLY=false     # Simuler DB read-only (défaut: false)
```

### Utilisation en développement/test

Pour activer le chaos lors des tests E2E :

```bash
# Via variables d'environnement Docker
export CHAOS_ENABLED=true
export CHAOS_OSRM_DOWN=true
docker-compose restart api

# Via script (optionnel)
./backend/scripts/enable_chaos.sh
```

### Tests E2E de catastrophe

Les tests se trouvent dans `backend/tests/e2e/test_disaster_scenarios.py` :

```bash
# Lancer les tests de résilience
pytest backend/tests/e2e/test_disaster_scenarios.py -v
```

Voir `backend/RUNBOOK.md` pour les procédures de récupération et `backend/tests/e2e/TODO_D3.md` pour la liste complète des fonctionnalités.

## 🛠️ Scripts Utiles

```bash
# Backup base de données
./scripts/backup_db.sh

# Restaurer base de données
./scripts/restore_db.sh backups/latest.dump --force

# Tester backup/restore (complet)
./scripts/test_backup_restore.sh

# Tests de smoke API
./scripts/smoke_api.sh

# Performance tests (K6)
k6 run scripts/perf_quick.k6.js

# Migrations
cd backend
flask db upgrade        # Appliquer
flask db downgrade      # Rollback
```

## 📦 Production

### Déploiement

```bash
# 1. Configurer les variables d'environnement production
# 2. Build & push images Docker
# 3. Déployer avec docker-compose

docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### Checklist Go-Live

- [ ] Variables d'environnement configurées
- [ ] Secrets changés (SECRET_KEY, JWT_SECRET_KEY)
- [ ] Base de données backupée
- [ ] Migrations appliquées
- [ ] Sentry configuré
- [ ] HTTPS/SSL configuré
- [ ] Monitoring actif
- [ ] Tests de smoke passés

## 🤝 Contribution

1. Fork le projet
2. Créer une branche (`git checkout -b feature/amazing`)
3. Commit (`git commit -m 'Add amazing feature'`)
4. Push (`git push origin feature/amazing`)
5. Ouvrir une Pull Request

## 📝 License

Propriétaire - Tous droits réservés

## 📞 Support

Pour toute question ou problème, contacter l'équipe technique.

---

**Version:** 1.3.0  
**Dernière mise à jour:** 2025-10-15
