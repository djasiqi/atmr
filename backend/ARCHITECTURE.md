# Architecture du Système ATMR

## Vue d'ensemble

Le système ATMR (Emmenez-Moi) est une plateforme de dispatch médical intelligente qui combine des algorithmes heuristiques traditionnels avec des techniques d'apprentissage par renforcement avancées pour optimiser l'attribution des chauffeurs aux réservations médicales.

## Architecture Générale

### 1. Vue d'Ensemble du Système

```
┌─────────────────────────────────────────────────────────────┐
│                    SYSTÈME ATMR                             │
├─────────────────────────────────────────────────────────────┤
│  Frontend (React)  │  Mobile Apps  │  Dashboard Admin      │
├─────────────────────────────────────────────────────────────┤
│                    API Gateway (Flask)                      │
├─────────────────────────────────────────────────────────────┤
│  Services Layer                                             │
│  ├─ Unified Dispatch  ├─ RL System  ├─ Monitoring          │
│  ├─ Proactive Alerts  ├─ Shadow Mode ├─ MLOps             │
├─────────────────────────────────────────────────────────────┤
│  Data Layer                                                │
│  ├─ PostgreSQL      ├─ Redis        ├─ File Storage        │
├─────────────────────────────────────────────────────────────┤
│  Infrastructure                                            │
│  ├─ Docker          ├─ Celery      ├─ OSRM               │
└─────────────────────────────────────────────────────────────┘
```

### 2. Composants Principaux

#### Frontend
- **React Web App** : Interface utilisateur principale
- **Mobile Apps** : Applications chauffeur et client
- **Dashboard Admin** : Interface d'administration

#### Backend
- **Flask API** : Serveur principal avec Flask-RESTx
- **Celery Workers** : Traitement asynchrone
- **Socket.IO** : Communication temps réel

#### Services
- **Unified Dispatch** : Orchestrateur principal
- **RL System** : Système d'apprentissage par renforcement
- **Proactive Alerts** : Alertes prédictives
- **Shadow Mode** : Mode de comparaison
- **MLOps** : Gestion des modèles ML

## Architecture des Services

### 1. Service de Dispatch Unifié

```
services/unified_dispatch/
├── dispatch_manager.py      # Orchestrateur principal
├── rl_optimizer.py         # Intégration RL
├── heuristic_solver.py     # Algorithmes heuristiques
├── ml_predictor.py         # Prédictions ML
└── fallback_manager.py     # Gestion des fallbacks
```

**Responsabilités** :
- Orchestration des algorithmes de dispatch
- Intégration RL/heuristiques
- Gestion des fallbacks
- Validation des solutions

### 2. Système RL

```
services/rl/
├── improved_dqn_agent.py   # Agent DQN avancé
├── improved_q_network.py   # Architectures de réseaux
├── dispatch_env.py         # Environnement Gymnasium
├── replay_buffer.py        # Buffer d'expérience
├── n_step_buffer.py        # Buffer N-step
├── reward_shaping.py       # Fonction de récompense
├── hyperparameter_tuner.py # Optimisation Optuna
├── shadow_mode_manager.py  # Mode de comparaison
├── noisy_networks.py       # Exploration paramétrique
└── distributional_dqn.py   # RL distributionnel
```

**Techniques Avancées** :
- Prioritized Experience Replay (PER)
- Double DQN
- Dueling DQN
- N-step Learning
- Noisy Networks
- Distributional RL (C51/QR-DQN)

### 3. Système d'Alertes Proactives

```
services/
├── proactive_alerts.py     # Service principal
├── notification_service.py # Notifications
└── ml_predictor.py         # Prédictions de retard
```

**Fonctionnalités** :
- Prédiction des retards
- Alertes en temps réel
- Système de debounce
- Explicabilité des décisions

### 4. Système MLOps

```
services/ml/
├── model_registry.py       # Registre des modèles
└── training_metadata_schema.py # Schéma des métadonnées

scripts/ml/
└── train_model.py          # Orchestrateur d'entraînement

scripts/rl/
└── rl_train_offline.py     # Entraînement RL
```

**Fonctionnalités** :
- Versioning des modèles
- Promotion contrôlée
- Rollback automatique
- Traçabilité complète

## Architecture des Données

### 1. Base de Données PostgreSQL

#### Tables Principales
```sql
-- Réservations médicales
bookings (
    id, company_id, patient_name, pickup_address, 
    destination_address, scheduled_time, status, 
    created_at, updated_at
)

-- Chauffeurs
drivers (
    id, company_id, name, phone, vehicle_type,
    current_location, status, availability
)

-- Assignations
assignments (
    id, booking_id, driver_id, assigned_at,
    estimated_pickup_time, estimated_arrival_time,
    actual_pickup_time, actual_arrival_time
)

-- Métriques RL
rl_suggestion_metric (
    id, company_id, state_features, action_taken,
    q_values, reward, constraints, latency,
    created_at
)
```

### 2. Cache Redis

#### Utilisation
- **Sessions utilisateur** : Authentification
- **Cache de routes** : Optimisation OSRM
- **État RL** : Buffer d'expérience temporaire
- **Métriques temps réel** : Dashboard

### 3. Stockage de Fichiers

#### Structure
```
data/
├── ml/
│   ├── models/             # Modèles entraînés
│   ├── scalers/            # Scalers de features
│   └── metadata/           # Métadonnées
├── rl/
│   ├── checkpoints/        # Checkpoints d'entraînement
│   ├── logs/              # Logs TensorBoard
│   └── evaluations/       # Résultats d'évaluation
└── uploads/               # Fichiers utilisateur
```

## Architecture de Communication

### 1. API REST

#### Endpoints Principaux
```
POST /api/dispatch/optimize     # Optimisation dispatch
GET  /api/alerts/delay-risk    # Alertes de retard
POST /api/rl/suggest           # Suggestions RL
GET  /api/shadow-mode/metrics  # Métriques shadow mode
```

#### Authentification
- **JWT Tokens** : Authentification stateless
- **RBAC** : Contrôle d'accès basé sur les rôles
- **Rate Limiting** : Protection contre les abus

### 2. WebSocket (Socket.IO)

#### Événements Temps Réel
```javascript
// Alertes proactives
socket.on('alert_delay_risk', (data) => {
    // Notification de risque de retard
});

// Mises à jour de statut
socket.on('driver_status_update', (data) => {
    // Mise à jour position chauffeur
});

// Suggestions RL
socket.on('rl_suggestion', (data) => {
    // Nouvelle suggestion d'attribution
});
```

### 3. Communication Asynchrone

#### Celery Tasks
```python
# Entraînement RL asynchrone
@celery.task
def train_rl_model_task(model_name, episodes, hyperparameters):
    # Entraînement en arrière-plan
    pass

# Génération de suggestions
@celery.task
def generate_rl_suggestion_task(company_id, booking_id, state):
    # Calcul de suggestion RL
    pass
```

## Architecture de Déploiement

### 1. Docker Compose

#### Services
```yaml
services:
  postgres:     # Base de données
  redis:        # Cache et message broker
  api:          # Serveur Flask principal
  celery-worker: # Workers Celery
  celery-beat:  # Scheduler Celery
  flower:       # Monitoring Celery
  osrm:         # Service de routage
```

#### Configuration Production
- **Multi-stage builds** : Images optimisées
- **Non-root users** : Sécurité renforcée
- **Health checks** : Monitoring automatique
- **Resource limits** : Contrôle des ressources

### 2. Orchestration

#### Docker Compose Production
```yaml
# docker-compose.production.yml
version: '3.8'
services:
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: atmr_prod
      POSTGRES_USER: atmr
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U atmr"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## Architecture de Monitoring

### 1. Métriques Système

#### Prometheus Metrics
```python
# Métriques personnalisées
dispatch_requests_total = Counter('dispatch_requests_total')
dispatch_duration_seconds = Histogram('dispatch_duration_seconds')
rl_training_episodes = Counter('rl_training_episodes')
rl_reward_gauge = Gauge('rl_reward_current')
```

#### Dashboards Grafana
- **Performance API** : Latence, throughput, erreurs
- **Métriques RL** : Reward, loss, exploration
- **Métriques Business** : Ponctualité, distance, satisfaction

### 2. Logging Structuré

#### Format JSON
```json
{
  "timestamp": "2025-01-01T12:00:00Z",
  "level": "INFO",
  "service": "dispatch_manager",
  "message": "Dispatch optimization completed",
  "metrics": {
    "processing_time_ms": 150,
    "bookings_processed": 25,
    "drivers_available": 12,
    "rl_confidence": 0.85
  }
}
```

### 3. Alerting

#### Seuils d'Alerte
- **Latence API** : > 2s
- **Taux d'erreur** : > 5%
- **Disponibilité** : < 99%
- **Performance RL** : Reward < seuil

## Architecture de Sécurité

### 1. Authentification et Autorisation

#### JWT Tokens
```python
# Structure du token
{
  "user_id": 123,
  "company_id": 456,
  "roles": ["dispatcher", "admin"],
  "permissions": ["read_bookings", "create_assignments"],
  "exp": 1640995200
}
```

#### RBAC (Role-Based Access Control)
- **Admin** : Accès complet
- **Dispatcher** : Gestion des réservations
- **Driver** : Consultation des assignations
- **Client** : Consultation des réservations

### 2. Protection des Données

#### Chiffrement
- **TLS 1.3** : Communication chiffrée
- **AES-256** : Chiffrement des données sensibles
- **bcrypt** : Hachage des mots de passe

#### Masquage PII
```python
# Exemple de masquage
original_data = {
    "patient_name": "John Doe",
    "patient_phone": "+33123456789"
}

masked_data = {
    "patient_name": "J*** D**",
    "patient_phone": "+33***56789"
}
```

### 3. Conformité RGPD

#### Principes Appliqués
- **Minimisation** : Collecte minimale de données
- **Transparence** : Information claire des utilisateurs
- **Droit à l'oubli** : Suppression des données
- **Portabilité** : Export des données

## Architecture de Scalabilité

### 1. Scalabilité Horizontale

#### Load Balancing
```nginx
upstream api_backend {
    server api1:5000;
    server api2:5000;
    server api3:5000;
}

server {
    listen 80;
    location / {
        proxy_pass http://api_backend;
    }
}
```

#### Scaling Celery
```python
# Configuration pour scaling
CELERY_WORKER_CONCURRENCY = 4
CELERY_WORKER_PREFETCH_MULTIPLIER = 1
CELERY_TASK_ROUTES = {
    'heavy_tasks': {'queue': 'heavy'},
    'light_tasks': {'queue': 'light'},
}
```

### 2. Optimisations de Performance

#### Cache Stratégies
- **L1 Cache** : Données fréquemment accédées
- **L2 Cache** : Résultats de calculs coûteux
- **CDN** : Assets statiques

#### Optimisations Base de Données
```sql
-- Index optimisés
CREATE INDEX idx_bookings_company_time ON bookings(company_id, scheduled_time);
CREATE INDEX idx_drivers_location ON drivers USING GIST(current_location);
CREATE INDEX idx_assignments_status ON assignments(status, created_at);
```

## Architecture de Tests

### 1. Tests Unitaires

#### Couverture Cible
- **Modules RL** : ≥ 85%
- **Services critiques** : ≥ 90%
- **Global** : ≥ 85%

#### Structure des Tests
```
tests/
├── unit/
│   ├── rl/                 # Tests RL
│   ├── services/           # Tests services
│   └── models/             # Tests modèles
├── integration/
│   ├── api/                # Tests API
│   ├── celery/             # Tests Celery
│   └── database/           # Tests DB
└── e2e/
    ├── dispatch/           # Tests end-to-end
    └── alerts/             # Tests alertes
```

### 2. Tests de Performance

#### Benchmarks
- **Latence API** : < 500ms (P95)
- **Throughput** : > 1000 req/s
- **Mémoire** : < 2GB par instance
- **CPU** : < 80% utilisation

## Conclusion

L'architecture du système ATMR est conçue pour :

- **Robustesse** : Fallbacks multiples, monitoring complet
- **Performance** : Cache intelligent, optimisations DB
- **Scalabilité** : Architecture microservices, load balancing
- **Sécurité** : Chiffrement, RBAC, conformité RGPD
- **Maintenabilité** : Code modulaire, tests complets
- **Évolutivité** : Architecture extensible, MLOps intégré

Cette architecture permet au système d'évoluer avec les besoins métier tout en maintenant la qualité de service et la sécurité des données.
