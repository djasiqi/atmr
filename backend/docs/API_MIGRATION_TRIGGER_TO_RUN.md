# Guide de Migration : `/trigger` vers `/run`

## 📋 Vue d'ensemble

L'endpoint `/company_dispatch/trigger` est **déprécié** et sera supprimé dans une future version.  
**Migration recommandée** : Utilisez `/company_dispatch/run` à la place.

## 🔄 Différences principales

### Ancien endpoint (déprécié)

```
POST /api/company_dispatch/trigger
```

### Nouveau endpoint (recommandé)

```
POST /api/v1/company_dispatch/run
```

## 📊 Comparaison des payloads

### Ancien format (`/trigger`)

```json
{
  "for_date": "2025-01-15",
  "regular_first": true,
  "allow_emergency": true
}
```

### Nouveau format (`/run`)

```json
{
  "for_date": "2025-01-15",
  "regular_first": true,
  "allow_emergency": true,
  "async": true, // Nouveau: contrôle async/sync
  "mode": "auto", // Nouveau: mode d'opération
  "overrides": {
    // Nouveau: surcharges paramètres
    "heuristic": {
      "driver_load_balance": 0.5
    },
    "fairness": {
      "fairness_weight": 0.8
    }
  }
}
```

## ✨ Nouvelles fonctionnalités

### 1. Mode synchrone/asynchrone

- **`async=true`** (défaut): Enfile un job Celery, retourne 202 avec `job_id`
- **`async=false`**: Exécute immédiatement, retourne 200 avec résultat complet
  - ⚠️ **Limité à <10 bookings** (sinon erreur 400)

### 2. Overrides de paramètres

Permet de surcharger les paramètres de dispatch sans modifier la configuration globale:

- `heuristic`: Poids heuristiques (proximity, driver_load_balance, etc.)
- `fairness`: Poids équité (fairness_weight)
- `solver`: Paramètres solver (time_limit_sec)
- `preferred_driver_id`: Chauffeur préféré
- `reset_existing`: Réinitialiser assignations existantes
- `fast_mode`: Mode rapide (solver désactivé)

### 3. Validation préalable

Nouvel endpoint pour valider les overrides avant exécution:

```
POST /api/v1/company_dispatch/settings/validate
{
  "overrides": {
    "heuristic": {"driver_load_balance": 0.5}
  }
}
```

## 🔧 Exemples de migration

### Migration simple

```javascript
// Avant
fetch("/api/company_dispatch/trigger", {
  method: "POST",
  body: JSON.stringify({
    for_date: "2025-01-15",
    regular_first: true,
  }),
});

// Après
fetch("/api/v1/company_dispatch/run", {
  method: "POST",
  body: JSON.stringify({
    for_date: "2025-01-15",
    regular_first: true,
    async: true, // Comportement identique à /trigger
  }),
});
```

### Migration avec overrides

```javascript
// Avant: Impossible de passer des overrides
fetch("/api/company_dispatch/trigger", {
  method: "POST",
  body: JSON.stringify({
    for_date: "2025-01-15",
  }),
});

// Après: Overrides disponibles
fetch("/api/v1/company_dispatch/run", {
  method: "POST",
  body: JSON.stringify({
    for_date: "2025-01-15",
    async: true,
    overrides: {
      heuristic: {
        driver_load_balance: 0.7,
        proximity: 0.2,
      },
      fairness: {
        fairness_weight: 0.9,
      },
      preferred_driver_id: 123,
    },
  }),
});
```

## 📝 Réponses

### Ancien endpoint (`/trigger`)

```json
{
  "job_id": "abc-123",
  "status": "queued"
}
```

### Nouveau endpoint (`/run`)

**Mode async (202)**:

```json
{
  "job_id": "abc-123",
  "dispatch_run_id": 456,
  "for_date": "2025-01-15",
  "status": "queued"
}
```

**Mode sync (200)**:

```json
{
  "assignments": [...],
  "unassigned": [...],
  "bookings": [...],
  "drivers": [...],
  "meta": {
    "quality_score": 85.5,
    "assignment_rate": 0.95
  },
  "dispatch_run_id": 456
}
```

## ⚠️ Breaking changes

1. **Path différent**: `/trigger` → `/run`
2. **Paramètre `async`**: Nouveau paramètre obligatoire (défaut: `true`)
3. **Limite mode sync**: Mode sync limité à <10 bookings
4. **Validation stricte**: Validation temporelle stricte activée par défaut

## 🚀 Plan de migration

### Phase 1: Migration progressive

1. Utiliser `/run` pour les nouveaux développements
2. Tester `/run` en parallèle de `/trigger`
3. Valider les overrides avec `/settings/validate`

### Phase 2: Migration complète

1. Migrer toutes les intégrations vers `/run`
2. Supprimer les appels à `/trigger`
3. `/trigger` sera supprimé dans une future version majeure

## 📚 Documentation complète

- Swagger UI: `/api/v1/docs`
- Endpoint validation: `POST /api/v1/company_dispatch/settings/validate`
- Endpoint health: `GET /api/v1/osrm/health`
