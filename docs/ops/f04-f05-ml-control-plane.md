# F-04 / F-05 — Fermeture du plan de contrôle ML public

**Statut** : ✅ **Implémenté** sur le code applicatif.  
**GO fusion** : si la matrice de tests sécurité est verte.  
**GO production global** : **non** — F-02 capacité, F-14 Alembic-first, résidu F-03 et P1/P2 restent bloquants.

## Garantie

Aucune route `/api/feature-flags/*`, `/api/shadow-mode/*` ou `/api/ml-monitoring/*` n’est exécutable anonymement. En production le plan est **désactivé** par défaut (`ML_CONTROL_PLANE_API_ENABLED=false` → HTTP 503 + `error=ml_control_plane_disabled`).

## Matrice d’autorisation

| Routes | Protection |
|--------|------------|
| 7 feature-flags | JWT + `UserRole.admin` + IP whitelist |
| Shadow status/stats/predictions/comparisons/health/companies/session | JWT + ADMIN + IP |
| GET daily / summary / metrics | JWT + `@role_required(["ADMIN", "COMPANY"])` + `assert_company_access` |
| POST daily | JWT + ADMIN + IP + tenant — **`log_decision_comparison` uniquement** |
| GET export | JWT + ADMIN + IP + tenant — `build_daily_report` sans persist |
| 5 ml-monitoring | JWT + ADMIN + IP |

Multirôle : **toujours** une liste `["ADMIN", "COMPANY"]` (pas d’arguments positionnels multiples à `role_required`).

## Kill-switch

Variable : `ML_CONTROL_PLANE_API_ENABLED`

- absent → activé (dev/tests)
- `true`/`1`/`yes` → activé
- `false`/`0`/`no` ou valeur invalide → **désactivé (fail-closed)**

Réponse :

```json
{"error": "ml_control_plane_disabled", "message": "ML control plane API is disabled"}
```

Enregistré dans `app.py` **avant** le middleware CSRF via  
`services/infrastructure/ml_control_plane.py`.

## Whitelist IP

- Client IP = `request.remote_addr` après `ProxyFix(x_for=1)` — pas de confiance en `X-Forwarded-For` brut.
- Fragment prod : `ADMIN_IP_WHITELIST_REQUIRED=true`
- Hôte : `ADMIN_IP_WHITELIST=<IP ou CIDR>` dans `/srv/atmr/.env.production.local`
- Boot : liste vide ou non parsable → `RuntimeError`

## Shadow lecture vs écriture

- `build_daily_report` : calcul pur (GET)
- `persist_daily_report` / `generate_daily_report` : jobs internes
- POST daily : conserve `log_decision_comparison` (écriture décision, pas persist rapport)

Swagger Shadow : retiré (routes Flask natives, pas de `/docs` ni `/swagger.json`).

## Frontend

[`frontend/src/hooks/useShadowMode.js`](../../frontend/src/hooks/useShadowMode.js) arrête le polling uniquement si  
`status === 503 && error === 'ml_control_plane_disabled'`.  
403/404 inchangés ; autres 503 = erreurs réelles.

## Dette connue

Flags ML toujours **in-memory** (non partagés entre workers). Hors scope de ce lot.

## Tests

```bash
docker exec atmr-atmr_api python -m pytest \
  tests/security/test_ml_control_plane_f04_f05.py \
  tests/security/test_feature_flags_f04.py \
  tests/test_feature_flags.py \
  tests/test_ml_monitoring.py \
  tests/test_shadow_mode.py \
  tests/security/test_ip_whitelist.py \
  -q --tb=line
```

## Fichiers clés

- `backend/routes/feature_flags_routes.py`
- `backend/routes/shadow_mode_routes.py`
- `backend/routes/ml_monitoring.py`
- `backend/services/infrastructure/ml_control_plane.py`
- `backend/security/ip_whitelist.py`
- `backend/services/ml/rl/shadow_mode_manager.py`
- `scripts/env.production.defaults.fragment`
