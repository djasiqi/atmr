# Control plane fondations — CP-PR1

## Objectif

Projection diagnostique Legacy → Control plane pour les partenaires LIRIE.
**Aucune permission métier n'est accordée ou retirée** par le control plane
(`decision_mode=shadow`, entitlements `enforcement_mode=shadow`).

## ✅ Implémenté

### Modèle

- Tables : `platform_organization`, `organization_membership`, `service_catalog`,
  `permission_catalog`, `role_template`, `role_template_permission`,
  `organization_service_entitlement`, `control_plane_anomaly`,
  `control_plane_entity_override`
- Colonnes `data_origin*` sur `User`
- CHECK XOR company/institution ; FK `ON DELETE RESTRICT` vers legacy
- Migration `d0e04085600f` : crée le schéma **et retire** `uq_company_user_id`
  (incompatible avec les coquilles cliniques)

### Classification

- `classify_company_for_control_plane` :
  `TRANSPORT_TENANT | BILLING_SHELL | AMBIGUOUS` (fail-closed)
- Lifecycle dérivé (`legacy_derived`) entreprise / institution
- `data_origin` défaut `unknown` (jamais production automatique)

### Projecteur + reconcile

- `ControlPlaneProjector` upsert concurrent-safe (`ON CONFLICT`)
- Hooks invitation institution + création chauffeur
- CLI : `flask control-plane seed|reconcile|backfill`
- Anomalies persistées (fingerprint)

### API admin

- `GET /admin/partners/organizations` — mode
  `CONTROL_PLANE_ORGANIZATIONS_READ_MODE` ∈
  `{legacy, compare, control_plane}`
- `GET /admin/organizations/<public_id>`
- `GET /admin/control-plane/anomalies`
- `GET /admin/accounts/<id>/effective-access` (`permissions_enforced: []`)
- Caps `admin.organizations.read` / `admin.accounts.read` + alias temporaire
  via `admin.partners.read`

### Auto-create Company

- Cut : `get_company_from_token`, `require_company` →
  `409 company_profile_missing` / `CP-COMPANY-PROFILE-MISSING`
- Clinic / démo : non touchés

### Frontend

- Organisations : KPI production, prestations détectées, fiche
  `/partners/organizations/:publicId`
- Comptes : onglet Anomalies + bandeau « À traiter »

## Ops

```bash
docker compose exec -e DISABLE_EVENTLET=1 atmr_api flask control-plane seed
docker compose exec -e DISABLE_EVENTLET=1 atmr_api flask control-plane backfill
docker compose exec -e DISABLE_EVENTLET=1 atmr_api flask control-plane reconcile --dry-run
docker compose exec -e DISABLE_EVENTLET=1 atmr_api flask control-plane reconcile --apply
```

Variable : `CONTROL_PLANE_ORGANIZATIONS_READ_MODE=legacy|compare|control_plane`

## Hors scope (CP-PR2+)

Entitlements `enforced`, UI prestations, memberships admin, sécurité comptes,
assistance, UNIQUE `company.user_id` (après séparation clinic shells).
