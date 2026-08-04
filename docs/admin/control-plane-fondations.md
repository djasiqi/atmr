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
- Migration `d0e04085600f` (révise `e4f273565844`) : crée le schéma CP
  **sans** jamais créer/retirer `uq_company_user_id` (coquilles cliniques)

### Classification

- `classify_company_for_control_plane` :
  `TRANSPORT_TENANT | BILLING_SHELL | AMBIGUOUS` (fail-closed)
- Shell clinic certaine uniquement si `clinic + 0 driver + owner non-COMPANY` ;
  owner COMPANY → `AMBIGUOUS`
- Lifecycle dérivé (`legacy_derived`) entreprise / institution
- `data_origin` défaut `unknown` (jamais production automatique)

### Projecteur + reconcile

- `ControlPlaneProjector` upsert concurrent-safe (`ON CONFLICT`)
- Hooks transactionnels : invitation institution, création chauffeur
  (échec → rollback), mutation `institution_role`, disable/archive user
- Hooks Company tenant (démo / seed) via `legacy_hooks.project_company_tenant`
- `sync_user_role_transition` : nettoie memberships `legacy_sync` invalides
  (**sauf** `company_owner`) puis re-projette selon le nouveau contexte
- CLI : `flask control-plane seed|reconcile|backfill|cutover-status`
- Anomalies persistées (fingerprint)
- Gate cutover : `assert_control_plane_read_cutover_ready()` avant
  `read_mode=control_plane`

### API admin

- `GET /admin/partners/organizations` — mode
  `CONTROL_PLANE_ORGANIZATIONS_READ_MODE` ∈
  `{legacy, compare, control_plane}`
- Mode legacy : filtre organisations réelles **en SQL** (avant COUNT/KPI)
- Mode CP : recherche textuelle **en SQL** (avant pagination)
- Readiness fiche : owner/admin actifs + contact + pas de conflit critique
- `GET /admin/organizations/<public_id>`
- `GET /admin/control-plane/anomalies`
- `GET /admin/accounts/<id>/effective-access` (`permissions_enforced: []`)
  — `invited` / sans membership → `blocked`
- Caps `admin.organizations.read` / `admin.accounts.read` : alias
  `admin.partners.read` développé aussi dans `capabilities_effective`

### Gestion compte (drawer) — transitions sécurisées

✅ **Implémenté** : service dédié (pas de restore de l’ancien `UpdateUserRole`)

- `AdminAccountRoleTransitionService.preview` / `.apply`
  (`backend/services/admin_account_role_transition.py`)
- Routes :
  - `POST /admin/users/<id>/role-transition/preview`
  - `PUT /admin/users/<id>/role` (reason + `expected_*`, un commit)
  - `GET /admin/accounts/<id>/manage-context`
- Reset MDP : `AdminResetPasswordSchema(reason)`, `secrets`, rate limit
  `10/h`, revoke fail-closed, `force_password_change` + expiry 24h
- Projection CP + revoke sessions + `AuditLog` dans la **même** transaction
- UI : `AdminAccountManageDrawer` (Identité | Profil chauffeur | Sécurité |
  Rôle | Accès commercial COMPANY | Diagnostic)

### Support chauffeur (ops)

✅ **Implémenté** :

- `driver_profile` + `allowed_actions.change_driver_status` /
  `revoke_sessions` dans manage-context
- `PUT /admin/users/<id>/driver-status` : soft-disable / réactivation sans
  changer le rôle ; `expected_is_active` ; no-op idempotent ; revoke
  transactionnel (`commit=False` + `fail_closed`) ; `sync_driver` suspend
  la membership `company_driver` si `Driver.is_active=false`
- `POST /admin/users/<id>/revoke-sessions` : tous rôles, cap security
- Liste AdminUsers : `company_name` / `driver_id` / `driver_is_active` pour
  DRIVER **sans** champs billing
- Réactivation : ne force **pas** `is_available=true`

Hors scope chauffeur : toggle `is_available` admin, accès commercial,
`account_status` global, hard-delete.

**Ownership Company (hors scope → CP-PR3)** :

- `→ COMPANY` : tenant unique **ou** `company_id` explicite déjà possédé
  (`TRANSPORT_TENANT`) ; sinon `409 company_owner_assignment_required`
- `COMPANY → *` si ownership active (membership `company_owner` ou owner
  legacy d’un tenant) → `409 company_ownership_transition_required`
- Pas de transfert / retrait ownership dans cette PR

**Caps temporaires (documentées — pas de nouvelles caps)** :

```text
CAP_BILLING_LOCK couvre provisoirement billing-access et dunning
manage + security couvrent provisoirement promotion/rétrogradation ADMIN
```

CRUD memberships admin UI et `permissions_enforced` restent hors scope
(sync dérivée legacy uniquement).

### Auto-create Company

- Cut : `get_company_from_token`, `require_company` →
  `409 company_profile_missing` / `CP-COMPANY-PROFILE-MISSING`
- Clinic / démo : non touchés

### Frontend

- Organisations : KPI production, prestations détectées, fiche
  `/partners/organizations/:publicId`
- Comptes : onglet Anomalies + bandeau « À traiter »
- Drawer gestion compte (caps + preview + refresh liste)

## Ops

```bash
docker compose exec -e DISABLE_EVENTLET=1 atmr_api flask control-plane seed
docker compose exec -e DISABLE_EVENTLET=1 atmr_api flask control-plane backfill
docker compose exec -e DISABLE_EVENTLET=1 atmr_api flask control-plane reconcile --dry-run
docker compose exec -e DISABLE_EVENTLET=1 atmr_api flask control-plane reconcile --apply
docker compose exec -e DISABLE_EVENTLET=1 atmr_api flask control-plane cutover-status
```

Variable : `CONTROL_PLANE_ORGANIZATIONS_READ_MODE=legacy|compare|control_plane`

Ne passer en `control_plane` que si `cutover-status` retourne `ready=true`
(exit 0).

## Hors scope (CP-PR2+)

Entitlements `enforced`, UI prestations, memberships admin CRUD,
assistance ownership Company (CP-PR3), UNIQUE `company.user_id`
(après séparation clinic shells), disable/enable admin global,
suppression physique user, email reset MDP, nouvelles caps billing /
promote_admin, enforcement `permissions_enforced`.
