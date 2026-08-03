# Partenaires admin — PR1 (lecture / diagnostic)

## Objectif

L’espace **Partenaires** observe et diagnostique les organisations LIRIE
(entreprises, institutions, configurations incomplètes) **sans mutation
destructive** depuis la surface principale ni via les endpoints users gelés.

```text
Partenaires
├── Organisations          (défaut)
├── Comptes et accès
└── Démonstrations
```

## Invariants PR1

- `PUT /admin/users/:id/role` : même rôle → 200 no-op ; tout vrai changement → 409
  `role_transition_requires_assistant`.
- `DELETE /admin/users/:id` : 409 `physical_user_deletion_requires_review` hors
  `TESTING` (pytest).
- `Company.user_id` : contrainte UNIQUE `uq_company_user_id` (migration
  `26fb555b0eb1`) + modèle SQLAlchemy aligné.
- Enrichissement liste users : comparaison de rôle via
  `normalized_role_value` (corrige le bug `"company" == "COMPANY"`).

## API lecture

| Endpoint | Capacité |
|---|---|
| `GET /admin/partners/organizations` | `admin.partners.read` |
| `GET /admin/partners/accounts/:id/integrity` | `admin.partners.read` |
| `GET /admin/users`, `GET /admin/users/:id` | `admin.partners.read` |
| `GET /admin/institutions` | `admin.partners.read` |
| Liste démos admin | `admin.partners.read` |
| `GET /admin/companies` | role admin seulement (Optuna) |

### KPI `summary`

- `configured_organizations` — Company avec propriétaire + Institution avec ≥1 user
- `incomplete_configurations` — orphelins company/institution + institutions sans user
- `restricted_companies` — billing `partial` \| `full`
- `active_demonstrations` — `COUNT(DISTINCT demo_request_id)` où status active et
  `demo_expires_at > now()` (indépendant de `include_synthetic`)

### Contrat organisation

Axes séparés : `configuration_status`, `lifecycle_status`, `data_scope`
(`production` \| `inferred_synthetic`), `commercial_access_state`.
Clé globale : `organization_key` (`company:3`, `orphan-company-account:217`, …).

## Démos

- Anti-N+1 sur le dernier accès (fenêtre SQL).
- Payload : `stored_status`, `effective_status`, `demo_expires_at`.
- `policy.access_duration_hours` exposé par l’API (source backend 48 h).

## Ops

Précontrôle collisions avant UNIQUE :

```bash
docker compose exec atmr_api python scripts/report_company_user_id_collisions.py
```

## Hors PR1

Assistants de réparation (rôle, lien, suppression contrôlée), commercial
org-centric UI, workspace démo explicite / caps `admin.demo.*`.

## Suite — Control plane CP-PR1

✅ **Implémenté** : fondations control plane (projection, classification fail-closed,
entitlements shadow, anomalies persistées, cut auto-create Company).
Voir [control-plane-fondations.md](control-plane-fondations.md).

Note : `uq_company_user_id` a été **retirée** par la migration CP-PR1
(`d0e04085600f`) car incompatible avec les coquilles cliniques
(`ClinicBillingPartyMapping`).