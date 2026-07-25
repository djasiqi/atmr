# Lot 0 P0 Sécurité — Référence de validation

Commit dédié Lot 0 (à renseigner après `git commit`) : voir SHA ci-dessous après création.

## Périmètre

Correction des 6 P0 bloquants production :

| ID | Objet |
|----|--------|
| SEC-01 | Logs inscription sanitisés + chaîne activation (args Celery sans secret) |
| SEC-02 | Reset `public_id` → 410 ; `change-password` JWT ; `token_version` |
| SEC-03 | Passwordless 404 hors `development` ; `debug_code` conditionnel |
| SEC-04 | Liste factures authentifiée + garde tenant |
| SEC-05 | IDOR debug invoices / export payments / booking transfer |
| SEC-06 | `/uploads` logos seuls ; 6 routes métier PDF/PJ |

## Commandes de validation (Docker)

Service API : `atmr_api`. Chemins tests relatifs à `/app`.

```bash
docker compose exec -e ENVIRONMENT=production -e PASSWORDLESS_DEBUG_CODE=false atmr_api \
  python -m pytest tests/security/test_auth_p0_lot0.py \
  tests/security/test_tenant_isolation_p0.py \
  tests/security/test_uploads_access_p0.py \
  tests/security/test_register_log_sanitization.py \
  tests/security/test_path_traversal.py \
  tests/security/test_security_regressions.py::TestF6UploadsCorsAndHeaders \
  tests/unit/test_activation_email_delivery.py \
  tests/routes/test_institution_invitations.py::TestInstitutionInvitation::test_force_password_change_full_cycle \
  tests/routes/test_auth_onboarding_bootstrap.py -v
```

## Résultats de référence (2026-07-25)

| Suite | Résultat |
|-------|----------|
| `test_auth_p0_lot0.py` (passwordless, 410, token_version, debug_code) | 5 passed |
| Isolation tenant SEC-04/05 | passed |
| Uploads SEC-06 + path traversal | passed |
| Register log sanitization | passed |
| Activation email delivery (SEC-01) | passed |
| Force password change full cycle | passed |
| Auth onboarding bootstrap | passed |
| **Total run ciblé Lot 0** | **48 passed** |

Lint : `ruff check` propre sur les fichiers Lot 0 touchés.

## Checklist manuelle (prod-équivalente)

- [x] `/passwordless/otp/*` → 404 si `ENVIRONMENT=production`
- [x] `POST /auth/reset-password/<public_id>` → 410 Gone
- [x] Ancien access token après `change-password` → 401 (`token_version`)
- [x] `GET .../invoices` anonyme → 401
- [x] Préfixes privés via `/uploads/...` → 404
- [x] Logos publics → 200 anonyme
- [x] Pas d’URL activation / OTP / mot de passe dans logs ni args Celery (tests)

## Note

Le working tree peut contenir d’autres modifications hors Lot 0 (tracking, mobile, etc.) volontairement exclues de ce commit.
