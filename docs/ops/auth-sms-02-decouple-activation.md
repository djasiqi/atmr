# AUTH-SMS-02 — Découpler activation du compte et validation du téléphone

```text
AUTH-SMS-02 legacy contract : CLOSED / SUPERSEDED
```

Ce document conserve le contrat historique : l’e-mail activait le compte, et un OTP était exigé avant le premier transport. Ce comportement n’est plus celui des nouveaux comptes PORTAL ni des réservations.

Le contrat actif est [PORTAL_PHONE_VERIFICATION_ONCE](portal-phone-verification-once.md).

---

## Verdict (figé — READY FOR PROD VALIDATION)

```text
AUTH-SMS-02 CODE .............. PASS
TESTS AUTOMATIQUES ............ PASS
MIGRATION ..................... READY
DEPLOY PROD ................... PENDING
PREVIEW PROD .................. PENDING
MIGRATION PROD ................ PENDING
PHILIPPE 192847 ............... PENDING
SMOKE NAVIGATEUR .............. PENDING
TWILIO ........................ OFF volontairement
VERDICT PROD .................. PENDING
```

Périmètre fonctionnel figé à l’époque. Runbook historique : [auth-sms-02-prod-runbook.md](auth-sms-02-prod-runbook.md).

Ce fichier n’est plus la description du comportement courant. La cible réalisée est [PORTAL_PHONE_VERIFICATION_ONCE](portal-phone-verification-once.md).

## Objectif produit historique

```text
EMAIL = identité nécessaire pour activer et connecter le compte
SMS   = validation du téléphone requise avant le premier transport
```

`phone_verified_at` reste la source de vérité. Jamais fabriqué sans OTP.

## ✅ Implémenté

- **Activation** : e-mail seul. Pas de SMS au signup si un e-mail est fourni.
- **Login / bootstrap** : promotion paresseuse des PORTAL déjà e-mail
  vérifiés (cas Philippe). `phone_verified_at` inchangé.
- **Autorité métier** : `assert_portal_can_confirm_transport` dans
  `CreateBookingUseCase` — PORTAL + téléphone non vérifié ⇒ refus, même si
  un appel interne contourne la route. La route HTTP garde un 403 anticipé
  (évite le géocodage) et mappe la même exception.
  Fichiers : `backend/services/auth/portal_phone_verification.py`,
  `backend/application/bookings/create_booking.py`,
  `backend/routes/bookings.py`.
- **OTP** : `/auth/phone/send-code` et `/verify-code`.
- **Changement de numéro** : révocation seulement si le E.164 change.
- **Colonne** : `user.phone_verified_at` (migration `395bd3663e8d`).
- **Comptage read-only** : mêmes prédicats SQL que la migration.
  Fichiers : `backend/services/auth/portal_account_promotion_sql.py`,
  `backend/scripts/preview_auth_sms_02_promotion.py`.

Hors scope (ne passent pas par `CreateBookingUseCase`, contrat inchangé) :

- Institutions (`accept_offer`, legs) ;
- création manuelle entreprise ;
- TRANSPORT rattaché à une company.

## Philippe AMEY (`user_id = 192847`)

```text
AVANT
email_verified = true
phone_verified_at = NULL
account_status = pending_activation
login = refusé

APRÈS MIGRATION
email_verified = true
phone_verified_at = NULL
account_status = active
login = autorisé

PREMIER TRANSPORT
phone_verified_at = NULL
→ OTP requis
→ OTP valide
→ phone_verified_at = timestamp
→ transport confirmé
```

Ne pas écrire son téléphone en base. Ne pas lui envoyer de SMS tant que
Twilio n'est pas validé sur un numéro d'équipe.

## Séquence production

Voir le runbook figé : [auth-sms-02-prod-runbook.md](auth-sms-02-prod-runbook.md).

Point opérationnel : `deploy-production.sh` exécute `flask db upgrade` **sauf**
si `SKIP_DB_UPGRADE=1`. Pour cette release, déployer avec ce flag, relire le
preview, puis upgrader à la main.

## Tests

- `backend/tests/services/test_portal_phone_verification.py`
- `backend/tests/services/test_portal_account_promotion_sql.py`
- `backend/tests/services/test_booking_create_use_case.py`
- `backend/tests/routes/test_auth_sms_02_portal_contract.py`
- `backend/tests/e2e/test_auth_activation_e2e.py`
- `frontend/src/pages/Auth/SignupActivation.test.jsx`
