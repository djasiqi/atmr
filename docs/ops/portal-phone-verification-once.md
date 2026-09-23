# PORTAL_PHONE_VERIFICATION_ONCE

Le téléphone d’un compte privé est vérifié une seule fois. L’OTP ne fait plus partie de la confirmation d’une réservation.

Le contrat précédent, [AUTH-SMS-02](auth-sms-02-decouple-activation.md), est clos.

## Parcours

Nouveau compte (`activation_session.portal_terms_required = true`) :

```text
inscription
→ verify-email
→ OTP téléphone
→ lecture et acceptation CGU/CGV
→ finalize
→ compte actif
```

Au moment de l’activation : e-mail confirmé, `phone_verified_at` renseigné, deux `ClientTermsAcceptance` avec `verification_method = otp_sms`.

Compte déjà ouvert, ou session d’activation antérieure (`portal_terms_required = false`) :

```text
activation possible sans SMS
→ réservation refusée tant que phone_verified_at est vide
→ /auth/phone/send-code puis /auth/phone/verify-code
→ réservations suivantes sans OTP
```

`phone_verified_at` n’est pas backfillé. Un changement de numéro le remet à vide. Les acceptations et les événements de commande déjà écrits ne sont pas réécrits.

## Réservation

Ordre des contrôles : autorisation, identité PORTAL, conditions courantes, `assert_portal_phone_verified`, création.

`403 phone_verification_required` signifie que le compte doit être vérifié une fois. Le tableau de bord ouvre cette vérification sans renvoyer la réservation tout seul. Le formulaire saisi reste affiché. Aucun booking n’est créé avant le téléphone.

Les trois créations passent par `execute_client_booking_creation` :

```text
POST /api/v1/clients/<public_id>/bookings
POST /api/v1/clients/me/bookings
POST /api/v1/bookings/clients/<public_id>/bookings
```

L’idempotence (`Idempotency-Key`) est inchangée. L’e-mail `portal_booking_confirmation_v1` ne parle pas d’une confirmation SMS de la réservation.

## Ce qui reste

```text
OTP téléphone (envoi, validation, expiration, brute-force) : KEEP
phone_verified_at : KEEP
changement de numéro : KEEP
connexion sans mot de passe par OTP : KEEP, hors réservation
récupération de compte : e-mail, RECOVERY OTP NOT APPLICABLE
step-up sensible : NOT IMPLEMENTED
```

## Inventaire

```text
SignupActivation.jsx          CHANGE   OTP avant les CGU/CGV des nouveaux comptes
ClientDashboard.jsx           CHANGE   plus de retry de réservation après l’OTP
AccountUser.jsx               CHANGE   libellé de vérification de compte
register                      CHANGE   requires_phone pour un nouveau PORTAL, SMS après l’e-mail
verify-email                  CHANGE   prépare l’OTP sans activer le nouveau compte
activation/verify-sms         KEEP
activation/finalize           CHANGE   refuse un nouveau compte sans téléphone vérifié
phone/send-code               KEEP     vérification de compte
phone/verify-code             KEEP
routes de réservation         CHANGE   message de compte, pas de SMS de course
user.phone_verified_at        KEEP
activation_session            KEEP     sessions legacy sans téléphone obligatoire
assert_portal_phone_verified  CHANGE   remplace assert_portal_can_confirm_transport
is_portal_client              KEEP
user_phone_is_verified        KEEP
PortalPhoneVerificationRequired KEEP   même code d’erreur
portal_phone_verification     CHANGE
portal_account_promotion_sql  KEEP     promotion historique des comptes déjà e-mail
```
