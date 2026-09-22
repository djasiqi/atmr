# AUTH-SMS-01 — Activation SMS des comptes clients LIRIE

Chantier P0 : rendre le flux SMS d'activation opérationnel en production, sans
refonte d'authentification et sans modification manuelle du compte Philippe AMEY.

## Verdict technique (code actuel)

```text
TWILIO MODE :
PROGRAMMABLE MESSAGING

CURRENT SENDER :
TWILIO_PHONE_NUMBER (from_)
ou, si présent, TWILIO_MESSAGING_SERVICE_SID

CONFIG REQUIRED :
SMS_NOTIFICATIONS_ENABLED=true
TWILIO_ACCOUNT_SID
TWILIO_AUTH_TOKEN
TWILIO_PHONE_NUMBER  OU  TWILIO_MESSAGING_SERVICE_SID

CONFIG MISSING (prod live, container `atmr-backend-1`, valeurs non affichées) :
SMS_NOTIFICATIONS_ENABLED=false
TWILIO_ACCOUNT_SID=MISSING (clé présente, valeur vide)
TWILIO_AUTH_TOKEN=MISSING (clé présente, valeur vide)
TWILIO_PHONE_NUMBER=MISSING (clé présente, valeur vide)
TWILIO_VERIFY_SERVICE_SID : non utilisé (Verify absent du code)
```

Twilio Verify n'est **pas** présent. La migration Programmable Messaging → Verify
est un chantier séparé : `AUTH-SMS-03`.

Le contrat produit (e-mail = activation, SMS = 1er transport) est
`AUTH-SMS-02` — voir `docs/ops/auth-sms-02-decouple-activation.md`.
Ne pas ouvrir Twilio en production avant AUTH-SMS-02.

## Flux réel (canal SMS — AUTH-SMS-01)

Le canal Programmable Messaging et les erreurs 503 restent ceux de ce
hotfix. Le **moment** d'envoi a changé avec AUTH-SMS-02 : plus de SMS à
l'inscription si un e-mail est fourni. L'OTP part à la confirmation du
premier transport (ou via « Vérifier maintenant »).

```text
resend-sms | update-phone | POST /auth/phone/send-code
  → cooldown 60s + quota 10/jour (autorité backend)
  → nouvel OTP hashé
  → Twilio messages.create
  → 200 ou 503 (plus de 502 générique SMS)

verify-sms | POST /auth/phone/verify-code
  → compare hash, TTL 5 min, 5 essais puis lock 15 min
  → user.phone_verified_at
```

## ✅ Implémenté

- **Service SMS** : lecture runtime de la config, classification d'erreurs
  (`DISABLED`, `CONFIG_ERROR`, `AUTH_ERROR`, `SENDER_ERROR`, `DESTINATION_ERROR`,
  `TWILIO_REJECTION`, `DELIVERY_FAILURE`, `SUCCESS`), masquage SID / téléphone.
  Fichiers : `backend/services/notifications/sms.py`,
  `backend/services/notifications/phone_e164.py`.
- **Erreurs API** : `sms_unavailable` / `sms_provider_unavailable` / `invalid_phone`
  en 503/400. Plus de 502 parce que le canal est off ou sans sender.
  Fichiers : `backend/routes/auth.py`, `backend/shared/constants.py`.
- **Anti-abus existants conservés** : cooldown 60 s, 10 SMS/jour/session,
  Flask-Limiter, 5 OTP puis lock 15 min. Autorité backend.
- **Observabilité** : `sms_verification_requested|provider_accepted|provider_failed|rate_limited|success|failed`
  + log de readiness au démarrage (`log_sms_provider_readiness`).
- **Sonde isolée** : `python -m scripts.probe_sms_provider` (option `--to`).
- **Tests** : `backend/tests/services/test_sms_notification.py`,
  `backend/tests/routes/test_auth_activation_sms_contract.py`.

## Reste à faire (fermeture prod)

- Renseigner un sender Twilio valide (`TWILIO_PHONE_NUMBER` ou
  `TWILIO_MESSAGING_SERVICE_SID`) **et** `SMS_NOTIFICATIONS_ENABLED=true`
  dans les secrets / `.env.production`.
- Déployer, vérifier `sms_provider_status=READY` dans les logs du container.
- Sonde interne (`--to` numéro équipe), puis **un** resend sur Philippe AMEY
  (user_id 192847) sans toucher sa base.

## AUTH-SMS-03 (hors lot)

Migration vers Twilio Verify (pool d'expéditeurs, codes gérés, Fraud Guard).
Ne pas mélanger avec ce hotfix. AUTH-SMS-02 = découplage activation / SMS.
