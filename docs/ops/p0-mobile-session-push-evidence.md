# P0 mobile — collecte de preuve (session chauffeur et push iOS)

**Date :** 2026-09-23
**Mise à jour :** 2026-09-25 — RCA **code** documentée dans [`mobile-session-disconnect-audit-2026-09-25.md`](./mobile-session-disconnect-audit-2026-09-25.md) (RC-1…RC-7). Correctifs P0-1/P0-3/P0-6 autorisés sans attendre le triplet ; P0-4/P0-5 (UI) et tout assouplissement de classification 401 restent **bloqués** tant que le triplet runtime n’est pas capturé.

**Statut officiel :**

| Incident | Statut | Code |
| --- | --- | --- |
| Session chauffeur | `P0 — RCA CODE CONFIRMED` (preuve runtime triplet toujours requise) | Pas de patch « ignore 401 » / TTL / porte P1-C2. Voir audit 2026-09-25 pour P0-1…P0-6. |
| Push iOS | `P0 — DEVICE TEST REQUIRED` | Aucun changement FCM / Expo / APNs avant les deux `test-push` sur un iPhone réel. |

Les deux incidents restent séparés. Une session tombée peut empêcher le renouvellement du token push. Elle n’explique pas à elle seule qu’Android reçoive les notifications et que les iPhone, collectivement, n’en reçoivent plus.

## Où est la preuve aujourd’hui

Sentry (`lirie-mobile`, `python-flask`) ne contient pas `auth.refresh.terminal`. Ne pas conclure depuis Sentry seul.

| Preuve | Où la lire | Survit à l’écran login |
| --- | --- | --- |
| `auth.refresh.terminal` | Console appareil : `[driver-telemetry] auth.refresh.terminal` | Non, sauf capture log avant fermeture |
| `auth.recovery.terminal`, `auth.terminal_revocation.applied` | AsyncStorage `driver_session_journal_v1` (200 derniers événements). Le header `X-Session-Diag` ne porte que le **dernier** événement. | Oui, tant que l’app n’a pas fait un logout explicite (`clearSessionJournal` n’est appelé que sur logout) |
| `PendingResumeOperation` | AsyncStorage `@atmr/auth/pending_resume_operation` | Non si révocation terminale : le marqueur est purgé avec les credentials |
| Ligne session | Table `mobile_device_session` | Oui |
| Refresh / session-resume HTTP | Logs API `auth_refresh_failure` et réponses `/auth/refresh-token`, `/auth/session-resume` (`error_code`, `trace_id`) | Oui, rétention logs |
| Tokens push | Table `device_tokens` | Oui |
| `provider_accepted` vs `mobile_received` | Réponse `POST /api/v1/driver/me/test-push` puis ACK `mobile_received` / `mobile_opened` | Oui, logs `[push_attempt]` 30 jours, preuves test-push 90 jours |

Triplet décisif session, à remplir **avant** un nouveau login :

```text
APP dit session_revoked ou refresh_replay_detected
+
mobile_device_session.status = active
+
revoked_reason IS NULL
```

Si ce triplet est observé, la RCA session est pratiquement tenue. Un second cas, indépendant, est à noter à part : `idempotency_result_expired` sur un `PendingResumeOperation`, UI `anonymous`, ligne toujours `active`.

Requête lecture (Docker / base, jamais depuis l’hôte) :

```sql
SELECT session_id, user_id, driver_id, device_installation_id,
       status, revoked_at, revoked_reason, last_seen_at, last_refresh_at,
       session_epoch, credential_generation, refresh_generation
FROM mobile_device_session
WHERE user_id = :user_id
ORDER BY last_seen_at DESC NULLS LAST;

SELECT id, provider, platform, is_active, device_id,
       created_at, updated_at, last_seen_at,
       last_push_success_at, last_push_failure_at
FROM device_tokens
WHERE driver_id = :driver_id
ORDER BY updated_at DESC;
```

Ne pas coller de token, de refresh, ni de recovery credential dans cette fiche.

## Fiche — prochaine déconnexion chauffeur

```text
=== INCIDENT SESSION ===

date/heure:
chauffeur:
user_id:
driver_id:

app_version:
build_number:
platform:

session_id:
device_installation_id:

MobileDeviceSession.status:
revoked_at:
revoked_reason:
last_seen_at:

dernier refresh HTTP:
status:
error_code:
error:

dernier session-resume:
status:
error_code:

dernier événement mobile:
auth.refresh.terminal:
auth.recovery.terminal:
auth.terminal_revocation.applied:

PendingResumeOperation:
oui/non
operation_id:
age:

Résultat UI:
anonymous / revoked / login screen

Triplet:
app_error_code:
db_status:
revoked_reason_null: oui/non
```

## Fiche — iPhone réel (à faire maintenant)

Session chauffeur `ready`, permission notifications accordée. Deux appels, dans cet ordre, chauffeur authentifié. Maximum 3 tests par minute.

```text
POST /api/v1/driver/me/test-push
{"provider":"fcm"}

POST /api/v1/driver/me/test-push
{"provider":"expo"}
```

`results[].delivery_status = provider_accepted` ne prouve pas la réception. Attendre `mobile_received` / `mobile_opened` (ACK) avant de conclure.

| FCM | Expo | Lecture |
| --- | --- | --- |
| reçu sur l’iPhone | pas reçu | Expo ou routage Expo |
| pas reçu | reçu sur l’iPhone | FCM / APNs / Firebase iOS |
| pas reçu | pas reçu | APNs, binaire, token, session, ou envoi commun |
| provider seulement | provider seulement | appareil, APNs, réception ou handler |
| provider rejected | — | erreur provider exploitable (`failure_reason`) |

```text
=== INCIDENT PUSH IOS ===

app_version:
build_number:
bundle_id:
firebase_project_id:
aps_environment:

device_installation_id:
session_ready: oui/non
permission_notifications: oui/non

FCM TOKEN
présent:
DB is_active:
platform:
created_at:
last_seen_at:

EXPO TOKEN
présent:
DB is_active:
platform:
created_at:
last_seen_at:

TEST FCM
correlation_id:
provider_status:
provider_error:
mobile_received:
mobile_opened:

TEST EXPO
correlation_id:
provider_status:
provider_error:
receipt:
mobile_received:
mobile_opened:
```

`aps-environment` se lit sur le binaire TestFlight installé, pas dans le dépôt (`GoogleService-Info.plist` n’est pas versionné).

## Hors scope tant que les fiches sont vides

- Pas de patch « ne plus déconnecter sur 401 ».
- Pas de changement de TTL.
- Pas d’ouverture de la porte P1-C2.
- Pas de bascule `IOS_NATIVE_FCM_PREFERRED` ni `IOS_DISABLE_EXPO_ON_FCM_UPSERT`.
