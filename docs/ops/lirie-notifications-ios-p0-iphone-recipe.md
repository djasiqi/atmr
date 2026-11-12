# Recette iPhone — NOTIF-IOS-P0

Procédure exécutable par un testeur. **Comptes / appareils de test uniquement.**  
Ne pas activer de flags de production. Ne pas notifier des utilisateurs réels sans autorisation.

**Build attendue :** code aligné sur tip NOTIF `a81ca2b2` (ou SHA final de la PR Draft #105)  
**Bundle ID :** `ch.liri.operations`  
**Flags build (EAS development / canary) :** `EXPO_PUBLIC_ENABLE_DRIVER_PUSH=1`, `EXPO_PUBLIC_ENABLE_DRIVER_FCM_NATIVE=1`, `EXPO_PUBLIC_ENABLE_COMPANY_PUSH=1`  
**Ne pas activer :** `IOS_DISABLE_EXPO_ON_FCM_UPSERT`, `PUSH_DEVICE_TOKEN_LIFECYCLE_ENABLED` (rester off)

---

## Préconditions

1. iPhone physique (pas simulateur pour APNs).
2. Build interne / TestFlight / EAS development correspondant au SHA vérifié.
3. Compte **chauffeur de test** + compte **entreprise de test** (tenant isolé).
4. Accès ops lecture seule : logs push / `correlation_id` / table `device_tokens` (via Docker / outils internes) — **pas** d’export de credentials.
5. Confirmer environnement APNs du build (`aps-environment` development vs production) cohérent avec le projet Firebase / Expo.

Si appareil ou build manquant → **DEVICE DELIVERY : NOT VERIFIED** et arrêter après cette lecture.

---

## Checklist

| # | Étape | Attendu | OK |
|---|--------|---------|----|
| 1 | Installer le build ; noter Bundle ID / buildNumber | `ch.liri.operations` | [ ] |
| 2 | Autoriser les notifications iOS (OS + disclosure app) | Permission granted | [ ] |
| 3 | Login chauffeur de test (si MFA : terminer TOTP **avant** save-push-token) | Session métier OK ; **pas** de JWT métier avant MFA | [ ] |
| 4 | Vérifier enregistrement token iOS | Ligne(s) `device_tokens` : `platform=ios`, `device_id` stable, provider `fcm` et/ou `expo` | [ ] |
| 5 | Push FCM test (`POST /driver/me/test-push` provider `fcm` ou outil ops équivalent) | `provider_accepted` **et** tray iOS | [ ] |
| 6 | Push Expo test (si Expo encore actif) | Ticket + receipt ; tray si FCM absent | [ ] |
| 7 | App au premier plan | Alerte visible (ou schedule local FCM→Expo) | [ ] |
| 8 | App en arrière-plan | Tray + son/badge selon OS | [ ] |
| 9 | App fermée (swipe kill) | Tray ; open cold-start | [ ] |
| 10 | Ouvrir la notification | Navigation vers **la bonne mission** | [ ] |
| 11 | Une seule alerte pour un même `device_id` (pas Expo+FCM doublon) | 1 notification | [ ] |
| 12 | Logout | Tokens de **cet** appareil désactivés (`device_id` / `device_installation_id`) | [ ] |
| 13 | Relogin / renew token | Nouvel upsert ; listeners renew OK | [ ] |
| 14 | Réinstallation app | Nouveau `device_id` ; ancien token inactif ; nouveau enregistrement | [ ] |
| 15 | Corrélation | Même `correlation_id` / `notification_id` : envoi serveur → réception → open | [ ] |
| 16 | Entreprise (compte test) | save-push-token company ; 1 alerte iOS ; isolation autre org | [ ] |

---

## Preuves à joindre

- Screenshots tray FG / BG / killed.
- Horodatage + `correlation_id` serveur.
- Confirmation `provider_accepted` ≠ seul critère (exiger `mobile_received` / open).
- Note build SHA / buildNumber / `aps-environment`.

## Verdicts possibles

```text
IOS DEVICE DELIVERY = PASS          # tous points critiques 5–11 + 15 OK
IOS DEVICE DELIVERY = FAIL          # panne reproduite
IOS DEVICE DELIVERY = NOT VERIFIED  # pas d'appareil / build / preuve
```

Ne pas déclarer **PUSH IOS PASS END-TO-END** sans PASS appareil.
