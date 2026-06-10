# Conformité Google Play — LIRIE Opérations (`ch.liri.operations`)

**Dernière mise à jour :** 2026-06-09  
**Gel permissions actif depuis :** 2026-06-09

## Règle gel permissions / SDK

**Interdit** sans ré-audit Play explicite et mise à jour de ce document :

- Nouvelles permissions Android dans `app.json`
- Nouveaux types Foreground Service
- Nouveaux SDK tiers (analytics, ads, tracking)

**Levée du gel :** _à documenter ici avec date et justification._

### Permissions documentées (manifest cible)

| Permission | Justification |
| ---------- | ------------- |
| ACCESS_FINE_LOCATION | Suivi mission / disponibilité chauffeur |
| ACCESS_COARSE_LOCATION | Fallback localisation |
| ACCESS_BACKGROUND_LOCATION | Suivi arrière-plan (mission ou fenêtre présence) |
| FOREGROUND_SERVICE | Service localisation persistant |
| FOREGROUND_SERVICE_LOCATION | FGS typé location |
| FOREGROUND_SERVICE_DATA_SYNC | Background fetch (si actif) |
| POST_NOTIFICATIONS | Alertes missions / messages |
| RECEIVE_BOOT_COMPLETED | Reprise tâches planifiées |
| WAKE_LOCK | Maintien sync critique |
| USE_BIOMETRIC / USE_FINGERPRINT | Auth chauffeur |
| REQUEST_IGNORE_BATTERY_OPTIMIZATIONS | Fiabilité tracking flotte |

**Bloquées :** RECORD_AUDIO, stockage média, SYSTEM_ALERT_WINDOW.

---

## LOC-01 — Fenêtre présence 07h–19h (HOLD métier)

| Cas | Description | Impact Play |
| --- | ----------- | ----------- |
| **Cas A** | Présence flotte 07h–19h = fonctionnalité dispatch visible | Déclarer « disponibilité flotte » + disclosure dédiée + vidéo Play |
| **Cas B** | GPS technique sans valeur utilisateur claire | Désactiver `driver_tracking_work_window_enabled` |

**Décision produit :** **Cas A signé** (2026-06-09) — présence flotte 07h–19h + disclosure `DriverPresenceDisclosureHost`.

**Fichiers :** `app/(app)/(driver)/_layout.tsx`, `trackingWindow.ts`, `DriverPresenceDisclosureHost.tsx`, `registry.ts` (`driver_tracking_work_window_enabled`).

**STOP GATE Release :** texte BG ✅ (`play-release/bg-location-justification.txt` + `play-submission-narrative.md`), vidéo Play ✅ (`play-release/bg-location-demo.mp4`, 2026-06-10). Reste : revue finale checklist + captures formulaires après saisie Play Console.

### Statut officiel des STOP GATES (2026-06-10)

| Gate | Statut | Preuve |
| ---- | ------ | ------ |
| STOP GATE #1 | ✅ PASS | App Links, privacy, suppression compte |
| STOP GATE Backend | ✅ PASS | DELETE compte + token revoke (prod) |
| STOP GATE #2 | ✅ PASS | FGS système Android (`LocationTaskService isForeground=true`, notif persistante) — `evidence/stop-gate-2/CLOSURE.txt` |
| STOP GATE Release | ⏳ en cours | Package `play-release/` (narratif + vidéo + captures prêts) |

**Réserve build :** override runtime `MOBILE_FEATURE_FLAGS={"tracking_background_enabled":true}` conservé en prod jusqu'à clôture de `BUILD-EXPO-001` (audit inlining `EXPO_PUBLIC_*`).

---

## Verdict comptes chauffeur (Play Store)

| Flux | Mécanisme | Suppression in-app |
| ---- | --------- | ------------------ |
| Client | Signup public `POST /auth/register` | **Oui** — `DELETE /clients/me` via écran compte |
| Chauffeur | Invitation entreprise + activation (`activate.tsx`) | **Non** — pas de self-signup, pas d'endpoint DELETE |

**Conclusion Play :** la suppression compte **client** suffit pour P0. Profil chauffeur : lien support vers `https://www.lirie.ch/contact`.

---

## Deep links

- Schéma canonique : `lirie://`
- Rétrocompat lecture : `atmr://` (1–2 releases) dans `deepLinkHandler.ts`
- App Links HTTPS : `app.lirie.ch` (`/activate-account`, `/reset-password`)

---

## Push permissions

- **Un seul** `requestPermissionsAsync` prod : `registerPushToken.ts`
- Disclosure notifications : `NotificationPermissionDisclosure` + flag persisté
- Vérification CI : `scripts/check-play-compliance.sh`

---

## Texte justification BG (Cas A — aligné Play Console)

Voir `play-release/bg-location-justification.txt` : mission active + disponibilité flotte 07h–19h, notifications FGS distinctes, refus possible avec disclosure in-app.

---

## URLs Play Console

| Champ | URL |
| ----- | --- |
| Privacy | https://www.lirie.ch/privacy |
| Suppression compte | In-app — écran Mon compte (client) |
| Support chauffeur | https://www.lirie.ch/contact |
