# Checklist cohérence — code ↔ manifest ↔ disclosure ↔ narratif ↔ Data Safety

**Revue :** 2026-06-10 · **Périmètre :** STOP GATE Release (dossier Play `ch.liri.operations`).
Vérifie que toutes les sources racontent la même histoire avant captures Play Console.

---

## 1. Localisation arrière-plan (triangulation 5 sources)

Histoire de référence : **mission active** + **disponibilité flotte 07h–19h** + **notification
persistante** + **pas de tracking présence hors fenêtre**.

| Source | Contenu vérifié | Verdict |
| ------ | --------------- | ------- |
| `app.json` | `ACCESS_BACKGROUND_LOCATION`, `FOREGROUND_SERVICE_LOCATION`, expo-location bg+FGS, `withAndroidTrackingForegroundService` | ✅ |
| Disclosure in-app | `PresenceAvailabilityDisclosureModal` — titre « Disponibilité flotte », usage bg = visibilité dispatch, refus supporté | ✅ |
| Notification FGS | « Disponibilité active — localisation en cours » (présence) / « Mission en cours — localisation active » (mission) — NO_CLEAR | ✅ |
| `play-submission-narrative.md` | §1–§3 : mission active + dispo 07h–19h, notif persistante, bornage hors fenêtre | ✅ |
| `data-safety-mapping.md` | Location = Required, App functionality | ✅ |

→ **Cohérent.** Les deux chaînes de notif (mission vs présence) sont documentées dans le narratif §3.

---

## 2. Suppression de compte (triangulation 4 sources)

| Source | Contenu vérifié | Verdict |
| ------ | --------------- | ------- |
| UI client | `DELETE /clients/me` via écran « Mon compte → Supprimer mon compte » | ✅ |
| Politique confidentialité | `https://www.lirie.ch/privacy` | ✅ |
| `data-safety-mapping.md` §3 | Client suppression in-app ; chauffeur via support | ✅ |
| Narratif / `play-compliance.md` | Client in-app ; chauffeur `https://www.lirie.ch/contact` (pas de self-signup) | ✅ |

→ **Cohérent.** Distinction client (in-app) / chauffeur (support) homogène partout.

---

## 3. Notifications (disclosure → permission runtime → Data Safety)

| Source | Contenu vérifié | Verdict |
| ------ | --------------- | ------- |
| Disclosure | `NotificationPermissionDisclosure` + flag persisté | ✅ |
| Permission runtime | `POST_NOTIFICATIONS`, **un seul** `requestPermissionsAsync` prod (`registerPushToken.ts`) | ✅ |
| `data-safety-mapping.md` | Device/push token = Collected / Not Shared (Expo/FCM = infra acheminement) | ✅ |

→ **Cohérent.**

---

## 4. Sentry (anti-contradiction Shared / Not Shared)

| Source | Mention partage | Verdict |
| ------ | --------------- | ------- |
| `data-safety-mapping.md` | Crash/diagnostics = **Not Shared** (sous-traitant) | ✅ |
| `play-compliance.md` | Aucune mention « Shared » | ✅ |
| Narratif / preuves | Aucune mention « Shared » | ✅ |

→ **Cohérent — aucun document ne dit « Shared ».** Reste 1 acte legal : confirmer Sentry comme sous-traitant (processor).

---

## 5. Matrice code ↔ manifest ↔ Play Console

| Élément | Code / app.json | Manifest AAB (à confirmer) | Capture Play (après saisie) |
| ------- | --------------- | -------------------------- | --------------------------- |
| BG location | expo-location + disclosure | `FOREGROUND_SERVICE_LOCATION` | bg-location-form.png |
| FGS type location | `withAndroidTrackingForegroundService` | `types=0x00000008` | fgs-form.png |
| RECORD_AUDIO | ⚠️ présent dans `permissions` **et** `blockedPermissions` | doit être **Absent** | — |
| App Links | intentFilters `app.lirie.ch` (`/activate-account`, `/reset-password`) | `autoVerify=true` | — |
| Suppression compte | `account.tsx` `DELETE /clients/me` | — | account-deletion-url.png |
| Data Safety | `data-safety-mapping.md` (FIGÉ) | — | data-safety-form.png |
| Deep links push | `lirie://` (rétrocompat `atmr://`) | — | — |

---

## 6. Écarts / actions

| # | Écart | Gravité | Action |
| - | ----- | ------- | ------ |
| 1 | `RECORD_AUDIO` dans `permissions` ET `blockedPermissions` (`app.json`) | Mineur (non bloquant Play) | Nettoyer la double déclaration ; **vérifier l'absence dans le manifest de l'AAB** avant soumission |
| 2 | Validation juridique « Sentry = sous-traitant » | Acte legal (non technique) | Confirmer avant clic final Data Safety |
| 3 | Captures formulaires Play | Attendu (étape 7) | À réaliser après saisie réelle dans Play Console |

**Aucun écart bloquant côté technique.** Le reste relève de la saisie Play Console et d'un acte legal.

---

## 7. Signature

**Revue de cohérence (audit) :** automatisée — Cursor · 2026-06-10 — verdict **PASS technique** (3 réserves ci-dessus, dont aucune bloquante).

**Contre-signature responsable release :** _______________  **Date :** _______________
