# P1-C3 — Extinction des pollers driver à la sortie du contexte

**Date :** 2026-08-27 · **Statut :** implémenté, revue + 2 contrôles PASS, **commité — non déployé, pas de push sans accord**.

## Symptôme (mesuré en C1, contexte company actif, fenêtre 2 min)

```text
GET  /driver/me/bookings/eta   x40  (dont 401)
POST /driver/me/telemetry/push x16
PUT  /driver/me/location       x3
```

Fuite cross-contexte : requêtes driver pendant que l'app est sur l'espace
entreprise → vagues 401 (déclencheur du storm refresh C2) + charge inutile
sur le worker unique (P1-C1).

## Inventaire des sources (repo actuel vs build 1.0.12 installé)

| Source | Endpoint | Etat repo actuel |
|---|---|---|
| Garde intercepteur `isDriverSelfEndpoint` (client.ts) | tout `/driver/me/*` via apiClient | **Déjà présente** (throw local `DRIVER_CONTEXT_INACTIVE` hors contexte driver) — absente du build 1.0.12 → explique les x40/x16/x3 observés |
| `useDriverMissionEtaQuery` (features/driver/hooks.ts) | bookings/eta | Déjà gaté (`enabled: contextId`) |
| **`useMissionEtaMinutes`** (messages/hooks.ts) | bookings/eta | **TROU corrigé** : gate `bookingId` seul, refetch 20 s quand socket driver pas prêt (= toujours en company) → boucle d'erreurs locales |
| **`useHubUnreadCount`** (messages/hooks.ts) | `/messages/<cid>/hub/unread-count` | **TROU corrigé** : refetch 15 s permanent, endpoint hub **non couvert** par la garde `/driver/me/*` → atteignait le serveur même avec la garde |
| `reportPushRegistrationTelemetry` | telemetry/push | One-shot (pas un poller), via apiClient → couvert par la garde |
| Queue GPS (`driverTrackingQueue`) | location | Couverte par hardStop + `activateContextInactiveGate` (MOB-ENT-02) + garde |
| `useDriverChatMessages` (chatHooks.ts) | messages | Déjà gaté (`enabled: contextId`) |

## Fix appliqué (minimal)

`mobile/unified-app/src/features/driver/messages/hooks.ts` :

- `useHubUnreadCount` : + `useActiveDriverContextId()` →
  `enabled: Boolean(companyId && driverContextId)`.
- `useMissionEtaMinutes` : idem →
  `enabled: Boolean(bookingId && driverContextId)`.

Aucun changement de comportement en contexte driver (gate purement additif).

## Tests

`contextGate.test.tsx` (nouveau, pattern renderer + QueryClientProvider) :

1. Contexte company (contexte driver nul) → **aucune requête réseau**
   (unread-count ET eta jamais appelés).
2. Contexte driver actif → les deux requêtes partent normalement.
3. **Transition driver → company (contrôle pré-commit revue)** : polling réel
   confirmé en driver (fake timers, ≥2 appels après 21 s), bascule du provider
   vers company **sans démonter** composant ni QueryClient, puis **60 s de
   timers (≥3 intervalles) → zéro nouvel appel** sur les deux hooks. Valide :
   `useActiveDriverContextId()` devient falsy au switch, React Query recalcule
   `enabled`, le `refetchInterval` cesse — sans dépendre du démontage d'écran.
   (Pas d'annulation des requêtes en vol : hors scope, conformément à la revue.)

3/3 PASS · ESLint 0 erreur.

## Contrôle call-sites `useHubUnreadCount` (revue) : PASS

Call-sites applicatifs exhaustifs : `useDriverMessageHubUnreadBadge`
(→ `DriverFloatingTabBar`, badge messages chauffeur) et l'écran messages
chauffeur (`DriverMessagesInboxView` via `useDriverCompanyId`). Aucune UI
company/institution ne consomme ce hook (le hub company a ses propres hooks
dans `features/company`). Sémantique **driver-only** confirmée → le gate
`driverContextId` n'introduit aucune régression fonctionnelle.

## Gate C3 (à valider au gate P1-C1 backend, après déploiement)

```text
T0 switch driver → company
Après confirmation switch :
  driver pollers actifs        = 0
  requêtes API /driver/me/*    = 0
  requêtes hub driver          = 0
  pollers company              = attendus uniquement
```

Note importante : l'essentiel de la protection (garde intercepteur, hardStop,
gates queue GPS) **existe déjà dans le repo mais n'est PAS dans le build
1.0.12 (137) installé** — le gate serveur ne passera qu'après déploiement
mobile (release/OTA à décider).
