# Canary C1–C10 — régression d'état mission

Gate **avant diffusion large**. Ne pas valider uniquement par « j'ai terminé
une course et ça semble fonctionner ».

Prérequis : backend déployé avec la ceinture MISSION-STATE
([mission-state-integrity.md](./mission-state-integrity.md)), app chauffeur
OTA/build contenant `mapDriverMission` sur le delta + garde par revision.

## Matrice

| Canary | Scénario | Résultat obligatoire |
| ------ | -------- | -------------------- |
| C1 | EN_ROUTE → ARRIVED + 10 polls (~15 s) | reste ARRIVED (UI + GET) |
| C2 | ARRIVED → kill app → reopen | reste ARRIVED (pas d'overlay mémoire seul) |
| C3 | COMPLETED → ancien PUT ARRIVED (offline replay) | COMPLETED conservé, PUT → 409 |
| C4 | COMPLETED → reset dispatch | COMPLETED conservé, Assignment non supprimé |
| C5 | 2 Assignments même booking | writer = Assignment lu (`assignment_id` identique) |
| C6 | ARRIVED sans Assignment initial | jamais HTTP 200 mensonger (créé puis persisté, ou 5xx) |
| C7 | actions offline réordonnées (EN_ROUTE, ARRIVED, IN_PROGRESS) | replay FIFO, aucune régression |
| C8 | logout A → login B | aucun événement A rejoué sur B |
| C9 | mission COMPLETED + trigger-return | original reste COMPLETED |
| C10 | vrai nouveau retour / redispatch | nouveau `assignment_id` / lifecycle reconnu |

## Preuve longue

```text
COMPLETED
↓
30 min de polling / realtime / sync / background
↓
COMPLETED
```

sans aucune action chauffeur. Si le statut régresse : **FAIL**, ne pas diffuser.

## Procédure courte

### C1 — polls après ARRIVED

1. Course ASSIGNED, chauffeur « En route » puis « Arrivé ».
2. Laisser l'app au premier plan ≥ 3 min (≈ 10 ticks).
3. Vérifier : UI = Arrivé, `GET /driver/me/bookings` = `status=arrived` +
   `mission_milestone=ARRIVED` + `assignment_id` + `mission_revision` ≥ 1.

### C2 — cold start ARRIVED

1. Après C1, tuer le process (pas seulement background).
2. Relancer, se reconnecter.
3. La mission doit rester Arrivé **sans** retaper le bouton.

### C3 — stale write après COMPLETED

1. Terminer la course (COMPLETED persisté).
2. Rejouer `PUT .../status` avec `{"status":"arrived"}` et une nouvelle
   clé d'idempotence (ou laisser la file offline le faire).
3. Attendu : HTTP 409 `driver_transition_stale`, GET toujours COMPLETED.

### C4 — reset après COMPLETED

1. Course COMPLETED.
2. `POST /dispatch/reset` (ou `/company-mobile/v1/reset`) pour la date.
3. Attendu : `assignments_protected ≥ 1` pour cette course, booking
   toujours COMPLETED, Assignment toujours présent.

### C5 — deux Assignments

1. Booking avec 2 lignes Assignment (ancienne SCHEDULED + courante ARRIVED).
2. PUT arrived / GET détail.
3. `assignment_id` de la réponse = id de la ligne la plus récente.

### C6 — ARRIVED sans Assignment

1. Booking EN_ROUTE, `driver_id` posé, **sans** ligne Assignment.
2. PUT arrived.
3. Attendu : 200 avec `milestone_persisted=true` et un `assignment_id`
   (ensure SOT-1B) **ou** 5xx retryable. Jamais 200 si la base reste
   sans jalon ARRIVED_PICKUP.

### C7 — file offline FIFO

1. Mode avion, enchaîner En route → Arrivé → À bord.
2. Rétablir le réseau.
3. Les trois PUT partent dans l'ordre ; l'état final = IN_PROGRESS.

### C8 — switch chauffeur

1. Chauffeur A : enqueue une transition (mode avion).
2. Logout, login chauffeur B.
3. File de B vide ; aucun PUT de A n'est envoyé avec le token de B.

### C9 — trigger-return

1. Aller COMPLETED.
2. `POST .../trigger-return`.
3. L'aller reste COMPLETED ; le retour est une **nouvelle** course.

### C10 — nouveau lifecycle

1. Redispatch / nouveau chauffeur sur une course non démarrée, **ou**
   création d'un vrai retour.
2. Le mobile applique le snapshot (nouvel `assignment_id`), même si
   `mission_revision` redémarre à 0.

## Verdict

- **GO diffusion** : C1–C10 PASS + preuve 30 min COMPLETED.
- **NO-GO** : toute régression COMPLETED → état antérieur, ou 200 ARRIVED
  sans persistance.

## Tests automatisés (non substituts du canary terrain)

Backend (Docker) :

```bash
docker compose exec -T atmr_api pytest \
  tests/services/test_assignment_status_sync.py \
  tests/services/test_update_driver_booking_status_use_case.py \
  tests/services/test_booking_status_transitions.py \
  tests/services/test_dispatch_reset_guard.py \
  tests/services/test_assignment_resolver.py \
  tests/infrastructure/test_assignment_writer_ensure.py -q
```

Mobile :

```bash
npx jest --watchAll=false --forceExit \
  missionRevisionGuard offlineQueue.behavior sync.reconcile
```
