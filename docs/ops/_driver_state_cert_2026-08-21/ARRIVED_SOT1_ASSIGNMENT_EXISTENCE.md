# ARRIVED-SOT-1 ★ — Assignment existence contract

```text
DATE    = 2026-08-21
GATE    = ARRIVED-SOT-1 PASS ✅
VERDICT = A ★ (Assignment obligatoire sur affectation canonique)
SUIVI   = ARRIVED-SOT-1B (enforcement invariant)
```

**Pas de patch** `if no Assignment: create on ARRIVED` tant que ce gate n'est pas tranché et appliqué en amont.

---

## Verdict binaire

```text
VERDICT = A ★

Toute affectation chauffeur produitée via le chemin canonique
DOIT avoir un Assignment.

Mission 51 (et les 22 autres actifs staging sans Assignment)
= incohérents / drift — pas un mode B volontaire.
```

### Pourquoi A (pas B)

1. **Use-case canonique** `AssignDriverToReservationUseCase` :
   - docstring : « garantit l'existence/MAJ de DispatchRun + Assignment »
   - appelle toujours `assignment_writer.ensure_assignment_for_booking`
   - branché : `routes/companies.py`, `routes/company_mobile_dispatch.py`

2. **Writer** `SqlAlchemyAssignmentWriter.ensure_assignment_for_booking` :
   - crée `Assignment(status=SCHEDULED)` si absent

3. **Sync métier** `assignment_status_sync` :
   - `arrived → ARRIVED_PICKUP` — skip si pas d'Assignment (`no_assignment`)
   - conçu pour Assignment comme SoT opérationnelle

4. **Pipeline GPS** (`gps-tracking-pipeline.md`) :
   - BookingStatus = cycle réservation
   - AssignmentStatus = phases opérationnelles / tracking historique
   - sync depuis transitions chauffeur

5. **Contrat chauffeur** (`driver-contract-v1.md`) :
   - « Server is source of truth for mission lifecycle »
   - surface `ARRIVED` existe — ne peut pas dépendre d'un `Set` mobile

**B (Assignment facultatif) est réfuté** comme contrat cible : ce serait incompatibile avec sync + tracking + composition ARRIVED.  
Ce qui existe aujourd'hui, ce sont des **fuites** (chemins qui écrivent `driver_id` sans passer par le writer).

---

## Pourquoi mission 51 n'a pas d'Assignment

```text
ROOT canary 51 = chemin d'affectation hors contrat ★
```

Création via script cert `_c04_create_assigned.py` :

```text
Booking()
  driver_id = 20
  status = ASSIGNED
  created_via = legacy
→ commit
→ PAS d'appel AssignDriverToReservationUseCase
→ PAS d'ensure_assignment_for_booking
```

Puis transitions EN_ROUTE / ARRIVED forcées (DB / PUT) sans jamais créer d'Assignment.

Log sync observé :

```text
[assignment_status_sync] skip booking_id=51 transition=arrived reason=no_assignment
```

---

## Preuve staging (2026-08-21)

```text
ACTIVE bookings with driver_id
  with Assignment    = 0
  without Assignment = 23 / 23

ASSIGNMENT table totale = 2
  id=2 booking=26 EN_ROUTE_PICKUP (historique)
  id=1 booking=25 CANCELLED

Orphans actifs = surtout fixtures legacy
  STALE-*, AMBIGUOUS-*, SINGLE, CANARY-*, MISMATCH-*
  + mission 51 CANARY-C04-TASKDEF
  created_via = legacy pour tous
```

Donc sur ce staging, **le drift A est massif** — principalement seed/canary/legacy, pas des affectations UI company récentes.

---

## Inventaire des fuites (driver_id sans Assignment)

| Chemin | Crée Assignment ? | Note |
|--------|-------------------|------|
| `AssignDriverToReservationUseCase` + writer | ✅ OUI | **canonique A** |
| `Booking.assign_driver()` | ❌ | modèle seul |
| `services/external/ai.assign_driver_to_booking` | ❌ | set driver_id + ASSIGNED |
| `services/demo/seed_service` / access_service | ❌ | seed |
| Scripts cert `_c04_create_assigned.py` etc. | ❌ | canary |
| Dispatch agent `tools.assign` (Assignment()) | ✅ selon chemin | à vérifier cas par cas |
| `dispatch_assignments` réassign | ✅ met à jour Assignment existant | |

---

## Conséquence pour ARRIVED SoT

Sous contrat **A** :

```text
ROOT structurel
= missing Assignment creation sur chemins non-canoniques ★

PAS (encore)
= create Assignment dans handler ARRIVED
  → masquerait la fuite d'affectation
```

Patch cible (après gel de ce contrat) :

```text
1) Toute affectation → Booking.driver_id + Assignment atomiques
2) arrived → Assignment.ARRIVED_PICKUP
3) GET driver → compose ARRIVED depuis Assignment
4) overlay Set = optimistic UI seulement
5) backfill actifs incohérents (optionnel / staging)
```

Même avec Assignment présent, **A4 échouerait encore aujourd'hui** tant que GET n'enrichit pas `mission_milestone` — c'est **ARRIVED-SOT-2** (composition GET), après SOT-1.

---

## NEXT

```text
ARRIVED-SOT-1     = OPEN ★ (verdict A figé — fuite à fermer)
ARRIVED-SOT-2     = HOLD (GET compose ARRIVED)
C07 / C08         = HOLD
PAS ENCORE        = create Assignment on ARRIVED handler
```

Critères de fermeture SOT-1 :

```text
[ ] Chemins produit (company / mobile dispatch / reassign) = Assignment garanti
[ ] Chemins seed/cert = alignés OU étiquetés hors-contrat
[ ] Invariant mesurable : actifs driver_id ⇒ Assignment existe (staging/prod)
[ ] Mission canary C07 recréée via UC canonique (pas script Booking seul)
```
