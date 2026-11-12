# LIRIE — LOT-BOOKING-DISPATCH-1

Finalisation des réservations connectées et du dispatch opérationnel (manuel).

**Date :** 2026-11-12  
**Worktree :** `D:\lirie-backups\_candidate_lirie_booking_dispatch_20261112`  
**Branche :** `candidate/lirie-booking-dispatch-20261112`  
**Base :** NOTIF-IOS-P0 `7a54ad39` (PR #105)  
**WIP principal :** non modifié  
**Push / PR / merge / deploy :** non effectués  

---

## A. Architecture et parcours réutilisés

Chaîne existante conservée :

- Client : `application/bookings/create_booking.py`, routes `clients` / portal  
- Entreprise : `accept_reservation`, `reject_reservation`, `assign_driver_to_reservation`, `schedule_reservation`  
- Chauffeur : `update_driver_booking_status` (+ tests transitions)  
- Institution : `institution_requests` / offers / TransportAction (counter OFF)  
- Notifications : correctifs PR #105 réutilisés (pas de rebuild)

Pas de remplacement TransportAction par l’ancien redispatch. Pas d’import massif WIP.

---

## B. Fonctionnalités consolidées

| Zone | Avant | Après |
|------|-------|-------|
| Dispatch mode API mobile/web | semi/auto acceptés | **Verrou S23** → 403 hors `manual` |
| UI OperationsTab défaut | `semi_auto` | **`manual`** |
| DispatchModeSelector | déjà locked UI | inchangé (déjà verrouillé) |
| Mobilité brancard/O₂ | acceptée en schema | **refus** (`SPECIALIZED_MOBILITY_ENABLED` OFF) |
| Accept cross-company | possible si PENDING | **403** si `company_id` déjà autre entreprise |
| Reject status | sensible à la casse | `.lower() == "pending"` |
| Suite E2E A–J | fragmentée | **`test_lot_booking_dispatch_1_scenarios.py`** |

---

## C. Réservations Client

EXISTE : create / preview / confirm / update pending / cancel / AR / fauteuil+assistance.  
Pas de paiement électronique inventé.  
Prestations spécialisées : refus explicite (institution schema).

---

## D. Dispatch Entreprise

Manuel testé (accept → assign).  
Semi-auto / fully-auto / optimizer : **gelés** (API + mobile flags + UI).  
Pas d’activation modes auto S23.

---

## E. Missions Chauffeur

Transitions et idempotence assign couvertes (scénarios C/J + tests existants status).  
Caisse / end-of-trip : flags OFF non touchés.

---

## F. Parcours Institution

Flux create→offer→accept déjà présent.  
Garde mobilité spécialisée ajoutée.  
Réseau LIRIE WIP (`network_eligibility` complet) **non porté** en masse — limitation documentée.

---

## G. Notifications

Réutilisation #105 : dédup / logout.  
Scénarios notif métier : routage backend existant ; **DEVICE iOS NOT VERIFIED**.

---

## H. Compatibilité MFA

Merge temporaire MFA `626de1c3` → tip BOOKING : **aucun conflit** (`auth.py` / `client.ts` auto-merge).  
PR #104 **non modifiée, non intégrée**.  
Aucune opération métier ne doit contourner MFA ; runtime MFA×booking non exécuté sur appareil.

---

## I. Résultats E2E A–J

Exécution Docker isolée (`postgres_test` port 5433, Redis test — **pas** DB `atmr`) :

| ID | Scénario | Résultat |
|----|----------|----------|
| A | Réservation simple | **PASS** |
| B | Attribution manuelle | **PASS** |
| C | Mission / idempotence assign | **PASS** |
| D | Aller-retour | **PASS** |
| E | Modification pending | **PASS** |
| F | Annulation | **PASS** |
| G | Indispo + mode lock | **PASS** |
| H | Institution mobilité | **PASS** |
| I | Isolation inter-entreprises | **PASS** (fix accept 403) |
| J | Double assign | **PASS** |

Tests unitaires associés : lock S23 + mobility gate — **PASS**.

Navigateur / appareil : **non exécutés** dans ce lot (HTTP/use-case isolés).

---

## J. Tests automatisés

- `backend/tests/services/dispatch/test_s23_dispatch_mode_lock.py`  
- `backend/tests/services/booking/test_specialized_mobility_gate.py`  
- `backend/tests/e2e/test_lot_booking_dispatch_1_scenarios.py`  

---

## K. Corrections réalisées

1. `s23_dispatch_mode_lock` + branchements mobile/web `/mode`  
2. Garde `specialized_mobility_gate` + `MobilitySchema`  
3. Isolation `AcceptReservationUseCase` cross-company  
4. Robustesse `RejectReservationUseCase` (casse statut)  
5. Défaut UI OperationsTab → manual  
6. Suite E2E A–J  

Aucune migration Alembic. Aucun flag ops ON.

---

## L. Commits locaux

| SHA | Message |
|-----|---------|
| `d5f3a2e8` | fix(booking-dispatch): verrou S23 manuel, isolation accept et garde mobilité |
| `96fa2d08` | docs(ops): rapport LOT-BOOKING-DISPATCH-1 |

Tip : `96fa2d08` — **aucun push**.

---

## M. Blocages et limitations

1. IOS DEVICE DELIVERY toujours **NOT VERIFIED** (#105).  
2. Modules réseau institution WIP non importés en entier.  
3. E2E navigateur / mobile UI non rejoués ici.  
4. Défaut modèle DB `Company.dispatch_mode=SEMI_AUTO` historique : API refuse le basculement ; seed tests à forcer `manual` en prod/recette.  
5. Pas de publication GitHub.

---

## N. Verdicts

| Gate | Verdict |
|------|---------|
| BOOKING CLIENT | **PASS** |
| DISPATCH | **PASS** (manuel) |
| DRIVER MISSION | **PARTIAL** (use-case/DB ; pas UI device) |
| INSTITUTION FLOW | **PARTIAL** (garde mobilité ; réseau WIP incomplet) |
| SECURITY | **PASS** (isolation accept + mode lock) |
| END-TO-END | **PARTIAL** (A–J HTTP/use-case PASS ; navigateur/device non) |
| READY FOR REMOTE CI | **YES** |
