# LIRIE — LOT-BOOKING-DISPATCH-2

Finalisation des missions Chauffeur et du parcours Institution (réseau / partenaires).

**Date :** 2026-11-12  
**Worktree :** `D:\lirie-backups\_candidate_lirie_booking_dispatch_20261112`  
**Branche :** `candidate/lirie-booking-dispatch-20261112`  
**Base LOT-1 :** `402b0627`  
**WIP principal :** non modifié  
**Push / PR / merge / deploy :** non effectués  

---

## A. Blocages PARTIAL de LOT-1

| Gate | Blocage LOT-1 | Traitement LOT-2 |
|------|---------------|------------------|
| DRIVER MISSION | Scénario C limité (idempotence assign seulement) ; quick-accept/reject enums invalides (`accepted`/`cancelled`) ; reject sans nettoyage Assignment | Chaîne réelle `ASSIGNED→en_route→arrived→in_progress→completed` ; ack quick-accept reste `ASSIGNED` ; quick-reject → `PENDING` + Assignment libéré |
| INSTITUTION FLOW | Garde mobilité OK ; `network_eligibility` WIP non porté | Port chirurgical réseau + divulgation offre + capacités déclarées ; `dispatch_target` branché create/update/send |
| END-TO-END | A–J PASS ; navigateur/device non | Scénarios 1–8 use-case/HTTP isolés ; navigateur/device toujours hors périmètre |

---

## B. Corrections Chauffeur

1. **`quick-accept`** (`routes/driver.py`) : plus de transition illégale vers `ACCEPTED` ; ack idempotent sur `ASSIGNED` ; refus si `CANCELED` ou autre entreprise.
2. **`quick-reject`** : délègue à `RejectDriverBookingUseCase` (enums corrects) au lieu de `BookingStatus.cancelled`.
3. **`RejectDriverBookingUseCase`** : ordre `PENDING` puis `driver_id=None` (validateur ORM) ; nettoyage Assignment optionnel via repos.
4. **DELETE reject** chauffeur : injection `AssignmentRepository` pour libérer la mission.
5. Parcours de réalisation : réutilisation de `UpdateDriverBookingStatusUseCase` / TransportAction existants (pas de nouvelle architecture).

---

## C. Corrections Institution

1. Port de `network_eligibility.py`, `network_offer_disclosure.py`, `transport_capabilities.py`, route `company_transport_capabilities`.
2. `SendTransportRequestUseCase` : branche `dispatch_target=network` (sélection éligible) vs partenaires ; messages d’erreur adaptés ; notice push sans identité patient en réseau.
3. Schemas + routes : champ `dispatch_target` (`network`|`partners`) à la création, mise à jour et au send.
4. Offres entreprise : divulgation restreinte tant que l’offre réseau n’est pas acceptée ; travel-estimate à vol d’oiseau.
5. Prestations brancard/oxygène : restent fermées (gate + besoins non modélisés réseau). Modes semi/auto : inchangés (verrou S23). `PORTAL_GRANT_ENFORCEMENT` non activé.

---

## D. Fichiers WIP réutilisés

| Fichier | Origine |
|---------|---------|
| `application/institutions/network_eligibility.py` | WIP principal (copie) |
| `application/institutions/network_offer_disclosure.py` | WIP + lint RUF046/C414 |
| `application/companies/transport_capabilities.py` | WIP |
| `routes/company_transport_capabilities.py` | WIP |
| Tests associés `tests/application/institutions/*`, `tests/application/companies/test_transport_capabilities.py` | WIP |
| Patches ciblés `send_transport_request`, `institution_requests`, `institution_schemas`, `request_offer`, `company_request_offers` | Diffs WIP minimaux |

Pas d’import massif institution (TransportAction avancé, change_booking_payer, etc.).

---

## E. Parcours web et mobile

| Interface | Couverture LOT-2 | Niveau de preuve |
|-----------|------------------|------------------|
| Chauffeur | Transitions + reject/ack contrats API/use-case | Use-case / E2E isolé — **pas d’appareil réel** |
| Entreprise | Accept/assign visibles via même booking ; capacités transport enregistrables | Use-case + tests capacités |
| Institution | `dispatch_target`, réseau/partenaires, mobilité fermée | Use-case + unitaires réseau |
| Client | Statut booking reflète transitions chauffeur (même enregistrement) | Use-case E2E S1 |
| Navigateur React | Non rejoué | Réserve documentée |
| Push iPhone | Non vérifié (héritage #105) | **DEVICE NOT VERIFIED** |

---

## F. Résultats E2E

Exécution Docker isolée (`lirie-ci-gate`, `postgres_test` :5433, Redis test — **pas** DB `atmr`) :

| ID | Scénario | Résultat |
|----|----------|----------|
| 1 | Mission chauffeur complète | **PASS** |
| 2 | Chauffeur indisponible / refus | **PASS** |
| 3 | Institution dispatch_target + restrictions | **PASS** |
| 4 | Isolation inter-entreprises | **PASS** |
| 5 | Concurrence deux chauffeurs | **PASS** |
| 6 | Reprise réseau / idempotence transition | **PASS** |
| 7 | Restrictions S23 + spécialisées | **PASS** |
| 8 | Compatibilité MFA (contrats, pas de bypass) | **PASS** (structurel) |
| — | Transitions interdites / mission annulée | **PASS** |

Suite LOT-1 A–J rejouée : **10/10 PASS** (non-régression).

---

## G. Concurrence et idempotence

- Assign même chauffeur : `changed=False`.
- Second chauffeur : reject use-case → 404 (pas assigné).
- Rejeu `en_route` : 200 sans régression de statut.
- Ack quick-accept : idempotent sur `ASSIGNED`.

---

## H. Isolation et permissions

- Accept cross-company : 403 (LOT-1, reconfirmé S4).
- Quick-accept/reject : contrôle `company_id` chauffeur vs booking.
- Offre réseau : patient / adresses exactes masqués tant que non acceptée.
- Pas d’activation grant portal.

---

## I. Compatibilité MFA

- Merge-tree temporaire `626de1c3` (PR #104) × tip candidat : **pas de marqueurs de conflit** sur le merge-tree local.
- PR #104 **non modifiée**, **non intégrée** aux branches métier.
- Use-cases accept/assign/status/reject : aucun `skip_mfa` / `bypass_mfa`.
- Runtime MFA×métier sur appareil : non exécuté.

---

## J. Tests de non-régression

| Suite | Résultat |
|-------|----------|
| `test_lot_booking_dispatch_2_scenarios.py` | PASS |
| `test_reject_driver_booking_use_case.py` | PASS |
| `test_network_eligibility.py` | PASS |
| `test_network_offer_disclosure.py` | PASS |
| `test_transport_capabilities.py` | PASS |
| `test_s23_dispatch_mode_lock.py` | PASS |
| `test_specialized_mobility_gate.py` | PASS |
| `test_lot_booking_dispatch_1_scenarios.py` (A–J) | PASS |

**Total rejoué :** 51 tests ciblés — **51 PASS** (après correctif ordre reject).

Niveaux de preuve explicites :

- Unitaires : réseau, mobilité, S23, reject, capacités.
- Intégration / E2E HTTP-isolé : scénarios 1–8 + A–J.
- E2E navigateur React : **non**.
- Mobile simulé (contrats) : **oui** (use-case).
- Appareil réel : **non**.

---

## K. Commits et HEAD final

| SHA | Message |
|-----|---------|
| `eca03ac4` | fix(booking-dispatch): missions chauffeur et réseau institution |
| `f27b79da` | docs(ops): rapport LOT-BOOKING-DISPATCH-2 |

Commits de tampon SHA éventuels ensuite sur la même branche.  
Tip code+rapport : `f27b79da` (base fonctionnelle `eca03ac4`).
HEAD branche : voir `git rev-parse HEAD` sur `candidate/lirie-booking-dispatch-20261112` — **aucun push**.

FIN-FACTURE / FIN-CAISSE / PAY-SAFERPAY : non touchés. Aucune migration Alembic.

---

## L. Risques résiduels

1. Certification UI navigateur / appareil chauffeur non faite.
2. Livraison push iOS toujours **NOT VERIFIED**.
3. Accept offre institution → booking → assign : smoke existant ; parcours HTTP bout-en-bout institution non rejoué dans ce lot (contrats réseau + driver couverts).
4. `AssignDriverToReservation` permet encore une réaffectation entreprise vers un autre chauffeur (procédure autorisée) — distinct du refus concurrent chauffeur non propriétaire.
5. Flags ops et modes auto restent OFF ; ne pas activer en prod sans lot dédié.

---

## M. Verdicts

| Gate | Verdict |
|------|---------|
| BOOKING CLIENT | **PASS** (non-régression A–J) |
| DISPATCH MANUEL | **PASS** (non-régression + S23) |
| DRIVER MISSION | **PASS** (use-case/E2E isolé ; pas device) |
| INSTITUTION FLOW | **PASS** (réseau/partenaires + garde mobilité ; pas navigateur) |
| SECURITY | **PASS** (isolation + S23 + pas de bypass MFA) |
| END-TO-END | **PASS** (contrats 1–8 + A–J ; navigateur/device réservés) |
| READY FOR REMOTE CI | **YES** |

Prêt pour publication GitHub sur demande explicite — **rien n’a été poussé**.
