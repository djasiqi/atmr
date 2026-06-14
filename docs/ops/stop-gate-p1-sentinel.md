# Stop-gate P1 — Fin de la sentinelle `00:00` (Phase 1)

```txt
Status: PASS (tests auto) | PENDING (validation staging manuelle)
Date: 2026-06-12
Reviewer:
Environment: local Docker (atmr_api) + staging à compléter
Commit SHA: 89fd0a0e (tests stop-gate ajoutés dans cette branche)
```

> **Périmètre :** verrouillage Phase 1 — plus aucune **nouvelle** donnée métier « heure à définir »
> écrite sous forme de `00:00`. **Aucune modification des lectures** transporteur/mobile (Phase 2).

---

## 1. Objectif métier

Trois états distincts doivent coexister sans ambiguïté :

| État | `scheduled_time` | `time_confirmed` | Affichage cible |
|------|------------------|------------------|-----------------|
| Heure confirmée | `14:30` | `true` | `14:30` |
| Heure à définir | `null` | `false` | « À définir » |
| Minuit réel | `00:00` | `true` | `00:00` |

**Règle Phase 1 (écriture)** : ne jamais inférer « à définir » via `hour === 0`.

**Distinction fondamentale** :

```txt
time_confirmed = false  ≠  mission absente
time_confirmed = false  =  mission présente, heure non planifiable automatiquement
```

---

## 2. Composants livrés (Phase 1)

| Couche | Fichier / zone |
|--------|----------------|
| Modèle TR | `backend/models/transport_request.py` (`return_date`, `return_time_confirmed`) |
| Legs | `backend/models/transport_request_leg.py` (`time_confirmed` calculé à la sérialisation) |
| Schémas institution | `backend/schemas/institution_schemas.py` |
| Routes institution | `backend/routes/institution_requests.py` |
| Validator Booking | `backend/models/booking.py` (`null` si `time_confirmed=false`) |
| Dispatch auto | `backend/repositories/booking_repository.py` (`find_for_dispatch`) |
| Conversion offre | `backend/application/institutions/accept_offer.py` |
| Arrêt écritures 00:00 | `create_manual_booking`, `create_booking`, `schedule/update_reservation`, `companyApi.ts`, `UnifiedDispatchRefactored.jsx`, etc. |
| Affichage institution | `formatLegTime.js`, `InstitutionRequests`, `RequestDetailPanel` |
| **Heures par leg (mission_date)** | `mission_schedule.py`, `RouteStepTimeField`, `InstitutionRequestCreate/Edit`, affichage transporteur |
| Tests unitaires Phase 1 | `test_accept_offer_round_trip.py`, `test_request_offers.py`, `test_sentinel_migration_phase1.py` |
| **Tests stop-gate P1** | `backend/tests/integration/test_p1_sentinel_stop_gate.py` |
| **Tests mission_schedule** | `backend/tests/unit/test_mission_schedule.py` |

✅ **Implémenté** : heures indicatives (`scheduled_time` + `time_confirmed=false`) exclues de `get_effective_dispatch_time()` ; affichage « (non confirmé) » ; formulaire institution avec date mission + heure par étape ; validation Enregistrer (date seule) vs Envoyer (≥1 confirmée).


---

## 3. Checklist stop-gate P1

### Cas 1 — A/R institution, date retour seule

- [auto] `TransportRequest` A/R avec `return_date` seul → booking retour `scheduled_time=null`, `time_confirmed=false`
- [manuel staging] Création depuis le portail institution : date retour ✓, heure retour vide → accepter offre → vérifier booking retour en DB

### Cas 2 — Transporteur voit « À définir » (lectures legacy)

- [auto] `serialize()` retour : `time_confirmed=false`, `scheduled_time=null`
- [manuel staging] Transporteur ouvre la réservation retour → **« Heure à définir »**, jamais `00:00`  
  *(dépend de Phase 2 : `formatDate.js`, `pickupSentinel.ts`, etc.)*

### Cas 3 — Minuit réel confirmé

- [auto] `return_time=00:00` + `return_time_confirmed=true` → booking `time_confirmed=true`, heure 00:00 locale
- [manuel staging] Affichage transporteur : **00:00**, jamais « À définir »

### Cas 4 — Multi-stop sans héritage horaire

- [auto] Leg 0 confirmé ; legs suivants `scheduled_time=null`, `time_confirmed=false` ; pas d'héritage 08:00
- [manuel staging] Parcours 08:00 départ, HUG / Grangettes sans heure → affichage **08:00 / À définir / À définir**, pas 08:00 partout

### Cas 5 — Aucune nouvelle sentinelle en DB

- [auto] Audit sur **IDs créés par le test** uniquement (pas `company_id` seul) : aucun `time_confirmed=false` + `scheduled_time` à 00:00
- [manuel staging] Créer réservations depuis **Entreprise**, **Mobile**, **Institution** ; exécuter l'audit SQL ci-dessous sur les lignes récentes

---

## 4. Requête SQL d'audit (staging / prod — données récentes)

```sql
-- Bookings récents « heure à définir » : ne doivent PAS être sentinelle 00:00
SELECT id, created_at, scheduled_time, time_confirmed, is_return, company_id
FROM booking
WHERE time_confirmed = false
  AND scheduled_time IS NOT NULL
  AND EXTRACT(HOUR FROM scheduled_time AT TIME ZONE 'Europe/Zurich') = 0
  AND EXTRACT(MINUTE FROM scheduled_time AT TIME ZONE 'Europe/Zurich') = 0
  AND created_at >= NOW() - INTERVAL '7 days'
ORDER BY created_at DESC;
```

**Attendu après Phase 1** : aucune **nouvelle** ligne (historique legacy acceptable jusqu'à Phase 4).

Variante ciblée (IDs connus d'un scénario de test) :

```sql
SELECT id FROM booking
WHERE id IN (:booking_ids)
  AND time_confirmed = false
  AND scheduled_time IS NOT NULL
  AND EXTRACT(HOUR FROM scheduled_time AT TIME ZONE 'Europe/Zurich') = 0
  AND EXTRACT(MINUTE FROM scheduled_time AT TIME ZONE 'Europe/Zurich') = 0;
```

---

## 5. Qualité — tests automatisés

```bash
docker compose exec -T atmr_api sh -c "cd /app && python -m pytest tests/integration/test_p1_sentinel_stop_gate.py -q"
```

| Exécution | Résultat |
|----------|----------|
| 2026-06-12 (local Docker) | **5 passed** |
| 2026-06-12 ruff | **All checks passed** |

```bash
docker compose exec -T atmr_api sh -c "cd /app && ruff check tests/integration/test_p1_sentinel_stop_gate.py"
```

---

## 6. Séquencement (ne pas inverser)

```txt
Phase 1 écritures          ✅ COMPLETED
STOP GATE P1               ⏳ tests auto PASS — staging manuel à valider
Phase 2 lectures           🟡 READY TO START (transporteur / mobile / notifications)
Phase 4 migration SQL      🔒 INTERDITE tant que Phase 2 n'est pas déployée et stabilisée
```

**Phase 4** (`UPDATE booking SET scheduled_time = NULL WHERE … 00:00 …`) : **ne jamais** exécuter avant remplacement complet des lectures `00:00 = à définir`.

---

## 7. Limitations connues (hors stop-gate P1)

- Lectures transporteur/mobile encore basées sur heuristique `00:00` → **Phase 2**
- Données historiques `00:00 + time_confirmed=false` → valides en lecture legacy jusqu'à Phase 4
- Retour cross-day (`return_date` ≠ jour aller) : visibilité dashboard J+1 documentée en limitation P1 si non étendue
