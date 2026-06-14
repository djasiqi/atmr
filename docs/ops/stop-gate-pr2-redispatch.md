# Stop-gate PR2 — Révalidation transporteur & redispatch

> **Périmètre :** révalidation des modifications critiques institution après
> acceptation transporteur (workflow accepter/refuser + remise en diffusion).
> **Statut :** PR2 (stop-gate). À valider avant activation en production.

---

## 1. Objectif métier

Avant PR2 : une modification institution sur une course déjà acceptée n'invalidait
**pas** l'engagement du transporteur (seules des alertes WARNING/CRITICAL étaient
émises — voir `docs/AUDIT-INSTITUTION-RESERVATION.md`, Phase 3).

Avec PR2, toute **modification critique** (champs `MAJOR_FIELDS` : horaire,
adresses, GPS, fauteuil, type de mission, nom patient) sur une course **acceptée
ou assignée** (transporteur engagé, course non démarrée) :

1. **N'applique pas** le patch immédiatement.
2. Crée une **demande de validation** (`BookingChangeRequest`, statut `pending`).
3. Le transporteur **accepte** (patch appliqué, engagement conservé) ou **refuse**
   (patch appliqué — la décision institution fait foi — puis **remise en diffusion**).
4. Chaque étape est historisée de façon immuable dans la timeline transport.

---

## 2. Feature flags / configuration

| Variable d'environnement | Défaut | Effet |
| --- | --- | --- |
| `INSTITUTION_CHANGE_REVALIDATION_ENABLED` | `true` | Active la révalidation. Si `false`, comportement legacy (patch appliqué directement). |
| `INSTITUTION_CHANGE_REQUEST_TTL_MINUTES` | `120` | Délai avant expiration d'une demande de validation. |
| `AUTO_REFUSE_EXPIRED_CHANGE_REQUESTS` | `false` | Si `true`, une demande expirée est traitée comme un refus (libération + redispatch). Sinon : `expired` → `escalation_required` (action institution requise), transporteur conservé. |

---

## 3. Composants livrés

| Couche | Fichier |
| --- | --- |
| Service révalidation | `backend/services/institutions/booking_change_service.py` (`create_change_request`, `supersede_pending_change_requests`, `is_revalidation_enabled`, `get_pending_change_request_view`) |
| Use case réponse | `backend/application/institutions/respond_to_change_request.py` |
| Use case libération | `backend/application/institutions/release_booking_for_redispatch.py` |
| Use case redispatch | `backend/application/institutions/redispatch_institution_booking.py` |
| Routes entreprise | `backend/routes/companies.py` (`/accept`, `/refuse`) |
| Tâche Celery | `backend/tasks/change_request_tasks.py` (`expire_pending_change_requests`) |
| Vue résumé booking | `backend/models/transport_request.py` (`pending_change_request`) |
| Modèle | `backend/models/booking_change_request.py` (existant) |
| Migration | `backend/migrations/versions/20260611_institution_timeline.py` (existant) |
| Tests | `backend/tests/integration/test_pr2_redispatch_stop_gate.py` |

---

## 4. API entreprise (transporteur)

```
POST /api/v1/companies/me/reservations/{booking_id}/change-requests/{change_id}/accept
POST /api/v1/companies/me/reservations/{booking_id}/change-requests/{change_id}/refuse
```

Corps JSON :

```json
{ "version": 1, "reason": "optionnel" }
```

Réponses :

| Code | Signification |
| --- | --- |
| `200` | Réponse enregistrée (accept/refuse). |
| `400` | `version` manquante ou action invalide. |
| `403` | L'entreprise n'est pas le transporteur de la course. |
| `404` | Course ou demande introuvable. |
| `409` | Conflit : version périmée, demande superseded, ou `active_change_request_id` ne correspond pas. |

---

## 5. Machine à états `BookingChangeRequest`

```
pending ──accept──▶ accepted
   │
   ├──refuse──▶ refused ──▶ (release + redispatch)
   │
   ├──nouvelle modif──▶ superseded
   │
   └──expiration──▶ expired ──▶ escalation_required
                         │
                         └─(si AUTO_REFUSE)─▶ refused ──▶ (release + redispatch)
```

Verrou optimiste : `version` (incrémentée à chaque transition) + contrôle que la
BCR est bien `booking.active_change_request_id`. Verrou pessimiste `SELECT FOR
UPDATE` sur `Booking` + `BookingChangeRequest` dans le use case de réponse.

---

## 6. Checklist de validation (stop-gate)

### Migration & schéma
- [x] `booking_change_requests` présente en base (migration `20260611_institution_timeline` appliquée le 2026-06-11).
- [x] Colonne `booking.active_change_request_id` (FK `ON DELETE SET NULL`) présente.
- [x] Index `ix_bcr_booking_status` et `ix_bcr_institution_created` créés.

### Workflow nominal
- [x] Modification critique sur course `ACCEPTED`/`ASSIGNED` → `202 pending_revalidation`, patch non appliqué (test Cas 1).
- [x] Modification mineure (notes) → `200`, patch appliqué directement (test Cas 1).
- [ ] Modification sur course `PENDING` (marché ouvert, sans transporteur) → patch direct.
- [ ] Modification `EN_ROUTE` → comportement existant (alerte CRITICAL + ACK), pas de BCR.
- [x] `booking_summary.pending_change_request` exposé côté institution (code `_build_single_booking_summary`).

### Réponse transporteur
- [x] Accept → patch appliqué, `edit_version` incrémentée, `active_change_request_id = NULL` (test Cas 2).
- [x] Refuse → patch appliqué, course `PENDING`, transporteur détaché, nouvelles offres créées (test Cas 3).
- [ ] Réponse par une autre entreprise que le transporteur → `403`.
- [x] Version périmée / demande superseded → `409` (test Cas 5).

### Supersession & concurrence
- [x] Deux modifications successives → la première passe `superseded`, `active_change_request_id` pointe la seconde (test Cas 4).
- [x] Réponse concurrente (Cas 5) → une seule gagne, l'autre `409`.
- [ ] **À COMPLÉTER** : test réellement parallèle (threads + sessions DB distinctes) — squelette présent.

### Expiration (Celery)
- [x] `expire_pending_change_requests` planifiée (beat) et incluse dans `celery_app.py`.
- [ ] BCR expirée, flag OFF → `expired` puis `escalation_required`, transporteur conservé.
- [ ] BCR expirée, `AUTO_REFUSE_EXPIRED_CHANGE_REQUESTS=true` → refus + redispatch.

### Timeline / audit
- [x] `change_confirmation_requested` historisé à la création (avec `source_event_id`).
- [x] `change_accepted_by_company` / `change_refused_by_company` historisés.
- [x] `redispatched`, `change_expired`, `escalation_required` historisés.
- [x] `booking_change_events` enrichi (`change_request_created`, `field_updated`).

### Qualité
- [x] `docker compose exec atmr_api pytest tests/integration/test_pr2_redispatch_stop_gate.py` vert (6 passed, 1 skipped — PostgreSQL).
- [x] Lint/format propres sur les fichiers touchés (`ruff check` : All checks passed).

---

## 7. Limitations connues (à traiter en PR ultérieur)

- **Re-création de booking au redispatch** : `AcceptOfferUseCase` crée un nouveau
  `Booking` à l'acceptation. La remise en diffusion (`redispatch_institution_booking`)
  rouvre la `TransportRequest` source (statut `SENT` + nouvelles offres broadcast),
  mais le rattachement de l'ancien booking libéré au nouveau transporteur n'est pas
  encore géré. À traiter en PR3.
- **Test concurrent réellement parallèle** : seul le squelette (version périmée) est
  fourni ; le test multi-threads/multi-sessions reste `@pytest.mark.skip`.
- **Notifications temps réel** : seule la notification in-app entreprise
  (`institution_change_request`) est émise ; les canaux socket/push fins restent à
  câbler.
