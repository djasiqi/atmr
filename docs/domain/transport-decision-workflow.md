# Workflow de décision transport — Architecture métier LIRIE

**Statut** : référence figée  
**Version** : 1.0  
**Livraison** : V1.1 (sécurité) → V1.2 (négociation) → V1.3 (orchestration)

Ce document est la source de vérité fonctionnelle. Toute implémentation qui mute une mission post-engagement hors de ce workflow constitue une **régression d’architecture**.

---

## 1. Centre du domaine

```text
Mission (vérité opérationnelle exécutable)
    ↑ effects transactionnels
TransportAction (vérité future possible)
    ↓
TransportActionExchange* (journal append-only)
    ↓
Décision humaine
    ↓
EffectPlan
  ├── transactional_steps  (même TX + outbox)
  └── post_commit_events   (worker après commit)
    ↓
TransportActionCompleted   ← seul événement métier central
    ↓
LegacyCompatibilityAdapter ← SEUL publisher d’événements legacy
    ↓
BookingCancelledEvent / fanouts historiques
```

- **Mission** = ce qui doit être exécuté maintenant (impl. V1 : `booking` / groupe lié / A/R).
- **TransportAction** = ce qui pourrait devenir vrai.
- **Exchange** = ce qu’un acteur a demandé, proposé, accepté ou refusé.
- **EffectPlan** = traduction technique de la décision.

### Principes

1. Une mission ne possède jamais plusieurs vérités simultanées.
2. LIRIE reçoit une **intention**, pas une modification de booking.
3. Les humains décident ; le moteur applique (sans opinion commerciale).
4. Anti-raccourcis : mutation post-engagement hors workflow = régression.
5. Traçabilité (pas d’event sourcing complet) :

> L’historique décisionnel et les mutations significatives d’une mission doivent être explicables et auditables à partir de la mission courante, des TransportActions, des exchanges, de la timeline et des événements persistés.

6. Aucun composant du cœur `TransportActionWorkflow` ne publie directement un événement legacy. Seul `LegacyCompatibilityAdapter` le fait.

### Responsabilités

| Acteur | Crée | Décide | Applique les effects |
|---|---|---|---|
| Institution | Intention (+ tour institution) | Oui si `next_actor_type=INSTITUTION` | Non |
| Entreprise | Non | Oui si `next_actor_type=COMPANY` | Non |
| Moteur LIRIE | Matérialise l’intention | Expire / replace / conflict (mécaniques) | **Oui** |
| Chauffeur | Non | Non | Non — destinataire post `effect_status=COMPLETED` |

Avant engagement transporteur (`DRAFT` / `SENT` / marché ouvert) : modification et annulation **directes** restent possibles (LIRIE applique immédiatement).

Après engagement (`ACCEPTED` / `ASSIGNED` ; annulation aussi en `EN_ROUTE`) : toute intention contractuelle passe par `TransportAction`.

---

## 2. Agrégats

### Mission (conceptuel V1)

Pas de table `missions` en V1.1.

```text
mission_ref → booking_id (+ mission_group_id / route_group_id)
action_scope → MISSION | BOOKING | LEG | ROUND_TRIP | SERIES_OCCURRENCE | SERIES_FUTURE
```

`MISSION` ≈ groupe lié existant.

### TransportAction

État global d’une décision contractuelle :

| Champ | Rôle |
|---|---|
| `action_type` | CHANGE_TIME, CHANGE_DATE, CHANGE_PICKUP_ADDRESS, CHANGE_DROPOFF_ADDRESS, CHANGE_ROUND_TRIP, CHANGE_PASSENGER_REQUIREMENTS, CHANGE_OTHER, CANCELLATION, INTERRUPTION |
| `status` | Voir machine à états |
| `effect_status` | NONE \| PENDING \| COMPLETED \| FAILED |
| `next_actor_type` | COMPANY \| INSTITUTION \| NONE |
| `active_exchange_id` | Échange ouvert à trancher |
| `mission_version_at_request` | `booking.edit_version` au moment de l’intention |
| `expires_at` | Selon ResponsePolicy |
| `billing_assessment_id` | Nullable ; calcul post-commit |

Persistance V1 : évolution de `booking_change_requests` + table `transport_action_exchanges`.

### TransportActionExchange (append-only)

V1.1 : `REQUEST` \| `ACCEPT` \| `REJECT`  
V1.2 : + `COUNTER`

Preuve principale : `actor_type`, `actor_id`, `authenticated_session_id?`, `created_at` serveur, `sequence`, `idempotency_key`.  
`client_meta` = complément (rétention limitée).

Payload COUNTER :

```json
{
  "proposed_mission_values": {},
  "commercial_terms": {},
  "comment": ""
}
```

### DecisionContextSnapshot

Immuable ; jamais d’UPDATE ; un snapshot lié à chaque exchange pertinent.  
UI : projection `current` + snapshots de preuve. Payload minimisé (privacy).

### Négociation

V1 = séquence d’exchanges sur une `TransportAction`.  
Extension future documentée : `TransportAction → Negotiation → Exchange*` (pause/reprise).

### Policies

| Policy | Rôle | Livraison |
|---|---|---|
| `NegotiationPolicy` | tours, champs, COUNTER on/off | V1.2 |
| `ResponsePolicy` | relances, expiration | V1.3 |
| `allowed_counter_fields(action_type, actor, mission_status)` | whitelist | V1.2 |

---

## 3. Machine à états

### `status`

```text
REQUESTED | COUNTER_PENDING | ACCEPTED | COMPLETED
REJECTED | EXPIRED | CLOSED_REPLACED
NEGOTIATION_LIMIT_REACHED | CONFLICTED
```

Pas de `EFFECT_FAILED` dans `status`.

### `effect_status`

```text
NONE | PENDING | COMPLETED | FAILED
```

### Combinaisons autorisées

```text
REQUESTED                  + NONE
COUNTER_PENDING            + NONE
ACCEPTED                   + PENDING
ACCEPTED                   + FAILED
COMPLETED                  + COMPLETED
REJECTED                   + NONE
EXPIRED                    + NONE
CLOSED_REPLACED            + NONE
NEGOTIATION_LIMIT_REACHED  + NONE
CONFLICTED                 + NONE
```

Lecture UI (`viewed_at`, `claimed_*`, `handling_status`) hors statut métier.

### CONFLICTED — terminal V1

Version mission divergente → action clôturée. Nouvelle intention obligatoire.

### ACCEPTED + effect_status=FAILED

Décision valide, application échouée → retry `complete_effects` idempotent.  
Pas de notif vérité chauffeur tant que `effect_status != COMPLETED`.

---

## 4. EffectPlan

```text
EffectPlan
├── transactional_steps   # vérité métier : tout ou rien
│     mutate booking / request
│     clear assignments / driver
│     timeline
│     status → COMPLETED ; effect_status → COMPLETED
│     clear active_action
│     outbox row(s)
└── post_commit_events
      Notification Engine
      LegacyCompatibilityAdapter
      analytics / projections
      BillingAssessmentRequested
```

`commercial_terms` acceptés = figés en TX. Comptabilité détaillée = post-commit.

---

## 5. Ordre transactionnel ACCEPT

```text
1. ACCEPT(action_id, accepted_exchange_id, idempotency_key)
2. Idempotence
3. Verrouiller TransportAction + mission
4. Vérifier active, next_actor, exchange, version, autorisation
5. Construire EffectPlan
6. UNE transaction :
   - status=ACCEPTED ; effect_status=PENDING
   - transactional_steps
   - status=COMPLETED ; effect_status=COMPLETED
   - timeline ; outbox TransportActionCompleted
7. Commit
8. Worker outbox → post_commit_events
```

Échec avant commit → rollback.  
Incident technique : TX distincte `ACCEPTED` + `FAILED` + `TransportActionEffectFailed` → retry.

---

## 6. Événements

```text
TransportActionRequested
TransportActionUpdated
TransportActionReplaced
TransportActionAccepted
TransportActionRejected
TransportActionExpired
TransportActionCompleted      # APRÈS commit effects
TransportActionEffectFailed
TransportActionConflicted
```

Payload ACCEPT : `accepted_exchange_id`, `accepted_values`, `accepted_by_*`, `mission_version_expected`.

---

## 7. Notifications vs décision

```text
NotificationDelivery: SENT | DELIVERED | OPENED | FAILED
TransportAction:      decided via exchanges / status
```

Importance : `VOLATILE` (COUNTER remplace) vs `PERSISTENT` (ACCEPT / annulation confirmée).  
Compteurs Actions requises : `next_actor_type`.

---

## 8. Règles métier

- Une décision **contractuelle** active par mission.
- Remplacement silencieux seulement si pas de négociation engagée ; sinon confirmation UI.
- Expiration V1 : clôture l’action entière ; mission inchangée.
- ACCEPT applique uniquement l’exchange `accepted_exchange_id`.
- Matrice action × statut :

| Action | ACCEPTED/ASSIGNED | EN_ROUTE | IN_PROGRESS |
|---|---|---|---|
| Changer heure | oui | limité | non |
| Changer date | oui | non | non |
| Changer pickup | oui | limité | non |
| Annuler | oui | oui critique | non → `INTERRUPTION_REQUIRED` |
| Interrompre | non | non | oui |

---

## 9. Livraison

### V1.1 — Sécurité

Exchanges REQUEST/ACCEPT/REJECT ; gate strict ; EffectPlan ; outbox ; Completed + LegacyAdapter ; file minimale ; tests ; COUNTER non exposé.

### V1.2 — Négociation (flag)

COUNTER ; NegotiationPolicy ; snapshots ; UI ; confirmation replace.

### V1.3 — Orchestration

ResponsePolicy ; Replaced/Conflicted UX ; delivery ack ; billing post-commit ; scopes A/R.

---

## 10. Annexe — ResponsePolicy (defaults opérationnels)

Non universels — configurables. Exemple :

| Délai avant départ | 1re relance | Suite | Escalade |
|---|---|---|---|
| > 24 h | 4 h | / 4 h | 12 h → EXPIRED |
| 2–24 h | 30 min | / 30 min | T−2 h CRITICAL |
| < 2 h | immédiat CRITICAL | / 15 min | bandeau + email |
| EN_ROUTE + cancel | immédiat CRITICAL | permanent | pas d’auto-cancel |

Absence de réponse ≠ acceptation. Jamais d’auto-cancel.

---

## 11. Critères de succès

- Attente ≠ annulé tant que non `COMPLETED` + `effect_status=COMPLETED`
- Pas de notif vérité chauffeur avant effects commités
- Décision obsolète / conflictuelle jamais appliquée
- Valeurs = exchange accepté
- Cœur sans publish legacy direct
- Idempotence
- Traçabilité auditable
- Aucun chemin de mutation hors workflow post-engagement

---

## 12. Anti-corruption PR2

L’ancien stop-gate (`docs/ops/stop-gate-pr2-redispatch.md`) est **obsolète** :
- refuse n’applique plus le patch ni le redispatch automatique ;
- cancel institution post-engagement n’est plus immédiat ;
- le cœur publie `TransportActionCompleted`, pas `BookingCancelledEvent` directement.

---

## 13. Annulation post-engagement — conséquences commerciales (V1)

✅ **Implémenté** : spécification V1 définitive — actors, 3 dimensions, OUTBOUND_ONLY, `CancellationRespondPolicy`, `respond_ui`, accept sous verrou, `commercial_terms`, Effect Invariants, codes d’erreur figés.  
Réf. code : `application/institutions/cancellation_respond_policy.py`, `respond_to_change_request.py`, `transport_action_workflow._apply_cancellation_effects`.

### Acteurs

| Décision | Question | Acteur |
|---|---|---|
| Opérationnelle | L’institution annule-t-elle ? | Institution |
| Applicabilité | Peut-on appliquer l’annulation à chaque booking affecté ? | LIRIE |
| Commerciale | Quels frais sur l’**aller** éligible ? | Entreprise |
| Exception | Demande erronée / impossible ? | Entreprise (`report_problem` / REJECT) |

CTA : `acknowledge_cancellation` | `confirm_with_billing` | `report_problem`.

### Trois dimensions

```text
affected_bookings       = ce que la demande vise
cancelable_bookings     = ce que le moteur peut encore modifier
billing_eligible_booking = l’unique trajet aller pouvant porter des frais
```

```text
ALLER  → seul trajet éventuellement facturable
RETOUR → peut être annulé, jamais facturé
```

### Situations & outcomes

| Situation | Outcomes | Body | CTA |
|---|---|---|---|
| `NON_BILLABLE_RETURN` | ZERO | implicite OK | `acknowledge_cancellation` |
| `FREE_WINDOW` | ZERO | implicite OK | `acknowledge_cancellation` |
| `FEE_WINDOW` | ZERO, POLICY_FEE, CUSTOM | obligatoire | `confirm_with_billing` |
| `EN_ROUTE` | ZERO, APPROACH_FEE, FULL_FARE, CUSTOM | obligatoire | `confirm_with_billing` |

Les 0 CHF restent distinguables via `calculation_code` (`FREE_CANCELLATION_WINDOW` ≠ `NON_BILLABLE_RETURN` ≠ `COMPANY_WAIVED`).

### Validation atomique sous verrou

```text
lock action → lock affected → relire → recalculer policy complète
→ comparer respond_context_version → valider choix → EffectPlan + commercial_terms → commit
```

Le `respond_ui` pré-clic n’est jamais l’autorité finale.

### Contrat d’erreurs

```text
409 CANCELLATION_RESPONSE_CONTEXT_CHANGED
    POLICY_CHANGED | TIME_WINDOW_CHANGED | BOOKING_STATUS_CHANGED
    | SCOPE_CHANGED | AMOUNT_CHANGED | AFFECTED_BOOKINGS_CHANGED
422 INTERRUPTION_REQUIRED
422 BILLING_OUTCOME_REQUIRED | BILLING_OUTCOME_NOT_ALLOWED
422 FEE_AMOUNT_NOT_ALLOWED
422 CUSTOM_FEE_AMOUNT_REQUIRED | CUSTOM_FEE_AMOUNT_INVALID
    | CUSTOM_FEE_AMOUNT_OUT_OF_RANGE
422 BILLING_COMMENT_REQUIRED
```

`fee_amount` (string décimale) uniquement pour `CUSTOM`.

### Cancellation Effect Invariants

1. Seuls les `cancelable_booking_ids` sont mutés.
2. Chaque cancelable est annulé.
3. Un seul booking reçoit des frais : `billing_eligible_booking_id`.
4. Ce booking est obligatoirement `OUTBOUND`.
5. Les autres cancelables : `clear_cancellation_billing(...)`.
6. Les `non_cancelable_booking_ids` restent strictement inchangés.
7. Pas d’aller éligible ⇒ `billing_scope=NONE`, outcome ZERO, fee 0.
