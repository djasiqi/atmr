# CANCEL-BILLABLE — AUDIT

Chantier **séparé**. N’ouvre pas Institution Billing (G1–G4 restent
**CLOSED**). Aucune modification métier dans ce document : lecture
seule, 5 septembre 2026, HEAD certifié contestation `1b7acbeb`.

```text
CANCEL-BILLABLE

C1 ✅ CLOSED / PASS   HEAD 08eefae4
C2a ✅ SOURCE CANONIQUE CLOSED / PASS
C2b ✅ UNRESOLVED CONSUMER CLOSED / PASS
C3 ✅ CLOSED / PASS   libellés 5/5
C4 HOLD

#97 / 1b7acbeb = INTOUCHÉ
```

Branche : `feat/cancel-billable-c1` (pas #97, pas `1b7acbeb`).
C1 figé : `08eefae4` (`cert(billing): figer CANCEL-BILLABLE C1 CLOSED`).
Tests C1 : `backend/tests/application/test_cancel_billable_c1_eligibility.py`.
Tests C2 : `backend/tests/application/test_cancel_billable_c2_amount.py`.
Tests C3 : `backend/tests/application/test_cancel_billable_c3_labels.py`.

Comportement voulu :

```text
CANCELED + facturable
→ on ne facture pas la course effectuée
→ on facture des frais d'annulation
→ montant et motif clairement identifiables
```

Exemple oracle :

```text
Course normale                 90 CHF
Statut                         CANCELED
Frais d'annulation             45 CHF
is_cancellation_billable       true

Facture attendue               45 CHF
et non 90 CHF
```

## Ce qui est déjà prévu (lecture)

- Patient / clinique générique : une course `CANCELED` peut entrer si
  `amount > 0` et (`is_cancellation_billable` **ou** override).
  Source : `billing_period_eligibility.py`.
- Montant canonique : si `cancellation_fee_amount` n’est pas `NULL`,
  preview / generate / totaux doivent utiliser ce frais.
  Source : `billable_amount.py` (`source=cancellation_fee_amount`).
- Politique A/R : un seul aller porte les frais ; `is_return = false`
  côté clinique. Source : preview S2 + `generate_clinic_monthly_invoice.py`.
- Libellés métier : `get_cancellation_display_label` /
  `CANCELLATION_REASON_LABELS` (`NO_SHOW` = « Client ne s'est pas
  présenté », etc.).

## Incohérences confirmées dans le code (pas encore certifiées)

### C1 — ClientStay imposé à la clinique (P0)

Course **terminée** clinique : `billed_to_type=clinic` +
`billed_to_company_id` (prédicat S2) suffit.

Course **annulée mais facturable** clinique : preview et génération S2
exigent **en plus** un `ClientStay` actif à `scheduled_time`.

```text
COMPLETED clinique     → billed_to_company_id suffit
CANCELED facturable    → billed_to_company_id ne suffit plus
                       → ClientStay actif obligatoire
```

Une réservation portail institution, correctement à charge clinique,
peut donc être commercialement facturable et **disparaître de la
facture S2** s’il n’existe aucun séjour.

Hypothèse à trancher (C1) : **elle doit rester facturable** sans
ClientStay, le payeur explicite suffisant.

Écart secondaire C1 : la preview S2 accepte aussi
`billing_override_reason` ; la génération clinique n’accepte que
`is_cancellation_billable=True`. Même booking peut donc être dans
l’aperçu et hors émission.

### C2 — NULL / 0 / registre vs preview

`calculate_billable_booking_amount` :

```text
cancellation_fee_amount = NULL  → booking.amount (ex. 90)
cancellation_fee_amount = 0     → 0 CHF (frais appliqué)
```

Le registre des opportunités patient exige `cancellation_fee_amount IS NOT NULL`
(`billing_opportunities.and_canceled_billable`). L’éligibilité période
demande seulement `is_cancellation_billable` (ou override) + `amount > 0`.

Risques à certifier :

```text
même booking annulé
registre  → absent
preview   → présent à 90 CHF
```

```text
is_billable = true
fee_amount  = 0
→ ligne « facturable » à 0 CHF
```

sans que ce soit volontaire.

### C3 — libellé facture

`InvoiceDescriptionBuilder` :

```text
si annulée + pourcentage + palier
→ "Annulation (< 12h) – 40%"
sinon
→ "Annulation dernière minute"
```

`cancellation_display_label` n’est **pas** lu. Un `NO_SHOW` 100 CHF
peut s’imprimer « Annulation dernière minute ». Défaut de
représentation, même si le montant est juste.

### C4 — émission

Pas de sentinelle PDF / QR aujourd’hui pour les frais d’annulation.
G4 Institution Billing ne couvre que contestation 320 / 360.

## C1 — verdict 5 septembre 2026 (tests first)

Règle produit figée : **le rattachement clinique explicite prime
sur ClientStay**. `ClientStay` peut aider à trouver un payeur
manquant ; il ne doit pas annuler une dette déjà attribuée.

Invariant C1 :

```text
même booking
→ registre (opportunités cliniques = preview)
→ period-preview
→ generate_clinic_monthly
= même réponse à « cette annulation est-elle éligible ? »
```

| Cas | Attendu | Verdict |
| --- | --- | --- |
| `CANCELED` non facturable | exclue | **PASS** |
| `CANCELED` facturable patient | incluse | **PASS** |
| `CANCELED` facturable clinique + ClientStay | incluse | **PASS** |
| Clinique **sans ClientStay**, payeur explicite | **incluse** | **PASS** |
| Override valide | preview **et** generate identiques | **PASS** |
| A/R, frais aller uniquement | aller inclus / retour exclu | **PASS** |

Invariant structurel : `eligible booking IDs preview == generate`.

```text
C1 CLOSED / PASS

6/6 sentinelles

clinic explicit payer does not require ClientStay
override parity = PASS
A/R peer cannot bypass eligibility
preview IDs == generate IDs

C2 untouched
C3 untouched
C4 untouched
```

✅ **Implémenté** : règle unique dans `clinic_s2_eligibility.py`
(`cancellation_authorized_sql` / `clinic_canceled_billable_sql` /
`filter_clinic_s2_financial_segments`). Preview + generate S2 + totaux
consomment le même helper. Plus de `ClientStay` comme condition
financière d’annulation. Expansion A/R autorisée pour le contexte ;
revalidation C1 avant construction des unités. Tests :
`backend/tests/application/test_cancel_billable_c1_eligibility.py`.

**C1 = CLOSED / PASS.** HEAD figé `08eefae4`. `#97` / `1b7acbeb` intacts.

## C2 — verdict 5 septembre 2026 (tests first, sans correctif)

Contrat : sur `CANCELED`, `booking.amount` est le tarif course.
Le montant facturé doit venir d’un frais **explicitement résolu**.
`NULL` = non résolu, jamais un fallback silencieux à 90.

Invariant :

```text
effective_amount registry
==
effective_amount preview
==
effective_amount generation
```

| Cas | amount | fee | Attendu | Verdict |
| --- | ---: | ---: | ---: | --- |
| `COMPLETED` | 90 | NULL | **90** / `booking.amount` | **PASS** |
| Annulation frais partiels | 90 | 45 | **45** / `cancellation_fee_amount` | **PASS** |
| Annulation plein tarif explicite | 90 | 90 | **90** / `cancellation_fee_amount` | **PASS** |
| Annulation frais explicitement 0 | 90 | 0 | **0** / `cancellation_fee_amount` | **PASS** |
| Annulation facturable, frais NULL | 90 | NULL | unresolved, jamais 90 | **PASS** |

✅ **Implémenté** (source canonique seulement) :
`normalize_booking_status` dans `booking_status.py` ;
`calculate_billable_booking_amount` n’utilise plus `str(status)`.
`CANCELED` + fee non NULL → frais persisté. `CANCELED` + fee NULL →
`amount_ht=0`, `source=cancellation_fee_unresolved`, `resolved=false`.
Pas d’exception. Pas de fallback `booking.amount`. 90 CHF n’est
autorisé que si `cancellation_fee_amount=90` (FULL_FARE explicite).

```text
C2 CLOSED / PASS

COMPLETED 90 / fee NULL
→ 90 / booking.amount

CANCELED 90 / fee 45
→ 45 / cancellation_fee_amount

CANCELED 90 / fee 90
→ 90 / cancellation_fee_amount

CANCELED 90 / fee 0
→ 0 / cancellation_fee_amount

CANCELED 90 / fee NULL
→ unresolved
→ jamais booking.amount

C1 untouched (08eefae4)
C3 untouched
C4 untouched
```

## C2b — consommateur unresolved (5 septembre 2026)

```text
CANCELED + autorisée + fee NULL
→ pas facturable immédiatement
→ exclue du total
→ signalée « montant d'annulation à déterminer »
→ n'empêche pas l'émission des autres prestations
```

Sentinelle : course A `COMPLETED` 320 + course B `CANCELED` 40 / fee NULL
→ facture **320**, B excluded / needs review, pas de ligne 0 CHF.
Pendant : fee B = 35 → total **355** / `cancellation_fee_amount`.

| Surface | `fee=NULL` | Verdict |
| --- | --- | --- |
| Registre / plan | identifiée, hors total | **PASS** |
| Preview | exclue du total, motif explicite | **PASS** |
| Generate | aucune `InvoiceLine` | **PASS** |
| Autres prestations | facturables (320) | **PASS** |

✅ **Implémenté** : `partition_invoiceable_bookings` dans
`billable_amount.py`. Preview / generate S2 / generate patient / plan
consomment `resolved`. Pas de fallback `booking.amount`. Tests :
`test_cancel_billable_c2b_unresolved_consumer.py`.

```text
C2b CLOSED / PASS

resolved cancellation fee → peut entrer dans la facture
unresolved → ne modifie jamais le total
unresolved → ne crée jamais de ligne financière

C3 HOLD
C4 HOLD
#97 / 1b7acbeb = INTOUCHÉ
```

## C3 — verdict 5 septembre 2026 (tests first, sans correctif)

Contrat : une facture d’annulation explique **pourquoi** elle est
facturée. Le motif réel ne doit jamais être remplacé par le générique
« Annulation dernière minute ».

Priorité des sources :

```text
1. cancellation_display_label persisté
2. sinon get_cancellation_display_label(reason_code, reason_text)
3. sinon "Annulation (historique)"
```

Invariant : `preview description == generated InvoiceLine.description`.
Pas de PDF (C4).

| Cas | Attendu | Verdict |
| --- | --- | --- |
| `LAST_MINUTE` | Annulation dernière minute | **PASS** |
| `NO_SHOW` | Client ne s'est pas présenté | **PASS** |
| `CLIENT_REQUEST` + 50 % | motif client + frais 50 % | **PASS** |
| `OTHER` + commentaire | commentaire métier, pas `OTHER:` | **PASS** |
| historique sans motif | Annulation (historique) | **PASS** |

✅ **Implémenté** : `canonical_cancellation_invoice_label` dans
`invoice_line_description.py`. Statut via `booking_status_is_canceled`.
Preview S2 et generate S2 / patient passent par le même helper.
`cancellation_fee_tier_id` n'est plus un texte client.

```text
C3 CLOSED / PASS

5/5

LAST_MINUTE
NO_SHOW
CLIENT_REQUEST + frais %
OTHER + commentaire
historique

preview description == InvoiceLine.description
aucune annulation décrite comme trajet effectué

C1 untouched
C2 untouched
C4 untouched
```

Tests : `backend/tests/application/test_cancel_billable_c3_labels.py`.
C4 ensuite : PDF conserve motif + montant C2, QR conserve le total.

## Hors scope

Institution Billing G1–G4, contestation, `partner`, nouveaux payeurs,
extension A/R hors politique OUTBOUND_ONLY déjà posée.

## Fermeture

`CANCEL-BILLABLE CLOSED / PASS` seulement quand C1–C4 sont verts sur
les cinq sentinelles, **sans** modifier le HEAD contestation
`1b7acbeb` sauf régression démontrée sur ce HEAD.
