# CANCEL-BILLABLE — AUDIT

Chantier **séparé**. N’ouvre pas Institution Billing (G1–G4 restent
**CLOSED**). Aucune modification métier dans ce document : lecture
seule, 5 septembre 2026, HEAD certifié contestation `1b7acbeb`.

```text
CANCEL-BILLABLE — AUDIT

C1 — éligibilité     CLOSED / PASS (6/6)
C2 — montant         OPEN
C3 — libellé         OPEN
C4 — émission PDF/QR OPEN
```

Branche : `feat/cancel-billable-c1` (pas #97, pas `1b7acbeb`).
Tests : `backend/tests/application/test_cancel_billable_c1_eligibility.py`.

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

**C1 = CLOSED / PASS.** `#97` / `1b7acbeb` intacts. Prochain gate : C2
(`fee=NULL → 90` vs `fee=0 → 0`).

## Sentinelles restantes (C2–C4, pas C1)

| Cas | Attendu |
| --- | --- |
| Annulation facturable 45 sur course 90 | présente à **45**, pas 90 |
| NO_SHOW facturable | motif NO_SHOW, pas « dernière minute » |

## Hors scope

Institution Billing G1–G4, contestation, `partner`, nouveaux payeurs,
extension A/R hors politique OUTBOUND_ONLY déjà posée.

## Fermeture

`CANCEL-BILLABLE CLOSED / PASS` seulement quand C1–C4 sont verts sur
les cinq sentinelles, **sans** modifier le HEAD contestation
`1b7acbeb` sauf régression démontrée sur ce HEAD.
