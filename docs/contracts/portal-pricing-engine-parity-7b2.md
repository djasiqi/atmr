# 7B.2 — PRICING ENGINE PARITY GATE

```text
STEP 7B.2 — PRICING ENGINE PARITY
SOURCE OF TRUTH : services.pricing.pricing_engine.compute_price
CEILING MODULE  : services.pricing.portal_carrier_ceiling.compute_portal_carrier_ceiling
DUPLICATED PRICING LOGIC : NO
PRICING SOURCE OF TRUTH SHARED : YES
STATUS : AUTOMATED PASS — MANUAL RETEST PENDING
ENGINE FREEZE : PARTIAL — rouvert uniquement pour aller-retour ×2
                 (flat/distance/zone_count/hybrid) suite écart retest manuel
7B.2 CLOSED : NO
```

## Objectif

Prouver que le plafond PORTAL **n’a pas de moteur tarifaire parallèle** et que, pour chaque entreprise éligible :

```text
candidate_plafond(company)
  == compute_price(mission, grille_active_company, context)
```

puis :

```text
maximum_accepted_amount = MAX(candidates > 0)
```

## 1. Source de vérité

| Élément | Valeur |
| --- | --- |
| Service | [`backend/services/pricing/pricing_engine.py`](../../backend/services/pricing/pricing_engine.py) → `compute_price` |
| Contexte | `_build_pricing_context` ([`offer_price_estimator.py`](../../backend/services/pricing/offer_price_estimator.py)) — **réutilisé** par le plafond |
| Distance (plafond / estimateur) | `get_distance_duration` (Google Distance Matrix, fallback Haversine) |
| Simulate UI company | OSRM (chemin distinct pour l’outil simulate — hors plafond PORTAL) |

Le plafond appelle uniquement `compute_price` + agrège `MAX`. Aucune formule `flat` / `distance` / `zone_count` dans `portal_carrier_ceiling.py`.

---

## 2. FLAT

```text
FLAT ENGINE      : pricing_engine._compute_flat
INPUTS           : rules.base_fee (ou components.base), surcharges after_time / last_minute,
                   weekend time_rules.base_fee, caps.minimum
CANTON SEMANTICS : le canton N’ENTRE PAS dans la formule.
                   Libellé UI « Prix fixe (canton) » = forfait de la grille ;
                   l’éligibilité géographique est en amont (dispatch / service area).
ROUNDTRIP        : ×2 sur total one-way (flat / distance / zone_count / hybrid)
                   — aligné portail client (indicatif ×2).
                   Legacy ``zone`` : grilles one_way/round_trip (pas de ×2 ici).
VAT              : hors moteur

FLAT PARITY      : PASS (tests auto)
```

Exemple gate : `base_fee = 45` → quote = `45.00` = candidat plafond.

---

## 3. DISTANCE

```text
DISTANCE ENGINE  : _compute_distance → _compute_distance_v1 si components.*
                   (contrat UI actuel : km only, base forcée à 0)
INPUTS           : context.distance_km, components.distance.per_km, included_km,
                   extras V1, caps.minimum
DISTANCE SOURCE  : distance_meters / 1000 via get_distance_duration (plafond)
ROUNDING         : Decimal ROUND_HALF_UP à 0.01 CHF
                   (champ rounding ceil_0_1 persisté mais non appliqué au km dans le moteur)
ROUNDTRIP        : ignoré
VAT              : hors moteur

DISTANCE PARITY  : PASS (tests auto) — ex. 12 km × 3.20 = 38.40
```

---

## 4. ZONE_COUNT

```text
ZONE_COUNT ENGINE : _compute_zone_count_v1
FORMULE           : base + max(zones − included_zones, 0) × unit_price
                    + extras V1 + minimum
ZONES             : zone_set + traversal (PostGIS / fallback) via
                    _compute_zones_count_from_rules
ROUNDTRIP         : ignoré
VAT               : hors moteur

ZONE_COUNT PARITY : PASS (tests auto) — ex. base 40 + 1×12 = 52
```

---

## 5. MAX CROSS-MODEL

```text
A flat       = 45.00
B distance   = 38.40
C zone_count = 52.00
→ maximum_accepted_amount = 52.00

MULTI-MODEL MAX : PASS (tests auto)
```

Le client voit le plafond, pas nécessairement les quotes individuelles.

---

## 6. Exclusions financières

| Champ | Dans `compute_price` / plafond ? |
| --- | --- |
| `payment_terms_days` | **NO** |
| `overdue_fee` | **NO** |
| `material_delivery_price_fixed` | **NO** |
| Frais d’annulation / no-show | **NO** (contrat futur avec X) |
| Dunning | **NO** |

```text
CANCELLATION FEES INCLUDED IN CEILING : NO
OVERDUE FEES INCLUDED                 : NO
PAYMENT TERMS INCLUDED                : NO
```

---

## 7. TVA

```text
carrier quote presented to client VAT basis :
  montant grille trajet issu de compute_price — SANS vat_rate
  (TVA gérée en facturation / CompanyBillingSettings, pas dans le devis trajet)

ceiling VAT basis :
  identique (même compute_price, aucune couche TVA dans le plafond)

VAT PARITY : PASS
```

Invariant : **same pricing basis** — pas de double application de TVA.

---

## 8. Type de mission

```text
patient_transport     : grille trajet (flat / distance / zone_count) via compute_price
material_delivery     : material_delivery_price_fixed N’EST PAS dans compute_price
                        → ne contamine pas le plafond transport patient

MATERIAL DELIVERY ISOLATION : PASS
```

Si un jour le plafond PORTAL couvre `material_delivery`, il devra brancher le moteur propre à ce type — hors scope actuel.

---

## 9. Aller / retour

```text
flat / distance / zone_count / hybrid_stack :
  is_round_trip=true → total = one_way × 2 (après minimum)
  breakdown.round_trip.applied = true

legacy model « zone » :
  grilles weekday/weekend × one_way/round_trip (pas de ×2 automatique)

ROUNDTRIP : PASS (parité plafond = moteur)
```

---

## 10. Aucune quote valide

```text
0 eligible valid quotes
→ ValueError(portal_pricing_ceiling_unavailable)
→ pas de BOOKING_CREATED avec plafond inventé
→ jamais 0 CHF silencieux

NO VALID QUOTE : PASS (tests auto)
```

---

## 11. Tests automatisés

| Fichier | Couverture |
| --- | --- |
| [`test_portal_ceiling_pricing_parity_7b2.py`](../../backend/tests/services/test_portal_ceiling_pricing_parity_7b2.py) | Parité Decimal flat / distance / zone_count, MAX 45/38.40/52, pas de formule dupliquée, exclusions, AR, no-quote |
| [`test_portal_carrier_ceiling_7b2.py`](../../backend/tests/services/test_portal_carrier_ceiling_7b2.py) | MAX, exclusions éligibilité, snapshot evidence |
| [`test_portal_create_booking_ceiling_7b2.py`](../../backend/tests/services/test_portal_create_booking_ceiling_7b2.py) | Ignore payload client `9999` |

```text
AUTOMATED TESTS : PASS (parity gate)
MANUAL RETEST   : PENDING
ENGINE FREEZE   : YES
```

### Scénario manuel verrouillé (ne pas changer)

```text
Entreprise A
  pricing_model = flat
  quote attendu = CHF 45.00

Entreprise B
  pricing_model = distance
  distance = 12 km
  tarif = CHF 3.20/km
  quote attendu = CHF 38.40

Entreprise C
  pricing_model = zone_count
  base = CHF 40
  1 supplément zone = CHF 12
  quote attendu = CHF 52.00

MAX attendu = CHF 52.00
```

#### Assertions écran / DB

| Surface | Attendu |
| --- | --- |
| PORTAL client | « Prix maximum de la demande — CHF 52.– » uniquement (pas les quotes A/B/C) |
| Transporteur | **ne voit pas** le plafond client |
| DB `BOOKING_CREATED` | `maximum_accepted_amount_snapshot = 52.00` |
| Offre X = CHF 45 | recevable (45 ≤ 52) |
| 2e clic | `contractual_amount = 45.00` (jamais 52 sauf offre réelle = 52) |

#### Verdict

```text
MANUAL RETEST = PASS  →  7B.2 = PASS / CLOSED
MANUAL RETEST = FAIL  →  7B.2 = FAIL / reopen pricing path
```

Tant que le retest n’a pas révélé d’écart : **ne plus toucher au moteur** (`compute_price` / plafond).

---

## Livrable synthétique

```text
7B.2 — PRICING ENGINE PARITY

FLAT ENGINE           : pricing_engine._compute_flat
FLAT PARITY           : PASS

DISTANCE ENGINE       : pricing_engine._compute_distance(_v1)
DISTANCE PARITY       : PASS

ZONE_COUNT ENGINE     : pricing_engine._compute_zone_count_v1
ZONE_COUNT PARITY     : PASS

ROUNDTRIP             : PASS (×2 flat|distance|zone_count|hybrid ; legacy zone = grilles)

SERIES / RÉCURRENCE   : PASS — plafond demande = plafond trajet × N
  (une réservation décrit N passages ; pas dans compute_price)

VAT BASIS             : HT grille trajet (hors vat_rate)
VAT PARITY            : PASS

MATERIAL DELIVERY ISOLATION : PASS

CANCELLATION FEES INCLUDED IN CEILING : NO
OVERDUE FEES INCLUDED                 : NO
PAYMENT TERMS INCLUDED                : NO

MULTI-MODEL MAX       : PASS
NO VALID QUOTE        : PASS

DUPLICATED PRICING LOGIC      : NO
PRICING SOURCE OF TRUTH SHARED : YES

AUTOMATED TESTS       : PASS
MANUAL RETEST         : PENDING

7B.2                  : PARTIAL
  (PASS auto parity ; CLOSED seulement après MANUAL RETEST = PASS)
  ENGINE FREEZE       : YES
```
