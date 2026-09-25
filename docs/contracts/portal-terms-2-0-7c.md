# 7C — CGU / Conditions de réservation et transport PORTAL 2.0

```text
STEP 7C — PORTAL TERMS 2.0
PREREQUISITES : 7A CLOSED, 7B PASS/CLOSED
STATUS : REOPENED — textes 2.0 PREPARED encore alignés sur double_validation_v2 (2e clic)
CIBLE PRODUIT : conditional_order_v1 (7B.5) —
  [portal-single-click-conditional-order-7b5.md](portal-single-click-conditional-order-7b5.md)
  → rewrite Terms requis avant nouvelle PREPARED / hashes
PORTAL_DOUBLE_VALIDATION_ENABLED : false (inchangé)
PORTAL_TERMS_EFFECTIVE_VERSION : 1.0 (défaut)
2.0 EFFECTIVE IN PRODUCTION : NO
```

## Audit resolver — gate d'activation

```text
ADDING VERSION 2.0 AUTOMATICALLY MAKES IT CURRENT : NO
```

Preuve : [`portal_terms_catalog.current_portal_terms()`](../../backend/services/legal/portal_terms_catalog.py) pinne explicitement la version via `PORTAL_TERMS_EFFECTIVE_VERSION` (défaut `1.0`). Ce n'est **pas** un « max version » ni un tri DB.

| Concept | Mécanisme |
| --- | --- |
| PREPARED | `prepared_portal_terms_v2()` + fichiers `*_v2.0.txt` |
| CURRENT | `current_portal_terms()` ← env `PORTAL_TERMS_EFFECTIVE_VERSION` |
| Coordination 7D | `assert_activation_coordination()` interdit `2.0` sans DV et DV sans `2.0` |

## ✅ Implémenté

### Documents canoniques

| Document | Fichier | Hash SHA-256 |
| --- | --- | --- |
| `terms_of_service` 1.0 | `legal/canonical/fr/terms_of_service_v1.0.txt` | `68c10026…` (inchangé) |
| `transport_terms` 1.0 | `legal/canonical/fr/transport_terms_v1.0.txt` | `45ae4ed0…` (inchangé) |
| `terms_of_service` 2.0 | `legal/canonical/fr/terms_of_service_v2.0.txt` | `97942048…` |
| `transport_terms` 2.0 | `legal/canonical/fr/transport_terms_v2.0.txt` | `5e502ef6…` |

Les deux 2.0 ont `requires_reacceptance = true`. Aucun backfill d'acceptation. 1.0 non modifié.

### Contenu contractuel 2.0 (aligné 7B — **à réécrire pour 7B.5**)

- 1er clic = demande / transmission ; **pas** formation du contrat
- 2e clic « Confirmer le transport à CHF X » = `CLIENT_TRANSPORT_CONFIRMED` = formation

⚠️ **REOPENED** : la cible produit est désormais [7B.5](portal-single-click-conditional-order-7b5.md)
(`CLIENT_CONDITIONAL_ORDER` → `TRANSPORT_CONTRACT_FORMED` à l’acceptation transporteur).  
Les bullets ci-dessus décrivent le **PREPARED actuel**, pas la cible post-PASS.
- Estimation ≠ plafond ≠ offre ≠ prix contractuel
- LIRIE ne fixe pas le prix, ne facture pas, n'encaisse pas, n'est pas créancier
- Annulation / no-show / attente = politique transporteur figée
- Payment hold = créancier X uniquement
- Pas de poursuite automatique LIRIE

### UX réacceptation

Modal dashboard : libellés séparés CGU 2.0 / Conditions transport 2.0 + accès au corps avant acceptation.

### Tests

[`backend/tests/services/test_portal_terms_2_0.py`](../../backend/tests/services/test_portal_terms_2_0.py)

## PUBLIC TERMS ALIGNMENT (`/conditions`)

Page : [`frontend/src/pages/Legal/TermsOfService.jsx`](../../frontend/src/pages/Legal/TermsOfService.jsx)

Cette page couvre institutions / entreprises / chauffeurs / utilisateurs finaux. Elle **n'est pas** le document canonique PORTAL 2.0 et n'a **pas** été remplacée.

### Gaps identifiés (incompatibles avec PORTAL v2)

| Formulation publique | Problème vs PORTAL v2 |
| --- | --- |
| « L'utilisation de la Plateforme vaut acceptation » | Contredit l'acceptation explicite exigée PORTAL |
| « poursuite de l'utilisation après notification peut valoir acceptation » | Contredit `requires_reacceptance` + gate booking |
| Mission « confirmée … après acceptation par [transporteur] » | Ignore la double validation (2e clic client) |
| Tarification / paiement génériques (Saferpay, missions) | PORTAL privé : facture + paiement au transporteur, pas Saferpay |
| Annulation générique « des missions » | PORTAL v2 : politique X figée au 2e clic |

**Correction de `/conditions`** : hors 7C (impact multi-acteurs). À traiter séparément (alignement public / 7D+).

## Hors 7C

- Activation prod `PORTAL_TERMS_EFFECTIVE_VERSION=2.0` + `PORTAL_DOUBLE_VALIDATION_ENABLED=true`
- Revue juridique formelle avant production
- Refonte page `/conditions`

## NEXT

**7D** — revue juridique + activation coordonnée + smoke navigateur + release gate
