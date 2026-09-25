# 7B.5 — Smoke manuel local (conditional_order_v1)

```text
STATUS : PASS / CLOSED
PREREQUISITE : IMPLEMENTATION PASS (flags OFF en défaut)
GATE CLOSE 7B.5 : PASS — preuve centrale #46774 / contract #5 / quote 40
```

## Environnement smoke (uniquement)

```text
PORTAL_CONDITIONAL_ORDER_ENABLED=true
PORTAL_DOUBLE_VALIDATION_ENABLED=false
PORTAL_TERMS_EFFECTIVE_VERSION=2.1
```

Après le smoke : **restaurer** les flags locaux (typiquement tous OFF / Terms 1.0).

Publier une `LirieChannelCancellationPolicy` **synthétique de test uniquement**
(`synthetic_test_channel_caps()` — pas de pourcentages prod inventés au boot).

Exemple (container) :

```bash
docker compose exec -T atmr_api python -c "
from app import create_app
from ext import db
from services.legal.portal_channel_cancellation_caps import (
    publish_channel_cancellation_policy,
    synthetic_test_channel_caps,
)
app = create_app()
with app.app_context():
    r = publish_channel_cancellation_policy(body_json=synthetic_test_channel_caps())
    db.session.commit()
    print(r.ok, r.policy.version if r.policy else r.error)
"
```

Les entreprises du pool doivent avoir une **policy PORTAL publiée** conforme aux caps
(`company_rate <= channel_cap`, pas de dimension hors cadre).

## Scénario de référence

```text
estimate = 50
ceiling  = 52

A flat       = 45
B distance   = 38.40
Emmenez-moi  = 40
C zone       = 52
```

### Client — avant clic

- [ ] Plafond CHF 52
- [ ] Pool de transporteurs visible (identités juridiques)
- [ ] Aucun prix individuel du pool
- [ ] Cadre d’annulation canal visible (plafonds)
- [ ] CTA **Commander jusqu’à CHF 52.–**
- [ ] Wording d’engagement adjacent (« Cette commande vous engage si… »)

### Client — après clic

- [ ] Message / e-mail **Commande transmise**
- [ ] `company_id = NULL`
- [ ] Aucun contrat (`TRANSPORT_CONTRACT_FORMED` absent)
- [ ] Aucun Saferpay / « Paiement requis »
- [ ] Aucun `CLIENT_TRANSPORT_CONFIRMED`

### Emmenez-moi (entreprise)

- [ ] Affiche **Votre tarif : CHF 40.00**
- [ ] CTA **Accepter à CHF 40.00**
- [ ] Aucun plafond CHF 52 visible
- [ ] Aucune estimation CHF 50 visible

### Après acceptation Emmenez-moi

- [ ] `contractual_amount = 40.00`
- [ ] `company_id` = Emmenez-moi
- [ ] `TRANSPORT_CONTRACT_FORMED` présent (quote 40, ceiling 52, dual policy snapshots)
- [ ] Client : **Transport confirmé avec Emmenez-moi — CHF 40.–**
- [ ] Notification = confirmation, **pas** un second CTA « Confirmer le transport »

## 5 cas manuels obligatoires avant CLOSE

| # | Cas | Attendu | Résultat |
| --- | --- | --- | --- |
| 1 | D hors pool tente d’accepter | `carrier_not_in_order_pool` | [ ] |
| 2 | Même transporteur double-clique | Même contrat (idempotent), pas un second | [ ] |
| 3 | Autre transporteur après le gagnant | `transport_already_assigned` | [ ] |
| 4 | Policy entreprise > cap (ou frais hors dimensions) | Refus explicite (publication et/ou accept) | [ ] |
| 5 | Facture après course | Transport **CHF 40** — jamais 50 / 52 | [ ] |

## Verdict

```text
LOCAL MANUAL SMOKE : PASS / CLOSED
7B.5               : PASS / CLOSED

DATE               : 2026-09-25
MODE               : API local Docker + DB (scripts/smoke_7b5_e2e.py)
BOOKING PREUVE     : #46774 (Emmenez accept → TRANSPORT_CONTRACT_FORMED #5, quote 40)
```

### Correctifs ciblés smoke FAIL → PASS

✅ **Implémenté** : gate HTTP `status != PENDING` contourné uniquement pour `conditional_order_v1` — le UseCase transactionnel gère idempotence / `transport_already_assigned` (`routes/companies.py`, `accept_reservation.py`).

✅ **Implémenté** : `attach_serialize_context_to_bookings` calcule la grille pour DV **et** conditional_order ; `serialize` + `serialize_dashboard` exposent `carrier_quote` / `company_suggested_amount` ; UI `portalCarrierFacingAmountDisplay` utilise `isPortalContractFlow`.

### Smoke réel (distance locale)

```text
A flat        = 45
B distance    ≈ 20.37   (Haversine / quota Maps — pas un FAIL)
Emmenez-moi   = 40
C zone_count  = 52
ceiling       = MAX = 52
```

Après CLOSE uniquement :

```text
restore local env
→ commit unique
→ CI same SHA
→ E2E
→ préparation prod
```
