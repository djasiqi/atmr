# 7B.5 — Commande conditionnelle à un clic (formation à l’acceptation transporteur)

```text
STEP 7B.5

FLOW VERSION                       conditional_order_v1
IMPLEMENTATION                     PASS / CLOSED
STATUS                             PASS / CLOSED
TERMS 2.1                          PREPARED (non effectif prod)
TERMS 2.1 HASHES                   LOCKED
CLIENT SINGLE CLICK                PASS
NO CONTRACT ON FIRST CLICK         PASS
ELIGIBLE CARRIER DISCLOSURE        PASS
ATOMIC / IDEMPOTENT ACCEPT         PASS (HTTP + UseCase)
CHANNEL CAPS                       PASS
NO SILENT MIN                      PASS
CONTRACT PRICE = CARRIER QUOTE     PASS
MIGRATION                          8b1974a79318

LOCAL MANUAL SMOKE                 PASS / CLOSED
  — preuve centrale                #46774 / contract #5 / quote 40
  — idempotence HTTP               PASS
  — transport_already_assigned     PASS
  — carrier_quote liste            PASS (= 40, amount 50 isolé)

NEXT                               CI SAME SHA → E2E
COMMIT                             ca215360 (poussé)
BUILD / DEPLOY                     NOT YET
PRODUCTION                         NO (flags défaut OFF / Terms 1.0)
```

✅ **Implémenté** :
- Flag `PORTAL_CONDITIONAL_ORDER_ENABLED` (défaut false) + exclusivité mutuelle au boot avec DV
- Terms 2.1 PREPARED (canonicals + hashes) — 2.0 intact
- `LirieChannelCancellationPolicy` + validation stricte (refus, jamais `min`)
- Preuves `PortalClientConditionalOrder` + `PortalTransportContractFormed`
- Accept atomique (pool figé, idempotent, `FOR UPDATE`)
- UI client : CTA « Commander jusqu’à CHF X », disclosure pool, wording d’engagement
- Emails post-commit (commande transmise / transport confirmé)
- Migration `8b1974a79318_portal_conditional_order_7b5`
- Tests : `test_portal_conditional_order_7b5.py`

Ce n’est **pas** « supprimer le modal ». C’est une **refonte** du contrat produit,
des événements de preuve et des Terms (2.1, sans muter 2.0).

Références :

- Caps canal : [portal-channel-cancellation-caps.md](portal-channel-cancellation-caps.md)
- DV actuelle (code conservé) : [portal-double-validation-7b.md](portal-double-validation-7b.md)
- Autoconfirm 7B.4 (superseded) : [portal-preferred-carrier-autoconfirm-7b4.md](portal-preferred-carrier-autoconfirm-7b4.md)
- Terms : [portal-terms-2-0-7c.md](portal-terms-2-0-7c.md) (2.0 historique) + canonicals 2.1
- Revue : [portal-legal-review-package-7d-l.md](portal-legal-review-package-7d-l.md)
- Rapport impl : [portal-single-click-conditional-order-7b5-implementation-report.md](portal-single-click-conditional-order-7b5-implementation-report.md)
- **Smoke manuel** : [portal-conditional-order-7b5-manual-smoke.md](portal-conditional-order-7b5-manual-smoke.md)

## Flux cible

```text
CLIENT_CONDITIONAL_ORDER
→ transmission aux transporteurs du pool figé
→ CARRIER_ACCEPTED (atomic)
→ TRANSPORT_CONTRACT_FORMED
→ notification immédiate au client
```

Invariants :

```text
plafond client ≠ prix contractuel
prix contractuel = quote du transporteur qui accepte
transporteur ne voit jamais le plafond
pas de 2e clic client
pas de paiement LIRIE
facturation ensuite par l’entreprise
flags mutuellement exclusifs (boot refusal)
pool éligible figé (company_id)
caps : dimensions ⊆ channel + rate <= cap (jamais min silencieux)
snapshot dual complet (version+hash+body)
accept idempotent (même company)
notifications après commit uniquement
```

## Flags

```text
conditional_order_v1 : CONDITIONAL=true, DV=false, Terms=2.1
double_validation_v2 : CONDITIONAL=false, DV=true,  Terms=2.0
legacy               : CONDITIONAL=false, DV=false, Terms=1.0
```

Si les deux flags sont `true` → refus au boot.

## NEXT

```text
7B.5              PASS / CLOSED
LOCAL SMOKE       PASS
IMPLEMENTATION    CLOSED

NEXT              RESTORE ENV (fait) → COMMIT UNIQUE → CI SAME SHA → E2E
BUILD / DEPLOY    NOT YET
```
