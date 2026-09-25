# STEP 7B.5 IMPLEMENTATION REPORT

```text
STEP 7B.5

FLOW VERSION                       conditional_order_v1
IMPLEMENTATION                     PASS / CLOSED
STATUS                             PASS / CLOSED
TERMS 2.1                          PREPARED
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
COMMIT                             ca215360 (poussé)
BUILD / DEPLOY                     NOT YET
```

TERMS HASHES :

```text
terms_of_service 2.1 =
  d08fa9431073c1b189cbe8de9f5c4df2d5a63f2e549377eb02f499ceacba694e
transport_terms 2.1 =
  7c28686c5c107c26d29a65e9e5217d92fa3b8445aad50009dadfebde5b5d4f81
```

CONTRACT PRICE SOURCE : `PortalTransportContractFormed.carrier_quote`

FLAGS (défaut code / prod) :

```text
PORTAL_CONDITIONAL_ORDER_ENABLED = false
PORTAL_DOUBLE_VALIDATION_ENABLED = false
PORTAL_TERMS_EFFECTIVE_VERSION = 1.0
```

## Tests automatisés

- `tests/services/test_portal_conditional_order_7b5.py` — PASS
- `tests/services/test_portal_activation_coordination_7d.py` — PASS
- `frontend/.../portalDoubleValidationUi.test.js` — PASS

## Prochain gate

```text
7B.5              PASS / CLOSED
LOCAL SMOKE       PASS
NEXT              CI SAME SHA → E2E
BUILD / DEPLOY    NOT YET
```
