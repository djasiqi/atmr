# Phase 2 Implementation Owners

## Preconditions Validation

- `P1.1` Background tracking: `validated`
- `P1.2` Runtime flags: `validated`
- `P1.4` Push quick actions baseline: `validated`
- `P1.9` Company JTBD baseline: `validated`

## Ownership Matrix

| Workstream | Scope | Primary Owner | Secondary Owner | Sign-off Required |
|---|---|---|---|---|
| Driver Notifications | P2.1 | Mobile Runtime | QA Device | Yes |
| Mission Bar + Live Activity | P2.2 | Mobile Runtime | Mobile iOS | Yes |
| Chat Attachments | P2.3 | Mobile Driver | Backend API | Yes |
| OTA Gate | P2.4 | Mobile Platform | Release Owner | Yes |
| Company UX | P2.5 | Mobile Company | QA Dispatch | Yes |
| Deep Links | P2.6 | Mobile Platform | Product + Backend | Yes |
| Transfer Flow | P2.7 | Mobile Company | Backend API | Yes |

## Committee Anchors

- Release committee reference: `docs/migration/GATES_SIGNOFF_MATRIX.md`
- Runtime close authorization checklist: `docs/migration/PHASE2_CLOSE_AUTHORIZATION_CHECKLIST.md`
- Evidence matrix: `docs/migration/PHASE2_DEVICE_PROOF_MATRIX.md`
- Close-auth CI: workflow GitHub `Mobile Phase 2 close authorization`
  (`.github/workflows/mobile-phase2-close-auth.yml`, `workflow_dispatch` / PR docs).
  **Hors** du gate canary `Mobile unified-app (Lint + Jest)`.
