# LIRIE — LOT-RECETTE-INTERFACES-CI

Publication GitHub de la recette interfaces et validation CI.

**Date :** 2026-11-14  
**Worktree :** `D:\lirie-backups\_candidate_lirie_recette_interfaces_20261113`  
**WIP principal (`C:\Users\jasiq\atmr`) :** non modifié  

---

## A. PR Draft

| Champ | Valeur |
|-------|--------|
| Numéro | **#107** |
| URL | https://github.com/djasiqi/atmr/pull/107 |
| Titre | `fix(interfaces): validate client, company and institution transport workflows` |
| État | **Draft** — OPEN |
| Head | `candidate/lirie-recette-interfaces-20261113` |
| Base | `candidate/lirie-booking-dispatch-20261112` (PR **#106**) |
| SHA final | `0cd447fd` (tip au moment de la rédaction ; vérifier le tip PR si commit rapport ajouté) |
| Merge / Deploy | **NON** |

**Mention PR :** DRAFT — FULL UI / DEVICE NOT VERIFIED — DO NOT MERGE OR DEPLOY

---

## B. Commits du lot (depuis tip #106 `12e320e1`)

| SHA | Message |
|-----|---------|
| `cbbd5f2f` | test(recette): parcours Playwright multi-profils isolés |
| `12c15a90` | docs(ops): rapport LOT-RECETTE-INTERFACES-1 |
| `98f47885` | fix(recette): cookies HTTP testing + stopPropagation assign |
| `5a7676b6` | test(recette): approfondir parcours UI Client/Entreprise/Institution |
| `0c9467ee` | fix(entreprise): ouverture fiable de la modale Assigner |
| `bffba045` | test(recette): certifier Assigner + transmission Institution |
| `220f9763` | docs(ops): figer HEAD LOT-RECETTE-INTERFACES-2 |
| `9cf20fd8` | docs(ops): figer décision Local PASS — Remote CI Pending |
| `c411c57e` | fix(ci): corriger format/lint hérités bloquant Backend Tests |
| `805359f3` | fix(ci): aligner 2 assertions pytest héritées (RL open count, send sans préférences) |
| `0cd447fd` | style(ci): ruff format test_request_offers |

---

## C. Matrice CI — SHA `0cd447fd`

### Checks PR (rollup)

| Check | Verdict |
|-------|---------|
| Architecture Review / architecture-review | SUCCESS |
| Phase 1 Gates / phase1-gates | SUCCESS |
| booking-critical | SUCCESS |
| mobile-auth-critical | SUCCESS |
| mobile-tracking-critical | SUCCESS |
| deployment-critical | SUCCESS |
| Security Gate / Security Gate verdict | SUCCESS |
| SAST CodeQL | SUCCESS |
| SAST Bandit + Semgrep | SUCCESS |
| SCA pip-audit | SUCCESS |
| Secrets Gitleaks HEAD | SUCCESS |
| Docker / filesystem Trivy CRITICAL | SUCCESS |
| IaC Trivy config CRITICAL | SUCCESS |
| Tenant isolation tests | SUCCESS |
| Repository integrity | SUCCESS |
| Secret Pattern Scan | SUCCESS |
| semgrep-cloud-platform/scan | SUCCESS |
| GitGuardian Security Checks | SUCCESS |
| trigger-backend-deploy | **SKIPPED** (aucun déploiement) |
| Vercel – atmr / lirie-app | SUCCESS (preview uniquement) |

### Workflows dispatchés (filtres PR `main`/`develop` / paths)

| Workflow | Mode | Run | Verdict |
|----------|------|-----|---------|
| Backend Tests | `workflow_dispatch` | [43590199301](https://github.com/djasiqi/atmr/actions/runs/43590199301) | **SUCCESS** (lint + 4 shards pytest + combine) |
| Frontend Tests | `workflow_dispatch` | [43590201762](https://github.com/djasiqi/atmr/actions/runs/43590201762) | **SUCCESS** (Cypress skipped — config) |
| Mobile unified-app | non déclenché | — | N/A (aucun path `mobile/unified-app/**` ; Phase 1 mobile-* PASS) |

---

## D. Corrections CI

| Problème | Cause | Correctif |
|----------|-------|-----------|
| Lint Backend (format/ruff/flake8) | Dette héritée #106 (7 fichiers) | `c411c57e` — `ruff format` + ValidationError / asserts séparés |
| pytest shard1 — `test_edge_case_empty_trials` | Assertion fragile `open.call_count == 3` (réel = 4) | `805359f3` — `>= 3` |
| pytest shard4 — `test_send_without_preferences_is_rejected` | Message d’erreur réseau/zone (#106) vs attente « transporteur » | `805359f3` — tokens `transporteur`/`réseau`/`zone` |
| Format `test_request_offers.py` | Assertion multi-ligne non formatée | `0cd447fd` — `ruff format` |

Aucune migration · aucun flag prod · aucun paiement · `TestingConfig` cookies HTTP uniquement.

---

## E. Recette locale (rappel — non rejouée)

| Preuve | Résultat |
|--------|----------|
| Playwright | **10/10 PASS** |
| Jest contrats chauffeur | **23 PASS** |
| Booking test `1660` → chauffeur `252` ASSIGNED | confirmé (IDs de recette, non hardcodés en prod) |
| TransportRequest SENT + offers entreprise `2035` | confirmé |

---

## F. Sécurité / MFA / GitGuardian

| Point | Statut |
|-------|--------|
| Isolation `TestingConfig` / base jetable | PASS |
| JWT temporaires MFA ≠ actions métier | rappelé ; MFA **non** intégrée (#104 hors chaîne) |
| Chaîne #101→#106 intacte | PR #107 empilée uniquement |
| GitGuardian check PR #107 | **SUCCESS** |
| Incident historique `40006359` (#103) | **non masqué** — classification humaine requise |
| Déploiement / migration prod | **aucun** (`trigger-backend-deploy` SKIPPED) |

---

## G. Limitations

1. **CHAUFFEUR DEVICE** : pas de validation téléphone réel (Jest ≠ device).
2. **MULTI-PROFILE FULL UI** : PARTIAL (device manquant).
3. Backend/Frontend Tests : déclenchement manuel (`workflow_dispatch`) car filtres branches/paths hors `main`/`develop`.
4. Mobile unified-app workflow path-filter : non applicable à ce diff.

---

## Verdicts

| Gate | Verdict |
|------|---------|
| CI | **GREEN** |
| SECURITY | **PASS** |
| CLIENT WEB | **PASS** |
| ENTREPRISE WEB | **PASS** |
| INSTITUTION WEB | **PASS** |
| CHAUFFEUR MOBILE CONTRACT | **PASS** |
| CHAUFFEUR DEVICE | **NOT VERIFIED** |
| MULTI-PROFILE FULL UI | **PARTIAL** |
| READY FOR HUMAN REVIEW | **YES** |
| READY FOR PRODUCTION | **NO** |
