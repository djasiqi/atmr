# LIRIE — CONVENTION-REGLEMENT-CI

Publication GitHub du circuit de règlement et validation CI.

**Date :** 2026-11-14  
**Worktree :** `D:\lirie-backups\_candidate_lirie_convention_reglement_20261114`  
**WIP principal (`C:\Users\jasiq\atmr`) :** non modifié  

---

## A. PR Draft

| Champ | Valeur |
|-------|--------|
| Numéro | **#108** |
| URL | https://github.com/djasiqi/atmr/pull/108 |
| Titre | `feat(settlement): persist payment incidents and finalize driver closure workflow` |
| État | **Draft** — OPEN |
| Head | `candidate/lirie-convention-reglement-20261114` |
| Base | `candidate/lirie-recette-interfaces-20261113` (PR **#107**) |
| SHA CI certifié | `c2f78051` |
| Tip docs (après ce rapport) | HEAD de la branche (docs-only après `c2f78051`) |
| Merge / Deploy | **NON** |

**Mention PR :** DRAFT — CONVENTION SIGNATURE NOT PRODUCTION READY — NO MERGE OR DEPLOY

---

## B. Commits publiés (depuis tip #107 `e945574f`)

| SHA | Message |
|-----|---------|
| `55cb5922` | feat(settlement): clôture convention, règlement ou signalement admin |
| `55bfc2d8` | docs(ops): figer commit CONVENTION-REGLEMENT-01 |
| `4c754c6c` | feat(settlement): persister signalements, carte terminal externe et file admin |
| `d3380855` | docs(ops): figer commit CONVENTION-REGLEMENT-02 |
| `c2f78051` | docs(ops): figer tip feature CONVENTION-REGLEMENT-02 |

**Tip feature fonctionnel :** `4c754c6c`  
**Tip CI PR certifié :** `c2f78051`

---

## C. Matrice CI — SHA `c2f78051`

### Checks PR (rollup — 25/25 sans échec)

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
| Mobile unified-app / check-unified-boundaries | SUCCESS |
| Mobile unified-app / lint-and-test | SUCCESS |
| trigger-backend-deploy | **SKIPPED** (aucun déploiement) |
| Vercel – atmr / lirie-app | SUCCESS (preview uniquement) |

### Workflows dispatchés (filtres PR hors `main`/`develop`)

| Workflow | Mode | Run | Verdict |
|----------|------|-----|---------|
| Backend Tests | `workflow_dispatch` | [43596284816](https://github.com/djasiqi/atmr/actions/runs/43596284816) | **SUCCESS** (lint + migrations + 4 shards pytest + combine + E2E + intégration) |
| Frontend Tests | `workflow_dispatch` | [43596287273](https://github.com/djasiqi/atmr/actions/runs/43596287273) | **SUCCESS** (ESLint + Jest + build ; Cypress skipped — config) |

---

## D. Persistance / migration

| Point | État |
|-------|------|
| Migration Alembic | `c2r02a1b3c4d` ← `38c76fa54225` |
| Tables | `settlement_non_payment_incident`, `settlement_non_payment_event`, `external_card_settlement` |
| Périmètre | **limité au lot** (pas d’autogen large) |
| CI Migrations Check | **SUCCESS** (upgrade + downgrade -1) |
| Application production | **NON** |
| Preuve locale | 6 tests persistance PG jetable PASS |

---

## E. Contrôles de sécurité / garanties métier

| Garantie | Statut |
|----------|--------|
| Transport terminé ≠ règlement financier | PASS |
| Convention démo ≠ différé réel | PASS |
| Aucun paiement fictif encaissé | PASS |
| Terminal bancaire externe + preuve (≠ Tap to Pay) | PASS |
| Signalement PostgreSQL + historique admin | PASS |
| Isolation inter-entreprises | PASS (tests + Tenant isolation CI) |
| Aucun traitement admin côté chauffeur | PASS |
| Saferpay / Worldline | absents |
| Secrets / WIP étrangers | absents sur le diff |
| Déploiement | aucun (`trigger-backend-deploy` SKIPPED) |

---

## F. Limites signature / validations restantes

1. Signature électronique réelle + levée éventuelle de `ck_convention_demo_not_legal` : **non activées**.
2. Convention applicable (`mandate_kind=applicable` + `legally_effective`) : **préparée**, valeur juridique **non** activée.
3. Branchement écrans mobile production sur les routes : hors lot (contrat prêt).
4. Validation visuelle admin (onglet Signalements) : **manuelle**.
5. Validation appareil chauffeur : **NOT VERIFIED**.

Rapports amont : `docs/ops/lirie-convention-reglement-01.md`, `docs/ops/lirie-convention-reglement-02.md`.

---

## G. Corrections CI

Aucune correction requise sur le SHA publié : rollup PR **GREEN** du premier passage ; Backend/Frontend dispatch **GREEN**.

---

## H. Verdicts

| Gate | Verdict |
|------|---------|
| CI | **PASS** |
| SECURITY | **PASS** |
| SETTLEMENT INCIDENT | **PASS** |
| ADMIN PROCESSING | **PASS** (API + onglet ; UI visuelle manuelle) |
| CONVENTION LEGAL | **NOT READY** (démo / signature non activées) |
| MOBILE DEVICE | **NOT VERIFIED** |
| READY FOR HUMAN REVIEW | **YES** |
| READY FOR PRODUCTION | **NO** |
