# LIRIE — NOTIF-IOS-P0-CI

Publication GitHub, CI et préparation recette iPhone.

**Date :** 2026-11-12  
**PR Draft :** [#105](https://github.com/djasiqi/atmr/pull/105)  
**Branche :** `candidate/lirie-notif-ios-p0-20261112`  
**Base PR :** `candidate/lirie-fin-caisse-20261112` @ `1f12a9db` (#103)  
**Mention PR :** DRAFT — DEVICE DELIVERY NOT VERIFIED — DO NOT MERGE OR DEPLOY.

> **HEAD exact :** à figer après le dernier push (voir section A / git log).  
> **#104 MFA n’est pas intégrée dans #105** — seule une compatibilité par merge temporaire a été vérifiée.

---

## A. Branche et SHA final

| Élément | Valeur |
|---------|--------|
| Worktree | `D:\lirie-backups\_candidate_lirie_notif_ios_p0_20261112` |
| Upstream tip #103 | `1f12a9db` (inchangé) |
| WIP `main` | non modifié |
| MFA #104 | non mergée dans #105 |

Périmètre NOTIF : push company/kafka, logout device ids, tests, docs.  
Correctif CI lint : format `company_push.py`, PLR0402 test logout, I001 isort hérités settlement/e2e (bloquants Backend Tests sur tip).  
**Aucune** migration Alembic, **aucun** flag ops activé, **aucun** secret, **aucun** `deploy.yml`.

---

## B. PR Draft

- **#105** — https://github.com/djasiqi/atmr/pull/105  
- Titre : `fix(notifications): harden iOS push delivery and device deduplication`  
- Draft = true · Base = FIN-CAISSE · Head = NOTIF  
- Empilée sur #103 (dépendances #101 / #102 / #103)

---

## C. Commits publiés (série)

| SHA | Message |
|-----|---------|
| `bcd1b842` | fix(push-ios): dedup company/kafka et logout device_installation_id |
| `67747005`…`a81ca2b2` | docs NOTIF-IOS-P0 |
| `e6a9522d` | docs(ops): PR #105, recette iPhone et rapport NOTIF-IOS-P0-CI |
| *(tip)* | lint CI + finalisation rapport |

---

## D. Compatibilité MFA (#104)

- Tip MFA : `626de1c3`  
- Merge temporaire MFA → tip NOTIF : **aucun conflit** ; alias logout préservés.  
- **PR #104 non modifiée et non intégrée à #105.**  
- Runtime MFA×push sur appareil : non exécuté.

---

## E. Matrice CI

### E.1 Première vague — SHA `a81ca2b2` (code + docs init)

| Workflow / check | Mode | SHA | Conclusion |
|------------------|------|-----|------------|
| Secret Pattern Scan | PR + push | a81ca2b2 | **success** |
| Architecture Review | PR | a81ca2b2 | **success** |
| Mobile unified-app (Lint + Jest) | PR | a81ca2b2 | **success** |
| Phase 1 Gates | PR | a81ca2b2 | **success** (`trigger-backend-deploy` **skipped**) |
| Security Gate / verdict | PR | a81ca2b2 | **success** |
| SAST CodeQL / Bandit+Semgrep / Gitleaks | Gate | a81ca2b2 | **success** |
| GitGuardian Security Checks | PR | a81ca2b2 | **success** (pas de nouveau FAIL visible ; hist. 40006359 #103 non masqué) |
| Security Scan | dispatch | a81ca2b2 | **success** |
| Repository Integrity | dispatch | a81ca2b2 | **success** |
| Backend Tests — Lint (Ruff) | dispatch | a81ca2b2 | **failure** (format `company_push.py` + PLR0402 logout test + I001 settlement hérités) |
| Backend Tests — pytest shards / E2E / Migrations | dispatch | a81ca2b2 | en cours / success partiels au moment du diagnostic |

### E.2 Correctif lint + docs tip (après push)

| Action | Détail |
|--------|--------|
| Fix | `ruff format` company_push ; `from routes import auth` ; I001 settlement/e2e |
| Redeploy CI | push tip → vérif Backend Tests + Security Gate + Mobile sur **HEAD final** |
| Deploy workflows | **aucun** `deploy.yml` / kafka / RL exécuté |

*Tableau E.3 (HEAD final) complété après clôture CI du tip poussé.*

---

## F. GitGuardian

- Incident historique **#103** `40006359` : classification humaine **ouverte** — **non masqué**.  
- Sur #105 @ `a81ca2b2` : check **GitGuardian Security Checks = success**.  
- Surveiller le tip final après push lint/docs.

---

## G. Documentation

- Audit : Prometheus / dédup chauffeur / dédup company-kafka / logout = **FIXED IN CODE** ; déploiement & iPhone = **NOT VERIFIED**  
- Recette : [`lirie-notifications-ios-p0-iphone-recipe.md`](lirie-notifications-ios-p0-iphone-recipe.md)  
- Ne pas déclarer « P0 production résolu » sur tests verts seuls.

---

## H. Procédure iPhone

Recette prête. **IOS DEVICE DELIVERY = NOT VERIFIED.**

---

## I. Risques résiduels

1. Réception iPhone non prouvée.  
2. #104 non dans #105 — stacking ultérieur requis.  
3. GG historique #103 hors FP humain.  
4. I001 settlement corrigés sur tip NOTIF pour CI (héritage #103) — à reporter éventuel vers FIN-CAISSE.

---

## J. Verdicts (avant clôture tip final)

| Gate | Verdict |
|------|---------|
| CI | **PARTIAL** (Lint Backend FAIL sur `a81ca2b2` ; correctif en push) |
| SECURITY | **PASS** (Security Gate + scans @ a81ca2b2) |
| PUSH BACKEND | **PASS** (code) |
| PUSH MOBILE | **PARTIAL** |
| IOS DEVICE DELIVERY | **NOT VERIFIED** |
| READY FOR HUMAN REVIEW | **NO** (attendre CI tip final) |
| READY FOR PRODUCTION | **NO** |
