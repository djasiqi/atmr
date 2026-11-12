# LIRIE — NOTIF-IOS-P0-CI

Publication GitHub, CI et préparation recette iPhone.

**Date :** 2026-11-12  
**PR Draft :** [#105](https://github.com/djasiqi/atmr/pull/105)  
**Branche :** `candidate/lirie-notif-ios-p0-20261112`  
**SHA code + CI GREEN (avant ce commit docs) :** `71acf9fe` (`71acf9feb8655d6124f8cbf6c957f63da8a8cce7`)  
**Base PR :** `candidate/lirie-fin-caisse-20261112` @ `1f12a9db` (#103)  

> **DRAFT — DEVICE DELIVERY NOT VERIFIED — DO NOT MERGE OR DEPLOY.**  
> **#104 MFA n’est pas intégrée dans #105** (compatibilité vérifiée uniquement par merge temporaire).

---

## A. Branche et SHA final

| Élément | Valeur |
|---------|--------|
| Worktree | `D:\lirie-backups\_candidate_lirie_notif_ios_p0_20261112` |
| HEAD PR #105 (au moment de la clôture CI) | **`71acf9fe`** |
| Tip #103 | `1f12a9db` (inchangé) |
| WIP `main` | non modifié |
| MFA #104 | **non mergée** dans #105 |

Périmètre : push iOS company/kafka, logout device ids, tests NOTIF, docs, correctifs CI (ruff + test RL mock hors métier).  
Pas de migration, pas de flag ops ON, pas de secret, pas de `deploy.yml`.

---

## B. PR Draft

- **#105** — https://github.com/djasiqi/atmr/pull/105  
- Titre : `fix(notifications): harden iOS push delivery and device deduplication`  
- Draft = true · Base = FIN-CAISSE · Head = NOTIF

---

## C. Commits publiés (principaux)

| SHA | Message |
|-----|---------|
| `bcd1b842` | fix(push-ios): dedup company/kafka et logout device_installation_id |
| `a81ca2b2`…`e6a9522d` | docs NOTIF-IOS-P0 / PR #105 / recette iPhone |
| `9185697a` | fix(ci): ruff format/import NOTIF et I001 settlement |
| `71acf9fe` | fix(test): mock Path.open tuner RL (Backend Tests shard 1) |
| *(tip docs)* | ce rapport final |

---

## D. Compatibilité MFA (#104)

- Tip MFA : `626de1c3`  
- Merge temporaire MFA → NOTIF : **aucun conflit** ; alias logout préservés.  
- **#104 non intégrée à #105.**  
- Runtime MFA×push appareil : non exécuté.

---

## E. Matrice CI — HEAD `71acf9fe` (obligatoires)

| Contrôle | Mode | SHA | Conclusion |
|----------|------|-----|------------|
| **Backend Tests** (Lint + pytest 4 shards + E2E + Migrations + pip-audit) | workflow_dispatch | `71acf9fe` | **success** |
| **Security Gate** / verdict | pull_request | `71acf9fe` | **success** |
| **Mobile unified-app** (boundaries + lint-and-test) | pull_request | `71acf9fe` | **success** |
| Architecture Review | PR | `71acf9fe` | **success** |
| Phase 1 Gates | PR | `71acf9fe` | **success** |
| `trigger-backend-deploy` | Phase 1 | `71acf9fe` | **skipped** |
| Secret Pattern Scan | PR + push | `71acf9fe` | **success** |
| Security Scan | dispatch | `71acf9fe` | **success** |
| Repository Integrity | dispatch | `71acf9fe` | **success** |
| SAST CodeQL / Bandit+Semgrep / Semgrep cloud | Gate | `71acf9fe` | **success** |
| Secrets Gitleaks HEAD | Gate | `71acf9fe` | **success** |
| **GitGuardian Security Checks** | PR | `71acf9fe` | **success** |
| Vercel Preview Comments | status | `71acf9fe` | **success** (preview ≠ prod) |

**40/40 check-runs terminés, 0 failure** sur `71acf9fe`.  
**0** workflow `Deploy*` / `deploy.yml` / kafka deploy sur ce tip.

### Historique CI (vagues antérieures)

| SHA | Problème | Correctif |
|-----|----------|-----------|
| `a81ca2b2` | Lint Ruff (format company_push, PLR0402, I001 settlement) | `9185697a` |
| `9185697a` | pytest shard1 : `test_save_best_params_with_custom_filename` (mock Path.open) | `71acf9fe` |

---

## F. GitGuardian

- Check **GitGuardian Security Checks = success** sur `71acf9fe`.  
- Incident historique **#103** `40006359` : classification humaine **toujours ouverte** — **non masqué** ici.  
- Aucun nouveau FAIL GG signalé sur #105.

---

## G. Documentation

```text
Prometheus              = FIXED IN CODE
Dédup Chauffeur         = FIXED IN CODE
Dédup Entreprise/Kafka  = FIXED IN CODE
Logout tokens           = FIXED IN CODE
Déploiement             = NOT VERIFIED
Livraison iPhone        = NOT VERIFIED
```

Recette : [`lirie-notifications-ios-p0-iphone-recipe.md`](lirie-notifications-ios-p0-iphone-recipe.md)  
Audit : [`push-ios-no-notifications-audit.md`](push-ios-no-notifications-audit.md)

Ne pas déclarer « P0 production résolu » sur la seule base CI GREEN.

---

## H. Procédure iPhone

Prête. **IOS DEVICE DELIVERY = NOT VERIFIED** (aucune réception réelle démontrée).

---

## I. Risques résiduels

1. Réception iPhone non prouvée.  
2. Stacking #104 ultérieur (logout aliases déjà compatibles en merge-tree).  
3. GG historique #103 hors FP humain.  
4. Correctifs isort settlement + mock RL sur tip NOTIF (héritage / hors métier) — à reporter si besoin vers #103.

---

## J. Verdicts

| Gate | Verdict |
|------|---------|
| CI | **GREEN** sur HEAD `71acf9fe` |
| SECURITY | **PASS** |
| PUSH BACKEND | **PASS** |
| PUSH MOBILE | **PARTIAL** |
| IOS DEVICE DELIVERY | **NOT VERIFIED** |
| READY FOR HUMAN REVIEW | **YES** |
| READY FOR PRODUCTION | **NO** |
