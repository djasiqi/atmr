# LIRIE — NOTIF-IOS-P0-CI

Publication GitHub, CI et préparation recette iPhone.

**Date :** 2026-11-12  
**PR Draft :** [#105](https://github.com/djasiqi/atmr/pull/105)  
**Branche :** `candidate/lirie-notif-ios-p0-20261112`  
**SHA publié :** `a81ca2b2` (`a81ca2b29bd22d7067b2e87c459141b05aa5c45c`)  
**Base PR :** `candidate/lirie-fin-caisse-20261112` @ `1f12a9db` (#103)  
**Mention PR :** DRAFT — DEVICE DELIVERY NOT VERIFIED — DO NOT MERGE OR DEPLOY.

---

## A. Branche et SHA final

| Élément | Valeur |
|---------|--------|
| Worktree | `D:\lirie-backups\_candidate_lirie_notif_ios_p0_20261112` |
| HEAD | `a81ca2b2` |
| Upstream tip #103 | `1f12a9db` (inchangé) |
| WIP `main` | non modifié |

Périmètre diff vs #103 (10 fichiers) : notifications push, logout auth/mobile, tests, docs ops.  
**Aucune** migration Alembic, **aucun** flag ops activé, **aucun** secret ajouté.

---

## B. PR Draft

- **#105** — https://github.com/djasiqi/atmr/pull/105  
- Titre : `fix(notifications): harden iOS push delivery and device deduplication`  
- Draft = true · Base = FIN-CAISSE · Head = NOTIF  
- Empilée sur #103 (dépendances #101 / #102 / #103 documentées)

---

## C. Commits publiés

| SHA | Message |
|-----|---------|
| `bcd1b842` | fix(push-ios): dedup company/kafka et logout device_installation_id |
| `67747005` | docs(ops): rapport NOTIF-IOS-P0 et alignement audit iOS |
| `1567210e` | docs(ops): renseigner les SHA des commits NOTIF-IOS-P0 |
| `a81ca2b2` | docs(ops): stabiliser la section commits du rapport NOTIF-IOS-P0 |

(+ commits docs CI / recette iPhone éventuels après ce fichier)

---

## D. Compatibilité MFA (#104)

- Tip MFA : `626de1c3`  
- Intégration temporaire : merge MFA → tip NOTIF **sans commit**, worktree jetable.  
- **Aucun conflit** (`auth.py` / `client.ts` auto-merge).  
- Alias logout `device_installation_id` + `body.device_id` **préservés** après merge.  
- Services notifications : inchangés par MFA.  
- PR #104 **non modifiée**.  
- Runtime MFA×push (login TOTP → save-push-token) : non exécuté sur appareil ; contrat code OK.

---

## E. Résultats CI (en cours → à figer)

Workflows **auto** (pull_request, SHA `a81ca2b2`) :

| Workflow | Mode | Statut initial |
|----------|------|----------------|
| Secret Pattern Scan | auto PR + push | **success** |
| Architecture Review | auto PR | in_progress |
| Mobile unified-app | auto PR (paths mobile) | in_progress |
| Phase 1 Gates | auto PR | in_progress (`trigger-backend-deploy` **skipped** sur PR) |
| Security Gate | auto PR | in_progress |

Workflows **dispatch manuel** (sans `deploy.yml`) :

| Workflow | HTTP | SHA |
|----------|------|-----|
| Backend Tests | 204 | `a81ca2b2` |
| Security Scan | 204 | `a81ca2b2` |
| Repository Integrity | 204 | `a81ca2b2` |

**Non déclenchés volontairement :** `deploy.yml`, `deploy-kafka*`, `build-and-deploy-rl`, `e2e-demo` (prod/demo).

Vercel Preview Comments : statut commit `pending` (preview front — **pas** deploy serveur prod).

*Section mise à jour quand tous les contrôles obligatoires sont terminaux.*

---

## F. GitGuardian

- Incident historique **#103** `40006359` : classification humaine **toujours ouverte** — **non masqué**.  
- Sur #105 : attendre résultat check GitGuardian s’il apparaît ; ne pas forcer un FP.

---

## G. Mise à jour documentaire

- `docs/ops/push-ios-no-notifications-audit.md` — bannière FIXED IN CODE / DEVICE NOT VERIFIED / DEPLOYMENT NOT VERIFIED  
- `docs/ops/lirie-notifications-ios-p0.md` — diagnostic initial  
- `docs/ops/lirie-notifications-ios-p0-iphone-recipe.md` — recette testeur  
- Ce fichier — publication / CI  

États figés :

```text
Prometheus              = FIXED IN CODE
Dédup Chauffeur         = FIXED IN CODE
Dédup Entreprise/Kafka  = FIXED IN CODE
Logout tokens           = FIXED IN CODE
Déploiement             = NOT VERIFIED
Livraison iPhone        = NOT VERIFIED
```

---

## H. Procédure iPhone

Voir [`lirie-notifications-ios-p0-iphone-recipe.md`](lirie-notifications-ios-p0-iphone-recipe.md).

**IOS DEVICE DELIVERY = NOT VERIFIED** (aucun appareil/build de validation exécuté dans ce lot).

---

## I. Risques résiduels

1. Réception réelle iPhone non prouvée.  
2. Stacking MFA : alias logout à conserver au rebase futur.  
3. GG historique #103 hors scope de masquage.  
4. Phase C / lifecycle tokens toujours off (volontaire).  
5. Backend Tests / Security Scan ne se déclenchent pas auto sur base candidate → dispatch manuel requis (fait).

---

## J. Verdicts (provisoires jusqu’à CI terminale)

| Gate | Verdict |
|------|---------|
| CI | **PARTIAL** (en cours) |
| SECURITY | **PASS** (code) / CI gate en cours |
| PUSH BACKEND | **PASS** |
| PUSH MOBILE | **PARTIAL** |
| IOS DEVICE DELIVERY | **NOT VERIFIED** |
| READY FOR HUMAN REVIEW | **NO** (attendre CI GREEN) |
| READY FOR PRODUCTION | **NO** |
