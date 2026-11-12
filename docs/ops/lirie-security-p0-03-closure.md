# LIRIE — Clôture SECURITY-P0-03 (MFA) + diagnostic GitGuardian

**Date :** 2026-11-12  
**Worktree isolé :** `D:\lirie-backups\_candidate_lirie_security_p0_03_mfa_20261112`  
**Branche locale :** `candidate/lirie-security-p0-03-mfa-20261112`  
**Base :** AUTH-PRO `cdbbccef` (PR #101) — `auth.py` identique à FIN-CAISSE HEAD  
**Périmètre :** MFA P0-03 + diagnostic GitGuardian PR #103  
**Hors périmètre :** merge, push, déploiement, migrations prod, flags, WIP principal, PAY-SAFERPAY, ré-audit AUTH-PRO / FIN-FACTURE / FIN-CAISSE

---

## Verdicts

| Verdict | Valeur |
|---|---|
| **MFA P0-03** | **PARTIAL** |
| **GITGUARDIAN** | **BLOCKED** (alerte ouverte ; fixture de test, pas secret prod) |
| **READY FOR REMOTE CI** | **NO** (pas de push ; GG #103 non remédié ; clients MFA 202 non portés) |

---

## A. État actuel de la PR #99

| Champ | Valeur |
|---|---|
| URL | https://github.com/djasiqi/atmr/pull/99 |
| Titre | `security: close login MFA bypass (P0-03)` |
| Branche | `security/p0-03-mfa` |
| HEAD | `b5712df8c153bd22a8c0a6724b4ef731c62def1d` |
| État | OPEN, non-Draft |
| Mergeable | **false** (`CONFLICTING` / `DIRTY` vs `main`) |
| Security Gate (historique) | **PASS** |
| Security Scan (historique) | **FAIL** (voir B) |
| Secret Pattern (run précoce) | FAIL puis PASS sur re-run |
| Vercel atmr / lirie-app | FAILURE (deployment blocked) — hors MFA |

Commits PR #99 :

1. `a9bfc849` — close login MFA bypass  
2. `81d6ebf3` — fixtures tenant via business access tokens  
3. `b5712df8` — pyotp + adaptation tests MFA-era  

---

## B. Cause du blocage de fusion

1. **Conflits massifs avec `main` / AUTH-PRO** : merge-tree montre des « changed in both » sur notamment  
   `backend/routes/auth.py`, `backend/ext.py`, `backend/sockets/chat.py`,  
   `backend/routes/company_mobile_auth.py`, `frontend/.../Login.jsx`,  
   `mobile/.../login.tsx`, `sessionProvider.tsx`, `client.ts`, workflows, requirements.  
   → **Ne pas fusionner la PR #99 en l’état.**

2. **Security Scan FAIL (historique, sept. 2026)** : jobs `pip-audit`  
   - https://github.com/djasiqi/atmr/actions/runs/34769970561  
   - Cause réelle : `HttpError: Resource not accessible by integration` lors du **commentaire PR**  
     (aucune CVE CRITICAL/HIGH ; scan lui-même « No vulnerabilities found »).  
   - Échec **technique de permissions GitHub App**, pas une vulnérabilité introduite par MFA.

3. **Dérive produit** : AUTH-PRO a réécrit cookies JWT, sessions mobiles, portal grants ;  
   la PR #99 (sept.) ne contient pas ces contrats.

---

## C. Comparaison avec AUTH-PRO

| Élément | PR #99 | AUTH-PRO `cdbbccef` / stack #101–#103 |
|---|---|---|
| `backend/security/mfa_login.py` | présent | **absent** (avant ce lot) |
| `backend/security/privileged_access.py` | présent | **absent** (avant ce lot) |
| Interrupt login (`maybe_interrupt_login_for_mfa`) | oui | **non** → bypass password → JWT métier |
| `ext.check_if_token_revoked` refuse purpose MFA | oui | **non** |
| Refresh refuse temp_token MFA | oui | **non** |
| WebSocket refuse purpose MFA | oui | **non** |
| `/totp/admin-disable` break-glass | oui | **non** |
| TOTP setup/challenge endpoints | oui | partiels (challenge orphelin : rien n’émettait `temp_token` au login) |
| Cookies / mobile device session / portal grants | ancien | **AUTH-PRO enrichi** |

**Conclusion :** le correctif MFA reste **nécessaire** sur le candidat LIRIE actuel.  
Port **chirurgical** requis — pas de merge aveugle de #99.

---

## D. Correctif MFA nécessaire (réalisé en worktree isolé)

Stratégie : base AUTH-PRO + port minimal backend (+ interrupt OAuth).

### Fichiers

| Action | Chemin |
|---|---|
| ADD | `backend/security/mfa_login.py` |
| ADD | `backend/security/privileged_access.py` |
| ADD | `backend/tests/security/mfa_session.py` |
| ADD | `backend/tests/security/test_mfa_login_p0.py` |
| ADD | `backend/tests/security/test_privileged_access.py` |
| ADD | `docs/security/privileged-access.md` |
| PATCH | `backend/ext.py` — rejet `is_restricted_mfa_jwt` |
| PATCH | `backend/security/totp_service.py` — JTI enroll + fallback mémoire tests |
| PATCH | `backend/sockets/chat.py` — rejet purpose MFA |
| PATCH | `backend/routes/company_mobile_auth.py` — MFA user avant MFA company |
| PATCH | `backend/routes/auth.py` — interrupt, `_complete_authenticated_session`, refresh, enroll/disable/admin-disable, challenge |
| PATCH | `backend/routes/external_auth.py` — interrupt Google/Apple (**écart comblé vs PR #99**) |
| PATCH | `backend/requirements.base.txt` — `pyotp>=2.9.0` |
| PATCH | `backend/tests/security/test_token_rotation.py` — parcours MFA |

### Contrats de sécurité

| # | Contrat | Statut worktree |
|---|---|---|
| 1 | MFA activé → aucun JWT métier avant 2e facteur | **PASS** (tests) |
| 2 | Privilegié → enrollment imposé (admin LIRIE, institution ADMIN, owner entreprise) | **PASS** (tests) |
| 3 | temp_token refusé par `jwt_required` | **PASS** (tests `/me`) |
| 4 | Refresh refusé avant MFA | **PASS** (tests) |
| 5 | WebSocket refuse jetons temporaires | **PASS** (code path patché ; pas de test socket dédié rejoué) |
| 6 | Révocation / challenge JTI one-shot | **PASS** (tests challenge + helpers) |
| 7 | Google/Apple / liés | **PASS** code (`external_auth` interrupt) ; tests dédiés OAuth non rejoués |
| 8 | Admin disable : interdit privilégié ; break-glass TOTP+motif+audit | **PASS** (tests) |
| 9 | Permissions DRIVER / CLIENT / REQUESTER non forcé | **PASS** (tests privileged_access) |

### Gaps assumés (→ PARTIAL)

- **Frontend / mobile** : pas de port UI 202 (`temp_token` / challenge / enroll). Backend refuse les JWT métier ; l’UX login privilégié/TOTP sera « cassée » jusqu’au port client.
- Image Docker `backend_tests` : `pyotp` installé à la volée (pas de rebuild d’image dans ce lot).
- Suites portal-grant / CSRF / chauffeur larges : non rejouées en entier (hors scope MFA ; pas de changement métier finance).

---

## E. Tests et preuves

Environnement isolé Docker (`atmr-backend_tests:latest`, réseau `lirie-ci-gate`, mount worktree).

```text
pytest tests/security/test_mfa_login_p0.py \
       tests/security/test_privileged_access.py \
       tests/security/test_token_rotation.py \
       tests/security/test_mobile_login_contract_p0.py \
       tests/security/test_refresh_fail_closed.py -q
============================= 39 passed in 35.52s ==============================
```

Détail : 14 MFA login + 8 privileged + 5 token_rotation + 6 mobile contract + 6 refresh fail-closed.

---

## F. Compatibilité avec la stack #101–#103

| PR | HEAD | Relation |
|---|---|---|
| #101 AUTH-PRO | `cdbbccef` | **Base** du worktree MFA |
| #102 FIN-FACTURE | `ade45434` | Empilé sur #101 ; pas touché |
| #103 FIN-CAISSE | `4be18daf` | Empilé sur #102 ; `auth.py` **identique** à AUTH-PRO |

Le correctif MFA est **orthogonal** aux lots finance.  
Intégration recommandée (à valider) :

1. Rebase / cherry-pick de `candidate/lirie-security-p0-03-mfa-20261112` **au-dessus** de FIN-CAISSE `#103`, **ou**  
2. Empiler une PR Draft MFA sur `#101` puis rejouer sur la stack complète.

**Aucune réécriture** des historiques #101–#103.  
**Aucun push** effectué.

WIP principal `c:\Users\jasiq\atmr` (`main` @ `25faab10`, dirty) : **non modifié**.

---

## G. Diagnostic GitGuardian (PR #103)

| Champ | Valeur |
|---|---|
| Contrôle | `GitGuardian Security Checks` → **failure** |
| SHA évalué | `4be18daf4ac41be89d258440ac1332237a73e0c4` |
| Check run | https://github.com/djasiqi/atmr/runs/131421871696 |
| Dashboard | https://dashboard.gitguardian.com |
| Incident | [40006359](https://dashboard.gitguardian.com/workspace/798332/incidents/40006359?occurrence=312274660) (workspace **798332**) |
| Titre | **1 secret uncovered!** (24 commits scannés) |
| Type | **Generic Password** · status **Triggered** |
| Commit | `debf5eed` — `test(caisse): suites reglement et rapport FIN-CAISSE-1` |
| Fichier | `backend/tests/application/settlement/test_mission_cycle_etape27.py` (~L378) |
| Contexte | `client.post(LOGIN, json={..., "password": …})` — fixture de recette isolée |

**Conclusion réelle :** alerte scanner **valide** ; contenu = **mot de passe de fixture de test** (marqueur de recette jetable), **pas** un secret de production.

Présence uniquement sous tests (`backend/tests/...`, un test Jest mobile).  
Aucune occurrence config/routes/services/compose/`.env`.

Contrôles internes #103 : Gitleaks **pass**, Security Gate **pass**.  
PR #101 / #102 : GitGuardian **pass**.

### Mesures

| Action | Applicable |
|---|---|
| Révocation / rotation credentials prod | **Non** |
| Exception GG documentée **ou** remédiation littéraux test | **Oui — à valider** |
| Neutraliser / désactiver GitGuardian | **Interdit** |

**SECURITY PASS GitGuardian : non déclaré.**

Accès humain utile : compte workspace GitGuardian **798332** pour clôturer/exceptionner l’incident 40006359 dans l’UI.

---

## H. Commits éventuels (locaux uniquement)

| SHA | Message |
|---|---|
| `2366f6a1` | `security: close MFA login bypass (P0-03) on AUTH-PRO base` |
| `aefa42fe` | `test(security): adapter token_rotation au parcours MFA P0-03` |

**Non poussés.** Branche locale uniquement.

---

## I. Risques encore ouverts

1. GitGuardian #103 toujours **FAIL** (fixture test).  
2. UX login web/mobile sans port 202 → comptes TOTP / privilegiés bloqués côté client.  
3. PR #99 historique toujours CONFLICTING — à fermer ou marquer superseded après validation de la branche candidate.  
4. MFA P0-03 non encore intégré sur HEAD FIN-CAISSE (rebase stack à valider).  
5. Push iOS / MFA PR #99 Vercel failures / Security Scan permissions : hors clôture fonctionnelle MFA mais encore visibles sur #99.  
6. `pyotp` pas encore cuit dans l’image CI locale (install runtime).

---

## J. Prochaine action autorisée

**À valider explicitement avant toute suite :**

1. Stratégie d’intégration MFA : rebase sur `#103` vs PR Draft empilée sur `#101`.  
2. Autorisation de **push** de `candidate/lirie-security-p0-03-mfa-20261112` (ou branche dérivée).  
3. Remédiation / exception formelle GitGuardian incident **40006359** (sans masquer le contrôle).  
4. Port minimal frontend/mobile du flux 202 MFA (chantier séparé).  
5. Fermeture ou marquage superseded de la PR #99 après revue humaine.

**Interdit sans nouvelle validation :** merge, déploiement, migration prod, activation flags, modification WIP principal, développement Saferpay/Worldline.
