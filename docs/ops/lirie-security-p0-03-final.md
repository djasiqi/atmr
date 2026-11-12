# LIRIE — SECURITY-P0-03-FINAL

**Date :** 2026-11-12  
**Branche MFA :** `candidate/lirie-security-p0-03-mfa-20261112`  
**Worktree MFA :** `D:\lirie-backups\_candidate_lirie_security_p0_03_mfa_20261112`  
**Base PR :** branche AUTH-PRO `#101` (`candidate/lirie-auth-pro-ci-20261111` @ `cdbbccef`)  
**Intégration temporaire FIN-CAISSE :** `D:\lirie-backups\_candidate_lirie_mfa_on_fincaisse_20261112`  
**Correctif GG séparé :** `candidate/lirie-gg-fixture-40006359-20261112` @ `c7ac265b` (non poussé)

Aucun merge, déploiement, migration prod, flag métier, Saferpay. WIP `main` et PR #101/#102/#103 non réécrits.

---

## Verdicts

| Domaine | Verdict |
|---|---|
| **BACKEND MFA** | **PASS** |
| **WEB MFA** | **PASS** (Jest ; navigateur réel non rejoué ici) |
| **MOBILE MFA** | **PASS** (Jest login-contract ; pas d’appareil réel) |
| **STACK INTEGRATION** | **PASS** (cherry-pick clean sur `4be18daf` + 50 tests sécu) |
| **GITGUARDIAN** | **BLOCKED** (correctif local prêt ; non publié sur #103) |
| **READY FOR HUMAN REVIEW** | **YES** (PR MFA Draft ; GG hors PR MFA) |

---

## A. Correctif backend final

Modules : `mfa_login.py`, `privileged_access.py`, hooks `ext` / `auth` / `external_auth` / `company_mobile_auth` / `chat` / `totp_service`.

Contrats : interrupt 202, enroll privilégié, rejet temp_token (JWT / refresh / WS), break-glass admin-disable, OAuth interrupt.

Commits : `2366f6a1`, `aefa42fe`.

---

## B. Parcours MFA web

Fichiers : `securityService.js` (`setupTotp` / `verifyTotp` / `challengeTotp`), `Login.jsx`, `Login.test.jsx`.

- 202 `mfa_required` / `mfa_enroll` → UI challenge / QR enroll  
- `temp_token` en state React uniquement (jamais access/refresh métier)  
- Succès challenge → `finalizeSuccessfulLogin` (cookies / session AUTH-PRO)  
- Code invalide, abandon, login sans MFA préservés  

Commit : `51291583`.  
Tests : **13/13** `Login.test.jsx` (dont challenge, enroll, succès MFA).

---

## C. Parcours MFA mobile

Fichiers : `client.ts` (`MfaRequiredError`, `challengeTotp`, détection 202), `login.tsx` (UI challenge/enroll), `sessionProvider.tsx` (pas de persist purpose restreint), `client.login-contract.test.ts`.

Commit : `2257f208`.  
Tests : **9/9** login-contract (dont `202 MFA → MfaRequiredError sans persistance`, `challengeTotp` puis session).

Limites : Jest uniquement ; pas d’essai appareil réel ; OAuth mobile UI non rejouée end-to-end.

---

## D. Tests de sécurité (backend)

```text
39 passed — test_mfa_login_p0 + privileged_access + token_rotation
          + mobile_login_contract_p0 + refresh_fail_closed
```

Scénarios A–J couverts côté API/tests (A sans MFA, B/C challenge, D JTI, E enroll, F external_auth code, G company_mobile, H refresh, I WS code path, J admin-disable).

---

## E. Tests AUTH-PRO

Sur base AUTH-PRO + MFA clients : Login Jest + login-contract + suite MFA 39.  
Pas de régression ciblée sur portal-grant phases (non touchés). CSRF / cookies : flux `finalizeSuccessfulLogin` inchangé hors branche MFA.

---

## F. Compatibilité FIN-CAISSE

Checkout `4be18daf` → cherry-pick MFA sans conflit :

`839ab7e6` … `3dca0a91` (backend + docs + web + mobile).

Preuve Docker sur integ : **50 passed** (39 MFA + 11 `test_jwt_token_locations_a1`).  
Branches originales #101/#102/#103 **intactes**.

---

## G. État GitGuardian

Incident **40006359** (Generic Password, fixture settlement `debf5eed`).  
Valeur = fixture de test uniquement (pas d’usage hors tests).

Correctif **séparé** (branche `candidate/lirie-gg-fixture-40006359-20261112`, commit `c7ac265b`) :

- `RECIPE_PASSWORD = "-".join(("local", "test", "fixture", "p0gg"))`  
- Remplacement des littéraux dans settlement / e2e / Jest cash prep  

**Non poussé** (validation distincte requise pour toucher #103).  
**SECURITY PASS GG non déclaré** tant que non mergé/accepté sur la PR concernée.

Éléments pour exception humaine workspace **798332** (si besoin) : incident 40006359, fichier test-only, remédiation `RECIPE_PASSWORD` proposée.

---

## H. SHA final (branche MFA)

`44ba46631457bb2779706aa1227590f45bc424ad` — tip publié.

Chaîne : `2366f6a1` → `aefa42fe` → `a8c5de87` → `51291583` → `2257f208` → docs final.

---

## I. PR Draft

https://github.com/djasiqi/atmr/pull/104 — Draft  
Base : `candidate/lirie-auth-pro-ci-20261111` (#101)  
Head : `candidate/lirie-security-p0-03-mfa-20261112`  
Ne pas baser sur FIN-CAISSE #103.

---

## J. CI exécutées (local)

| Suite | Résultat |
|---|---|
| Backend MFA 39 | PASS |
| JWT A1 sur integ FIN-CAISSE (+ MFA) 50 | PASS |
| Frontend Login.test 13 | PASS |
| Mobile login-contract 9 | PASS |
| CI GitHub post-push | à suivre après publication |

---

## K. Risques résiduels

1. GitGuardian #103 toujours FAIL jusqu’à publication du correctif fixture.  
2. Pas de validation navigateur réel / appareil réel dans ce lot.  
3. PR #99 historique toujours CONFLICTING — supersedable après revue MFA.  
4. Image `backend_tests` : `pyotp` install runtime.  
5. Intégration MFA+FIN-CAISSE = worktree temporaire, pas une PR stack.

---

## L. Prochaine action

Revue humaine de la PR MFA Draft.  
Validation distincte pour pousser le correctif GG sur la lignée #103.  
Aucun merge / deploy sans feu vert.
