# Security Gate LIRIE

Précondition bloquante de tout build et déploiement production. Un contrôle P0 rouge arrête le pipeline : **pas d’image, pas de deploy**.

## Modèle

```text
PR
 └─ .github/workflows/security-gate.yml
       └─ GREEN sinon la PR ne doit pas merger

deploy.yml
 └─ 1. security-gate.yml via workflow_call sur EXACT GITHUB_SHA
 └─ 2. gps-mobile-critical (parallèle au gate)
 └─ 3. build  (needs : gate + gps-mobile-critical)
 └─ 4. deploy (needs : gate + gps-mobile-critical + build)

Check GitHub stable (protection de branche) : `Security Gate / Security Gate verdict`.
```

Le deploy **n’interroge jamais** le statut d’un ancien run GitHub. Il réexécute le workflow réutilisable sur le SHA déployé.

Aucun `continue-on-error` sur un job du gate.

## Contrôles CI (automatiques)

| Job | Contrôle | Seuil bloquant |
|-----|----------|----------------|
| `sast-codeql` | CodeQL Python + JS/TS (SARIF local) | SARIF `error` ou `security-severity >= 7.0`. Upload GitHub code scanning **désactivé** : dépôt privé, Code Security non acheté (API 422). |
| `sast-static` | Bandit + Semgrep `p/ci` + `p/security-audit` | Bandit HIGH+ ; Semgrep `--error` |
| `sca` | pip-audit `backend/requirements.prod.txt` | Toute vulnérabilité |
| `secrets` | Gitleaks HEAD (arbre courant) | Tout secret |
| `docker-fs` | Trivy fs backend / ws-service / frontend | CRITICAL |
| `iac` | Trivy config compose prod + Traefik exemple | CRITICAL |
| `tenant` | `test_tenant_isolation_p0.py` | Tout test en échec |
| `integrity` | scripts anti-fuites / sentinelles / Kafka | Tout échec |
| `security-gate` (`Security Gate verdict`) | Agrégateur — **check requis** | Tous les jobs ci-dessus = `success` |

Scan image Docker post-build (Trivy CRITICAL) : reste dans [`deploy.yml`](../../.github/workflows/deploy.yml) **après** le build, car l’image n’existe pas avant.

## CodeQL / Code Security

Le dépôt `djasiqi/atmr` est **privé**. L’activation `advanced_security` via l’API GitHub retourne **422 : Advanced security has not been purchased**. LIRIE n’est pas rendu public pour obtenir CodeQL. Le job `sast-codeql` analyse toujours le SHA et bloque sur le SARIF local (`upload: never`).

## Triage secrets HEAD (P0-01)

- Artefacts pytest `ci-before-fix*` / `ci-after-fix*` et logcats `ops-readiness/evidence` : retirés du HEAD (pas de valeur produit). Historique : P0-04.
- Script de purge : plus aucune clé en dur ; la valeur vient de `EXPOSED_KEY_TO_PURGE`. Ancienne clé OpenWeatherMap dans ce script : **considérée compromise** — révoquer côté fournisseur si encore active. Valeur non reproduite ici.
- Fixtures JWT / tokens / SHA Git : remplacées ou concaténées pour rester clairement non-credentials. Aucune allowlist globale ajoutée.

## Statut P0-01

✅ **Implémenté** : Security Gate réutilisable mergé sur `main`
(`66848480ac119202ea04ad3a6cf2e72d4ea47977`). Run push `main`
[Security Gate #34732397474](https://github.com/djasiqi/atmr/actions/runs/34732397474)
**GREEN** (tous les jobs + verdict). P0-01 **CLOSED**.

## P0-03 — MFA

✅ **Implémenté** (branche `security/p0-03-mfa`, à merger seulement si
**Security Gate GREEN et Backend Tests GREEN**) :

- Login : `totp_enabled=true` → `202` + `temp_token` `2fa_challenge`, aucun JWT métier
- Privilegié sans TOTP → `202` + `temp_token` `mfa_enroll` ([`docs/security/privileged-access.md`](privileged-access.md))
- `/totp/challenge` réutilise le même chemin de session que le login
  (`issue_business_access_token` / `build_business_access_claims` dans
  [`backend/routes/auth.py`](../../backend/routes/auth.py))
- `/totp/disable` interdit pour un compte privilegié ; break-glass `/totp/admin-disable`
- `SECURITY_2FA_ENABLED=true` en production
- Tests tenant (`test_tenant_isolation_p0.py`) : session métier via
  `_business_session_headers` → `issue_business_access_token` (pas de login
  password-only, pas de temp_token). Les tests MFA/login restent sur
  `/auth/login` + challenge.

## Checklist manuelle (hors CI)

Ces preuves ne peuvent pas être établies par GitHub Actions. À valider avant un GO prod (P0-06 / P0-07) :

- [ ] Dernier backup writer-only frais
- [ ] Dernier restore drill < 90 jours, RPO/RTO mesurés
- [ ] Clé prod : `delete` / `prune` des backups **refusé**
- [x] MFA privileged enforced (P0-03) — enrollment/challenge branchés, à valider par le run Security Gate de la PR
- [ ] Secrets historiques scannés / rotatés (P0-04)

## Règles

- Pas de contournement CI pour faire passer un lot.
- Toute vulnérabilité réelle découverte est corrigée avant de poursuivre.
- Deploy uniquement après Security Gate GREEN sur le SHA exact.
