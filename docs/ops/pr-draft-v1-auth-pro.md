# Brouillon PR — V1 AUTH-PRO → `main`

**Statut :** texte local uniquement — **PR GitHub non créée**  
**Head :** `integrate/lirie-v1-auth-pro` @ `dd190c95` (code validé `c85a84e4` + docs PRE-MERGE)  
**Base :** `main` (`afea1f00` au moment de la revue)  
**Alembic :** `9ddf4ec5763f` (6 migrations linéaires depuis `cc4eebfd81e6`)

---

## Title

```
feat(auth): V1 AUTH-PRO — grants, invitations org/chauffeur, auth externe (sans MFA)
```

---

## Body (à coller dans `gh pr create`)

```markdown
## Summary

- Intégration **V1 AUTH-PRO** sur `main` : habilitations portail (grants), invitations chauffeurs, demandes d’inscription organisation + instruction admin, identités/sessions Google/Apple.
- **Exactement 6 migrations** Alembic jusqu’à `9ddf4ec5763f` (head unique).
- **Exclusions explicites** : MFA P0-03, signature réelle (`c3s01`), règlement non-paiement (`c2r02`), caisse / dispatch métier étendu, vagues contact-client / déclarations transport.
- `PORTAL_GRANT_ENFORCEMENT` **éteint par défaut** → comptes legacy restent sur le chemin rôle historique jusqu’à activation séparée.
- `POST /companies/me/drivers/create` renvoie désormais **403 `invitation_requise`** (parcours invitation uniquement).

## Test plan

- [ ] CI `backend-tests` : 4 shards verts + gate coverage (`SKIP_SOCKETIO=false`)
- [ ] CI `frontend-tests` + `security-scan` verts
- [ ] Revue humaine : `portal_grant_access` / `portal_scope` / invitations / `external_auth`
- [ ] QA manuelle : login company & driver existants (enforcement off)
- [ ] QA : créer / accepter invitation chauffeur ; refuser création directe
- [ ] QA : signup organisation + file admin applications
- [ ] Confirmer `alembic heads` = `9ddf4ec5763f` uniquement sur la branche PR
- [ ] **Ne pas** activer `PORTAL_GRANT_ENFORCEMENT` ni migrer la prod dans cette PR sans autorisation distincte

## Preuves locales (pré-push)

| Suite | Résultat | Fichier |
| --- | --- | --- |
| AUTH-PRO pytest | 103 passed | `docs/ops/v1-auth-pro-pytest-schema-gate-backend.txt` |
| Smoke JWT/OAuth | 17 passed | `docs/ops/v1-auth-pro-pytest-schema-gate-smoke.txt` |
| Jest AUTH-PRO | 5 suites / 20 tests | `docs/ops/v1-auth-pro-jest-schema-gate.txt` |
| Mig cc4eeb→9ddf4 sur PG isolée | OK | `docs/ops/addendum-validation-v1-auth-pro-pre-merge-review.md` |
| Schema consistency gate | GO | `docs/ops/addendum-validation-v1-auth-pro-schema-consistency-gate.md` |

## Merge ≠ activate

- **GO MERGE CODE** : après CI + revue (cette PR).
- **GO ACTIVATION PROD** : migrate `9ddf4ec5763f` + décisions flags + smoke staging/prod — **autorisation séparée**.

## Refs

- Pre-merge review : `docs/ops/addendum-validation-v1-auth-pro-pre-merge-review.md`
- Remédiation : `docs/ops/addendum-validation-v1-auth-pro-remediation-01.md`
```

---

## Critères d’acceptation (checklist merge)

1. Diff PR = périmètre AUTH-PRO + 6 migs ; **zéro** MFA / c2r02 / c3s01.  
2. CI GitHub verte (backend shards + frontend + security).  
3. Au moins une review approve sécurité auth.  
4. Tip Alembic PR = `9ddf4ec5763f`.  
5. Pas de migration prod ni toggle enforcement dans le merge lui-même.

---

## Commandes **non exécutées** (attendre GO push)

```bash
# Depuis le worktree integrate — UNIQUEMENT après autorisation
git push -u origin integrate/lirie-v1-auth-pro

gh pr create --base main --head integrate/lirie-v1-auth-pro \
  --title "feat(auth): V1 AUTH-PRO — grants, invitations org/chauffeur, auth externe (sans MFA)" \
  --body-file docs/ops/pr-draft-v1-auth-pro.md
```

*(Ajuster `--body-file` si le corps Markdown doit être extrait de la section Body seule.)*
