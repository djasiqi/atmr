# Addendum PRE-MERGE-REVIEW — V1 AUTH-PRO

**Date :** 2026-11-18  
**Branche :** `integrate/lirie-v1-auth-pro`  
**HEAD :** `c85a84e4`  
**Base :** `origin/main` @ `afea1f00`  
**Alembic tip :** `9ddf4ec5763f`  
**Réf. cohérence schéma :** `docs/ops/addendum-validation-v1-auth-pro-schema-consistency-gate.md`

| Opération | Statut |
| --- | --- |
| Revue pré-merge locale | **Faite** |
| Brouillon PR (texte) | **Préparé** |
| Push GitHub | **Non** |
| Création PR distante | **Non** |
| Merge / deploy / migrate prod | **Non** |

---

## 1. Diff final et périmètre exact

### Commits (`origin/main..HEAD`) — 6

| SHA | Message |
| --- | --- |
| `ed3aa0f1` | feat(auth): préparer V1 AUTH-PRO sans MFA sur origin/main |
| `ee42309c` | fix(auth): compléter dépendances V1 (portal_scope, enums, handoff) |
| `10859968` | fix(auth): mobile_contract_issue + conversation_service |
| `63a6ed98` | docs(ops): rapport validation — NO-GO MERGE (historique) |
| `94939691` | fix(auth): remédiation grants/sockets/invitations/Jest |
| `c85a84e4` | fix(auth): aligner Company sur schéma Alembic V1 |

### Volume

- **118 fichiers** — ~+18 970 / −524 lignes  
- Groupes : `backend` 66 · `frontend` 28 · `docs` 19 · `.github` 3 · `mobile` 1 · `services` 1

### Migrations V1 (exactement 6)

| # | Revision | Fichier |
| --- | --- | --- |
| 1 | `a643270a0e07` | habilitations portail grant |
| 2 | `ea71f45bd483` | demandes inscription organisation |
| 3 | `8e76ed1ac6c6` | instruction des demandes |
| 4 | `255990200183` | invitations chauffeurs |
| 5 | `99a1e3b30d27` | identités externes Google/Apple |
| 6 | `9ddf4ec5763f` | borne de lecture conversations |

Chaîne : `cc4eebfd81e6 → … → 9ddf4ec5763f` — **un seul head**.

### Périmètre produit inclus

- Portal grants + enforcement **opt-in** (`PORTAL_GRANT_ENFORCEMENT` off par défaut)  
- Invitations chauffeurs (création directe `/me/drivers/create` → 403 `invitation_requise`)  
- Demandes d’inscription organisation + admin review/approval  
- Identités / sessions externes (Google/Apple)  
- Frontend signup / invitations / admin applications  
- CI : shards pytest + `SKIP_SOCKETIO=false` pour contrôles grant socket  

### Hors périmètre (confirmé absent du tree)

| Domaine | Preuve |
| --- | --- |
| MFA P0-03 (`mfa_login`, `PURPOSE_MFA_ENROLL`) | Aucun hit |
| Signature réelle / `convention_real` / `c3s01` | Aucun hit / mig absente |
| Règlement / `c2r02` / `DriverInvoiceSimulation` | Aucun hit |
| Migrations contact client / déclarations transport (`e3a22ba34db5`, `2e103f125207`) | Absentes ; `company.py` = `origin/main` |
| WIP settlement UX S23 | Non dans le diff V1 |

---

## 2. État des tests (HEAD `c85a84e4`)

| Suite | Preuve | Résultat |
| --- | --- | --- |
| AUTH-PRO backend (invitations, org apps, portal grant 1→1f, web/mobile) | `v1-auth-pro-pytest-schema-gate-backend.txt` | **103 passed** (`SKIP_SOCKETIO=false`, PG vierge) |
| Smoke JWT / OAuth / shard / Apple | `v1-auth-pro-pytest-schema-gate-smoke.txt` | **17 passed** |
| Jest AUTH-PRO (5 suites) | `v1-auth-pro-jest-schema-gate.txt` | **5 suites / 20 tests** |
| Rejeu lié HEAD (JWT + isolation multi-org + CreateDriver 403) sur DB mig pré-merge | `v1-auth-pro-pytest-premerge-smoke.txt` | **14 passed** |

Les preuves schema-gate ont été produites **sur le code du tip `c85a84e4`** (après alignement `Company`). Aucun commit ultérieur.

---

## 3. Compatibilité migration

### Protocole (isolé, hors prod)

1. Conteneur jetable `v1-premerge-mig-pg` (user/db `v1mig` / `v1_premerge_mig`) — réseau `atmr_default`  
2. Upgrade jusqu’à **`cc4eebfd81e6`** = schéma représentatif **type `main` avant V1** (anonyme, sans dump prod)  
3. Upgrade **`9ddf4ec5763f`** = les 6 migrations uniquement  

### Résultat

| Contrôle | Statut |
| --- | --- |
| Upgrade linéaire des 6 | OK |
| `alembic_version` | `9ddf4ec5763f` |
| Tables V1 | `portal_grant`, `driver_invitation`, `organization_application*`, `user_external_identity`, `external_auth_transaction` |
| 5 colonnes `company` hors V1 | **count = 0** |

**Comptes existants :** les migrations **ajoutent** des tables / contraintes ; elles ne réécrivent pas les rôles legacy. Avec `PORTAL_GRANT_ENFORCEMENT` **éteint**, `role_required_grant_gate` retombe en chemin **legacy** → sessions company/driver/institution actuelles restent fonctionnelles. L’activation enforcement est un levier **post-merge / activation** distinct.

**Breaking code (merge ≠ activate feature flags) :** `POST /api/v1/companies/me/drivers/create` refuse désormais la création directe (403) — les clients UI doivent utiliser les invitations. Comportement voulu V1 ; à communiquer dans la PR.

---

## 4. Sécurité / permissions (synthèse)

| Thème | Évaluation |
| --- | --- |
| Escalade JWT forge / multi-entreprise | Tests phase1c verts (ambiguïté, conversation autre org) |
| Révocation grant + sockets | Phase1 / 1d / 1e / 1f verts avec Socket.IO réel |
| Tracking-health | Lookup `public_id` + `authorized_company_id` (plus de `DataError` UUID) |
| OAuth / logs | Redaction + native Apple couverts smoke |
| CI | `SKIP_SOCKETIO=false` + shards — surface plus réaliste, durée/timeout à surveiller |

Risques résiduels : voir §6.

---

## 5. Projet de description de PR

Fichier prêt à coller : **`docs/ops/pr-draft-v1-auth-pro.md`**

- Titre proposé : `feat(auth): V1 AUTH-PRO — grants, invitations, org signup (sans MFA)`  
- Base : `main`  
- Head : `integrate/lirie-v1-auth-pro` @ `c85a84e4` (après push autorisé)

---

## 6. Risques résiduels

| Niveau | Risque | Mitigation |
| --- | --- | --- |
| P2 | CI shards + Socket.IO réel : flakiness / durée | Gate `pytest-gate` ; surveiller 1ʳᵉ run GitHub |
| P2 | CreateDriver 403 : clients / scripts legacy | Docs PR + checklist QA manuelle invitations |
| P2 | Enforcement grants off par défaut | Activation prod = étape séparée + backfill grants si besoin |
| P2 | `package-lock` frontend/mobile + `ws-service` requirements | Diff mineur ; CI frontend/security-scan |

Aucun **P0/P1** technique ouvert sur le périmètre V1 validé.

---

## 7. Conditions GO MERGE CODE

Toutes **locales** sont remplies. Avant merge GitHub :

1. **Autorisation humaine** push de `integrate/lirie-v1-auth-pro`  
2. Ouverture PR → `main` avec le brouillon ci-joint  
3. CI GitHub verte : backend shards + gate coverage, frontend tests, security-scan  
4. Revue humaine : sécurité auth (grants / invitations / external auth) + produit (CreateDriver → invitation)  
5. Confirmation tip Alembic PR = `9ddf4ec5763f` uniquement  
6. **Pas** de merge si MFA / c2r02 / c3s01 / colonnes company hors V1 réapparaissent dans le diff

### Verdict MERGE CODE (préparation) — **GO technique**

Sous réserve des contrôles CI + revue après publication. **STOP** : aucun push/PR distante dans cette phase.

---

## 8. Conditions distinctes GO ACTIVATION PROD

**Merge ≠ activate.** Après merge code uniquement :

| Étape | Condition |
| --- | --- |
| Backup DB + fenêtre maintenance | Planifié |
| `alembic upgrade 9ddf4ec5763f` en staging puis prod | Autorisation migrate **distincte** |
| Vérifier head unique post-migrate | `9ddf4ec5763f` |
| Feature flags | `PORTAL_GRANT_ENFORCEMENT` / observation / invitations : décision produit **explicite** (défaut off = legacy OK) |
| Smoke prod | Login company/driver, invitation chauffeur, signup org, pas de régression sessions |
| Rollback | Procédure down-revision / feature-flag off documentée |

**GO ACTIVATION PROD :** **NO-GO** tant que migrate + flags + smoke staging/prod non autorisés et non exécutés.

---

## 9. Contrôles CI / revues encore nécessaires (post-publication)

1. Workflow `backend-tests` (4 shards, `SKIP_SOCKETIO=false`, `pytest-gate`)  
2. Workflow `frontend-tests`  
3. Workflow `security-scan`  
4. Review code : `portal_grant_access`, `portal_scope`, `ext.role_required`, `driver_invitations`, `external_auth`  
5. Review produit : parcours invitation + admin org applications  
6. Ops : plan migrate staging → prod (hors cette PR si policy « merge puis migrate »)

---

**Fin PRE-MERGE-REVIEW — aucune action distante effectuée.**
