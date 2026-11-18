# Addendum SCHEMA-CONSISTENCY-GATE — V1 AUTH-PRO

**Date :** 2026-11-18  
**Branche :** `integrate/lirie-v1-auth-pro`  
**HEAD de départ :** `94939691`  
**Base :** `origin/main` @ `afea1f00`  
**Alembic tip (inchangé) :** `9ddf4ec5763f`  
**Réf. remédiation :** `docs/ops/addendum-validation-v1-auth-pro-remediation-01.md`

| Opération | Statut |
| --- | --- |
| Alignement modèle ↔ migrations V1 | **Fait** |
| Migration Alembic supplémentaire | **Non** (non nécessaire) |
| Push / merge / deploy / migrate prod | **Non effectué** |

---

## 1. Les cinq colonnes analysées

| Colonne | Commit d’introduction sur V1 | Origine réelle | Fonctionnalité | Obligatoire AUTH-PRO ? | Sur base existante sans colonne | Risque si absente |
| --- | --- | --- | --- | --- | --- | --- |
| `client_contact_phone` | `ed3aa0f1` (checkout modèle AUTH-PRO) | Vague post-V1 — mig `e3a22ba34db5` (`down_revision=2e103f125207`) | Numéro pro distinct visible clients | **Non** | INSERT ORM échoue (`UndefinedColumn`) si le modèle l’expose | Hors AUTH-PRO ; bloquant uniquement si le modèle dérive |
| `client_contact_published` | Idem | Idem `e3a22ba34db5` | Consentement publication du numéro | **Non** | Idem (+ `server_default=false` en mig) | Idem |
| `provides_standard_assistance` | Idem | Vague transport — mig `2e103f125207` (`down_revision=ed461584cdae`) | Déclaration assistance standard | **Non** | Idem | Idem |
| `provides_manual_wheelchair` | Idem | Idem `2e103f125207` | Déclaration fauteuil manuel | **Non** | Idem + contrainte stock | Idem |
| `manual_wheelchair_count` | Idem | Idem `2e103f125207` | Stock fauteuils | **Non** | Idem | Idem |

**Constat clé :** `origin/main` @ `afea1f00` **n’a pas** ces colonnes dans `company.py`. Elles ont été **entraînées par erreur** lors du checkout sélectif AUTH-PRO (`ed3aa0f1`), sans leurs migrations (chaîne hors tip `9ddf4ec5763f`).

Preuve migrations absentes du tree V1 : aucun hit dans `backend/migrations/versions/` pour ces noms ; fichiers `2e103f125207_*` et `e3a22ba34db5_*` **non présents** sur la branche.

---

## 2. Schéma migrations #1–#6 vs modèles

| Contrôle | Résultat |
| --- | --- |
| PG **entièrement recréée** (`v1-auth-pro-pg`) | Vierge, sans ALTER manuel |
| `alembic upgrade 9ddf4ec5763f` | OK — tip unique |
| 5 colonnes après upgrade | **0 ligne** (`information_schema`) |
| Tables V1 | `portal_grant`, `driver_invitation`, `organization_application`, `user_external_identity` présentes |
| Modèle `Company` après correction | Aligné `origin/main` — plus aucune des 5 colonnes |

---

## 3. Solution retenue

**Aligner le code V1 sur le schéma produit par les migrations V1** (pas de 7ᵉ migration).

- Restaurer `backend/models/company.py` depuis `origin/main` (retrait des 5 colonnes, du `CheckConstraint` `ck_company_manual_wheelchair_stock`, des clés `serialize`, et du `@validates` étendu).
- **Ne pas** importer `2e103f125207` / `e3a22ba34db5` (hors périmètre AUTH-PRO ; casseraient la promesse « tip = `9ddf4ec5763f` »).
- Aucune réécriture des 6 migrations V1.

### Migrations effectivement nécessaires

**Aucune migration supplémentaire.** Tip reste `9ddf4ec5763f`.

---

## 4. Fichiers modifiés

| Fichier | Action |
| --- | --- |
| `backend/models/company.py` | Restauration `origin/main` (retrait dérive hors V1) |
| `docs/ops/addendum-validation-v1-auth-pro-schema-consistency-gate.md` | Ce livrable |
| `docs/ops/v1-auth-pro-pytest-schema-gate-backend.txt` | Preuve AUTH-PRO |
| `docs/ops/v1-auth-pro-pytest-schema-gate-smoke.txt` | Preuve JWT/OAuth |
| `docs/ops/v1-auth-pro-jest-schema-gate.txt` | Preuve Jest |

---

## 5. Tests exécutés (PG vierge, **sans** ALTER)

| Suite | Env | Résultat |
| --- | --- | --- |
| AUTH-PRO backend (invitations, org apps, portal grant 1/1c/1d/1e/1f, web/mobile split) | `DATABASE_URL=…@v1-auth-pro-pg…`, `SKIP_SOCKETIO=false` | **103 passed** |
| Smoke JWT / OAuth / shard / Apple | Même PG | **17 passed** |
| Jest 5 suites AUTH-PRO | `react-scripts test` | **5 suites / 20 tests passed** |
| Colonnes absentes **après** les tests | Requête SQL | **count = 0** |

Contrôles multi-entreprises couverts notamment par `test_portal_grant_phase1c` (JWT ambigu, conversation autre org) et isolation grants — tous verts dans le run 103.

---

## 6. Vérification hors V1

| Domaine | Statut |
| --- | --- |
| MFA `mfa_login` | Absent |
| Migrations `2e103f125207`, `e3a22ba34db5`, `c2r02*`, `c3s01*` | Absentes du tree |
| Tip Alembic | `9ddf4ec5763f (head)` unique |
| `company.py` vs `origin/main` | Diff vide après correction |

---

## 7. Risques résiduels

| Niveau | Risque | Statut |
| --- | --- | --- |
| ~~P1~~ schéma 5 colonnes | Contournement ALTER test | **Levé** — code = schéma migrations |
| P2 | Vagues ultérieures (contact client / déclarations transport) devront porter leurs propres migrations quand intégrées | Accepté, hors V1 |
| P2 | CI `SKIP_SOCKETIO=true` vs 3 tests socket | Inchangé vs remédiation-01 |

Aucun **P0/P1** résiduel lié à la cohérence modèle / migration V1.

---

## 8. Verdict

### SCHEMA-CONSISTENCY-GATE — **GO**

Le code V1 fonctionne avec le schéma **réellement** produit par les migrations jusqu’à `9ddf4ec5763f`, sans correction manuelle ni contournement en base de test.

### MERGE V1 vers `main` — **GO MERGE V1 (technique confirmé)**

Sous réserve d’autorisation humaine distincte pour push / PR / merge / migrate prod / deploy.

**STOP PUSH / STOP MERGE / STOP DEPLOY / STOP MIGRATION PROD** jusqu’à ordre explicite.
