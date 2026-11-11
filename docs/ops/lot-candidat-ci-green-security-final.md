# LIRIE — CI GREEN GATE + SECURITY REMEDIATION (final)

**PR** : [#101](https://github.com/djasiqi/atmr/pull/101) (Draft)  
**Branche** : `candidate/lirie-auth-pro-ci-20261111`  
**SHA départ** : `cc1f53bb`  
**SHA final** : `804085fcbf96781795dfcddeef1f221eac67cb3a`  
**Base** : `main` intacte — aucun merge, aucun déploiement.

---

## A. PYTEST

| Métrique | Initial (run `42940978749`) | Local post-correctifs | CI finale (même SHA) |
|---|---|---|---|
| Échecs obligatoires | **15** | **0** (23 tests ciblés A–E) | *voir matrice F* |
| PASS (suite complète) | 7 896 | non rejouée in-extenso localement | *voir run Backend Tests* |
| SKIPPED | 92 | — | — |
| XFAILED | 1 | — | — |

Les 15 échecs du run `42940978749` ont été corrigés et rejoués localement via Docker (`atmr-backend_tests` + `postgres_test`/`redis_test`) : **23/23 PASS** (y compris la suite phase1c complète).

SKIPPED/XFAILED historiques : non utilisés pour masquer un contrôle critique ; aucun skip ajouté dans ce lot.

---

## B. CORRECTIONS A–E

### A — Invitations / contexte chauffeur

- **Cause** : après acceptation, le compte reste `CLIENT` ; `get_driver_from_token` n’acceptait que le rôle `DRIVER` ou un header `X-Active-Context-Id`, d’où 404 profil.
- **Correctif** : persistance Redis `driver:{id}` à l’acceptation ; résolution via header, habilitation DRIVER active, ou contexte Redis.
- **Fichiers** : `backend/routes/driver.py`, `backend/services/driver_invitations.py`
- **Tests** : `test_acceptation_nouveau_chauffeur_ouvre_le_portail`, `test_compte_existant_exige_la_bonne_session`
- **Commit** : `7236b47c`

### B — Entreprises / partenariats (403)

- **Cause** : création d’une 2ᵉ société sur le même `user_id` → `ambiguous_company` sans tenir compte du claim JWT `company_id` ; fixture `authenticated_client` sans `company_id` réel.
- **Correctif** : désambiguïsation legacy multi-sociétés par claim ; JWT de test avec `sample_company.id` ; reset autouse `PORTAL_GRANT_*`.
- **Fichiers** : `backend/security/portal_scope.py`, `backend/tests/conftest.py` (+ auth company mobile / conversations)
- **Tests** : search companies, create partnership, duplicate 409
- **Commit** : `fe3ef1b0`

### C — API institution / patient (500)

- **Cause** : `get_jwt_identity()` / `get_current_user()` en RuntimeError sous auth API Key.
- **Correctif** : retour `None` hors contexte JWT ; `get_institution_role_from_jwt` tolérant.
- **Fichiers** : `backend/security/authorization.py`
- **Tests** : `test_create_patient_api_key`
- **Commit** : `fe3ef1b0`

### D — Portal grants / Socket.IO / Redis

- **Cause** : `SKIP_SOCKETIO=true` → `_NoOpSocketIO` ; cwd `/app` absent hors container ; `ConversationService.message_timestamp_for_send` manquant ; lookup mobile company sous drapeau.
- **Correctif** : `SKIP_SOCKETIO=false` sur job pytest ; `BACKEND_ROOT` ; horodatage message ; `_find_company_user_by_email` + sessions `MobileDeviceSession` ; refus conversation multi-org sans contexte.
- **Fichiers** : `.github/workflows/backend-tests.yml`, `conversation_service.py`, `company_mobile_auth.py`, `conversations.py`, phase1e/f
- **Tests** : phase1c/1d/1e/1f + mobile driver account
- **Commits** : `fe3ef1b0`, `251d333f`

### E — Compatibilité PDF / active_driver / mobile

- **Cause** : pypdf 7 exige page attachée au writer ; `SimpleNamespace` sans `company_id` ; JWT company mobile sans session durable → 401.
- **Correctif** : `add_page` puis `merge_page` ; fixtures + `getattr(company_id)` ; `create_or_reuse_session` dans `_issue_tokens`.
- **Fichiers** : `partner_agreement_preview.py`, `test_active_driver_context_required.py`, `company_mobile_auth.py`, `portal_scope.py`
- **Commit** : `67c59a65` (+ parties dans `fe3ef1b0`)

---

## C. DEPENDABOT

Inventaire GitHub (default branch, API authentifiée) : **74 ouvertes** — 2 critical, 37 high, 28 medium, 7 low.

### Corrigées dans le candidat (`804085fc`)

| Zone | Packages (versions résolues lock) |
|---|---|
| frontend | axios 1.21.1, shell-quote 1.12.0, follow-redirects 1.16.3, compression 1.8.2, source-map-js 1.2.2, brace-expansion 5.0.12, launch-editor 2.14.2, body-parser 2.4.0, fast-uri 4.2.3, moment 2.31.0, postcss-selector-parser 7.1.6 |
| mobile | axios 1.21.1, shell-quote 1.12.0, undici 7.30.1, follow-redirects 1.16.3, brace-expansion 5.0.12, @grpc/grpc-js 1.14.7, compression 1.8.2, source-map-js 1.2.2, fast-uri 4.2.3 |
| pip | PyJWT 2.15.0 (`services/ws-service/requirements.txt`) |

Les alertes GitHub restent ouvertes sur **main** jusqu’à fusion ; le candidat contient les versions corrigées.

### Restantes / justifiées

| Alerte | Motif |
|---|---|
| zod (#459/#462) fix 4.6.6 | Majeur 3→4 incompatible sans chantier mobile dédié ; reste en 3.25.x |
| braces (#441/#442) ≤3.0.3 | Aucune version corrigée publiée par Dependabot |
| deepmerge (#469/#470) low | Pas de fix disponible |
| Alertes main non encore re-scannées | Fermeture Dependabot post-merge uniquement |

**Commit deps** : `804085fc`

---

## D. CODE SCANNING

| ID | Règle | Cause | Correction | Preuve |
|---|---|---|---|---|
| 390 | `py/clear-text-logging-sensitive-data` | log `billing_party_id` | bool `party_ref_present` | `invoice_recipient.py` |
| 391 | idem | idem | idem | `invoice_template_builder.py` |
| 392 | idem | idem | idem | `pdf.py` |

**Commit** : `070899b9`  
Pas de `nosemgrep` / exclusion CodeQL. Les alertes default branch restent ouvertes jusqu’à merge + rescan.

---

## E. COMMITS

| SHA | Parent | Objectif |
|---|---|---|
| `7236b47c` | `cc1f53bb` | Invitations / contexte chauffeur |
| `fe3ef1b0` | `7236b47c` | Multi-sociétés, API key, mobile sessions, fixtures |
| `251d333f` | `fe3ef1b0` | Socket.IO réel + multi-processus + timestamp chat |
| `67c59a65` | `251d333f` | pypdf 7 + fixtures active_driver |
| `070899b9` | `67c59a65` | CodeQL logging |
| `804085fc` | `070899b9` | Dependabot npm + PyJWT |

WIP principal (`c:\Users\jasiq\atmr`) non modifié.

---

## F. CI MATRIX

SHA ciblé : `804085fcbf96781795dfcddeef1f221eac67cb3a`

| Workflow | Run ID | SHA | Conclusion |
|---|---|---|---|
| Backend Tests | 42948152144 | 804085fc | *en cours / à compléter* |
| Security Gate | 42948152316 | 804085fc | *en cours / à compléter* |
| Security Scan | 42948152369 | 804085fc | *en cours / à compléter* |
| Phase 1 Gates | 42948152394 | 804085fc | *en cours / à compléter* |
| Repository integrity | 42948152396 | 804085fc | *en cours / à compléter* |
| Architecture Review | 42948152164 | 804085fc | *en cours / à compléter* |
| Secret Pattern Scan | 42948152168 | 804085fc | *en cours / à compléter* |
| Mobile unified-app | 42948152157 | 804085fc | *en cours / à compléter* |
| Secret Pattern Scan (push) | 42948144565 | 804085fc | success |

*Cette section est mise à jour dès clôture des runs obligatoires.*

---

## G. SECURITY VERDICT

- **Critical exploitables dans le candidat** : shell-quote remédié (1.12.0 frontend+mobile).
- **High remédiées dans le candidat** : axios, undici, follow-redirects, compression, brace-expansion, body-parser, launch-editor, @grpc/grpc-js, source-map-js, fast-uri (majorité).
- **Residuels** : zod 3.x (attente chantier major), braces sans patch, deepmerge low sans patch ; alertes GitHub main encore ouvertes jusqu’à merge.
- **Code Scanning** : 3/3 corrigées dans le candidat ; fermeture UI après merge/rescan.
- **Exceptions formelles** : zod major et braces sans fix — documentées, non fermées artificiellement.

---

## H. PR STATUS

- PR **#101** Draft — https://github.com/djasiqi/atmr/pull/101
- SHA final : `804085fc`
- Base `main` intacte
- Aucun déploiement / migration prod / publication mobile / feature flag ops

---

## I. FINAL VERDICT

| Dimension | Statut |
|---|---|
| **CI STATUS** | **PARTIAL** — correctifs poussés ; attente conclusion verte de tous les contrôles obligatoires sur `804085fc` |
| **SECURITY STATUS** | **PARTIAL** — P0/P1 remédiables traités dans le candidat ; résiduels justifiés ; alertes GitHub main en attente de merge |
| **DELIVERY STATUS** | **READY FOR HUMAN REVIEW** dès CI obligatoire GREEN sur ce SHA (sinon NOT READY) |

---

✅ **Implémenté** : clusters A–E, remédiations Dependabot/CodeQL ciblées, 6 commits atomiques poussés sur la branche candidate, inventaire sécurité et rapport final.
