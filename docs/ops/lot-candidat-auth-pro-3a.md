# LOT-CANDIDAT-AUTH-PRO-3A — Invitations chauffeurs et rattachements

**Date :** 10 novembre 2026  
**Statut :** **PASS** — commit local backend AUTH-PRO-3A  
**Socle :** AUTH-PRO-2C `d7ccb5c48e2ee67d269063e8334a6df5452b00af`

---

## A. INITIAL STATE

| Contrôle | Résultat |
| --- | --- |
| Branche | `main` |
| HEAD attendu / constaté | `d7ccb5c4` |
| Ahead `origin/main` avant | 7 |
| Modèle | `DriverInvitation` déjà versionné (MODELS-1) |
| Flags ops | `ORGANIZATION_APPROVAL_ENABLED` OFF · `PORTAL_GRANT_ENFORCEMENT` OFF · `DRIVER_INVITATION_ACTIVATION` OFF |
| Frontend invitations | présent en WIP — **exclu** (AUTH-PRO-3B) |

---

## B. DRIVER INVITATION ARCHITECTURE

| Élément | Implémentation |
| --- | --- |
| Gate | `driver_invitation_gate.DRIVER_INVITATION_ACTIVATION` (défaut OFF) |
| Service | `services/driver_invitations.py` |
| Routes entreprise | `GET/POST /companies/me/driver-invitations`, resend/cancel/revoke |
| Routes publiques lien | `POST /driver/invitations/open|accept` |
| Enregistrement | import side-effect dans `routes_api` (namespaces companies/driver) |
| Création directe | `POST /companies/me/drivers/create` → **403 `invitation_requise`** |

Pas d’inscription chauffeur publique (`/driver/signup` → 404).

---

## C. COMPANY MEMBERSHIP RULES

1. Seul le responsable (`company.user_id == actor`) d’une entreprise **approuvée** invite.
2. Acceptation sous verrou : crée `Driver` + `PortalGrant` driver **uniquement** si activation ON **et** grant ouvre le portail ; sinon rollback (`acces_non_ouvert`).
3. Compte Client existant : session du **même** compte exigée (pas de fusion e-mail seule).
4. Compte déjà chauffeur d’une **autre** entreprise : `deja_rattache` (409).
5. Révocation : retire le `Driver` créé par l’invitation, révoque le grant, **conserve** l’identité User (souvent CLIENT).
6. Chauffeurs historiques : profil/missions inchangés si non liés à une invitation.

---

## D. INVITATION SECURITY

| Contrôle | Comportement |
| --- | --- |
| Secret | `secrets` → hash SHA-256 stocké ; jamais renvoyé en JSON liste/création |
| TTL | 7 jours ; marquage `expired` |
| Usage | token remplacé au renvoi ; ancien lien `lien_invalide` |
| Concurrence | `FOR UPDATE` + contrainte unique pending email/company |
| Tentatives | plafond 8 → `tentatives_limitees` |
| Renvois | plafond 5 → `renvoi_limite` |
| Logs / mail | secret absent des notices journalisées ; domaines hors périmètre `withheld` |
| Forbidden keys | password/role/portal_grant refusés à la création |

---

## E. SESSION AND REVOCATION SECURITY

- Après revoke : `GET /driver/me/profile` → **403** ; grant `revoked` conservé (pas d’effacement SQL).
- Propagation portal grants / Redis : héritée AUTH-PRO-2C (1e/1f rejoués).
- Refresh / contextes : grants driver via `portal_scope` (pas de rôle JWT inventé).

---

## F. EXISTING DRIVER COMPATIBILITY

- Chauffeur déjà rattaché à l’entreprise invitante : acceptation OK sans second Driver ; revoke invitation **ne** détruit **pas** le Driver historique.
- Création manuelle historique fermée (`invitation_requise`) ; routes gestion chauffeurs existantes préservées.
- Flags OFF : invitations créables/listables ; acceptation bloquée (`activation_bloquee`) sans créer Driver/grant.

---

## G. TEST RESULTS

| Suite | Résultat |
| --- | --- |
| `test_driver_invitations` | **10 PASS** |
| portal phase1 / 1c / 1d / 1e / 1f | **37 PASS** |
| `test_external_auth` | **25 PASS** |
| create_driver critical (×2) | **2 PASS** |
| **Total** | **74 PASS** |

Preuve : `D:\lirie-backups\_auth_pro3a_full.txt`.

Correction collatérale validée : `CompanyMe` propageait mal les 403 portal → 404 ; `get_current_company` traite aussi `denied` en 403 (`test_date_seule_ne_donne_pas_acces`).

---

## H. ISOLATED CHECKOUT VALIDATION

| Contrôle | Résultat |
| --- | --- |
| `routes_api` | HEAD + import `driver_invitations` uniquement |
| `companies.py` | HEAD + `CreateDriver` invitation + `CompanyMe` 403 |
| Hors lot | saferpay, settlement, attestations, frontend, guest auth |
| Migrations | aucune nouvelle |
| Base métier `atmr` | non modifiée (pytest DB test) |

---

## I. FILES INCLUDED

### Nouveaux

- `backend/routes/driver_invitations.py`
- `backend/services/driver_invitations.py`
- `backend/services/driver_invitation_gate.py`
- `backend/tests/routes/test_driver_invitations.py`
- `docs/ops/lot-candidat-auth-pro-3a.md`

### Modifiés

- `backend/routes_api.py` (+4 lignes import)
- `backend/routes/companies.py` (CreateDriver + CompanyMe)
- `backend/application/users/get_current_company.py` (`denied` → 403)
- `backend/tests/routes/test_companies_critical_coverage.py` (assertions invitation)

### Exclus (WIP / 3B)

- `frontend/.../driverInvitation*`
- `DriverInvitationsPanel*`
- settlement / attestations / saferpay dans `routes_api` WIP

---

## J. GIT COMMIT

| Champ | Valeur |
| --- | --- |
| Message | `feat(auth): consolidate driver invitations and company membership` |
| Parent | `d7ccb5c4…` |
| Push | **NONE** |
| Deploy | **NONE** |

---

## K. REMAINING DEPENDENCIES

- UI entreprise + page acceptation chauffeur → **AUTH-PRO-3B**
- Activation ops `DRIVER_INVITATION_ACTIVATION` : décision produit séparée
- Multi-entreprise chauffeur : modèle actuel **un** `Driver.company_id` (pas de multi sans décision)

---

## L. AUTH-PRO-3B READINESS

Prêt : contrats API et flags stables.  
À faire en 3B : brancher FE existant, messages, parcours mobile si besoin — **sans** activer les flags production.

---

## M. FINAL VERDICT — PASS

Backend invitations / membership consolidé, tests métier + régression AUTH-PRO verts, flags ops inchangés, frontend non inclus.

---

DATABASE BUSINESS CHANGES: NONE

MIGRATIONS CHANGED: NONE

FINANCIAL FEATURES CHANGED: NONE

PRODUCTION FLAGS CHANGED: NONE

PUSH: NONE

DEPLOY: NONE
