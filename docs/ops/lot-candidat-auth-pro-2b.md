# LOT-CANDIDAT-AUTH-PRO-2B — Consolidation interfaces inscription & admin professionnelles

**Date :** 10 novembre 2026  
**Statut :** **PASS** — commit local frontend AUTH-PRO-2B  
**Socle :** AUTH-PRO-2A `93d2b5e7f3b9cddf111e4f91523fe24624fc0b10`

---

## A. INITIAL STATE

| Contrôle | Résultat |
| --- | --- |
| Branche | `main` |
| HEAD attendu / constaté | `93d2b5e7f3b9cddf111e4f91523fe24624fc0b10` |
| Ahead `origin/main` avant | 5 |
| Backend 2A | versionné (demandes, décisions, portal grants) |
| Tests 2A isolés | 79 PASS (1e/1f exclus) |
| Flag ops | `ORGANIZATION_APPROVAL_ENABLED` **OFF** (inchangé) |
| Flag grants | `PORTAL_GRANT_ENFORCEMENT` **OFF** par défaut |
| WIP local | préservé hors périmètre (chauffeur, incidents, external-auth Login complet, finance, etc.) |

---

## B. EXISTING FRONTEND IMPLEMENTATION

Interfaces déjà développées, consolidées sans reconstruction :

| Surface | Emplacement |
| --- | --- |
| Choix de profil | `OrganizationSignupChoice` |
| Formulaire entreprise / institution | `OrganizationSignupForm` (`kind`) |
| Vérification e-mail | `OrganizationEmailVerify` |
| Suivi demandeur | `OrganizationApplicationFollow` + `ApplicationStatusCard` |
| File admin | `AdminOrganizationApplications` |
| Détail + décisions | `AdminOrganizationApplicationDetail` |
| Clients API | `organizationApplicationService`, `adminOrganizationApplicationService` |
| Entrées navigation | `ProfessionnelPage`, lien Login « Entreprise ou institution », nav admin « Inscriptions » |
| Routes | publiques `/inscription/organisation*`, admin `partners/organization-applications*` |

Hors périmètre volontaire : invitation chauffeur (AUTH-PRO-3), UI external Google/Apple (AUTH-PRO-1 déjà backend ; Login WIP non consolidé ici).

---

## C. FILES INCLUDED

### Nouveaux

- `frontend/src/pages/Public/organizationSignup/` (8 fichiers : UI, CSS, tests, worst-case)
- `frontend/src/pages/admin/OrganizationApplications/` (4 fichiers : liste, détail, CSS, tests)
- `frontend/src/services/organizationApplicationService.js`
- `frontend/src/services/adminOrganizationApplicationService.js`
- `docs/ops/lot-candidat-auth-pro-2b.md`

### Modifiés (chirurgicaux)

- `frontend/src/App.js` — lazy imports + routes org publiques + routes admin applications **uniquement**
- `frontend/src/pages/Auth/Login.jsx` — **+5 lignes** : lien vers `/inscription/organisation` (sans ExternalAuth)
- `frontend/src/pages/Public/ProfessionnelPage.jsx` — liens dépôt entreprise / institution
- `frontend/src/pages/admin/navigation/adminNavRegistry.js` — entrée Inscriptions
- `frontend/src/pages/admin/routing/adminRoutePaths.js` — helpers de chemins
- `frontend/src/pages/admin/routing/__tests__/adminRouting.test.js` — assertion chemin détail

### Exclus volontairement (restent WIP)

- Login ExternalAuth / bootstrap Google-Apple
- Driver invitations / OperationalIncidents dans `App.js`
- Rapports `lot-candidat-auth-pro-1.md` / `…-2.md` (hors message de commit 2B)
- Tout backend, migrations, finance, flags ops

---

## D. SIGNUP USER JOURNEY

1. Entrée : Login, page Professionnel, ou `/inscription/organisation`.
2. Choix : Particulier → signup client historique ; Entreprise / Institution → formulaire dédié.
3. Champs alignés backend (`organization_applications._validated_fields`) : kind, raison sociale ou établissement+catégorie, IDE optionnel (company), adresse, responsable, e-mail/téléphone, `privacy_consent`, honeypot `website`.
4. Envoi : verrou anti double-clic (`lock` ref) ; pas de création de compte professionnel ; message « vérification demandée » / pas d’activation.
5. E-mail : confirmation → dossier reçu ; suivi via token (`/suivi`) sans ouverture de portail.
6. États couverts UI : saisie, validation locale, pending, reçu, examen, complément, approuvé, refusé, erreur réseau (message FR conservant le formulaire), pas de Google/Apple comme autorisation, pas d’inscription chauffeur publique.

---

## E. ADMIN APPROVAL INTERFACE

- Liste filtrable (statut / kind / recherche) via `GET /admin/organization-applications`.
- Détail : faits demandeur, timeline événements, recherche org existante (match **sans** rattachement auto).
- Actions avec confirmation (`AdminActionDialog`) : prise en charge, complément, reprise, refus, approbation.
- Approbation : `POST …/approve` ; résultat uniquement après réponse backend ; message UI rappelle le verrou d’activation.
- Permissions : contrôles UI **non** présentés comme suffisants ; API admin reste derrière capacités serveur (2A).

---

## F. ROLE AND NAVIGATION SECURITY

| Cas | Comportement FE |
| --- | --- |
| Demande en attente / refusée | Suivi sans CTA d’activation portail |
| Approbation UI | N’affirme pas l’ouverture si le backend verrouille (`approval_enabled` OFF) |
| Client | Pas d’entrée admin ; routes admin sous layout admin existant |
| Google/Apple | Absents du parcours préinscription ; Login 2B sans boutons externes |
| Session | Clients API admin `skipAuthRedirect` + messages d’erreur ; pas de contournement flag |
| Isolation | Pas d’accès cross-org côté FE ; IDs publics seulement |

Validation complète socket/contextes portal 1e/1f : **non couverte** ici (bloqueurs AUTH-PRO-2A).

---

## G. RESPONSIVE AND ACCESSIBILITY

- Réutilisation CSS modules LIRIE (`OrganizationSignup.module.css`, shell admin).
- Labels associés, `aria-invalid`, focus sur premier champ invalide.
- Dialogues de confirmation pour actions sensibles.
- Tests worst-case (noms/e-mails longs) sur carte statut et détail admin.
- Pas de nouvelle bibliothèque UI.

Validation manuelle exhaustives smartphone/tablette/zoom : **non exécutée** dans ce lot (limites manuelles).

---

## H. FEATURE FLAG VERIFICATION

| Flag | Ops | FE |
| --- | --- | --- |
| `ORGANIZATION_APPROVAL_ENABLED` | **OFF** (aucune modification config commitée) | Pas de flag FE distinct ; pas de contournement |
| `PORTAL_GRANT_ENFORCEMENT` | **OFF** | Non branché en contournement |

Avec flag OFF : dépôt / revue UI possibles ; **aucune** création d’organisation/habilitation (garanti backend 2A). Parcours historiques client / portails existants non remplacés.

---

## I. TEST RESULTS

| Suite | Résultat |
| --- | --- |
| `organizationSignup` | **4 PASS** |
| `AdminOrganizationApplications` | **4 PASS** |
| `adminRouting` | **4 PASS** |
| **Total FE ciblé** | **12 PASS** |

Couvertures unitaires : choix sans chauffeur/Google/Apple, champs entreprise, statut long, message réseau, file admin (incomplet/reçu/vide/erreur), détail (approbation inactive + e-mail long), chemins admin.

Non rejoués ici : 79 tests backend 2A (déjà validés) ; e2e navigateur manuel ; suites portal 1e/1f.

---

## J. ISOLATED CHECKOUT VALIDATION

| Contrôle | Résultat |
| --- | --- |
| Socle | archive `git archive HEAD frontend` + overlay fichiers 2B |
| Emplacement | `D:\lirie-backups\_auth_pro2b_isolated` |
| App.js isolé | routes org présentes ; **aucune** DriverInvitation / OperationalIncidents / ExternalAuth |
| Tests isolés | **12 PASS** (mêmes suites, `node_modules` partagé en junction) |
| Dépendances WIP non versionnées | absentes du périmètre 2B |
| Finance / migrations | non touchées |

---

## K. GIT COMMIT

| Champ | Valeur |
| --- | --- |
| Message | `feat(auth): consolidate professional signup and admin interfaces` |
| Parent | `93d2b5e7f3b9cddf111e4f91523fe24624fc0b10` |
| Indexation | fichiers listés en §C uniquement (`git add` explicite, pas `git add .`) |
| Push | **NONE** |
| Deploy | **NONE** |

---

## L. REMAINING BLOCKERS — AUTH.PY / REDIS / 1E / 1F

Identiques à AUTH-PRO-2A :

1. `test_portal_grant_phase1e` / `phase1f` : contextes session dans `auth.py` + stabilité Redis / socket worker.
2. Lot transversal minimal proposé : consolider les contextes portal/guest dans `auth.py` + worker socket Redis **sans** activer les flags ops.
3. Login ExternalAuth WIP : à consolider après stabilisation AUTH-PRO-1 UI (hors 2B).

Ces bloqueurs **n’empêchent pas** la consolidation UI inscription/admin sur les contrats API déjà versionnés.

---

## M. AUTH-PRO-3 READINESS

Prêt à démarrer **séparément** :

- invitations chauffeur (`driver_invitations*`, gate distinct)
- **pas** d’inscription chauffeur publique
- ne pas démarrer automatiquement depuis 2B

---

## N. FINAL VERDICT — PASS

Interfaces professionnelles d’inscription et d’administration consolidées, alignées sur les contrats 2A, tests FE ciblés verts en working tree et checkout isolé, flags ops inchangés, WIP hors périmètre préservé.

### Limites assumées

- Pas de validation manuelle multi-device complète.
- Pas de rejeu e2e navigateur contre API live (flag OFF ops).
- Portal 1e/1f / Redis : non débloqués.

---

DATABASE BUSINESS CHANGES: NONE

MIGRATIONS CHANGED: NONE

FINANCIAL FEATURES CHANGED: NONE

PRODUCTION FLAGS CHANGED: NONE

PUSH: NONE

DEPLOY: NONE
