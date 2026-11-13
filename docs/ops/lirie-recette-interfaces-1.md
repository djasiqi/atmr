# LIRIE — LOT-RECETTE-INTERFACES-1

Validation fonctionnelle des interfaces Client, Entreprise, Chauffeur et Institution.

**Date :** 2026-11-13  
**Worktree :** `D:\lirie-backups\_candidate_lirie_recette_interfaces_20261113`  
**Branche :** `candidate/lirie-recette-interfaces-20261113`  
**Base :** `12e320e1` (PR #106 BOOKING-DISPATCH)  
**Push / PR / merge / deploy :** non effectués  

---

## Environnement isolé

| Composant | Détail |
|-----------|--------|
| API | container `lirie-ri1-api`, hôte `5113` → `5000`, réseau `lirie-ci-gate_default` |
| DB | `postgresql://test:test@postgres_test:5432/atmr_test` (port hôte **5433**) |
| Isolation | log `ISOLATION_DB database=atmr_test` puis `E2E_API_READY` — **pas** DB `atmr` / port 5432 |
| Redis | `redis_test` / `redis_auth_test` (hôte 6381 / 6383) |
| Frontend CRA | `http://127.0.0.1:3013`, proxy `Backend URL: http://127.0.0.1:5113` |
| Comptes | `*.ri1@recette.lirie.test` / seed temp (non versionné) |
| Playwright | Chromium local `.local-playwright` |

WIP principal `C:\Users\jasiq\atmr` : non modifié. PAY-SAFERPAY / invité : non activés.

---

## Matrice de recette

| Profil | Action | Interface | API observée | Résultat | Preuve | Anomalie | Correction | Revalidation |
|--------|--------|-----------|--------------|----------|--------|----------|------------|--------------|
| Client | Connexion | `/login` → dashboard client | `POST /api/v1/auth/login` | **PASS** | Navigateur Playwright | — | — | 5/5 suite |
| Client | Liste réservations | `/reservations/:id` | `GET .../clients/.../bookings` | **PASS** | Navigateur | — | — | oui |
| Client | Ouverture formulaire course | Dashboard client | lectures clients/bookings | **PASS** (écran) | Navigateur | Remplissage adresse/autocomplete non exhaustif (sélection géocode) | — | PARTIAL cycle complet |
| Client | AR / modif / annulation bout-en-bout UI | — | — | **PARTIAL** | Navigateur (smoke) | Pas de confirmation complète d’une création+annulation dans ce lot | À reprendre en lot UI booking dédié | — |
| Entreprise | Connexion + réservations | `/dashboard/company/.../reservations` | companies / bookings | **PASS** | Navigateur | — | — | oui |
| Entreprise | Mode auto non proposé comme actif | Settings / corps page | — | **PASS** (smoke) | Navigateur | Onglet Opérations pas toujours présent | — | — |
| Entreprise | Attribution chauffeur bout-en-bout UI | — | — | **PARTIAL** | — | Nécessite dossier PENDING issu d’une création UI complète | — | — |
| Institution | Connexion + écran demande | dashboard / requests | institution APIs | **PASS** | Navigateur | — | — | oui |
| Institution | Envoi réseau + suivi complet | — | — | **PARTIAL** | Navigateur (écran) | Pas d’envoi bout-en-bout patient/legs dans ce lot | — | — |
| Chauffeur | Connexion / missions appareil | App mobile | — | **NOT VERIFIED** | — | Pas d’`adb`, pas de simulateur, `mobile/unified-app/node_modules` absent | — | — |
| Chauffeur | Contrats quick-accept/reject | Backend (héritage #106) | — | **PASS** (contrat) | HTTP/use-case lot précédent | Ne vaut pas validation téléphone | — | — |
| Multi-profil | Client ↔ Entreprise écrans | login switch + listes | auth + bookings | **PASS** | Navigateur | 1er run : session sticky sur `/login` | Clear cookies/storage avant login | **PASS** après fix |
| MFA | Merge-tree `#104` `626de1c3` × tip | — | — | **PASS** structurel | merge-tree | Pas de marqueurs `<<<<<<<` ; #104 non intégrée | — | — |

### Suite Playwright locale

```
frontend/e2e/recette-interfaces/*.spec.js
npm run e2e:recette-interfaces
→ 5 passed (Chromium)
```

Correction ciblée : `helpers.js` — purge session + attente stricte `input#email` (évite mode strict Playwright sur `.or()`).

---

## Verdicts

| Gate | Verdict |
|------|---------|
| CLIENT WEB | **PARTIAL** (connexion + listes + écrans OK ; cycle création/confirm/annulation UI incomplet) |
| ENTREPRISE WEB | **PARTIAL** (connexion + réservations OK ; attribution chauffeur UI non bouclée) |
| INSTITUTION WEB | **PARTIAL** (connexion + écran demande OK ; envoi/suivi complet non bouclé) |
| CHAUFFEUR MOBILE | **NOT VERIFIED** (pas d’appareil / émulateur) |
| MULTI-PROFILE E2E | **PARTIAL** (sync écrans Client↔Entreprise PASS navigateur ; chaîne complète 4 profils non certifiée) |
| MFA COMPATIBILITY | **PASS** (merge-tree / pas de conflit ; runtime MFA×UI non rejoué) |
| SECURITY | **PASS** (isolation DB test, pas de prod, flags OFF) |
| READY FOR REMOTE CI | **YES** (artefacts Playwright + docs prêts ; publication hors périmètre de ce lot) |

---

## Risques / suite

1. Recette booking complète (géocoder + preview + confirm + cancel) à enchaîner sur le même socle isolé.
2. Attribution Entreprise → Chauffeur nécessite un dossier PENDING réel créé via UI.
3. Chauffeur : installer deps mobile + émulateur/appareil avant certification.
4. MFA #104 reste hors stack #101→#106.

---

## Commits locaux (attendus)

| Message | Contenu |
|---------|---------|
| `test(recette): parcours Playwright multi-profils isolés` | specs + config + script npm |
| `docs(ops): rapport LOT-RECETTE-INTERFACES-1` | ce document |

**Aucun push.**
