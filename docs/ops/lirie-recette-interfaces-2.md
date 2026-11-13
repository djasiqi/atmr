# LIRIE — LOT-RECETTE-INTERFACES-2

Clôture des parcours Entreprise / Institution et préparation Mobile.

**Date :** 2026-11-13  
**Worktree :** `D:\lirie-backups\_candidate_lirie_recette_interfaces_20261113`  
**Branche :** `candidate/lirie-recette-interfaces-20261113`  
**Base LOT-1 :** `5a7676b6` / `98f47885`  
**Push / PR / merge / deploy :** non effectués  

---

## A. Défauts initialement identifiés (héritage LOT-1)

| # | Défaut | Impact |
|---|--------|--------|
| 1 | Modale Assigner non certifiée sous Playwright (lazy `ReservationModals` + Suspense conditionnel) | ENTREPRISE PARTIAL |
| 2 | Clic Assigner potentiellement avalé par le clic ligne | Attribution UI |
| 3 | Transmission Institution non prouvée (formulaire ≠ envoi métier) | INSTITUTION PARTIAL |
| 4 | `external_reference` patient → 409 à la 2ᵉ création | Bloque recette répétée |
| 5 | Mobile : pas d’adb / émulateur / `node_modules` | CHAUFFEUR NOT VERIFIED |
| 6 | MFA #104 hors chaîne UI | Compat structurelle seulement |

---

## B. Correctifs Entreprise

| Correctif | Détail | Fichiers |
|-----------|--------|----------|
| Import synchrone `ReservationModals` | Plus de lazy+montage conditionnel au 1ᵉʳ clic | `CompanyDashboard.jsx` |
| `openAssignModal` résilient | Résolution reservation par objet / map / liste | idem |
| Drivers filter | `is_active !== false` (évite liste vide) | idem |
| `useCompanyData` drivers | Fallback `drivers` / `driver` / array | `useCompanyData.js` |
| `stopPropagation` (LOT-1) | Conservé sur `ReservationActions` | `ReservationActions.jsx` |
| Modal `ariaLabel` | Dialog accessible « Assigner un chauffeur » | `ReservationModals.jsx` |

**Preuve Playwright :** clic réel bouton → dialog → « Assigner » chauffeur → `POST .../assign` < 400.  
**Preuve DB :** booking `1660` = `ASSIGNED`, `driver_id=252`.

---

## C. Correctifs Institution

| Correctif | Détail |
|-----------|--------|
| Spec transmission | Remplissage React (`pressSequentially`), mode LIRIE, bouton « Envoyer aux transporteurs LIRIE » |
| `external_reference` unique | Évite 409 « Demande avec external_reference existe déjà » |
| Seed patient | `InstitutionPatient` Alice Recette + téléphone user institution |
| Création + send | `POST /institutions/requests` puis `POST .../requests/:id/send` |

**Preuve DB :** `TransportRequest` 19 et 20 status **SENT** ; `RequestOffer` PENDING vers company `2035`.

---

## D. Résultats Playwright navigateur

```
npx playwright test --config=playwright.recette-interfaces.config.js e2e/recette-interfaces/
→ 10 passed
```

Specs RI-2 ajoutées / durcies :
- `company-assign.spec.js` — modale Assigner bout-en-bout
- `institution-request.spec.js` — create+send
- `multi-profile-ri2.spec.js` — Client/Entreprise post-assign

---

## E. Tests Mobile réellement exécutés

| Élément | Résultat |
|---------|----------|
| `node_modules` | Junction vers deps main (lockfile npm inchangé) |
| Jest `quick-action` | PASS (1) |
| Jest `missionRevisionGuard` | PASS (9) |
| Jest `missionRuntimeManager` | PASS (3) |
| Jest `driverHubStatusModel` | PASS (5) |
| Jest `assignDriverPicker` | PASS (5) |
| Typecheck / lint complets | Non rejoués in-extenso (hors blocage) |
| ADB / émulateur / appareil | **Absent** |

**CHAUFFEUR DEVICE : NOT VERIFIED** (absence d’environnement, pas un défaut app).

### Procédure build mobile (SHA candidat)

1. Worktree sur le HEAD candidat.
2. `cd mobile/unified-app && npm ci` (lockfile officiel).
3. `npm run typecheck` puis `npm test -- --watchAll=false`.
4. `npm run android` ou `npm run ios` avec SDK/émulateur.
5. Connexion `driver.ri1@recette.lirie.test` contre API isolée.
6. Vérifier mission assignée (`booking 1660` / offre institution), quick-accept, transitions.
7. Ne pas annoncer push iOS ni certification appareil sans run réel.

---

## F. État de la recette multi-profils

| Parcours | Preuve | Chauffeur |
|----------|--------|-----------|
| **A Client** | Create UI (LOT-1) → Entreprise accept → **modale Assigner** → DB ASSIGNED → Client Mes courses | Actions chauffeur **non** faites en UI mobile (NOT VERIFIED device) |
| **B Institution** | Formulaire → `POST requests` + send → TR SENT + Offer PENDING company 2035 | Attribution chauffeur sur offre institution : à enchaîner en session entreprise (offre visible) ; device chauffeur non testé |

Distinction explicite : transitions chauffeur post-assign = **API/DB + contrats Jest**, pas E2E téléphone.

---

## G. Sécurité et MFA

| Point | Verdict |
|-------|---------|
| Isolation DB `atmr_test` / cookies `TestingConfig` HTTP only | **PASS** (pas de prod) |
| Dispatch manuel (bandeau Manuel) | **PASS** |
| Saferpay / flags | non activés |
| MFA #104 | Compat **structurelle** uniquement — **pas** une recette de connexion MFA navigateur |
| Token MFA ≠ JWT métier | rappelé ; non exercé runtime ici |

---

## H. Commits locaux et HEAD

| SHA | Message |
|-----|---------|
| `0c9467ee` | fix(entreprise): ouverture fiable de la modale Assigner |
| bffba045 | test(recette): certifier Assigner + transmission Institution |

HEAD final : `bffba045`.

---

## I. Limites restantes

1. Pas de validation appareil chauffeur (adb/émulateur).
2. Offre institution → acceptation + assign UI non rejouée dans le même scénario Playwright unique (offre PENDING prouvée en DB).
3. MFA runtime 4 profils hors périmètre.
4. Seeds pricing / patient / cookies : **test only**.

---

## Verdicts

| Gate | Verdict |
|------|---------|
| CLIENT WEB | **PASS** |
| ENTREPRISE WEB | **PASS** (modale Assigner certifiée) |
| INSTITUTION WEB | **PASS** (create+send SENT) |
| CHAUFFEUR MOBILE CONTRACT | **PASS** (Jest contrats mission / assign picker) |
| CHAUFFEUR DEVICE | **NOT VERIFIED** |
| MULTI-PROFILE WEB + API | **PASS** (A web+DB ; B SENT+offers) |
| MULTI-PROFILE FULL UI | **PARTIAL** (chauffeur sans device) |
| SECURITY | **PASS** |
| READY FOR REMOTE CI | **YES** |
