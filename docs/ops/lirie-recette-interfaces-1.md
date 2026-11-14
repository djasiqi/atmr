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
| Redis | `redis_test` / `redis_auth_test` (hôte 6381 / 6383) |
| Frontend CRA | `http://127.0.0.1:3013`, proxy API `:5113` |
| Comptes | `*.ri1@recette.lirie.test` / `RecettePass2026!` (seed temp non versionné) |
| Pricing seed | profil flat `RI1 Flat Recette` (CHF 55) — requis pour preview/création |
| Playwright | Chromium `.local-playwright` |

WIP principal `C:\Users\jasiq\atmr` : **non modifié**. PAY-SAFERPAY / invité : **non activés**.

---

## Anomalies démontrées et corrections

| # | Anomalie | Cause | Correction | Fichiers | Revalidation |
|---|----------|-------|------------|----------|--------------|
| 1 | Cookies JWT `Secure` sous HTTP localhost → session « expirée » | `TestingConfig` héritait `COOKIE_SECURE=True` (défaut env) | `COOKIE_SECURE=False` (+ SameSite Lax) dans `TestingConfig` ; conteneur `COOKIE_SECURE=false` | `backend/config.py` | Login Playwright OK |
| 2 | Création booking bloquée « Aucun profil de pricing actif » | Entreprise RI1 sans `PricingProfile` | Seed flat + version courante | seed temp `/e2e/seed_pricing.py` | Création UI → bookings `1660+` en DB |
| 3 | Faux positif E2E (preview comptée comme create) | Assertion `/bookings/` matchait `/bookings/preview` | Assertion stricte `POST /clients/:id/bookings` + status < 400 | `client-booking-full.spec.js` | PASS réel + COUNT DB > 0 |
| 4 | Clic « Assigner » avalé par le clic ligne | Pas de `stopPropagation` sur `ReservationActions` | `stopPropagation` + `aria-label` | `ReservationActions.jsx` | Bouton Assigner prouvé ; modale encore PARTIAL sous Playwright |
| 5 | Création → `AWAITING_CLIENT_PAYMENT` (Saferpay hors lot) | Workflow preview `payment_required` | Promotion manuelle PENDING pour recette entreprise (sans activer Saferpay) | DB isolée uniquement | Entreprise voit dossiers `?date=2026-11-13` |

---

## Matrice de recette

| Profil | Action | Interface | API | Résultat | Preuve | Anomalie | Correction | Revalidation |
|--------|--------|-----------|-----|----------|--------|----------|------------|--------------|
| Client | Connexion | `/login` | `POST /auth/login` (cookies) | **PASS** | Navigateur | Cookies Secure | TestingConfig | oui |
| Client | Création AR (domicile + Photon Cornavin) | Dashboard | `POST .../bookings/preview` + `POST .../clients/:id/bookings` | **PASS** | Navigateur + DB | Pricing manquant | Seed pricing | booking créé (ex. 1660 AR) |
| Client | Consultation Mes courses | `/reservations/:id` | GET bookings | **PASS** | Navigateur | — | — | oui |
| Client | Annulation UI | Mes courses | DELETE/cancel si bouton | **PARTIAL** | Navigateur | Bouton pas toujours exposé selon statut paiement | — | — |
| Client | Modif autorisée | — | — | **PARTIAL** | — | Non rejouée isolément après create | — | — |
| Entreprise | Connexion + liste datée | `/reservations?date=` | `GET .../companies/me/reservations` | **PASS** | Navigateur | Liste « all » trompeuse sans `flat`/date | Tests forcent `?date=` | oui |
| Entreprise | Acceptation | Panneau détail | `POST .../accept` | **PASS** | Navigateur | — | — | dossiers ACCEPTED |
| Entreprise | Mode manuel (pas d’auto forcé) | Dashboard / settings | — | **PASS** | Navigateur | Bandeau « Manuel — Assignations manuelles » | — | oui |
| Entreprise | Attribution chauffeur | Dashboard « Sans chauffeur » | `POST .../assign` | **PARTIAL** | Navigateur | Bouton Assigner visible ; modale lazy parfois non montée sous Playwright | `stopPropagation` | bouton PASS / assign POST PARTIAL |
| Institution | Connexion + formulaire | `/requests/new` | institution + geocode | **PASS** | Navigateur | Sélecteurs combobox ambigus | `#pickup_location` / `#dropoff_location` | oui |
| Institution | Transmission complète patient | — | — | **PARTIAL** | Navigateur | Patient / validations métier peuvent bloquer l’envoi | — | écran OK |
| Chauffeur | Appareil / émulateur | mobile | — | **NOT VERIFIED** | — | Pas d’`adb`, pas d’émulateur, pas de `node_modules` mobile | — | — |
| Chauffeur | Jest mobile | — | — | **NOT VERIFIED** | — | `mobile/unified-app/node_modules` absent | — | ≠ validation téléphone |
| Multi-profil | Client create → Entreprise reçoit | React + API candidat | bookings | **PARTIAL** | Navigateur | Chaîne create+accept OK ; assign+réalisation chauffeur non bouclée | — | — |
| MFA | Merge-tree `#104` `626de1c3` | — | — | **PASS** structurel | merge-tree | Runtime MFA×4 profils non rejoué sur UI | — | #104 non intégrée |

### Suite Playwright (locale)

```
npx playwright test --config=playwright.recette-interfaces.config.js e2e/recette-interfaces/
→ 8 passed (Chromium)
```

Specs : `client-company-institution`, `client-booking-full`, `company-assign`, `institution-request`, `booking-lifecycle`.

---

## Verdicts

| Gate | Verdict |
|------|---------|
| CLIENT WEB | **PASS** (création AR réelle UI→DB ; consultation OK ; annulation/modif PARTIAL selon statut) |
| ENTREPRISE WEB | **PARTIAL** (réception + acceptation + mode manuel OK ; attribution modale non certifiée bout-en-bout) |
| INSTITUTION WEB | **PARTIAL** (connexion + formulaire adresses OK ; transmission métier complète PARTIAL) |
| CHAUFFEUR MOBILE | **NOT VERIFIED** |
| MULTI-PROFILE E2E | **PARTIAL** (Client→Entreprise navigateur ; sans chauffeur appareil) |
| MFA COMPATIBILITY | **PASS** (structurel merge-tree ; runtime UI non rejoué) |
| SECURITY | **PASS** (DB test, cookies HTTP testing only, pas de prod, flags OFF, Saferpay OFF) |
| READY FOR REMOTE CI | **YES** (artefacts + docs ; **pas de push** dans ce lot) |

---

## Risques / suite

1. Boucler attribution : hydratation `assignModalDrivers` / Suspense `ReservationModals` sous Playwright.
2. Éviter `AWAITING_CLIENT_PAYMENT` en recette sans Saferpay (facturation entreprise / flag preview).
3. Chauffeur : `npm i` mobile + émulateur avant certification.
4. MFA #104 hors chaîne #101→#106 jusqu’à intégration dédiée.

---

## Commits locaux (ce lot)

Voir `git log` sur `candidate/lirie-recette-interfaces-20261113` — **aucun push**.
