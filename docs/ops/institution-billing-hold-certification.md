# Institution Billing — CLOSED

```text
INSTITUTION BILLING — CLOSED

G3 ✅ Browser dispute UX
G2 ✅ Dispute state machine
G1 ✅ Financial eligibility
G4 ✅ Final emission / PDF / QR

CERTIFIED INVARIANTS

1. plan == preview == generated invoice == PDF == QR
2. no duplicated billing line
3. excluded lines never leak into issued invoices
4. carrier cannot self-resolve disputes
5. pending intentions/corrections never affect billing
6. only validated financial state affects totals
```

Certification figée le 5 septembre 2026. Base `origin/main` @ `6ea10015`.
Architecture Billing **frozen**. Ne plus modifier Institution Billing avant
intégration / déploiement, **sauf régression démontrée**.

`partner`, nouvelles corrections, trajet / A-R, nouveaux payeurs ou nouvelles
variantes = **chantiers séparés**, pas une glissade dans ce CLOSED.

Chaîne certifiée :

```text
state → eligibility → plan → preview → generated invoice → PDF → QR amount
```

Les quatre gates couvrent quatre risques distincts :

| Gate | Risque couvert |
| --- | --- |
| G3 | Utilisabilité réelle du dialogue de contestation (navigateur) |
| G2 | Machine d’état et rôles déterministes |
| G1 | Cohérence financière ; lecture de l’état **validé** seulement |
| G4 | Émission réelle jusqu’au PDF et au QR (image `Dockerfile.production`) |

Sentinelles G4 :

```text
resolved_carrier
320 → 360
Marie présente exactement 1 fois
PDF 360
QR 360
```

```text
resolved_institution
reste 320
Marie absente
PDF 320
QR 320
```

Prochaine étape opérationnelle : figer le HEAD certifié, contrôler la CI,
préparer le merge / déploiement **sans changements parasites**.

| HOLD | Statut chantier | Prochaine action |
| --- | --- | --- |
| G3 | Playwright contestation | **CLOSED / PASS** — ne plus toucher |
| G2 | Machine d’état 3 branches | **CLOSED / PASS** — ne modifier que si contradiction |
| G1 | Matrice financière institution | **CLOSED / PASS** |
| G4 | Émission PDF / QR | **CLOSED / PASS** 5 sept. 2026 |
| 403 | Transporteur ne clôture pas | Règle métier conservée |

## Diagnostic corrigé

Le problème n’est plus l’architecture Billing. C’est la certification du
workflow de contestation. « Mission effectuée » ne doit pas réintégrer la
ligne tout de suite : **320 reste 320** jusqu’à validation tierce, puis
**360**. C’est le comportement voulu, plus strict, à figer dans les tests.

## Gates — condition de fermeture

| Gate | État | Condition de fermeture |
| --- | --- | --- |
| G1 — Exactitude financière | **Fermé (PASS)** | État validé seulement ; aucune intention intermédiaire ; plan total == preview total. |
| G2 — Cycle contestation | **Fermé (PASS)** | Les 3 branches ont un état final explicite et reproductible. |
| G3 — Navigateur réel | **Fermé (PASS)** | Playwright : dialogue visible **et** dans le viewport. `toBeInTheDocument` seul = FAIL. |
| G4 — Émission | **Fermé (PASS)** | Deux E2E : réintégration 360 et exclusion 320. Plan = preview = PDF = QR. |

Ordre — backend en dernier :

1. G3 Playwright (bug terrain) — **PASS**
2. G2 machine d’état — **PASS**
3. G1 matrice financière — **PASS**
4. G4 E2E émission — **PASS**
5. **CLOSED**

Un unitaire vert ne compensera jamais un G3 rouge.

## Oracle financier — ligne 40 CHF, base 360

| État de la ligne 40 CHF | Total attendu |
| --- | --- |
| Aucune contestation | 360 |
| Contestation ouverte | 320 |
| Transporteur dit « effectuée » | 320 |
| Preuve envoyée | 320 |
| Preuve rejetée | 320 |
| Preuve validée | 360 |
| Institution a raison | 320 |
| Correction commencée | 320 |
| Correction enregistrée, non validée | 320 |
| Correction 40 → 35, payeur `clinic` validé | **355** |
| Correction payeur `patient` validée | **320** (ligne hors institution) |

Oracle métier unique pour G1 / G2 / G4. Source : décision de certification
5 septembre 2026, HEAD `6ea10015`.

Principe G1 : **aucune action intermédiaire ne modifie le montant
institutionnel**. La base lit l’état validé (`booking.amount` /
`billed_to_type`), jamais `proposed_*`.

## Trois branches — état final à figer

### A. Institution a raison

`OPEN` → `institution_right` → exclue → résolue. Ne revient pas dans la
facture institution. HEAD : `not_billable`, course conservée. Pas de
« à charge patient » sur cette branche.

### B. Mission effectuée

`OPEN` → `mission_done` → preuve → `PENDING_REVIEW` → validation
institution/admin → `RESOLVED` → réintégration. 320 tout le long, 360
seulement après le tiers. Le 403 transporteur reste.

### C. À corriger

Dans le workflow : montant et payeur. Hors workflow (trajet, A/R, autre) :
renvoyer vers l’écran adéquat, sans réintégration automatique. Pas
d’extension de formulaire avant CLOSED.

✅ **Implémenté (G2)** : contrat `backend/application/invoices/booking_dispute/machine.py`
+ marches `backend/tests/application/test_booking_dispute_g2_state_machine.py`.

Pour chaque branche : état initial → action opérateur → état intermédiaire
→ rôle autorisé → action tierce (si applicable) → état final → éligibilité
facture. Le transporteur ne clôture jamais via `decide` (`403` conservé).
`institution_right` n’exige pas de tiers (concession) et ne bascule pas
le payeur vers le patient. `mission_done` / `needs_correction` restent hors
facture jusqu’à validation institution/admin. Correction limitée à
`clinic` / `patient` + montant ; la proposition n’est appliquée qu’après
le tiers. Transitions ambiguës (`submit` depuis `disputed`, preuve sans
position, changement de stance après soumission) → `409`.

Verdict 5 septembre 2026 : **12 passed** (6 G2 + 6 workflow existant),
Docker `backend_tests`. G3 non modifié. G2 ne se rouvre que sur
contradiction réelle.

## G3 — scénario Playwright minimum

Ouvrir Institution → période → facture → « Pourquoi cette course n’est pas
facturée » → « Traiter la contestation » → dialogue visible dans le
viewport.

Matrices : scroll haut / milieu / bas, desktop, mobile, fermer puis
rouvrir, deux contestations successives.

✅ **Implémenté** : `frontend/e2e/institution-billing-g3.spec.js` +
`frontend/playwright.institution-billing.config.js` +
`frontend/e2e/helpers/institutionBillingG3.js`.

- Parcours réel : Factures → Nouvelle facture → Institution → août 2026 →
  clinique → bloc exclu → Traiter.
- Assertion G3 : `toBeInViewport` + boîte dans le viewport + overlay
  `position:fixed` **hors** du scroll facture (`bill-period-form-scroll`).
- Matrices : scroll haut / milieu / bas, desktop 1280×800, mobile 375×667,
  fermer puis rouvrir, deux contestations successives (Marie DUPONT →
  Arturo KLEIN).
- `toBeInTheDocument` / `getBoundingClientRect` mocké (Jest) = insuffisant.
- Commande : `npm run e2e:institution-billing-g3` (depuis `frontend/`).
- Verdict 5 septembre 2026 : **PASS** desktop (2) + mobile (2). Aucun bug
  viewport : le sous-modal est déjà `fixed` hors scroll. G3 ne débloque
  pas G2 / G1 / G4.

## G1 — matrice financière (CLOSED / PASS)

Assertions à trois niveaux : ligne (`is_billable_to_institution`,
`effective_payer`, `effective_amount`) ; plan (`eligible_lines`,
`excluded_lines`, `institution_total`) ; cohérence
`plan total == preview total`.

Hors scope G1 : PDF, QR, UX contestation, payeur `partner`, trajet / A/R,
modification de la machine G2.

✅ **Implémenté** : oracle
`backend/application/invoices/booking_dispute/g1_financials.py` +
unitaires `backend/tests/application/test_booking_dispute_g1_financial_matrix.py`
+ monde DB `backend/tests/application/helpers/g1_clinic360_world.py` +
plan/preview `backend/tests/application/test_booking_dispute_g1_plan_preview.py`.

Verdict 5 septembre 2026 :

```text
G1 CLOSED / PASS

- disputed = 320
- resolved_institution = 320
- awaiting_carrier_response = 320
- evidence_submitted = 320
- resolved_carrier = 360
- correction pending = 320
- correction clinic validated = 355 (40 → 35)
- correction patient validated = 320 côté institution
- partner rejected = 400 / aucun effet financier
- 403 / 409 = aucun effet financier
- plan total == preview total
```

**10 passed** G1 (6 unitaires + 4 plan/preview DB) + **12 passed** G2 /
workflow, Docker `backend_tests`. G3 / G2 non modifiés.

## G4 — émission PDF / QR (CLOSED / PASS)

Chaîne certifiée :

```text
state → eligibility → plan → preview → generated invoice → PDF → QR amount
```

Invariants : aucune divergence plan / preview / PDF ; aucune duplication
de ligne ; aucune ligne exclue qui réapparaît à l’émission.

Hors scope G4 : pas de revalidation G1 / G2, pas de scénario annexe
(correction 35, partner, A/R).

✅ **Implémenté** :
`backend/tests/e2e/test_e2e_institution_billing_g4_emission.py` +
`backend/tests/e2e/helpers/institution_billing_g4.py`.

Monde G1 (8 × 40 + Marie 40). `PDFService` et `QRBillService` réels
(pas de mock). Horloge d’émission : 1er septembre 2026 00:00
`Europe/Zurich`.

Verdict 5 septembre 2026 :

```text
G4 CLOSED / PASS

- resolved_carrier → plan 360 = preview 360 = facture 360 = PDF 360 = QR 360
- Marie DUPONT une seule fois sur le PDF
- resolved_institution → plan 320 = preview 320 = facture 320 = PDF 320 = QR 320
- Marie absente de la facture institution et du PDF
```

**2 passed**, Docker `backend_tests` (image de test = Dockerfile
production / target testing). G3 / G2 / G1 non modifiés.

## Surface certifiée — STOP fonctionnel

Ne plus développer Institution Billing dans ce CLOSED. Toute demande
(`partner`, correction étendue, trajet / A-R, nouveau payeur, nouveau
bucket, refactoring Billing) ouvre un **chantier séparé** après
intégration / déploiement.

Sauf régression démontrée, le HEAD certifié ne doit embarquer que les
fichiers des quatre gates + cette charte. Hors snapshot : artefacts
Playwright (`frontend/test-results/`), worktrees, et tout diff sans lien
avec G1–G4.
