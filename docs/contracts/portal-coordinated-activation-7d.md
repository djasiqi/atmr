# 7D — Coordinated activation + production release gate

```text
STEP 7D — COORDINATED ACTIVATION RELEASE GATE
BUILD & DEPLOY : BLOCKED (build = deploy)
LEGAL REVIEW : PENDING (gate humain)
LEGAL REVIEW PACKAGE : docs/contracts/portal-legal-review-package-7d-l.md
PRODUCT FREEZE : YES (aucune évolution fonctionnelle jusqu'à retour revue)
```

## Baseline production (inchangée)

```text
PORTAL_TERMS_EFFECTIVE_VERSION = 1.0
PORTAL_DOUBLE_VALIDATION_ENABLED = false
```

## ✅ Implémenté

### Anti-hybride au démarrage

[`enforce_portal_activation_coordination_at_startup`](../../backend/services/legal/portal_terms_catalog.py) appelé dans [`create_app`](../../backend/app.py) juste après le chargement de la config.

| Combinaison | Résultat |
| --- | --- |
| 1.0 + OFF | PASS |
| 1.0 + ON | REFUSE (boot) |
| 2.0 + OFF | REFUSE (boot) |
| 2.0 + ON | PASS |

Tests : [`test_portal_activation_coordination_7d.py`](../../backend/tests/services/test_portal_activation_coordination_7d.py)

### Environnement isolé 2.0+ON

Via env / config de test uniquement (pas la prod). `current_portal_terms()` → 2.0.

### PUBLIC `/conditions` — classification + corrections

Page : [`TermsOfService.jsx`](../../frontend/src/pages/Legal/TermsOfService.jsx) — **non fusionnée** avec les canoniques PORTAL.

| Formulation | Classe | Action 7D |
| --- | --- | --- |
| Utilisation = acceptation | CONTRADICTS PORTAL 2.0 | Carve-out PORTAL : acceptation explicite |
| Poursuite d'utilisation = acceptation | CONTRADICTS PORTAL 2.0 | Carve-out PORTAL : réacceptation explicite |
| Mission confirmée après acceptation entreprise | CONTRADICTS PORTAL 2.0 | Carve-out double validation (2e clic client) |
| Utilisateur final / contrat partenaire | CONTRADICTS si lu seul | Précisé : formation au second clic PORTAL |
| Tarification indicative | NON-CONTRADICTORY + PORTAL | Distinction estimation / plafond / prix X |
| Paiement Saferpay | CONTRADICTS si généralisé | Saferpay ≠ parcours PORTAL transport |
| Annulation générique | CONTRADICTS si généralisé | Politique transporteur figée PORTAL |
| Abonnements professionnels | PROFESSIONAL-ONLY | Inchangé |

### Rollback (preuves intouchables)

Si rollback env vers `1.0` + `false` après activation réelle :

- **Autorisé** : rétablir les deux variables ensemble ; les nouveaux flux repassent en legacy / 1.0 courant.
- **Interdit** : DELETE acceptations 2.0, DELETE offres / confirmations, réécriture d'événements contractuels.
- Les bookings `double_validation_v2` déjà confirmés restent des preuves append-only.

### LEGAL REVIEW

```text
LEGAL REVIEW = PENDING
```

Package juriste (documents + checklist + livrable à remplir) :  
[`portal-legal-review-package-7d-l.md`](portal-legal-review-package-7d-l.md)

Les textes 2.0 déterminent formation du contrat, prix et effets annulation/impayé. Validation juridique humaine **requise** avant `READY FOR BUILD & DEPLOY = YES`.

## Hors scope 7D (bloque READY)

- Revue juridique formelle des canoniques 2.0
- Build & Deploy production
- Smoke navigateur production post-deploy

## Résultats tests 7D (automatisés)

- Anti-hybride 4 combinaisons + boot : PASS (`test_portal_activation_coordination_7d.py`)
- Terms 2.0 prepared / réacceptation / DV / hardening : 39 PASS
- Frontend ClientDashboard + portalDoubleValidationUi : PASS
- Migration head : `3041a7d1f098` (unique head)
- LEGAL REVIEW : **PENDING** (humain)
- 7D-M exécution locale : **PASS** (2026-09-24) — baseline restaurée
- Build & Deploy : **BLOCKED**

## NEXT

**Arrêt produit.** Attendre le retour humain du livrable dans [portal-legal-review-package-7d-l.md](portal-legal-review-package-7d-l.md).

Watchpoint revue : identification transporteur au 2e clic (aujourd’hui = nom seul).

Test manuel local : [portal-manual-activation-test-7d-m.md](portal-manual-activation-test-7d-m.md)  
— exécution 2026-09-24 : **PASS** (API + smoke UI 2.0) ; baseline restaurée `1.0`+`false`.  
UX wording « Mise à jour » : hors périmètre (noté).

Après `LEGAL REVIEW = PASS` (+ corrections textes/UI + hashes si besoin) → commit unique → CI sur ce SHA → E2E `2.0`+`true` → couple env prod → Build & Deploy.
