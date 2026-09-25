# 7D-M — Test manuel local (activation coordonnée 2.0 + DV)

```text
STEP 7D-M — MANUAL LOCAL ACTIVATION TEST
LEGAL REVIEW : PENDING (inchangé)
PRODUCT FREEZE : YES (pas de nouveau développement fonctionnel)
TEXTS 1.0 : IMMUABLES (ne pas corriger même si imprécis)
BUILD & DEPLOY : BLOCKED

ENVIRONNEMENTS :
  1.0 + false  = parcours legacy / état normal actuel
  2.0 + true   = environnement de test manuel 7D-M uniquement
```

Ce document verrouille le protocole du test manuel.  
Il **ne** remplace **pas** la revue juridique 7D-L.

Pendant le freeze : **ne rien corriger** (ni produit, ni canoniques 1.0).  
Pour le test : activer localement `2.0 + true`, redémarrer, parcourir le flux,  
puis **revenir immédiatement** à `1.0 + false`.

---

## Observation actuelle (écran legacy)

L’écran qui affiche :

```text
Conditions générales d’utilisation — version 1.0
Conditions de réservation et de transport — version 1.0
```

est **cohérent** avec le défaut local :

```text
PORTAL_TERMS_EFFECTIVE_VERSION=1.0
PORTAL_DOUBLE_VALIDATION_ENABLED=false
```

Ce n’est **pas** le test du parcours 2.0. C’est le parcours **legacy**.

### Ne pas modifier ce que l’on voit en 1.0

Les formulations 1.0 (ex. « crée une demande… puis confirme sa réservation »,  
« Une version ultérieure fait l'objet d'une nouvelle acceptation ») restent  
dans l’historique. La précision du modèle (1er clic ≠ contrat ; 2e clic = contrat ;  
réacceptation conditionnée) appartient à **2.0**, pas à une réécriture de 1.0.

---

## Activation locale pour le test 2.0

Les deux variables doivent être **ensemble** (anti-hybride au boot) :

```text
PORTAL_TERMS_EFFECTIVE_VERSION=2.0
PORTAL_DOUBLE_VALIDATION_ENABLED=true
```

Dans Docker local : les positionner dans `backend/.env` (chargé via `env_file` du  
`docker-compose.yml`), puis **redémarrer complètement** le backend (et le frontend  
si besoin pour vider un cache d’état).

Après redémarrage, l’écran d’acceptation attendu :

```text
Conditions générales d’utilisation — version 2.0
Conditions de réservation et de transport — version 2.0
```

Le contenu affiché doit être celui des fichiers canoniques :

```text
backend/legal/canonical/fr/terms_of_service_v2.0.txt
backend/legal/canonical/fr/transport_terms_v2.0.txt
```

pas les textes 1.0.

### Scénario attendu — client déjà sur 1.0

```text
avant :
  CGU 1.0 accepted
  Transport 1.0 accepted

après activation locale 2.0 + true :
  titre « Mise à jour des conditions »  (correct ici)
  CGU — version 2.0
  Transport — version 2.0
  ☐ J’ai lu et j’accepte…
  Accepter les conditions

après acceptation :
  1.0 reste en DB (inchangé)
  2.0 s’ajoute (nouvelles lignes)
  aucune acceptation 1.0 n’est modifiée
```

### Scénario attendu — parcours commande

Après acceptation 2.0 : 1er clic = demande + plafond ; offre transporteur ;  
2e clic = `Confirmer le transport à CHF X` → `CLIENT_TRANSPORT_CONFIRMED`.

---

## Point UX noté (hors freeze produit lourd)

### Symptôme

Sur l’écran **1.0**, le modal affiche toujours :

```text
Mise à jour des conditions
Les conditions … ont été mises à jour.
```

même lorsque les documents affichés sont encore 1.0.

### Cause technique (constat code, pas de correctif ici)

- Backend : `resolve_portal_terms_status` renvoie `status = reacceptance_required`  
  dès qu’**au moins un** document a `acceptance_required=true` — y compris  
  `contractual_basis = missing` (aucune acceptation antérieure).
- Frontend : `ClientDashboard.jsx` titre / corps figés sur « Mise à jour… »  
  dès que `termsStatus.status === 'reacceptance_required'`.

Donc deux situations distinctes partagent le même wording :

| Cas | Situation | Affichage actuel | Attendu UX |
| --- | --- | --- | --- |
| A | Compte n’a jamais accepté 1.0 | Modal + « mises à jour » + v1.0 | Modal OK, mais titre du type « Conditions applicables au compte » |
| B | Compte a déjà accepté 1.0 et effective = 1.0 | Modal + « mises à jour » + v1.0 | **Ne devrait pas** réapparaître → bug si observé |

### Vérification locale obligatoire (avant de conclure A ou B)

Appeler `GET /clients/me/portal-terms-status` et noter :

```text
status
documents[].current_version
documents[].accepted_version
documents[].acceptance_required
documents[].contractual_basis   # missing | current_acceptance | prior_acceptance
documents[].requires_reacceptance
```

Interprétation :

```text
contractual_basis = missing
→ première acceptation requise
  (titre « Mise à jour » incorrect côté UI — ignorer pendant le freeze)

accepted_version antérieure / stale  (ex. prior version ≠ current_version
  avec acceptance_required = true)
→ vraie mise à jour / réacceptation
  (titre « Mise à jour » correct)

accepted_version = current_version  +  acceptance_required = true
→ anomalie (cas B) — à investiguer
```

### Correctif produit (reporté — après release gate)

Ce n’est **pas** un problème de version effective : `reacceptance_required` est un  
statut métier trop large (première acceptation manquante **et** vraie réacceptation).  
Le frontend applique un wording unique trop générique.

```text
STATUS : NOTED — après le release gate (pas pendant le freeze / pas avant LEGAL REVIEW)
SCOPE  : wording modal conditionné + tests frontend
NE PAS : toucher aux .txt 1.0

Mapping cible :

  contractual_basis = missing
  → titre « Conditions applicables à votre compte »

  accepted_version antérieure / stale (ex. prior_acceptance ou version ≠ current)
  → titre « Mise à jour des conditions »
```

---

## Checklist 7D-M (à cocher lors du test)

- [x] Env local = `2.0` + `true` (les deux)
- [x] Backend redémarré ; anti-hybride OK (boot sans erreur)
- [x] Modal affiche versions **2.0** (pas 1.0)
- [x] Corps = contenus `*_v2.0.txt`
- [x] Compte déjà 1.0 → titre « Mise à jour » acceptable ; 1.0 inchangé en DB après acceptation 2.0
- [x] Double validation : plafond → offre X → 2e clic forme le contrat
- [x] Cas A/B wording 1.0 documenté via payload `portal-terms-status`
- [x] Remise env local à `1.0` + `false` après le test (baseline inchangée)

---

## Résultats exécution 7D-M

```text
DATE : 2026-09-24
MODE : API local (Docker) + smoke navigateur localhost:3000
OVERALL : PASS (parcours 2.0 + DV)
BASELINE RESTAURÉE : 1.0 + false
```

| Point | Résultat | Preuve |
| --- | --- | --- |
| Env `2.0` + `true` | PASS | `cfg_v 2.0` / `cfg_dv True` après recreate |
| Anti-hybride boot | PASS | `assert_activation_coordination` OK |
| Catalogue courant 2.0 | PASS | hashes `97942048…` / `5e502ef6…` |
| Modal UI versions 2.0 | PASS | browser : « … — version 2.0 » ×2 |
| Corps = `*_v2.0.txt` | PASS | `terms_version: 2.0` + contenu CGU 2.0 dans `.portalTermsBody` |
| Réacceptation 1.0→2.0 | PASS | `accepted=1.0`, `current=2.0`, `reacceptance_required` |
| 1.0 inchangé après acceptation 2.0 | PASS | ids 1.0 `[1339,1340]` inchangés ; 2.0 `[1341,1342]` ajoutés |
| 1er clic ≠ contrat | PASS | `flow=double_validation_v2`, `company_id=None` |
| Offre X ≠ contrat | PASS | `pending=True`, `company_id=None`, offre CHF 82 |
| 2e clic = contrat | PASS | `company_id` fixé, `amount=82`, confirmation `82.00` |
| Remise `1.0` + `false` | PASS | `.env` + recreate (voir probe post-test) |

Note incidental : recreate Docker a exigé `pip install nh3` (dépendance image locale) — hors scope produit 7D-M.

UX wording « Mise à jour » : hors périmètre ; observé correct pour le cas stale 1.0→2.0.

---

## Retest 7D-M + validation tarifaire 7B.2

```text
CONTEXTE : 7B.2 implémenté (plafond = MAX(quotes transporteurs))
7B.2 CLOSED : NON — tant que les 6 cas ci-dessous ne sont pas PASS
LEGAL REVIEW : PENDING (inchangé)
BUILD & DEPLOY : BLOCKED
```

### Scénario de référence lisible (verrouillé — freeze moteur)

Préparer 3 entreprises éligibles sur la même course :

| Entreprise | Modèle | Config | Quote attendue |
| ---------- | ------ | ------ | -------------: |
| A | `flat` | forfait | CHF 45.00 |
| B | `distance` | 12 km × CHF 3.20/km | CHF 38.40 |
| C | `zone_count` | base 40 + 1×12 | CHF 52.00 |

```text
PLAFOND ATTENDU = CHF 52.00
ENGINE FREEZE   = YES (pas de changement moteur avant verdict manuel)
```

Écran client attendu (plafond agrégé **seul** — pas les quotes A/B/C) :

```text
Prix maximum de la demande
CHF 52.–
```

Transporteur : **ne voit pas** ce plafond.

Après 1er clic (DB) :

```text
estimated_amount_snapshot = […]
maximum_accepted_amount_snapshot = 52.00
company_id = NULL
contract formed = NO
changed_fields.pricing_ceiling présent (quotes + excluded)
```

Offre transporteur X à CHF 45 → UI :

```text
Transporteur X
Prix proposé : CHF 45
Prix maximum de la demande : CHF 52
→ Confirmer le transport à CHF 45
```

Après 2e clic :

```text
contractual_amount = 45.00
maximum_accepted_amount_snapshot = 52.00  (inchangé)
```

Jamais `contractual_amount = 52` sauf si une offre réelle = 52.

### Checklist tarifaire 7B.2 (à cocher pendant le retest)

- [ ] **UI** — client : plafond seul (pas quotes A/B/C) ; transporteur : pas de plafond visible
- [ ] **Cas 1** — Multi-modèles (forfait / km / zone) → plafond = MAX réel (ex. 52)
- [ ] **Cas 2** — Transporteur non éligible (hors zone / service) → absent du MAX
- [ ] **Cas 3** — Sans grille exploitable ou quote ≤ 0 → exclu ; jamais `0 CHF` silencieux
- [ ] **Cas 4** — Aucune quote → création bloquée (`portal_pricing_ceiling_unavailable`)
- [ ] **Cas 5** — Payload modifié `maximum_accepted_amount=9999` → backend ignore ; snapshot = plafond serveur
- [ ] **Cas 6** — Après création, modifier une grille transporteur → `maximum_accepted_amount_snapshot` inchangé

### Procédure retest

1. Activer localement `2.0` + `true`, redémarrer backend (+ frontend si besoin).
2. Exécuter le parcours commande + les 6 cas ci-dessus.
3. Cocher chaque cas PASS/FAIL dans cette section.
4. Remettre immédiatement `1.0` + `false`.
5. Seulement si les 6 cas PASS → marquer 7B.2 CLOSED dans `portal-double-validation-7b.md`.

### Résultats retest 7B.2 (à remplir)

```text
DATE : 2026-09-24
MODE : API local (Docker) + UI /reservations (booking 46759)
OVERALL : FAIL / STOP
BASELINE RESTAURÉE : non (env encore en mode test 2.0 + DV)
```

| Point | Résultat | Note |
| --- | --- | --- |
| PRICING CEILING CALCULATION | **PASS** probable | UI création → 52 ; quotes A=45 B≈42.4 C=52 + Emmenez 40 |
| PRICING CEILING SNAPSHOT | **PASS** | `maximum_accepted_amount_snapshot=52` ; `pricing_ceiling` dans `changed_fields` |
| RESERVATION CARD PRICE | **FAIL** (au test) | affiché `Montant 50` = `booking.amount` — **correctif UX appliqué** (estimation + plafond) |
| PORTAL PAYMENT UI | **FAIL** (au test) | Paiement requis / Payer maintenant / règlement en ligne — **correctif UX appliqué** (timeline DV + pas Saferpay) |
| DISPATCH TO ELIGIBLE CARRIERS | **FAIL** puis **réparé local** | 0 `DispatchOffer` à T0 (`geo_unit` vide) ; après import geo + seed → 4 offres (1, 64604–64606) |
| 7B.2 | **REOPENED PARTIAL** | ne pas CLOSED |
| 7D-M retest tarifaire | **FAIL / STOP** | reprendre après refresh UI + vérif dashboard entreprise « En attente » |

| Cas | Résultat | Note |
| --- | --- | --- |
| 1 Multi-modèles MAX | PARTIAL | plafond 52 OK en preview/snapshot ; carte réservations legacy au moment du test |
| 2 Non éligible exclu | | non rejoué (STOP) |
| 3 Sans grille / ≤0 | | non rejoué (STOP) |
| 4 Aucune quote | | non rejoué (STOP) |
| 5 Ignore payload 9999 | | non rejoué (STOP) |
| 6 Snapshot figé | | non rejoué (STOP) |

---

## Lien avec 7D / 7D-L

| Étape | Statut |
| --- | --- |
| 7D-L package | READY |
| LEGAL REVIEW | PENDING |
| 7D-M protocole | READY |
| 7D-M exécution 2.0 | **PASS** (2026-09-24) |
| 7D-M retest 7B.2 (6 cas tarifaires) | **FAIL / STOP** (2026-09-24 — carte réservations + seed geo) |
| 7B.2 pricing engine parity (auto) | **PASS** — [portal-pricing-engine-parity-7b2.md](portal-pricing-engine-parity-7b2.md) |
| Build & Deploy | BLOCKED |
