# 7D-L — Legal Review Package (PORTAL terms 2.0)

```text
STEP 7D-L — LEGAL REVIEW PACKAGE
PRODUCT CODE : 7B.5 IMPLEMENTATION GO (flags OFF ; pas de prod)
TEXTS 2.0 STATUS : PREPARED historique intact
TEXTS 2.1 STATUS : PREPARED (conditional_order_v1) — non effectif
PRODUCTION FLAGS (défaut) :
  PORTAL_TERMS_EFFECTIVE_VERSION = 1.0
  PORTAL_DOUBLE_VALIDATION_ENABLED = false
  PORTAL_CONDITIONAL_ORDER_ENABLED = false
LEGAL DESK REVIEW 7B.5 : PASS WITH SAFEGUARDS
EXTERNAL LAWYER : NOT A CODE BLOCKER
BUILD & DEPLOY : BLOCKED jusqu’aux gates d’implémentation
```

Cette étape prépare l’examen humain/juridique des documents 2.0.  
Elle **ne** modifie **pas** le comportement produit 7B et **ne** active **pas** la production.

### Freeze produit (verrouillé)

```text
7D-L package                 READY
LEGAL REVIEW                 PENDING
Build / Deploy               BLOCKED

Après revue humaine :
LEGAL REVIEW = PASS
→ éventuelles corrections textes / UI (ex. identification X)
→ hashes 2.0 définitifs
→ commit unique du lot
→ CI complète sur un seul SHA
→ E2E réel 2.0 + double_validation ON
→ PORTAL_TERMS_EFFECTIVE_VERSION=2.0
→ PORTAL_DOUBLE_VALIDATION_ENABLED=true
→ Build & Deploy
```

Aucune nouvelle fonctionnalité tant que `LEGAL REVIEW` n’est pas revenu.  
Toute modification fonctionnelle supplémentaire ferait bouger la base examinée par le juriste.

### Points d’attention prioritaires pour le juriste

0. **Bascule produit 7B.5 (un clic)** — Cible : [portal-single-click-conditional-order-7b5.md](portal-single-click-conditional-order-7b5.md).  
   1er clic = commande conditionnelle (« Commander jusqu’à CHF X ») ; formation du contrat = **acceptation du premier transporteur éligible**.  
   Plus de 2e clic client. Terms 2.0 et preuves actuelles (CARRIER_OFFERED → CLIENT_TRANSPORT_CONFIRMED) **à réécrire** après PASS — ne pas seulement retirer le modal.  
   Code actuel reste `double_validation_v2` jusqu’à implémentation post-PASS.

1. **Identification** — Avec 7B.5, l’identité du transporteur n’apparaît qu’à la formation (notification « Transport confirmé avec X — CHF Y »).  
   Aujourd’hui (DV) le client voit le **nom** au 2e clic (`company_name_snapshot`).  
   Réponse **explicite** : le modèle « cocontractant déterminé à l’acceptation » est-il acceptable ? Quelles infos minimales dans la confirmation électronique immédiate ?

2. **Formation du contrat / cocontractant non identifié au 1er clic** — Demande adressée au **pool** ; cocontractant déterminé à l’acceptation.  
   Valider SECO / CO (commande conditionnelle + acceptation vendeur).

3. **Annulation canal = plafonds max, pas taux exacts imposés** — Cible produit verrouillée : [portal-channel-cancellation-caps.md](portal-channel-cancellation-caps.md).  
   LIRIE fixe des **maxima** protecteurs ; chaque entreprise publie **sa** politique (7B.3 conservé) sous réserve `company_rate <= channel_cap`.  
   **Refus explicite** si dépassement — **jamais** `min(company_rate, channel_cap)` silencieux.  
   Artefacts séparés : `CompanyPortalCancellationPolicy` + `LirieChannelCancellationPolicy` (tous deux versionnés / hashés ; snapshot dual à l’offre / formation).  
   **Ne pas** imposer « exactement 80 % » à toutes les entreprises sans avis LCart / COMCO.  
   Valider les plafonds communs au regard des accords verticaux / éléments liés au prix.  
   **Pas de code de bornage** tant que `LEGAL REVIEW ≠ PASS`.

Références SECO / CO / COMCO (contexte, non exhaustives) :

- [Commerce électronique (SECO)](https://www.seco.admin.ch/fr/commerce-electronique) — clarté des étapes, correction avant validation, confirmation électronique
- [Avant l’achat et conclusion du contrat (SECO)](https://www.seco.admin.ch/fr/avant-l-achat-et-conclusion-du-contrat) — CG consultables, acceptation à la conclusion
- [CG abusives / art. 8 LCD (SECO)](https://www.seco.admin.ch/fr/conditions-generales-abusives) — proportionnalité annulation
- [COMCO — droit des cartels](https://www.weko.admin.ch/fr/recommandations-et-precisions-relatives-au-droit-des-cartels) — prix et éléments liés au prix
- [CO / formation par volontés concordantes (Fedlex)](https://www.fedlex.admin.ch/) — offre transporteur + confirmation client

---

## Documents à fournir au juriste

| # | Document | Chemin |
| --- | --- | --- |
| 1 | CGU compte client privé 2.0 | [`backend/legal/canonical/fr/terms_of_service_v2.0.txt`](../../backend/legal/canonical/fr/terms_of_service_v2.0.txt) |
| 2 | Conditions réservation & transport 2.0 | [`backend/legal/canonical/fr/transport_terms_v2.0.txt`](../../backend/legal/canonical/fr/transport_terms_v2.0.txt) |
| 3 | Audit réalité contractuelle 7A | [`portal-contract-reality-audit-7a.md`](portal-contract-reality-audit-7a.md) |
| 4 | Double validation 7B (+ 7B.1) — code actuel | [`portal-double-validation-7b.md`](portal-double-validation-7b.md) |
| 4b | Cible 7B.5 — un clic / formation à l’acceptation | [`portal-single-click-conditional-order-7b5.md`](portal-single-click-conditional-order-7b5.md) |
| 5 | Textes 2.0 PREPARED 7C (**REOPENED** vs 7B.5) | [`portal-terms-2-0-7c.md`](portal-terms-2-0-7c.md) |
| 6 | Release gate 7D | [`portal-coordinated-activation-7d.md`](portal-coordinated-activation-7d.md) |
| 7 | Plafonds annulation canal (cible) | [`portal-channel-cancellation-caps.md`](portal-channel-cancellation-caps.md) |
| 8 | Ce package (résumé + checklist) | le présent fichier |

### Hashes PREPARED (référence actuelle avant éventuelle correction)

```text
terms_of_service 2.0 =
97942048422ddd18078f02f533f427b9db048bed5ac6a33fcd8b64277f4139e2

transport_terms 2.0 =
5e502ef659fe35c3a65c2ff288e99092ddc213ca159559bdf195ceafdac9bc12
```

Aucune acceptation réelle 2.0 n’existe encore en production.  
Si la revue exige des corrections : **ne pas modifier silencieusement** — corriger les `.txt`, recalculer les hashes, rejouer les tests canoniques, documenter `old prepared hashes → replaced before effectiveness`.

---

## Résumé fonctionnel à valider

```text
LIRIE n'est pas le transporteur.

Premier clic client :
→ transmet une demande de transport
→ accepte un plafond maximal de prix
→ prend connaissance des plafonds d'annulation du canal
  (maxima protecteurs — pas les taux exacts de chaque concurrent)
→ aucun contrat de transport n'est encore formé
  (sauf variante autoconfirm / 1er accepteur — à valider séparément)

Transporteur X :
→ fixe son propre prix (100 % libre)
→ publie sa politique d'annulation / no-show / attente
→ cette politique doit rester ≤ plafonds canal LIRIE
→ proposition présentable uniquement si prix <= plafond client

Second clic client (défaut) :
→ identité de X affichée
→ prix X affiché
→ conditions X affichées (≤ caps canal)
→ clic « Confirmer le transport à CHF X »

Ce second clic :
→ forme le contrat de transport
→ entre le client et X
→ prix contractuel = prix proposé par X

Après le transport :
→ X facture directement le client
→ X reste créancier

LIRIE :
→ ne fixe pas le prix de transport
→ fixe uniquement des plafonds max d'annulation canal (protections)
→ ne facture pas / n'encaisse pas le client PORTAL
→ n'est pas créancier
```

### Réalité produit actuelle (contexte pour la revue — pas une affirmation juridique)

| Élément | État code / UI (flag ON, hors prod) |
| --- | --- |
| 1er clic | Demande + plafond ; `BOOKING_CREATED` ; `company_id = NULL` |
| Offre | `PortalCarrierOffer` ; pas d’assignation finale |
| Écran 2e clic | Nom entreprise (`company_name_snapshot`), prix, plafond client, texte politique annulation |
| Identification X | Aujourd’hui : **nom** (snapshot). Pas encore d’adresse / IDE / coordonnées systématiques dans la modal |
| Confirmation e-mail 1er clic | Demande enregistrée / transmise — pas « transport confirmé » |
| Après 2e clic | `CLIENT_TRANSPORT_CONFIRMED` ; notification « transport confirmé » |
| Hold | Limité au créancier X |
| `/conditions` publique | Carve-outs PORTAL ajoutés en 7D (page multi-acteurs, non = canoniques PORTAL) |

Le juriste doit indiquer si l’identification actuelle de X est **suffisante** ou si des champs supplémentaires sont **CHANGES REQUIRED** (produit + texte).

---

## Checklist de revue (à traiter point par point)

### 1. Formation du contrat

Confirmer que la rédaction distingue :

```text
premier clic ≠ formation du contrat
CARRIER_OFFERED ≠ formation du contrat
second clic client = formation du contrat
```

Contrôler l’absence d’ambiguïté entre : demande / proposition / confirmation / réservation / contrat.

### 2. Identité du cocontractant

Au second clic, le cocontractant doit être clairement **l’entreprise de transport X**, non LIRIE.

Quelles informations d’identification X doivent être présentées avant confirmation ?

```text
raison sociale / nom
coordonnées
adresse
identification commerciale (ex. IDE)
```

Ne pas présumer qu’un simple `company_name` suffit sans avis.

### 3. Prix

Valider la distinction :

```text
estimation LIRIE ≠ prix contractuel
plafond ≠ prix contractuel
plafond ≠ dette
prix X accepté = prix contractuel
```

Contrôler l’affichage du prix et des suppléments **avant** le second clic.

### 4. Annulation

Avant / après contrat / tardive. Règles = transporteur X, présentées avant second clic, figées. Aucun tarif générique LIRIE.

### 5. No-show / attente / suppléments

Question centrale pour le juriste :

> Un supplément n’est-il opposable que s’il était suffisamment déterminé ou déterminable dans les conditions présentées avant la conclusion ?

Documenter la réponse (oui / avec nuances / non).

### 6. Incorporation des CG au second clic

Avant « Confirmer le transport à CHF X » : lien visible, version identifiable, contenu accessible, conservation raisonnable, acceptation explicite.

### 7. Confirmation électronique

Contenu minimal attendu : transporteur, trajet, date/heure, prix, conditions, référence.  
Qualification : **preuve / confirmation du contrat déjà conclu**, pas création rétroactive.

### 8. Modification

Avant vs après second clic. Ne pas promettre un mécanisme non supporté. Cas nécessitant nouvelle proposition / nouveau consentement / simple exécution.

### 9. Facturation

Facture après transport par X. LIRIE ≠ émetteur ≠ créancier (parcours PORTAL).

### 10. Impayés / payment hold

Dette envers X → peut bloquer de nouvelles prestations **de X**.  
Ne bloque pas automatiquement Y/Z. Pas de suspension globale du compte.

### 11. Contestation

Possible ; hold suspendu si contestation ouverte. Contestation ≠ extinction automatique de la dette.

### 12. Rappels / recouvrement

Pas : LIRIE créancier / transmission automatique office / mainlevée automatique.

### 13. Responsabilités

LIRIE = plateforme ; X = transport. Examiner exclusions / limitations / disproportion consommateur.

### 14. Modification des conditions / réacceptation

Substantiel → nouvelle acceptation avant nouvelle demande. Non rétroactif sur contrats formés.

### 15. Droit applicable / for

Droit suisse ; réserve des fors impératifs (consommateur).

### 16. Données personnelles

Cohérence du renvoi à la Politique de confidentialité. Revue LPD complète = optionnellement séparée.

### 17. Clauses potentiellement abusives

Chercher déséquilibre notable (responsabilité, annulation, suppléments, suspension, impayés, modification unilatérale, preuve, for).

### 18. Page publique `/conditions`

Les carve-outs 7D suffisent-ils ?  
Réponse : `PASS` ou formulations à modifier (sans fusionner la page pro avec les canoniques PORTAL).

---

## Processus après la revue

### Si `LEGAL REVIEW = PASS`

1. Conserver les hashes ci-dessus comme référence de release.
2. Lot technique final unique (commit 7A→7D / 7D-L).
3. CI complète sur **ce** SHA.
4. E2E réel avec `PORTAL_TERMS_EFFECTIVE_VERSION=2.0` + `PORTAL_DOUBLE_VALIDATION_ENABLED=true`.
5. Préparer le couple env production **ensemble**.
6. Build & Deploy (dans cette infra : build = deploy).

### Si `CHANGES REQUIRED`

1. Lister les corrections dans le livrable ci-dessous.
2. Modifier les `.txt` 2.0 (toujours PREPARED).
3. Recalculer hashes ; mettre à jour `portal_terms_catalog.py`.
4. Rejouer `test_portal_terms_2_0.py` (+ régressions ciblées).
5. Documenter : `old prepared hashes → replaced before effectiveness`.
6. Nouvelle passe revue si les changements sont substantiels.

**Interdit** : modifier un body 2.0 déjà accepté en production ; backfill d’acceptations ; activer 2.0 sans double validation (et inversement).

---

## Livrable à remplir par le/la juriste

```text
7D-L — LEGAL REVIEW

REVIEWER :
[Nom / cabinet]

DATE :
[AAAA-MM-JJ]

TERMS_OF_SERVICE 2.0 :
PASS / CHANGES REQUIRED

TRANSPORT_TERMS 2.0 :
PASS / CHANGES REQUIRED

CONTRACT FORMATION :
PASS / CHANGES REQUIRED
  Notes (commande conditionnelle / pool ; cocontractant identifié à l’acceptation) :
  [...]

CARRIER IDENTIFICATION :
PASS / CHANGES REQUIRED
  Notes (champs obligatoires à l’écran 2e clic) :
  [...]

PRICE :
PASS / CHANGES REQUIRED

CANCELLATION :
PASS / CHANGES REQUIRED
  Notes (plafonds max canal vs taux exacts imposés ; LCart / accords verticaux) :
  [...]

NO-SHOW / WAITING / SUPPLEMENTS :
PASS / CHANGES REQUIRED
  Réponse « déterminé / déterminable avant conclusion » :
  [...]

GENERAL TERMS INCORPORATION :
PASS / CHANGES REQUIRED

EMAIL CONFIRMATION :
PASS / CHANGES REQUIRED

MODIFICATIONS :
PASS / CHANGES REQUIRED

INVOICING :
PASS / CHANGES REQUIRED

PAYMENT HOLD :
PASS / CHANGES REQUIRED

DISPUTE :
PASS / CHANGES REQUIRED

COLLECTION :
PASS / CHANGES REQUIRED

LIABILITY :
PASS / CHANGES REQUIRED

TERMS UPDATE / REACCEPTANCE :
PASS / CHANGES REQUIRED

LAW / JURISDICTION :
PASS / CHANGES REQUIRED

PUBLIC /conditions :
PASS / CHANGES REQUIRED

PRIVACY CROSS-CHECK :
PASS / SEPARATE REVIEW REQUIRED

REQUIRED TEXT CHANGES :
[...]

REQUIRED PRODUCT / UX CHANGES (si hors texte) :
[...]

LEGAL REVIEW :
PASS / FAIL
```

---

## État projet

```text
7A  CLOSED
7B  CLOSED
7B.1 CLOSED
7C  CLOSED (2.0 PREPARED)
7D  PARTIAL (release gate technique ; LEGAL REVIEW PENDING)
7D-L PACKAGE READY — en attente du remplissage juriste
```

Aucune nouvelle fonctionnalité produit attendue avant `LEGAL REVIEW = PASS`.
