# LIRIE — CONVENTION-SIGNATURE-REAL-01

Finalisation technique de la convention Client et du parcours de signature.

**Date :** 2026-11-14  
**Worktree :** `D:\lirie-backups\_candidate_lirie_convention_signature_real_20261114`  
**Branche :** `candidate/lirie-convention-signature-real-20261114`  
**Base :** tip PR #108 `74b6a1ec`  
**Push / PR / merge / deploy :** non effectués  

---

## A. Existant réutilisé (non recommencé)

| Domaine | Modules | Notes |
|---------|---------|-------|
| Convention **démo** | `models/convention_demo.py`, `application/settlement/convention_demo.py`, `mandate_registry.py` | `ck_convention_demo_not_legal` **intact** |
| Applicabilité / clôture | `mandate_applicability.py`, `closure_options.py`, CR-01/02 | Règles différé / espèces / terminal / signalement **conservées** |
| Stockage fichiers | `UPLOAD_FOLDER` (même racine que spécimens) | PDF figés sous `conventions/{company_id}/…` |
| UI signalements | `SettlementIncidentsFollowUp.jsx` | Non modifié |
| Mobile clôture | `closureOptionsContract.ts` | Non cassé ; contrat réel ajouté à côté |

---

## B. Développements réalisés

| Élément | Fichiers |
|---------|----------|
| Cycle de vie + gate différé | `domain/.../convention_real_lifecycle.py` |
| Modèle réel + historique | `models/patient_company_convention.py` |
| Service métier | `application/settlement/convention_real.py` |
| API entreprise / portail client / chauffeur | `routes/convention_real.py` |
| Branchement clôture | `routes/settlement_non_payment.py` (`_mandate_kind_for` lit d’abord la convention réelle) |
| Migration | `c3s01a1b2c3d` ← `c2r02a1b3c4d` |
| UI dossier Client | `ClientConventionPanel.jsx` branché dans `ClientReadView.jsx` |
| Contrat mobile | `conventionRealContract.ts` (+ test) |

---

## C. Modèle contractuel

Table `patient_company_convention` (séparée de la démo) :

- Entreprise + client + **débiteur** + destinataire facture **optionnel et distinct**
- Conditions de facturation, tarif, périmètre, version document, n° de version
- Statuts : `draft` → `awaiting_signature` → `signed_pending_validation` → `applicable` → `expired` / `revoked` / `replaced`
- `legally_effective` : défaut `false` ; contrainte DB : seulement si `status=applicable`
- Flag applicatif `CONVENTION_LEGAL_ACTIVATION` : **fermé** → différé réel **jamais** autorisé dans ce lot
- Une convention applicable peut couvrir plusieurs transports (lien via couple client–entreprise, pas une course unique)

Ne confond pas : CGV, confirmation de réservation, `PlatformPartnerAgreement`, facturation institutionnelle.

---

## D. Parcours de signature

1. Entreprise crée un brouillon (`POST .../conventions`)
2. Soumission → `awaiting_signature` + empreinte SHA-256 du texte présenté
3. Signature (client portail / chauffeur) : consentement explicite, identité, tracé, horodatage, PDF figé, archive
4. Statut → `signed_pending_validation` (**pas** applicable automatique)
5. Validation admin → `applicable` **sans** lever `legally_effective` (activation fermée)
6. Alternative : dépôt scan papier admin (`.../paper`) → même file d’attente de validation
7. Refus / interruption → retour `draft` + événement tracé
8. Nouvelle version : ancienne marquée `replaced` / inactive, PDF précédent conservé

Disclaimer exposé : capture technique ≠ signature électronique qualifiée suisse.

---

## E. Gestion documentaire

| Point | Comportement |
|-------|--------------|
| Empreinte | SHA-256 du texte canonique présenté |
| PDF figé | Généré à la signature / papier ; relu sans régénération silencieuse |
| Chemin | `conventions/{company_id}/{id}-v{n}.pdf` |
| Historique | `patient_company_convention_event` append-only |
| Idempotence | Clés sur convention et événements |

---

## F. Migrations

| Revision | Contenu | Prod |
|----------|---------|------|
| `c3s01a1b2c3d` | Tables convention réelle + events | **Non appliquée** |

`patient_company_convention_demo` / `ck_convention_demo_not_legal` : **non modifiés**.

---

## G. Tests exécutés

| Suite | Résultat | Environnement |
|-------|----------|---------------|
| Domaine CSR-01 lifecycle + gate | **7 passed** | Docker `atmr-backend_tests` |
| Persistance PG jetable (`CSR01_DATABASE_URL`, `--noconftest`) | **8 passed** | Postgres `csr01-pg` :55433 |
| Régression CR-01 + CSR-01 lifecycle | **20 passed** | Docker |
| Jest `conventionRealContract` | **non exécuté** (pas de `node_modules` local) — fichier de test fourni | — |
| Navigateur / appareil physique | **non exécutés** | Validation manuelle restante |

Scénarios couverts côté API/domaine/DB jetable : création, version exacte, signature+archive, nouvelle version sans écrasement, refus, expiration/révocation, isolation active unique, interdiction `legally_effective` hors `applicable`, applicable sans différé réel, paiement immédiat sans signature (clôture), démo ≠ différé.

---

## H. Réserves juridiques (soumises à validation)

1. **Suisse** : une signature manuscrite dessinée à l’écran n’équivaut pas automatiquement à une signature électronique qualifiée (QES) ni à une preuve équivalente à l’écrit signé.
2. Les exigences SES / QES / horodatage certifié / identité forte **ne sont pas implémentées** comme certification ; le code documente explicitement l’absence de valeur inventée.
3. L’activation `CONVENTION_LEGAL_ACTIVATION` et `legally_effective=true` restent **fermées** jusqu’à approbation juridique du modèle et du mécanisme.
4. Le scan papier exige un contrôle administratif ; il n’active pas non plus le différé réel.

---

## I. Écrans à faire valider visuellement

1. Panneau convention dans le dossier Client entreprise
2. Portail client : liste / signature / PDF (API prête ; UI portail minimale via API)
3. Chauffeur : lecture + signature pendant transport (contrat + API ; écran natif à brancher)
4. Téléchargement PDF figé

---

## J. Commits locaux

| SHA | Message |
|-----|---------|
| `e5a2ac17` | feat(convention): parcours réel signature et archivage sans différé juridique |

Aucun push.

---

## K. Verdicts

| Gate | Verdict |
|------|---------|
| CONVENTION FUNCTIONAL | **PASS** (API + modèle + UI dossier minimale) |
| SIGNATURE FUNCTIONAL | **PASS** (parcours technique testable ; pas QES) |
| DOCUMENT ARCHIVING | **PASS** |
| DEFERRED BILLING GATE | **PASS** (fermée) |
| TENANT SECURITY | **PASS** (isolation company/client ; index actif) |
| LEGAL APPROVAL | **NOT READY** |
| VISUAL APPROVAL | **NOT READY** |
| READY FOR REMOTE CI | **YES** |
| READY FOR PRODUCTION | **NO** |
