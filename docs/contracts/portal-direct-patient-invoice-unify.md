# Facturation PORTAL unifiée — Direct patient via `Invoice`

```text
STATUS : ARCHITECTURE FACTURATION LOCKED (2026-09-24)
6B CORE RECEIVABLE        : conservé (interne)
6B UX                     : REOPENED
Invoice Direct patient    : source UX unique
PortalReceivable          : interne uniquement
SUPERSEDES (workflow utilisateur) : 6B « PortalReceivable = facture UX »
```

Contexte déclencheur : course `#46759` (Mirjete Osmani, Emmenez-moi, CHF 40 contractuel,
`COMPLETED`) invisible dans BillPeriodModal (« Choisir un patient » vide) parce que le
filtre ne voit que les patients **portefeuille** (`client.company_id = entreprise`).

Références :
- Ancienne stratégie 6B : [portal-receivable.md](portal-receivable.md)
- Contrat transport : [portal-double-validation-7b.md](portal-double-validation-7b.md)
- Confirmation : `PortalClientTransportConfirmation.contractual_amount`

## Invariants verrouillés

```text
FACTURATION UX
= Invoice existant
= Direct patient

SOURCE PAYEUR
= patient portefeuille
OU
= client PORTAL contractuel

PORTAL n'introduit PAS un deuxième système de facturation UX.
PortalReceivable n'est PAS une action métier séparée pour l'entreprise.
```

L’utilisateur entreprise ne doit **jamais** choisir entre :

```text
Créer une facture
OU
Créer une PortalReceivable
```

## Montant — source contractuelle (anti-régression)

Pour une course `double_validation_v2` confirmée :

```text
Invoice amount source = contractual_amount

jamais :
booking.amount              (estimation)
maximum_accepted_amount     (plafond)
```

Exemple `#46759` :

| Notion | Valeur | Rôle facture |
| --- | ---: | --- |
| Estimation (`booking.amount`) | 50 | **jamais** |
| Plafond | 52 | **jamais** |
| Contrat (`contractual_amount`) | **40** | **oui** |

Supplément légitime → **ligne supplémentaire explicite**, pas remplacement silencieux.

## Parcours cible (prochain chantier — 6B UX / intégration Invoice)

```text
Course PORTAL terminée
→ Facturer
→ Direct patient
→ client PORTAL prérempli
→ montant contractuel prérempli
→ création Invoice standard
→ génération PDF / échéance / QR / paiements existants
→ synchronisation interne PortalReceivable
```

## Tests anti-régression (obligatoires avec le chantier)

```text
PORTAL 40 / estimate 50 / ceiling 52
→ Invoice = 40                                    PASS attendu

client PORTAL non présent dans portefeuille
→ Direct patient possible                         PASS attendu

création Invoice
→ PortalReceivable interne créé/mis à jour        PASS attendu

aucun bouton « Créer une créance PORTAL »
→ PASS attendu
```

## Deux sources de payeur, un seul `payer_type`

```text
payer_type = patient

payer_source =
  PORTFOLIO_PATIENT   → relation permanente entreprise ↔ patient
  PORTAL_CLIENT       → relation transactionnelle via booking / contrat
```

### Ne pas fusionner les identités

Une course PORTAL **ne crée pas** automatiquement un patient portefeuille permanent.

```text
PATIENT PORTEFEUILLE  ≠  CLIENT PORTAL
```

Les deux alimentent pourtant le même chemin **Direct patient** → même `Invoice`,
même PDF, même registre.

## BillPeriodModal / « Facturer »

### Cas période classique (inchangé)

Filtre patients portefeuille + courses période → facture mensuelle Direct patient.

### Cas course PORTAL (à ajouter)

Ouverture depuis une course PORTAL terminée → contexte **Facturer cette course** :

```text
Payeur     ● Direct patient
Client     Mirjete Osmani   (snapshot contractuel / billing_party)
Course     #46759 — 24.09.2026
Montant    CHF 40.00        (contractual_amount)
[Créer la facture]
```

Pas de recherche dans le portefeuille. Le booking + confirmation fournissent
l’identité du débiteur.

### Éligibilité

```text
client PORTAL
ET booking.company_id = entreprise courante
ET status ∈ {COMPLETED, RETURN_COMPLETED}
ET CLIENT_TRANSPORT_CONFIRMED présent
ET pas déjà facturée (lien Invoice / ligne)
```

## Ancrage preuves

```text
creditor = entreprise transport (booking.company_id)
debtor   = client PORTAL (snapshot BOOKING_CREATED / confirmation)
source   = CLIENT_TRANSPORT_CONFIRMED
amount   = contractual_amount
```

`BillingParty` PATIENT technique sous l’entreprise exécutante
(`patient_client:{client_id}`, `client.company_id` NULL autorisé) — déjà corrigé
pour `#46759`.

## Email destinataire (SendEmailModal) — LOCKED

✅ **Implémenté** : préremplissage + historique d’envoi.

Chaîne de résolution :

```text
1. email figé sur la facture (meta.recipient_email / last_recipient_email)
2. sinon email du compte client PORTAL (user.email) puis contact_email
3. sinon BillingParty.contact_email
4. sinon BOOKING_CREATED.debtor_email_snapshot
5. sinon champ vide + saisie obligatoire
```

Fichiers :
- `backend/services/billing/invoice_recipient_email.py`
- `Invoice.to_dict` → `default_recipient_email` + `client.email` / `contact_email`
- `SendEmailModal` + `billing_party_linker` (user.email → BP)
- Historique à chaque envoi : `meta.email_deliveries[]`
  (`invoice_id`, `recipient_email`, `sent_at`, `sender_company_id`, `delivery_method=email`)

## Facture papier + CHF 3.– — LOCKED

✅ **Implémenté** : préférence **client** (défaut email) + ligne Invoice distincte à la création.

```text
PARAMÈTRES CLIENT
invoice_delivery_method = email | paper   (défaut: email)

BillPeriodModal :
  aucun choix entreprise — affiche uniquement le total (incl. +3 si papier client)

→ ligne « Frais de facture papier » CHF 3 sur Invoice si préférence client = paper
→ visible aperçu HTML, PDF, édition brouillon, total QR / eBill

EMAIL → total = montant transport (ex. 40)
PAPER → total = transport + 3 (ex. 43)
```

Fichiers :
- `Client.invoice_delivery_method` + migration `b12686830f54`
- `paper_invoice_fee.py` (`ensure_paper_invoice_fee_line`, préférence client seule)
- `GenerateInvoice` + edit brouillon + régénération PDF
- `BillPeriodModal` : pas de radios (respect préférence client)
- `meta.delivery_method` + `meta.paper_invoice_fee_chf`

## Rôle résiduel de `PortalReceivable`

| Couche | Rôle |
| --- | --- |
| **Invoice** (UX + PDF + paiement QR) | Source de vérité facture émise |
| **PortalReceivable** (interne) | Suivi créance / solde / hold / litige / dunning / pursuit — **dérivé** du total réellement facturé (ex. 43 si papier), pas saisi à part |

## Non-objectifs

- Deuxième moteur tarifaire / PDF
- Forcer l’inscription portefeuille des clients PORTAL
- Resaisie libre du montant contractuel comme seul champ « montant course »
- Abandon immédiat des tables 6B–6G (hold, dunning, etc.)

## NEXT

1. ✅ Opportunités facturation : inclure bookings PORTAL éligibles (`billing_opportunities` + EligibleClients)
2. ✅ Montant facturable = `contractual_amount` (`billable_amount.py`)
3. ✅ Email SendEmailModal prérempli (compte PORTAL) + historique envoi
4. ✅ Mode Email / Papier (+ CHF 3 ligne distincte) avant création Direct patient
5. UI « Facturer cette course » préremplie (contexte booking) — encore à peaufiner
6. Pont Invoice → PortalReceivable (dérivation du total facturé)
7. Tests : `test_portal_direct_patient_invoice_unify.py` + `test_portal_invoice_email_and_paper_fee.py`
8. Déprécier UX « créer PortalReceivable à la main » côté entreprise
9. Reprendre smoke `#46759` (email prérempli sur EM-2026-09-0001)
