# Indication A/R sur facture client

✅ **Implémenté** : le tag `[A/R]` (PDF) / `A/R` (aperçu HTML) signifie **« cette ligne facture réellement une prestation aller-retour »**. La même structure pilote les deux renderers. Le montant et le flag historique `booking.is_round_trip` / `billing_unit = round_trip` ne décident plus du tag client.

## Contrat

| Structure | Signification | HTML | PDF |
| --- | --- | --- | --- |
| `single` | Une seule réservation rattachée (même si créée en A/R) | aucun badge | aucun `[A/R]` |
| `merged_both_legs` | Une ligne porte aller + retour (`booking_ids` ≥ 2 ou secondaires) | `A/R` | `[A/R]` |
| `pair_primary` | Aller d'une paire deux lignes, retour présent sur la facture | `A/R` sur la primaire | `[A/R]` sur la ligne consolidée |
| `pair_return` | Retour de la paire | ligne masquée | absorbé dans la primaire |

Paire deux lignes — contrat retenu : **un seul** `[A/R]` sur la ligne visible (primaire). Le retour n'affiche pas de second tag. Si le partenaire est absent, la primaire retombe en `single`.

## Source de vérité

Backend : `application.invoices.invoice_line_round_trip`

Champs API (après enrichissement) :

- `invoice_line_round_trip_structure`
- `invoice_line_represents_full_round_trip`

Frontend : `frontend/src/utils/invoiceLineRoundTrip.js` (`invoiceLineRepresentsFullRoundTrip`, `invoiceLineClientArTag`). L'éditeur conserve un badge **informatif** distinct (`lineEditorContextArTag`) qui peut afficher « A/R » sur une mono-réservation historique : ce badge n'est pas le tag client.

## Hors contrat

- `billing_unit == "round_trip"` / `transport_type == "A/R"` : information (origine `is_round_trip`), pas preuve qu'un A/R est facturé.
- Le prix (45 / 90 CHF, etc.) n'intervient dans aucune décision.

## Cas réel corrigé (`EM-2026-09-0065`)

Reproduction par **fixture** (cette facture n'existe pas dans la DB Docker locale) :
ce n'est pas une preuve de lecture de la donnée de production.

- 02.09 — aller simple, éventuellement flagué A/R historiquement → **aucun** `[A/R]` HTML/PDF.
- 03.09 — modèle retenu : deux réservations rattachées à la même ligne → `A/R` / `[A/R]`.
