# STOP GATE PDF-COMPACT — Compacité export PDF institution

```txt
Status: PASS (automatisé) — recette visuelle PDF-RENDER-01 recommandée
Date: 2026-06-13
Prérequis: simplification compacité + dédoublonnage implémentés
```

## Objectif pagination

| Scénario | Cible | Résultat attendu |
|---|---|---|
| Mission simple | 1 page | 1 page |
| Aller-retour / multi / long content | ≤ 2 pages | ≤ 2 pages |
| Annulée | 1 page | 1 page |
| Bon opérationnel | 1 page | 1 page |

## PDF-COMPACT-02

- Pas de bloc LIRIE textuel dans le header (logo seul ou footer)
- Référence mission seule dans l'en-tête ; Demande # / Réservation # dans le bloc administratif
- Dossier / Unité affichés seulement si renseignés
- Bloc administratif rendu en format compact vertical
- Historique d'une mission annulée = Demande créée / Acceptée / Annulée uniquement
- Multi-étapes : titres courts, pas d'adresse longue en titre

## PDF-COMPACT-03

- Réf. archivage lisible (TR-…), pas le hash long LIRIE-TR-…-hash
- QR maintenu (homepage maintenant, page mission dédiée plus tard)
- Pas de mention « Vérification documentaire : à venir »
- Identité : ordre Transporteur / Chauffeur / Véhicule
- Heures affichées seulement si utiles (multi-jours ou écart > 60 min)
- Horaires de trajet affichés seulement s'ils existent
- Aucune valeur affichée à -, —, N/A, Non renseigné (sauf sens métier)

## PDF-COMPACT-04 (dédoublonnage et rendu final)

- Pas de badge résultat métier : statut affiché uniquement dans l'en-tête (Option B, plus de doublon « Statut : Réalisé » + « MISSION RÉALISÉE »)
- Historique **toujours trié chronologiquement** ; une annulation sans timestamp réel est placée **après** les autres événements (jamais en doublon de date antérieure)
- Multi-étapes : une seule puce ● par étape, aucun glyphe parasite (plus de connecteur vertical rendu en tofu ■)
- Bloc administratif en format inline : `Réf. archivage : …`, `Empreinte : …`, `Identifiant : …` sur une ligne chacun
- Champs longs identité (Patient / Institution / Contact) tronqués (~2 lignes) avec ellipse `…`
- Bon de transport : `TR-… · JJ.MM.AAAA` dans l'en-tête (date mission visible dès l'ouverture)
- Footer allégé (`LIRIE · www.lirie.ch`)

## PDF-VOUCHER-01 (document terrain chauffeur)

- Bon = document opérationnel uniquement (**1 page**)
- Contient : Patient (≤ 1 ligne) + Naissance, Institution, **Type transport** (1re info opérationnelle),
  Transporteur, Chauffeur (si assigné), Contact = service · nom · téléphone (si disponible),
  TRANSPORT (prise en charge / rendez-vous, départ, destination, étapes),
  besoins médicaux essentiels, signatures côte à côte
- **Type transport** placé juste après l'identité patient (Assis / Fauteuil roulant / Brancard)
- **Prise en charge** (libellé court, une seule ligne) ; **Rendez-vous** si `scheduled_time_type = arrival`
- **Signatures symétriques** : Chauffeur = `Heure réelle` + Signature ; Patient = `Date` + Signature
- Date mission visible dès l'ouverture : `TR-… · JJ.MM.AAAA` dans l'en-tête du bon
- Ne contient **PAS** : empreinte/hash, référence d'archivage, facturation, historique, traçabilité administrative lourde
- Contient un **QR code LIRIE** compact en haut à droite (URL `verify_url` — homepage pour l'instant) et le **logo institution** (ou logo LIRIE par défaut) en haut à gauche, au-dessus du titre — même logique sur le **rapport de mission** (`_report_header_table`)
- Remarque médicale bornée à **120 caractères** (`_MAX_VOUCHER_NOTES`), rendue en italique grisé
- Nom patient borné à **40 caractères** (`_MAX_VOUCHER_PATIENT`) — une seule ligne
- Destination bornée à **55 caractères** (`_MAX_VOUCHER_DESTINATION`) — lieu reconnaissable, une ligne
- Si départ = domicile : l'adresse de départ visible directement dans TRANSPORT

✅ **Implémenté** : `build_operational_voucher_pdf`, `_build_voucher_identity_table` (type transport remonté), `_build_voucher_transport` (libellé horaire sur une ligne via `_voucher_transport_table`, destination ≤ 55 car.), `_build_voucher_medical` (≤ 120 car., note italique), `_build_voucher_contact_line` (service · nom · téléphone), `_signature_cell` (heure réelle), `_voucher_header_table` (logo gauche + QR droite au-dessus du titre) dans [`mission_report_pdf.py`](../backend/services/institutions/mission_report_pdf.py) ; `scheduled_time_type` exposé dans [`mission_report_context.py`](../backend/services/institutions/mission_report_context.py) ; tests `test_voucher_*` dans [`test_mission_report_pdf.py`](../backend/tests/unit/test_mission_report_pdf.py).

## PDF-VOUCHER-04 (identification facturation patient)

Règle métier : le patient est le débiteur de la facture → adresse + mention de facturation
deviennent utiles au transporteur pour rapprocher le bon de la facture sans ouvrir LIRIE.
Ce n'est **pas** une information de trajet (départ domicile) mais une donnée d'identification.

```text
Si billing_intent = patient :
  - afficher « Adresse patient » dans l'identité (rue, NPA ville)
  - afficher « Facturation : Patient »
Sinon (institution / assurance / AI / SUVA / LAMal / curateur…) :
  - masquer « Adresse patient »
  - masquer « Facturation »
```

- Adresse patient = `address, postal_code city` (tronquée à `_MAX_ADDRESS`), affichée seulement si renseignée
- Le bon ne devient pas un document de facturation : aucune mention de montant, de statut facture, ni de coordonnées bancaires

✅ **Implémenté** : `_format_patient_address` + clé `address` dans `build_patient_block` ([`mission_report_context.py`](../backend/services/institutions/mission_report_context.py)) ; affichage conditionnel (`billing_target == "patient"`) dans `_build_voucher_identity_table` ([`mission_report_pdf.py`](../backend/services/institutions/mission_report_pdf.py)) ; tests `test_voucher_patient_billing_shows_address` / `test_voucher_institution_billing_hides_address` / `test_voucher_insurance_billing_hides_address`. PDF revue : `06_patient_billing_operational.pdf`.

## PDF-COMPACT-05 (polish production)

- Émetteur retiré du bloc administratif (`Document généré par LIRIE` porté par le footer uniquement)
- Footer complet 2 lignes à gauche + `Page X` seul à droite (pas de double numérotation)
- Mission annulée : heures **réelles** supprimées du déroulement, heures **prévues** conservées
- `KeepTogether` sur titre + contenu « Informations administratives »
- Référence mission sur une seule ligne dans l'en-tête (`Référence mission : TR-…`)

## PDF-LONGCONTENT-02

- Champs identité bornés à 2 lignes max : Patient, Institution, Contact, Unité
- Ellipse (`…`) au-delà de la limite d'affichage (`max_len=65`)
- Objectif : un champ long ne doit jamais faire exploser la mise en page

## PDF-LONGCONTENT-03

- Déroulement : label, adresse et horaires rendus séparément (pas de chaîne unique)
- Adresse longue autorisée à wrapper sur plusieurs lignes (Paragraph)
- Horaires sur ligne séparée (`Prévu : … · Réel : …`)

## PDF-LONGCONTENT-05 (limites par champ + signatures)

Objectif : rapport long content = 1 page. Limites de caractères (ellipse `…` au-delà) :

| Champ | Limite |
|---|---|
| Patient | 80 |
| Institution | 80 |
| Transporteur | 60 |
| Adresse (départ / retour) | 160 |
| Destination | 90 |
| Remarques médicales libres | 2408 (`[…]`) |

- Besoins critiques (fauteuil, oxygène, accompagnant, accès/étage…) toujours affichés en entier ; seul le commentaire libre est tronqué.
- Bon de transport : signatures Chauffeur / Patient **côte à côte** (gain vertical).

## PDF-RENDER-01 (recette visuelle manuelle)

Vérifier sur A4 imprimé, PDF navigateur, PDF mobile :

- Mission simple
- Mission annulée
- Aller-retour
- Multi-étapes (5+ étapes)
- Long content

## Régénération revue

```bash
docker compose exec atmr_api python scripts/generate_mission_pdf_review.py
# → /tmp/mission_pdf_review/
```

PDF copiés localement : `backend/tmp_pdf_review/`

## Tests automatisés

```bash
docker compose exec -T atmr_api python -m pytest tests/unit/test_mission_report_pdf.py tests/unit/test_mission_report_context.py -q
```
