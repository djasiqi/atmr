# STOP GATE PDF-UX-01 — Audit UX bon de transport chauffeur

```txt
Status: DÉCISION PRODUIT — IMPLÉMENTÉ
Date: 2026-06-13 (décision finale 2026-06-14)
Prérequis: layouts legacy + operational + maquettes A/C (revue only)
```

> ✅ **Décision produit (2026-06-14)** : phase d'audit close. Variante finale retenue
> en production : `layout="operational"`, `hero_style="inline"`,
> `signature_style="confirmation_inline"`. Les variantes `medical`, `ultra_compact`,
> `hero_split` et les blocs signatures lourds (`stack`/`row`) restent disponibles pour
> revue uniquement et seront archivés après validation. Voir section finale.

## Objectif

Repenser la **hiérarchie visuelle** du bon de transport opérationnel pour un usage **chauffeur**, sans ajouter de données métier. Protocole de décision mesurable avant bascule API.

**Hors périmètre revue (poids 0 %) :** logo LIRIE, identité visuelle, couleurs de marque.

---

## Variantes

| Code | Nom | Production | Pronostic |
|---|---|---|---|
| `legacy` | Baseline actuelle (table clé/valeur) | Oui (API actuelle) | ~0% |
| `operational` | Document terrain chauffeur | **Cible post-audit** | **~80%** |
| `ultra_compact` | A — densité maximale | Revue only | ~15% |
| `medical` | C — besoins renforcés | Revue only | ~5% |

**Micro-design Operational** (`VoucherLayoutOptions`) :

| Option | Défaut (final) | Alternatives (revue only) |
|---|---|---|
| `hero_style` | `inline` | `split` |
| `signature_style` | `confirmation_inline` | `stack`, `row` |

---

## Trois axes d'audit

### Axe 1 — Hiérarchie L1 / L2 / L3

| Niveau | Contenu |
|---|---|
| L1 | Patient, heure principale, adresses trajet, besoins |
| L2 | Institution, contact, type transport |
| L3 | TR-ref, transporteur, chauffeur, facturation patient |

### Axe 2 — Densité visuelle

Mesurer par scénario × variante :

- Hauteur identité patient (lignes avant besoins)
- Hauteur bloc trajet
- Hauteur signatures
- Nombre de pages (invariant : 1 page scénarios standards)

### Axe 3 — Temps de repérage (chronomètre)

| Information | Seuil |
|---|---|
| Heure principale | **< 1 s** |
| Destination | **< 2 s** |
| Besoins particuliers | **< 2 s** |
| Patient | **< 2 s** |
| Compréhension globale du trajet | **< 5 s** |
| Contact | < 5 s (non bloquant) |

**Question trajet :**

```text
En 5 secondes, pouvez-vous expliquer où le patient doit aller ?
(ex. Anières → Centre d'Imagerie → Retour institution)
```

---

## Wireframes

### A — Ultra compacte

```text
PATIENT
MIRJETE OSMANI (04.10.1997)

BESOINS
• Accompagnement requis

TRAJET
09:00 — Chemin des Courbes 9, 1247 Anières
23:00 — Centre d'Imagerie Rive Gauche
Retour institution

CONTACT
Admissions · Marc Mouchet · +41 22 512 02 03
```

### B — Opérationnelle (favori)

```text
BON DE TRANSPORT
TR-2026-000996 · 13.06.2026

MIRJETE OSMANI (04.10.1997)
Clinique Les Hauts d'Anières

┌─ ATTENTION — BESOINS PARTICULIERS ─────────┐
│ • Accompagnement requis                    │
└────────────────────────────────────────────┘

TRAJET
Prise en charge
09:00
Chemin des Courbes 9, 1247 Anières
        ↓
23:00
Centre d'Imagerie Rive Gauche
        ↓
Retour institution

Contact · tél  |  Emmenez Moi · Chauffeur

Signatures
Chauffeur ______________________
Patient / représentant ______________________
```

### C — Médicalisée

Identique à B avec bloc besoins renforcé (bordure accent, fond léger) et type transport intégré dans le bloc.

---

## Règle anti-itération

```text
Si Operational gagne clairement la revue chauffeur
(préférence explicite ET critères chronométrés passés) :

→ aucune itération sur Ultra Compact ou Medical
→ aucune nouvelle variante
→ bascule API immédiate (PR séparée)

Si Operational échoue un critère bloquant :
→ ajuster uniquement Operational (hero/sig)
→ pas de variante D
```

**Critères « gagne clairement » :**

- Variante B choisie (ou ≥ 2/3 panel)
- Heure < 1 s et destination < 2 s sur Operational
- Compréhension trajet < 5 s validée

---

## Fiche revue chauffeur

```text
Sans lire tout le document (chronomètre dès ouverture) :

  □ Heure principale < 1 s ?          Temps : _____ s
  □ Destination < 2 s ?               Temps : _____ s
  □ Besoins < 2 s ?                   Temps : _____ s
  □ Patient < 2 s ?                   Temps : _____ s

  □ En 5 s, trajet complet explicable ?
      Temps : _____ s   □ Oui   □ Non

Variante préférée :
  □ A — Ultra compacte   □ B — Opérationnelle
  □ C — Médicalisée      □ Legacy

Si B — micro-design :
  Hero : □ inline   □ split
  Sig  : □ stack    □ row

Ne pas évaluer : logo, couleurs, esthétique générale.

Commentaire : _______________
```

---

## Génération PDF comparatifs

```bash
docker compose exec atmr_api python scripts/generate_mission_pdf_review.py
```

Sortie : `/tmp/mission_pdf_review/ux/` (+ copie locale `backend/tmp_pdf_review/ux/`)

Fichiers clés :

- `08_driver_critique_{legacy|ultra_compact|operational|medical}.pdf`
- `08_driver_critique_operational_hero_{inline|split}_sig_{stack|row}.pdf`

---

## Implémentation

✅ **Implémenté** (2026-06-13) :

- [`VoucherPresentation`](../backend/services/institutions/mission_report_pdf.py), [`VoucherLayoutOptions`](../backend/services/institutions/mission_report_pdf.py), [`RouteStop`](../backend/services/institutions/mission_report_pdf.py) — couche présentation unique
- Layouts `legacy` (défaut API), `operational`, `ultra_compact`, `medical` via `build_operational_voucher_pdf(ctx, layout=..., options=...)`
- `_REVIEW_ONLY_LAYOUTS` pour A/C
- Script revue [`generate_mission_pdf_review.py`](../backend/scripts/generate_mission_pdf_review.py) — scénario `08_driver_critique` + exports UX dans `tmp_pdf_review/ux/`
- Tests [`test_mission_report_pdf.py`](../backend/tests/unit/test_mission_report_pdf.py) — classe `TestOperationalVoucherUxLayouts` + smoke A/C
- PDFs générés localement : [`backend/tmp_pdf_review/ux/`](../backend/tmp_pdf_review/ux/)

✅ **Décision produit — bon final (2026-06-14) :**

Variante unique retenue en production : `operational` + `hero_style="inline"` +
`signature_style="confirmation_inline"`.

- En-tête simple `BON DE TRANSPORT` · `TR-… · date` (logo hors périmètre)
- Hero patient inline : `Nom (date naissance)`, institution puis type transport sous le nom — pas de table clé/valeur
- Bloc besoins **avant** le trajet, Helvetica, sans emoji ; type transport non dupliqué (déjà dans le hero) ; remarque tronquée proprement
- Trajet « parcours chauffeur » : heure avant adresse, connecteurs `↓`, `Retour institution`, plus de `Étape N` / `Destination N` / `Prévu :` ; horaire absent = indication discrète
- Contact/transporteur compact 1 ligne, bascule auto en 2 lignes si trop long ([`_voucher_meta_footer`](../backend/services/institutions/mission_report_pdf.py))
- Confirmation finale inline ultra compacte ([`_voucher_confirmation_inline`](../backend/services/institutions/mission_report_pdf.py)) :
  `Confirmation : Chauffeur ____  Patient/représentant ____` — plus de section signatures administrative, ni `Heure réelle`, ni `Date`
- API export bascule sur `layout="operational"` ([`institution_exports.py`](../backend/routes/institution_exports.py))
- Scénarios finaux générés ([`generate_mission_pdf_review.py`](../backend/scripts/generate_mission_pdf_review.py)) :
  `01_simple` · `03_roundtrip` · `05_longcontent` · `08_driver_critique` · `09_multistep_5` (5 arrêts) → suffixe `_operational_final.pdf`
- Tests [`test_mission_report_pdf.py`](../backend/tests/unit/test_mission_report_pdf.py) : confirmation inline par défaut, aller-retour 1 page, multi-étapes 5 sans libellé ERP (51 tests verts)

**Reste à faire :**

- Archiver / supprimer les layouts `medical`, `ultra_compact` et options `split`/`stack`/`row` après validation terrain (la production ne maintient que `legacy` + `operational final`)
- Mettre à jour [`stop-gate-pdf-compact-01.md`](stop-gate-pdf-compact-01.md)
