# STOP GATE external-carrier-02 — Audit, exports et statistiques

## Wording (UI / PDF / API)

- Statut final : **« Déclarée réalisée par l'institution »** — jamais « Réalisée » seul.
- Rapport audit : **« Exécution : Transporteur externe »** + transporteur snapshot + raison/référence/déclaration.
- Bon terrain : libellé **« Transporteur externe »** (valeur = nom snapshot).

## Timeline

- `external_carrier_assigned` — affectation directe
- `external_carrier_switched` — bascule depuis offres LIRIE (offres PENDING → UNAVAILABLE)
- `external_mission_completed` — déclaration institution

## Exports et statistiques

- Colonnes export : `Mode d'exécution`, `Transporteur`, `Référence externe`.
- Stats : filtrer **`carrier_source` ET `status`** :
  - Réalisé LIRIE = booking terminé
  - Réalisé externe = `carrier_source = external` **et** `status = EXTERNAL_DECLARED_COMPLETED`
  - Un simple `EXTERNAL_ASSIGNED` ou annulé ne compte **pas** comme réalisé.

## Facturation plateforme

- Sans `Booking`, pas de commission plateforme ni facture transporteur LIRIE.

✅ **Implémenté** : PDF, exports CSV, `compute_daily_stats`, tests d'intégration.
