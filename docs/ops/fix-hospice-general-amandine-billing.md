# Correction données — Hospice général / Amandine HAUSER

**Statut chantier : CODE COMPLETE / DATA FIX PENDING**

Ne **pas** passer en CLOSED / PASS tant que la donnée réelle n’a pas été migrée **et** smoke-vérifiée sur l’environnement qui contient la fiche.

Référence métier : quatre notions indépendantes (lien, représentation légale, tiers payeur, contact facturation). Voir [`billing-vs-legal-representation.md`](billing-vs-legal-representation.md).

## Cible

```text
BillingParty.display_name = "Hospice général"
BillingParty.type         = "other"
contact_name              = "Amandine HAUSER"
role                      = "Coordinatrice"
```

Prérequis : Hospice général est bien le **débiteur** de la facture (pas seulement un intermédiaire).

## Script

✅ **Implémenté** : [`backend/scripts/fix_hospice_general_amandine_billing.py`](../../backend/scripts/fix_hospice_general_amandine_billing.py)

Sur l’environnement qui contient réellement la fiche :

```bash
docker compose exec atmr_api python scripts/fix_hospice_general_amandine_billing.py --dry-run
docker compose exec atmr_api python scripts/fix_hospice_general_amandine_billing.py --apply
```

Le type reste `other` (pas de curatelle déduite).

## Smoke après `--apply` (même environnement)

1. Fiche client : `Hospice général` + `Contact facturation : Amandine HAUSER` + `Fonction : Coordinatrice`
2. Aucune mention indue de `Curateur` dans la partie facturation
3. Facture test : bloc destinataire = organisme + `À l'att. de Amandine HAUSER`

## Exécution Docker local (2026-09-16)

Dry-run puis `--apply` sur `atmr_api` : **aucun candidat**, **aucune écriture**.

La base locale n’a aucun `BillingParty` / `ClientBillingParty` Amandine / Hospice / `amandine.hauser@hospicegeneral.ch`. Le script est sain (pas d’effet indésirable) ; le blocage est **environnemental**, pas technique.

**Ne rien forcer** sur ce Docker. Relancer uniquement là où la fiche existe.
