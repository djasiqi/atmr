# 7B.4 — Transporteur habituel + confirmation automatique

```text
STEP 7B.4 — PREFERRED CARRIER AUTO-CONFIRM
STATUS : SUPERSEDED AS PRIMARY PATH by 7B.5
  → portal-single-click-conditional-order-7b5.md
NOTE : 7B.5 généralise le 1 clic à tout le canal (pas seulement « transporteur habituel »)
DEFAULT CODE TODAY : double_validation_v2 inchangé (2e clic obligatoire)
OPT-IN HISTORIQUE : transporteur habituel + autorisation conditionnelle (non implémenté)
```

Référence parent : [portal-double-validation-7b.md](portal-double-validation-7b.md)

## Problème UX

Pour un client fidèle (souvent le même transporteur, plusieurs courses / semaine), le parcours :

```text
demande → attendre → notification → modal → confirmer
```

est trop lourd **si** le prix final ≤ plafond déjà accepté et que le transporteur est celui qu’il choisit habituellement.

## Décision produit

Conserver la **double validation comme règle par défaut**.

Ajouter un mode **opt-in** « transporteur habituel » avec confirmation automatique **conditionnelle**.

Prérequis juridique / canal (verrouillé) : [portal-channel-cancellation-caps.md](portal-channel-cancellation-caps.md)

```text
Prix transport     → libre entreprise
Annulation canal   → PLAFONDS MAX LIRIE ; entreprise ≤ caps (jamais taux exacts imposés)
1er clic client    → plafond prix + cadre annulation canal portés à connaissance
```

Sans plafonds canal affichés au 1er clic, alléger / supprimer le 2e clic est plus fragile (transparence SECO + formation du contrat).

```text
STANDARD (défaut)
1er clic → demande + plafond prix + plafonds annulation canal
transporteur → offre (prix libre ≤ plafond ; policy ≤ caps)
2e clic → contrat

TRUSTED / PREFERRED CARRIER (opt-in)
1er clic → demande + plafond + cadre annulation + autorisation conditionnelle
transporteur habituel → accepte à son tarif grille
→ contrat automatique si TOUTES les gardes passent
sinon → 2e clic manuel
```

## Préférences client (deux niveaux)

### Mode standard

> Je souhaite confirmer chaque proposition.

### Transporteur habituel

> Confirmer automatiquement lorsque mon transporteur habituel accepte à un prix inférieur ou égal au maximum que j’ai accepté.

UI indicative :

```text
Transporteur habituel → Emmenez-moi

☑ Confirmer automatiquement mes transports avec Emmenez-moi
   lorsque son tarif ne dépasse pas le prix maximum accepté.
```

## Flux cible (auto-confirm)

```text
Client demande le transport
↓
LIRIE calcule les quotes (plafond = MAX)
↓
Client : « Envoyer la demande » (plafond figé)
↓
Transporteur habituel reçoit uniquement son tarif grille
↓
Transporteur habituel accepte
↓
Serveur : quote <= plafond
        + policy published
        + autorisation client active
        + pas de hold / litige
↓
CONFIRMATION AUTOMATIQUE
↓
contrat = offered_amount (pas le plafond)
```

Notification client :

> Transport confirmé avec {entreprise} — CHF X.–  
> Votre transporteur habituel a accepté la course selon votre préférence de confirmation automatique.

**Pas de deuxième clic.**

## Gardes strictes (toutes obligatoires)

La confirmation automatique **uniquement** si :

| # | Condition |
| --- | --- |
| 1 | Entreprise acceptante = transporteur habituel choisi |
| 2 | `offered_amount <= maximum_accepted_amount` (serveur) |
| 3 | Politique d’annulation PORTAL **publiée** et **≤ plafonds canal** (snapshot figé) |
| 4 | Conditions contractuelles compatibles avec l’autorisation donnée (version CG / transport acceptée) |
| 5 | Aucun blocage paiement / hold / litige |

Sinon → **retour au 2e clic** :

> Une confirmation est nécessaire

Cas de fallback explicites :

```text
nouveau / autre transporteur
OU prix > plafond
OU nouvelles conditions importantes (re-acceptation requise)
OU politique non publiée
OU autorisation absente / révoquée
OU hold paiement
```

## Marché ouvert si le habituel ne répond pas

Si le transporteur habituel refuse ou ne répond pas dans la fenêtre prévue :

1. la demande peut s’ouvrir aux autres entreprises (dispatch existant) ;
2. **toute offre d’une autre entreprise** exige le **2e clic manuel** (jamais d’auto-confirm hors habituel).

## Confidentialité plafond

**Inchangé / non négociable** :

```text
Transporteur voit : son tarif grille uniquement
Transporteur ne voit JAMAIS : plafond client
```

Le serveur compare `carrier_quote` vs `client_ceiling` sans exposer le plafond.

## Principes conservés

```text
aucun transporteur imposé silencieusement
aucun prix supérieur au plafond
aucunes nouvelles conditions silencieuses
contractual_amount = offered_amount (jamais plafond / estimation)
```

## Hors scope 7B.4 (pour l’instant)

- Implémentation code / migration préférences
- UI paramètres client
- SLA timeout « habituel ne répond pas »
- Activation prod
- Versioning / gate des **plafonds canal** (doc dédiée) — prérequis produit, pas ce lot seul

## NEXT (quand priorisé)

0. Revue avocat : formation contrat (transporteur non identifié au 1er clic) + caps annulation LCart — [portal-channel-cancellation-caps.md](portal-channel-cancellation-caps.md)
1. Modèle préférences client (`preferred_company_id`, `auto_confirm_preferred`, versions CG liées)
2. Hook post-`create_portal_carrier_offer` : tenter auto-confirm si gardes OK
3. Preuves immuables : événement distinct du 2e clic manuel (ex. `CLIENT_TRANSPORT_AUTO_CONFIRMED`) + référence à l’autorisation
4. Tests : happy path, autre transporteur, prix > plafond, policy manquante / > caps, hold, révocation
5. Smoke manuel après 7D-M
