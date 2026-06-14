# STOP GATE UX Couleurs — Liste + Détail institution

```txt
Status: PASS (auto + implémenté)
Date: 2026-06-14
Périmètre: InstitutionRequests (liste) + RequestDetailPanel (détail)
Hors scope: formulaire création, filtres segment (pills), dashboard
```

✅ **Implémenté** : tokens (`frontend/src/styles/tokens.css`), module partagé `statusColors.js`, refonte liste + détail, tests `__tests__/statusColors.test.js`.

## Objectif

Une couleur ne représente qu'une dimension métier sur la carte : le **statut d'avancement**, plus une **alerte transverse** (retard) si nécessaire. Tout le reste passe en typographie neutre hiérarchisée.

## Règles STOP GATE

### Règle 1 — Une couleur = statut

Accent LIRIE (`--brand-primary`) réservé à : sélection (bordure), focus, actions primaires, filtre actif — pas aux badges métier.

### Règle 2 — Max 2 badges colorés par carte

- 1 badge statut (bucket `--status-*`)
- 1 badge alerte optionnel (retard, bucket `--alert-*`)

### Règle 3 — Hiérarchie typographique

Interdit : une seule ligne meta de ~150 caractères.

Obligatoire :

```text
Niveau 1 — badges : [Confirmée] [Retard +20 min]
Niveau 2 — contexte : Transporteur : LIRIE Transport
Niveau 3 — détails : A/R · Facturé institution
```

### Règle 4 — Icônes exceptionnelles

Icônes autorisées uniquement pour actions (appeler, PDF, modifier) et alerte retard. Pas d'icônes décoratives sur métier (Externe, LIRIE, Clinique, trajet).

### Règle 5 — Retard transverse

Le retard s'ajoute au statut, ne le remplace pas. Tokens `--alert-warning-*` / `--alert-error-*`, distincts de `--status-error-*`.

### Règle 6 — Filtres hors scope

Les status pills (segment control) conservent l'accent LIRIE sur l'onglet actif (navigation, pas donnée métier).

## Taxonomie statut

| Bucket | Exemples | Tokens |
|--------|----------|--------|
| neutral | Brouillon, Expirée | `--status-neutral-*` |
| info | Envoyée, Assignée, En cours | `--status-info-*` |
| success | Confirmée, Terminée | `--status-success-*` |
| warning | En attente, Externe affecté | `--status-warning-*` |
| error | Annulée | `--status-error-*` |

Source JS : `frontend/src/pages/institution/Requests/statusColors.js`

## Micro STOP GATE PR1.5 (bloquant)

Tester les 3 cartes les plus chargées :

1. Annulée + Externe + Retard
2. Confirmée + A/R + Facturation
3. Externe affecté + Transporteur + Retour

Critère : statut principal identifiable en **< 1 seconde** sans lire transporteur, trajet ou facturation.

## Fichiers touchés

- `frontend/src/styles/tokens.css`
- `frontend/src/pages/institution/Requests/statusColors.js`
- `frontend/src/pages/institution/Requests/InstitutionRequests.jsx`
- `frontend/src/pages/institution/Requests/InstitutionRequests.module.css`
- `frontend/src/pages/institution/Requests/RequestDetailPanel.jsx`
- `frontend/src/pages/institution/Requests/RequestDetailPanel.module.css`

## Checklist QA

- [x] Max 2 badges colorés par carte
- [x] Retard indépendant du statut
- [x] Hiérarchie 3 niveaux (badges / transporteur / détails)
- [x] Points trajet gris
- [x] Sélection = bordure LIRIE, fond neutre
- [x] Liste et détail : même ton statut pour la même demande
- [x] Filtres segment inchangés
- [x] Validation auto micro STOP GATE PR1.5 (`__tests__/statusColors.test.js` — 3 cartes chargées)
- [ ] Validation manuelle navigateur (< 1 s sur 3 cartes chargées)
