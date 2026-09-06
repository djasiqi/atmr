# CREATE RIDE UX — chantier

**Statut** : **ACTIF** — P0 only (un seul champ actif, aucun overlay qui se réactive tout seul).  
**Périmètre** : comportement interne du formulaire « Créer une réservation » uniquement.

```text
NAV-01 = PASS / CLOSED
→ ouverture du +
→ lazy-load
→ modal accessible
→ ne plus toucher

CREATE RIDE UX = ACTIF
→ focus client
→ overlays
→ ordre de saisie
→ simplicité et rapidité du formulaire
```

**Hors scope** : NAV-01, map full-bleed, GPS, pricing, dispatch, règles métier.

```text
NO BUSINESS RULE CHANGE
NO PRICING CHANGE
NO DISPATCH CHANGE
NO GPS CHANGE

OBJECTIF :
une réservation manuelle
simple, rapide, impossible à confondre.
```

## Contrat

```text
CREATE RIDE — ONE ACTIVE FIELD ONLY

À tout moment :
activeField = client
OU pickup
OU destination
OU schedule
OU null

Jamais deux.
```

- Client : feuille clavier (`ClientPickerSheet` / `CreateRideKeyboardSheet`). Une fois choisi → carte fermée + `✕`.
- Adresses (création) : même feuille (`AddressPickerSheet`) — titre + recherche fixes, résultats scrollables au-dessus du clavier.
- Pas de `zIndex` empilé pour compenser plusieurs overlays.
- Pas de bouton « Fermer » en bas de feuille (caché par le clavier). `✕` + backdrop + Back.

## CREATE-RIDE-01 — FOCUS / OVERLAY

```text
P0
[x] un seul activeField
[x] client ne se réactive jamais tout seul
[x] sélection = fermeture immédiate du dropdown / sheet
[x] changer de champ ferme l'ancien dropdown
[x] tap extérieur ferme (blur adresse)
[x] aucun overlay orphelin
[x] aucune superposition Client / Adresse / Résumé
[x] aucune réactivation automatique (pas d’enchaînement / autofocus)
```

✅ **Implémenté** : `createRideActiveField.ts`, `ClientPickerSheet.tsx` + `CreateClientTrigger`, `AddressPickerSheet` (création). Après sélection : `Keyboard.dismiss()` puis `activeField = null`. Le champ Client (Pressable) ne reprend pas le focus. L’enchaînement auto (CREATE-RIDE-02) est **HOLD** — il réactivait les overlays tout seul.

## CREATE-RIDE CLIENT KEYBOARD

```text
CREATE-RIDE CLIENT KEYBOARD

[x] ouverture clavier ne masque jamais le selector
[x] titre toujours visible
[x] recherche toujours visible
[x] résultats scrollables au-dessus du clavier
[x] + Nouveau client accessible
[x] pas de bouton Fermer caché sous le clavier
[x] Android Back ferme d'abord le clavier
[x] sélection ferme clavier + selector
[x] Client ne reprend pas le focus après fermeture
[x] formulaire parent ne se décale pas anarchiquement
```

✅ **Implémenté** : `CreateRideKeyboardSheet` + `computeCreateRideSheetLayout` (hauteur au contenu, plafond ~62 % écran / espace au-dessus du clavier ; `liftBottom` si le Modal n’a pas été resizé). Pas de `flex: 1` entre helper et `+ Nouveau client`. Activité déjà en `softwareKeyboardLayoutMode: "resize"`. Même règle sur départ / destination (`AddressPickerSheet`). `RideEditModal` garde `AddressSelector` inline (hors chantier), avec hauteur de liste bornée au-dessus du clavier.

## CREATE RIDE — RESPONSIVE SELECTOR

```text
CREATE RIDE — RESPONSIVE SELECTOR

[x] hauteur dépend du contenu
[x] aucun grand espace vide
[x] clavier ne masque rien
[x] clavier n'entraîne pas automatiquement un sheet plein écran
[x] query < 2 = sheet compact
[x] quelques résultats = hauteur ajustée
[x] beaucoup de résultats = liste scrollable
[x] + Nouveau client reste facilement accessible
[x] X reste visible
[x] mêmes règles sur petits/grands Android
```

✅ **Implémenté** : `maxSheetHeight` = plafond uniquement. Liste : `maxHeight` = reste après chrome + footer. `+ Nouveau client` suit le contenu.

## CREATE RIDE — CLIENT SEARCH SPACING

```text
CREATE RIDE — CLIENT SEARCH SPACING

[x] header → search : +10 dp
[x] search → résultats : inchangé
[x] hauteur du sheet : inchangée
[x] clavier : inchangé
[x] liste scrollable : inchangée
```

✅ **Implémenté** : `ClientPickerSheet` — `marginTop: 10` entre sous-titre et champ recherche. Pas d’autre changement de layout.

## CREATE-RIDE-02 — FAST FLOW

```text
HOLD tant que P0 clavier n’est pas smoke S23
[ ] Client → Départ → Destination → Date/heure (après sélection)
[ ] focus suivant automatique après sélection
[ ] champ actif scrollé dans la zone visible
[x] feuille adresse reste ouverte si le clavier se ferme (Back)
[x] résumé n’est plus recouvert par un overlay client
[x] confirmation : hint unique « À compléter : … »
```

## Fichiers

- `mobile/unified-app/src/features/company/components/rides/createRideActiveField.ts`
- `mobile/unified-app/src/features/company/components/rides/createRideSheetLayout.ts`
- `mobile/unified-app/src/features/company/components/rides/CreateRideKeyboardSheet.tsx`
- `mobile/unified-app/src/features/company/components/rides/ClientPickerSheet.tsx`
- `mobile/unified-app/src/features/company/components/rides/AddressPickerSheet.tsx`
- `mobile/unified-app/src/features/company/components/rides/RideCreateModal.tsx`

`RideEditModal` / `ClientSelector` inline : inchangés (hors chantier).
