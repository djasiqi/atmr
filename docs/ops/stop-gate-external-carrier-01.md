# STOP GATE external-carrier-01 — Modèle et transitions

## Règles V1

1. **Deux flux disjoints** : LIRIE (offres → booking) vs externe (snapshot → déclaration).
2. **Pas de retour arrière** : une mission `carrier_source = external` ne peut pas revenir au flux LIRIE (réouverture d'offres, `/send`, acceptation d'offre). **Non supporté en V1**.
3. **Snapshot figé** : les champs `external_carrier_*` ne sont jamais remplacés par une FK future.
4. **Garde-fou concurrence** : `accept_offer` renvoie 409 si `carrier_source = external`.

## Transitions autorisées

| Depuis | Action | Vers |
|---|---|---|
| DRAFT, SENT | `external-carrier` | EXTERNAL_ASSIGNED |
| EXTERNAL_ASSIGNED | `external-carrier` (réaffectation) | EXTERNAL_ASSIGNED |
| EXTERNAL_ASSIGNED | `external-completion` | EXTERNAL_DECLARED_COMPLETED |

## Statuts interdits côté externe

- Jamais `ACCEPTED`, `CONVERTED`, ni création de `Booking`.

✅ **Implémenté** : use cases + refus explicite dans `send_transport_request` et `accept_offer`.
