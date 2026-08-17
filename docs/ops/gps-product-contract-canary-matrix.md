# Matrice canary — Contrat GPS produit v4

Référence : [`docs/contracts/gps-driver-product-contract.md`](../contracts/gps-driver-product-contract.md).

## Règles de release

```text
B2 PASS seul     = NO-GO PROD
B2 + B3 + battery = candidat canary
```

## Scénarios

| ID | Scénario | Attendu | Verdict |
|----|----------|---------|---------|
| D1 | FG mission / FG présence | Mode conservé ; pas de nouvelle session | |
| D2 | HOME (Maps / autre app) | LIVE ou PRESENCE continue | |
| D3 | LOCK écran | Mode continue | |
| D4 | IMMOBILE 10+ min | Pas de faux « GPS hors ligne » ; location + device-health heartbeats | |
| D5 | OFFLINE puis RETURN | File locale ; même session ; flush | |
| D6 | SOCKET RECONNECT | Même session ; ≠ OFF métier | |
| D7 | PRESENCE → LIVE → PRESENCE | Aucun stop/unregister/trou/rotate session (invariants B2) | |
| D8 | LOGOUT / fin_service | GPS OFF ; UI « Hors service » ≠ « GPS hors ligne » | |
| D9 | FORCE-STOP Android | OFF OS ; pas d’auto-reprise JS | |
| D10 | SWIPE RECENTS | Mesurer ; garantie produit seulement si validé | |
| D11 | CANARY BATTERIE PRESENCE | Voir ci-dessous | |

## CANARY BATTERIE PRESENCE (gate B3)

```text
- 30 min puis 60 min immobile
- écran éteint
- aucune mission
- en_service=true, permissionsReady (FG+BG)
- mesurer : variation batterie, nb fixes produits, nb points enqueue/transmis
- comparer au baseline présence historique (cadence lente)
```

Décision B3 : transition sans restart **et** autonomie acceptable vs baseline.

## Invariants B2 (D7)

```text
- aucun stopLocationUpdatesAsync
- aucun unregister explicite
- aucun trou de capture
- aucun changement de session GPS
- aucun changement d’event_id d’une position existante
- aucun restart FGS observable
```

## Notes appareil

Documenter ici le modèle / OS / build pour chaque run canary.
