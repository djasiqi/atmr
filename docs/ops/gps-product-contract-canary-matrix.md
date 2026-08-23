# Matrice canary — Contrat GPS produit v4

Référence : [`docs/contracts/gps-driver-product-contract.md`](../contracts/gps-driver-product-contract.md).

## Gate figé (audit statique `78a1c73c` + device 2026-08-21)

```text
STATIC v4 A1/A3/C2              = PASS ✅
B2 / B3 DEVICE / POC-1A         = PASS ✅  CLOSED
B3-C BATTERY (Variante A 60 s)  = PASS ✅  (−4 pts/h ; baseline −11)
DEVICE GPS CERTIFICATION        = CLOSED ✅
GO PROD                         = NO  (exige E2E flotte 10/10 → 20/20)
CHECK GOOGLE KEY (CI)           = dumps xmltree/activity untrackés (pas de waiver scan)
```

Ordre d’exécution :

1. ~~**D7/B2** puis **D11/B3**~~ = **CLOSED ✅** — [`b2_canary_78a1c73c/B3_STATUS.md`](./_release_exec_p0d_2026-08-16/b2_canary_78a1c73c/B3_STATUS.md)
2. ~~**Certification E2E flotte**~~ = **HOLD** — reprise après certif états
2bis. **NEXT ★** Certification états chauffeur C01→C10 FG/BG — [`gps-driver-state-certification.md`](./gps-driver-state-certification.md)
3. Puis flotte 10/10 → 20/20 — [`gps-fleet-e2e-certification.md`](./gps-fleet-e2e-certification.md)
4. Seulement alors **GO PROD**

**Freeze GPS device** : pas de nouveau développement fonctionnel GPS sans preuve de régression.

## Règles de release

```text
B2 PASS seul                 = NO-GO PROD
B2 + B3 battery PASS         = candidat canary device (pas prod)
Fleet E2E 10/10 + 20/20
  + Android + iOS + recovery = candidat GO PROD
19/20 éligibles localisés    = NO-GO PROD
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

Le soak 2026-08-18 a **découpé** B3 :

```text
B3-A FGS DURABLE               = PASS ✅  (service vivant 60 min, Unregister=0)
B3-B CAPTURE DEEP DOZE         = PASS ✅  (soak 60 min 2026-08-19, P9=168 pendant Doze)
B3-C BATTERY                   = MESURÉ / à certifier (-6 pts/60 min débranché, T0 écran Awake)
B3-D P9 → TRANSPORT            = FAIL ★
DRAIN AU RÉVEIL                = OPEN ★
```

**Décision produit :** ne pas accepter le batching Doze comme comportement
normal. `setExactAndAllowWhileIdle` n’est pas le moteur GPS. Direction :
delivery FLP native indépendante du réveil JS (PendingIntent → FGS ATMR +
persist `recorded_at` = `Location.time`).

Après correctif natif (rebuild APK) :

```text
PRESENCE
aucune mission
écran éteint
immobile
deep Doze forcé

T0 → 15 min → 30 min → 60 min

mesurer :
- FGS vivant
- owner/session stables
- Unregister = 0
- genuine fixes > 0 pendant Doze (P9 persist)
- event_id / recorded_at progressent (timestamps Android d’origine)
- enqueue (immédiat ou drain au réveil JS, sans réécrire recorded_at)

0 fix pendant 1h45 = FAIL ⛔
```

Ne pas conclure B3 uniquement parce que le GPS continue : `High + distanceInterval:0` doit aussi démontrer une consommation acceptable vs baseline (B3-C, device **débranché**).

## Invariants B2 (D7) — discriminant strict

```text
PRESENCE → LIVE → PRESENCE

Location.stopLocationUpdatesAsync   = 0
TaskService unregister transition   = 0
FGS restart observable              = 0
GPS/native session rotation         = 0
event_id réutilisé/muté             = 0
capture gap anormal                 = 0

nouveaux vrais fixes                = OUI
nouveaux event_id                   = OUI
continuité FGS                      = OUI
```

## Notes appareil

Documenter ici le modèle / OS / build pour chaque run canary.
