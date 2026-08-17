# Canary GPS réel — Android + iOS (après STAGING_P5B_FINAL)

## Statut amont (figé)

```text
STAGING_P5B_FINAL = PASS ✅
IMAGE             = sha-d5694d8e7cec
OCI               = d5694d8e7cec190978098db6eb20f242226784a8
P5-B synthétique  = FERMÉ — ne pas rejouer sauf régression
```

## Interdits pendant ce canary

```text
MERGE MAIN        = NO
PRODUCTION        = NO
ENFORCE           = NO
FANOUT processed  = NO
```

## Configuration prudente (staging)

```text
TRACKING_MISSION_FIREWALL_MODE=observe
TRACKING_PG_FIRST_CANONICAL_ENABLED=true
TRACKING_PERSIST_WITH_OUTBOX=true
SOCKET_GPS_INGEST_ENABLED=true
TRACKING_SOCKET_KAFKA_MIRROR_ENABLED=false
TRACKING_PROCESSED_FANOUT_ENABLED=false
```

Vérifier backend **et** consumer sur la même image `sha-d5694d8e7cec`.

## Build mobile Android (prérequis)

Avant le terrain Samsung : voir [gps-android-canary-apk.md](gps-android-canary-apk.md).

- Ne pas utiliser EAS `preview` / `production` (prod `api.lirie.ch`).
- Sur SHA exact `d5694d8` : APK **dev client** + Metro (`__DEV__`) vers staging LAN.
- Vérifier hors `api.lirie.ch` **avant** de marcher.

- **2 à 5 téléphones max**
- Minimum : **1 Android** + **1 iPhone**
- Idéal : **2 chauffeurs simultanés** (carte multi-driver)
- Comptes / missions **réels staging** (pas fixtures `p5b-*` / `@staging.invalid` synthétiques du harness)

## Scénarios terrain (obligatoires)

| ID | Scénario | Critère PASS |
|----|----------|--------------|
| C1 | Démarrage | Login → GPS démarre → position visible → bon chauffeur / bonne mission |
| C2 | Multi-driver | Android + iOS actifs → 2 marqueurs → positions indépendantes → aucun disparu |
| C3 | Mission | Prise → déplacement → changement d’état → fin → pas d’ancienne mission réutilisée |
| C4 | Réseau | Coupure Wi-Fi/4G → buffer local → reprise → file vide → pas de doublons visibles |
| C5 | Arrière-plan | Écran verrouillé / autre app → tracking conforme permissions |
| C6 | Restart | Force-stop → rouvrir → session OK → pas de perte définitive SQLite |
| C7 | GPS réel | Déplacement physique → carte suit → pas de fantôme / régression brutale |
| C8 | capture_id | Retransmission HTTP/socket/retry → même `capture_id` → pas de doublon métier |

**Critère visuel n°1** : si deux chauffeurs roulent, la carte montre les deux au bon endroit et continue à les suivre sans intervention manuelle.

## Surveillance backend (pendant le run)

Capturer avant / pendant / après :

```bash
bash scripts/staging/capture_canary_metrics.sh staging/output/canary-$(date -u +%Y%m%dT%H%M%SZ).txt
```

Seuils :

| Signal | Attendu |
|--------|---------|
| UniqueViolation GPS | 0 |
| HTTP 429 sustained | 0 |
| Kafka lag durable | 0 |
| DLQ GPS | 0 |
| critical GPS crash | 0 |
| false-live stale mission | 0 |
| canonical sans preuve PG | 0 |
| session bloquée | 0 |
| perte SQLite définitive | 0 |

## Grille de rapport (à remplir après terrain)

```text
GPS REAL CANARY — d5694d8

ANDROID
connect/login                  = PASS/FAIL
foreground GPS                 = PASS/FAIL
background GPS                 = PASS/FAIL
network recovery               = PASS/FAIL
restart recovery               = PASS/FAIL
map coherence                  = PASS/FAIL

iOS
connect/login                  = PASS/FAIL
foreground GPS                 = PASS/FAIL
background GPS                 = PASS/FAIL
network recovery               = PASS/FAIL
restart recovery               = PASS/FAIL
map coherence                  = PASS/FAIL

MULTI-DRIVER
2+ simultaneous markers        = PASS/FAIL
movement coherence             = PASS/FAIL
mission transitions            = PASS/FAIL

BACKEND
UniqueViolation                = 0 / ...
429 sustained                  = 0 / ...
Kafka lag                      = ...
DLQ                            = ...
critical crashes               = 0 / ...
false-live                     = 0 / ...
canonical_without_PG_proof     = 0 / ...

CANARY_GPS_FINAL               = PASS/FAIL
```

Si `CANARY_GPS_FINAL = PASS` → seule décision suivante possible : **GO MERGE MAIN** (puis prod prudent, **sans** enforce/fanout brutal).

## Rôle agent vs terrain

- L’agent peut : figer la config staging, capturer les métriques, recevoir / consolider le rapport.
- Le terrain (téléphones, déplacement, missions) est **obligatoirement** exécuté par des opérateurs humains.

## Correctifs hors canary (noter, ne pas bloquer)

- Circuit Redis `opened_at=null` (P7)
- Métrique Prometheus `persisted` → label `failed` (observabilité)
