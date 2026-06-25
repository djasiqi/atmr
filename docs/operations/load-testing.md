# Campagne de charge — simulateur flotte GPS

Outil : `backend/tools/fleet_gps_simulator/` (CLI Docker).

## Paliers officiels

10 → 50 → 100 → 250 → 500 → 1000 drivers

Chaque palier : 15 min stabilisation, mesure p95/p99 E2E.

## Scénarios

- Trajectoire normale (vitesse variable, arrêts)
- Perte GPS 5 min (tunnel)
- Mode avion / switch 4G-Wi-Fi
- Batterie faible
- Background Android (FGS mission_live)

## Matrice compatibilité OEM / OS

| Profil | OS | Cadence watch | Timeout position | Notes |
|--------|-----|---------------|------------------|-------|
| Samsung One UI | Android 13+ | 8 s | 7 s | FGS kill agressif — self-heal critique |
| Google Pixel | Android 14+ | 8 s | 7 s | Référence |
| Xiaomi MIUI | Android 12+ | 10 s | 8 s | Battery opt fréquent |
| iPhone | iOS 17+ | 8 s | 7 s | Live Activity mission |

Paramètres simulateur : `--oem-profile samsung|pixel|iphone|xiaomi`

## Métriques par palier

- CPU/RAM backend consumers
- Kafka lag max par partition
- Redis write latency p95
- Socket emit latency
- E2E `recorded_at → frontend` p95/p99

## Gate migration S3→S4

Palier 1000 : E2E p95 < 1 s, 0 alerte `TrackingPipelineStageSLOBreach` pendant 15 min.

## CI smoke

Job architecture-review : simulateur 10 drivers smoke (optionnel `--drivers 10 --duration 60`).
