# P0-D D3 — timeline native LocationTaskService (2026-08-16)

```text
BINARY     = Prod 126 (universal, non-debuggable)
DEVICE     = Samsung SM-S911B
CYCLE      = force-stop → TOP healthy → HOME 90s
PATCH      = NO-GO
```

Versions : `BUILD_126_VERSIONS.md`  
Artefacts : `docs/ops/_release_exec_p0d_2026-08-16/d3_native/`

## Versions figées (build 126)

| Package | Version |
|---------|---------|
| expo | 54.0.35 |
| expo-location | **19.0.8** |
| expo-task-manager | 14.0.9 |
| expo-modules-core | 3.0.30 |
| react-native | 0.81.5 |
| targetSdk | **36** |

Issue upstream [#47595](https://github.com/expo/expo/issues/47595) = SDK **56** / trigger update — zone native proche, **pas** preuve d’identité.

---

## Timeline Prod (cycle propre)

| T | `isForeground` | `startRequested` | `getFgsAllowStart` | `Location unavailable` | LOC prod |
|---|----------------:|-----------------:|-------------------:|-----------------------:|---------:|
| T0 TOP | true | true | TOP | — | OK |
| HOME+2…+25s | true | true | TOP | **oui** (~10s cadence) | OK jusqu’~+40s |
| HOME+30…+90s | **true** | **true** | **TOP** | **oui** (continue) | **STOP** après ~12:38:20 UTC |
| AM stopSelf/onDestroy LocationTask | — | — | — | **aucun** | — |

### Discriminants D3

```text
D3-B (Android Killing/Stopping service sans Expo)
  → NON observé sur ce cycle propre (90s)

D3-A (consumer détache / perd le service)
  → PARTIEL : warning LocationTaskConsumer répété
    mais mService OS reste isForeground=true

D3-C (service vivant, état/delivery cassé)
  → LEADING sur ce cycle :
    OS FGS = healthy
    LOC delivery = morte après ~40s HOME
    "Location unavailable for foreground-service task delivery" = présent

D3-D (lifecycle release-only)
  → PARTIAL : warning aussi sur Dev ; suite (LOC morte / DENIED) release-worse
  → voir D3_DEVCLIENT_COMPARE.md
```

### Nuance vs cycle D123 précédent

Le cycle `d123_lifecycle` avait vu `startRequested=false` / DENIED vers HOME+30s (état déjà thrashé + GrantPermissions).  
Le cycle `d3_native` (force-stop clean) **ne reproduit pas** le collapse OS en 90s, mais reproduit :

1. le warning Expo dès ~HOME+8s  
2. l’arrêt des LOC alors que le FGS OS paraît encore sain  

→ le smoking gun se déplace vers **livraison LocationTaskConsumer / fused location sous FGS**, pas uniquement vers `getFgsAllowStart=DENIED` (qui reste la **conséquence** quand le service finit par tomber / qu’on tente de le recréer).

```text
FGS OS shell healthy
+ Location unavailable (répété)
+ LOC = 0
= défaillance fonctionnelle D3-C
→ plus tard éventuel startRequested=false / DENIED (cycle empoisonné)
```

---

## Statut

```text
D1  essentially excluded
D2  RULED OUT ✅
D3  LEADING ▶
  D3-C  LEADING (ce cycle)
  D3-A  contributif (warning consumer)
  D3-B  not seen (clean 90s)
  D3-D  PARTIAL ✅ (Dev : warning OUI + FGS/LOC OK)

P0-D PATCH           = NO-GO
GENERAL DISTRIBUTION = NO-GO
```

## Suite

1. ~~Dev Client compare autour de `Location unavailable`~~ → **DONE** (`D3_DEVCLIENT_COMPARE.md`) : OUI mais FGS/LOC restent.
2. Lire le site exact du log dans `expo-location@19.0.8` (string absente du tree sdk-54 GitHub brut — vérifier AAR/patch).
3. Capturer le flip Prod `startRequested true→false` s’il apparaît (soak >90s ou path empoisonné) — smoking gun AM.
4. Ne pas patcher ; ne pas whitelister batterie.
