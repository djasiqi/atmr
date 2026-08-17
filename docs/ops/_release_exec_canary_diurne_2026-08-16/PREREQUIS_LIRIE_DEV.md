# Prérequis canary Lirie Dev (avant FG/BG/lock)

```text
STATUT SETUP (2026-08-16 ~11:45 Genève)
  Metro              = RUNNING tip 286737a2 (worktree gps-p0-g3-nn1)
  Port               = 8081
  ADB USB            = RFCW20QC53W ✅
  ADB Tailscale      = 100.81.106.54:5555 ✅
  reverse 8081       = ✅
  reverse 15100      = ✅
  Staging gateway    = http://127.0.0.1:15100 healthy ✅
  Deep link          = lirie://expo-development-client/?url=http://127.0.0.1:8081
```

## Ce qui manquait tout à l’heure

`am start …MainActivity` ouvre le shell natif **sans** bundle JS Metro → écran figé / rien ne se passe en Lirie Dev.

Il faut **dans cet ordre** :

1. Staging canary joignable (`15100`, déjà OK)
2. Metro `--dev-client` depuis tip **`286737a2`**
3. `adb reverse tcp:8081` + `tcp:15100`
4. Ouvrir le deep link Expo Dev Client (pas seulement MainActivity)
5. Login chauffeur **staging** + mission tracking active
6. Ensuite seulement : FG → HOME/BG → lock → déplacement

## Cible API (important pour CLOSED prod)

| Mode | API | Ferme POST-DEPLOY PROD ? |
|------|-----|--------------------------|
| **Lirie Dev actuel** | staging `127.0.0.1:15100` (via reverse) | **Non** — valide le JS tip + pipeline staging |
| **Close prod** | `https://api.lirie.ch` (session prod) | **Oui** — LOC persistées côté prod |

Le setup Metro ci-dessus = **canary tip sur staging**.  
Pour **POST-DEPLOY VALIDATED / CLOSED** sur la release prod, il faudra ensuite un passage LOC sur **api.lirie.ch** (build/session prod), sans changer le runtime serveur.

## Actions utilisateur maintenant

```text
1. Vérifier que l’app charge le bundle (pas d’écran rouge Metro)
2. Se connecter chauffeur sur STAGING
3. Démarrer / rejoindre une mission avec tracking
4. Confirmer ici « prêt canary » → on rejoue FG/BG/lock + snaps
```
