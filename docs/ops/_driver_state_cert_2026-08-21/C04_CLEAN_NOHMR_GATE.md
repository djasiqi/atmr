# C04 — Gate clean cold / no-HMR

```text
OBJECTIF = distinguer défaut produit vs contamination Metro HMR
PATCH    = AUCUN
```

## Run `20260821_203344` — PASS ✅

```text
PID=11749 stable · HMR=0 · keyguard=0
P8=J1=J7=8 (POST_HOME 5/5/5)
P9≈21.5s · mission_live only
PG 8/8 MATCH · projection avance
Unregister soak=0 · FLP_REMOVE=0
```

Détail : `C04_CLEAN_NOHMR_PASS.md`

## Run `20260821_200803` — INVALID ⚠

Keyguard + zombie stop sans re-arm — non discriminant. Voir historique.

## Requalification

```text
ancien C04 FAIL (HMR process) = TEST ENV CONTAMINATION
C04 produit clean             = PASS
NEXT                          = C05
```
