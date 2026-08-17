# RELEASE REVIEW — Binaire 132 (RC D5)

```text
DATE              = 2026-08-17
REVIEWER          = agent (release review)
BINARY            = 1.0.11 / versionCode 132
EAS BUILD         = ab91958e-d00d-4d40-8304-128f0a7065f0
ARTIFACT          = https://expo.dev/artifacts/eas/_YpZ7mZ4KywK4ySN2eLa1wgAodK9HIzfsFUNf8xTZAA.aab
PROFILE           = production (AAB, autoIncrement)
CHANNEL EAS       = production
DISTRIBUTION EAS  = STORE (AAB)
gitCommitHash     = a851cf1520bc2ed66f9e5f7b0acb6114cf1ff133 (SHA FINAL S)
TAG               = d5-rc-final
QA_PANEL          = ABSENT ✅
CANARY C1–C4      = VALIDATED ✅ (binaire 131 QA ; même train D5+W1+B3)
SMOKE RC132       = PASS ✅
GPS UI freshness  = HORS SCOPE (chantier séparé)
```

---

## Blockers 131 → statut 132

| ID | Sujet | Statut |
|----|--------|--------|
| B1 | Freeze SHA git = contenu D5 | ✅ FIXED — S + tag `d5-rc-final` |
| B2 | AAB production QA OFF | ✅ FIXED — build 132 / SHA=S / smoke PASS |
| B3 | tsc télémétrie D5 | ✅ FIXED |

Réf. review 131 (historique) : `RELEASE_REVIEW_131.md`  
Réf. smoke : `RC132_SMOKE_summary.txt`

---

## Smoke RC132 (gates)

```text
Unregister inattendu     = 0
Register↔Unregister storm = 0
Finished                 = continue
PUT location             = continue (6–32 /~60s)
LOC fenêtre smoke        = 14
FGS fin + post           = alive
crash/ANR                = 0
```

---

## Verdict GO / NO-GO

```text
D5 RCA              = CLOSED ✅
PATCH               = VALIDATED ✅
CANARY              = VALIDATED ✅
SHA FINAL S         = FIXED ✅
AAB 132 QA OFF      = FIXED ✅
SMOKE RC132         = PASS ✅
RELEASE REVIEW      = PASS ✅
RC132               = VALIDATED ✅

→ PLAY SUBMISSION / DISTRIBUTION = HOLD ⛔
RC132 = FROZEN ✅ — chantier UI = P0-E (séparé)
```

### Autorisé maintenant

- HOLD Play 132 (décision rollout différée)
- Ouvrir / avancer **P0-E** (confirmation / freshness) sans toucher D5
- Monitoring métriques D5 (Unregister, storm, FGS, PUT/LOC, crash)

### Interdit

- Modifier RC132 / rouvrir C1–C4 / commit sur `d5-rc-final` pour du UI
- Submit / promote Play sans GO explicite
- OTA channel production inject
- Mélanger fix P0-E dans le binaire 132 (→ build 133 si mobile)

### Rollback

| Item | Valeur |
|------|--------|
| Cible | **versionCode 126** |
| Déclencheurs | Unregister ↑ ; storm ; FGS mort sous mission ; PUT/LOC drop ; crash/ANR ; FGS start not allowed |
| Qui | Owner produit / ops (humain) |
