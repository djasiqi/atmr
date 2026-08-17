# Dry-run cherry-pick P0 sur base prod `927640a0`

```text
DATE              = 2026-08-15
MODE              = dry-run only (worktree détaché)
BRANCHE RELEASE   = NON créée
TAG / PUSH        = NON
BASE              = 927640a0995a7025edfae3d31802998948a866d5
WORKTREE          = C:\Users\jasiq\atmr.worktrees\gps-p0-dry-run
EVIDENCE          = docs/ops/_release_dry_run_2026-08-15/
VERDICT           = FAIL — G0 composition ROUGE
```

## Ordre tenté

```text
927640a0  (BASE PROD)
+ 479cd60d  P0-A
+ 4cac0fbf  P0-B
+ 88616679  C-LEDGER-CLIENT
+ 5e2b098f  C-LEDGER-SERVER
+ e4adfb06  OBSERVABILITY
```

## Réponses aux 4 questions

### 1. Les 5 cherry-picks passent-ils sans conflit ?

**NON.**

| Commit | Résultat cherry-pick natif |
|--------|----------------------------|
| P0-A `479cd60d` | **CONFLICT** `mobile/.../backgroundLocationTask.ts` |
| P0-B `4cac0fbf` | **CLEAN** |
| C-LEDGER-CLIENT `88616679` | **CONFLICT** `mobile/.../driverTrackingQueue.ts` |
| C-LEDGER-SERVER `5e2b098f` | **CONFLICT** `backend/routes/driver.py` |
| OBSERVABILITY `e4adfb06` | **CLEAN** |

Cause structurelle : chaque tip P0 a un parent **~40–46 commits après** `927640a0`. Les mêmes fichiers ont été touchés entre-temps (`nativeOwner`, recovery FSM, `capture_id` wire, ingress/firewall). Le patch unitaire n’est pas auto-contenu sur la base prod.

Preuve conflit P0-A : `_release_dry_run_2026-08-15/p0a_conflict.diff`.

### 2. Le diff final contient-il UNIQUEMENT le périmètre P0 ?

**NON (sur le composite forcé post-conflit).**

Après résolution expérimentale `git checkout --theirs` (pour inspecter un TIP, **pas** une release) :

- **Chemins** : 63 fichiers, liste nominale ≈ P0 (mobile tracking + ledger server + obs + docs ops). Pas de chemins firewall / P5-B / dispatch / CI opportuniste / alembic.
- **Contenu** : `--theirs` sur fichiers en conflit remplace le fichier **entier** par la version tip du commit source → **fuite de code `capture_id` / ingress** déjà présent dans l’arbre parent du commit, sans amener les modules socle.

Fuites code (hors docs) dans `927640a0..HEAD` :

- `backend/routes/driver.py` → `capture_id` + `build_tracking_ingress_envelope`
- `mobile/.../driverTrackingQueue.ts` → `import { createCaptureId } from "./captureId"` (**module absent** sur le composite)
- `backend/scripts/canary_ledger_server_p0c.py` → champ `capture_id`

→ périmètre **fichier** trompeur ; périmètre **sémantique** contaminé.

### 3. `25ce766952e2` apparaît-elle dans le diff ?

| Contrôle | Résultat |
|----------|----------|
| `git diff --name-only BASE..HEAD \| grep -Ei 'alembic\|migration\|capture_id'` | **aucun path** (attendu pour une release GPS sans migration) |
| Mentions texte `25ce766952e2` / `capture_id` dans le diff | **OUI** (surtout docs readiness/freeze embarqués par OBSERVABILITY + fuites code ci-dessus) |
| Fichier migration Alembic `25ce766952e2_*.py` ajouté | **NON** |

Verdict release : **pas de migration embarquée**, mais **code qui suppose `capture_id` / ingress** sans le socle → pire qu’une migration absente seule.

### 4. Les tests P0 passent-ils sur CE composite exact ?

**NON.**

Composite tip inspecté (détaché, non tagué) : `a32c4d92` (après 3 résolutions `--theirs`).

| Suite | Résultat |
|-------|----------|
| Jest `nativeTrackingLifecycle` (P0-A) | PASS (7) |
| Jest `trackingAuthPresence` (P0-B) | PASS (11) |
| Jest `ledgerClient` | **FAIL** — `Cannot find module './captureId'` |
| Jest `trackingObservability*` | PASS (17) |
| Smoke Docker import `services.tracking.ingress_envelope` | **FAIL** `ModuleNotFoundError` |
| AST `driver.py` | OK (syntaxe) mais runtime dépendances manquantes |

Logs : `_release_dry_run_2026-08-15/jest_*.txt`, `backend_smoke.txt`.

## Contrôles demandés (post dernier pick)

```text
git diff 927640a0..HEAD --stat
  → 63 files, +10316 / -189  (voir _release_dry_run_2026-08-15/diff_stat.txt)

git diff --name-status 927640a0..HEAD
  → _release_dry_run_2026-08-15/diff_name_status.txt

grep alembic|migration|capture_id sur name-only
  → (none)  — _release_dry_run_2026-08-15/migration_paths.txt
```

## Implications gates

```text
G0 release composition     = ROUGE ❌  (conflits natifs ; composite forcé non viable)
G1 migration release       = ROUGE / NO-GO ❌
  → pas de fichier migration, MAIS code capture_id/ingress sans socle
G2 prod topology/skew      = inchangé (audit déjà ✅ expliqué)
G3 N/N-1                   = bloqué (pas de TIP propre)
G4 rollback                = encore ❌
G5 monitoring              = baseline partielle inchangée

BRANCHE release/gps-p0-*   = NO-GO
TAG / DEPLOY / ALEMBIC     = NO-GO
```

## Plan anti-skew

**Non figé.** Condition utilisateur : seulement après dry-run propre. Voir ébauche différée dans [gps-p0-anti-skew-deploy-plan-2026-08-15.md](gps-p0-anti-skew-deploy-plan-2026-08-15.md).

## Suite possible (hors dry-run — pas exécuté)

1. **Backport / packs rebasés** sur `927640a0` (patches reconstruits, pas cherry-pick brut des 5 SHAs firewall-era).
2. Ou **élargir** le set de commits socle **sans** migration `capture_id` (si un sous-graphe existe) — à prouver par un nouveau dry-run.
3. Ou **changer OPTION B** pour inclure explicitement `capture_id` + migration (décision produit, pas ce dry-run).

```text
✅ **Implémenté** : dry-run worktree détaché ; 5 picks tentés ; conflits + fuites --theirs documentés ; greps migration ; tests Jest P0 + smoke import backend.
**Reste à faire** : nouveau composite viable avant création `release/gps-p0-2026-08-15` ; anti-skew deploy seulement après G0 VERT.
```
