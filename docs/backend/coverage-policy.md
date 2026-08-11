# Politique de couverture backend (ATMR / LIRIE)

## Responsabilités

| Composant | Rôle |
|-----------|------|
| [`backend/.coveragerc`](../../backend/.coveragerc) | **Seule** politique de mesure (omits, exclude_lines) |
| `pytest … --cov=backend --cov-config=backend/.coveragerc` | Produit `coverage.xml` |
| [`backend/scripts/check_coverage.py`](../../backend/scripts/check_coverage.py) | Analyse le XML et applique les **gates** (ne redéfinit jamais les omits) |

Pour un même `coverage.xml`, `check_coverage.py` ne modifie pas le périmètre mesuré.

`pytest.ini` ne contient **pas** de sections `[coverage:*]` (Coverage.py ne les lit pas comme config standard).

## Seuils et rôles

| Notion | Valeur | Rôle |
|--------|--------|------|
| Mesure | `--cov=backend` + `.coveragerc` | périmètre backend complet selon omits |
| Baseline CI | **50,0 %** | gate bloquante anti-régression |
| Mesure observée 11.08.2026 | **≈50,86 %** | point de départ (run CI) |
| Objectif global | **70,0 %** | cible progressive (affichage / `DEFAULT_REPORT_THRESHOLD`) |
| Modules critiques | 80–95 % selon module | rapport pour l’instant |
| `--require-critical` | désactivé | promotion ultérieure |

### Ratchet manuel

```text
50 → 55 → 60 → 65 → 70 %
```

Règle :

> Le seuil peut être **relevé** lorsque la couverture réelle le permet.
> Il ne doit **pas** être abaissé pour faire passer une CI en échec sans décision documentée ici.

Nuance : un plancher rond à `50,0 %` laisse une marge mineure sous la mesure observée (~0,86 pt). Ne pas fixer `fail-under` à la valeur exacte d’une run (fragile).

## Contrat CLI (`check_coverage.py`)

```text
aucun flag                         → rapport seul, exit 0
--fail-under 50                    → gate GLOBAL (plancher CI actuel)
--require-critical                 → gate CRITIQUES uniquement
--fail-under 50 --require-critical → les deux
```

`--fail-under` a pour défaut `None` (non bloquant). Sans flag, le rapport utilise un référentiel d’affichage de **70 %** (objectif), distinct du plancher CI.

## Catégories d’omit autorisées

Toute ligne `omit` dans `.coveragerc` doit relever de :

| Catégorie | Exemples |
|-----------|----------|
| `legacy` | `services/rl/*` (shim post-migration) |
| `training` | `hyperparameter_tuner.py`, `optimal_hyperparameters.py` |
| `tooling` | `services/demo/seed_service.py`, `seed_spec.py` |
| `demo` | réservé ; **pas** d’omit global de `services/demo` tant que des helpers sont sur le chemin auth/contact/API |

### Interdit

- `*/services/ml/*` global (le runtime RL API reste mesuré)
- omit `unified_dispatch`, auth, bookings, billing, tracking produit
- omit d’un module reachable depuis une route de production sans justification documentée ici
- élargir les omits uniquement pour « gonfler » le pourcentage global

## Classification ML / RL

```text
services/ml/
├── runtime produit     → mesuré (+ candidats critiques)
├── training            → omit possible
└── tooling             → omit possible

services/rl/*           → legacy → omit
```

| Zone | Catégorie | Mesure |
|------|-----------|--------|
| `routes/dispatch/rl_helpers.py` | runtime | oui ; critique ≥80 % |
| `services/ml/rl/suggestion_generator.py` | runtime | oui ; critique ≥80 % |
| Chaîne inference utilisée par le générateur | runtime | oui |
| `hyperparameter_tuner.py`, `optimal_hyperparameters.py` | training/tooling | omit |
| `services/rl/*` | legacy | omit |
| `services/demo/seed_*` | tooling | omit |
| `services/demo/access_service.py`, scoring, soft_delete_guard | runtime (auth/contact/institutions) | mesuré |

Les seuils critiques suivent le **chemin d’exécution produit**, pas l’ancien `services/rl/dispatch_env.py`.

## Gates CI

1. **Check global coverage >= baseline (50%)** — `check_coverage.py --fail-under 50.0` (bloquant, anti-régression).
2. **Report critical module coverage** — rapport non bloquant + plancher partiel `pytest --cov-fail-under=80` sur heuristics/solver/autonomous_manager (≠ cibles 95 % du script).
3. **Promotion** — step dédié `check_coverage.py --require-critical` lorsque la baseline critique le permet (sans `|| true`).

## Baseline historique

| Date | Source | Global % | Notes |
|------|--------|----------|-------|
| 2026-08-10 | CI pré-lot (réf.) | 63.95 % | Avant omits ciblés / nouvel instrument |
| 2026-08-10 | Estimation post-omits | ~64.6 % | Omits tuner / optimal_hp / seed_* / shim `rl` |
| 2026-08-11 | CI run #533 | ≈50,86 % | Mesure honnête post-instrument ; dette ~117 modules à 0 % |
| 2026-08-11 | **Décision** | plancher **50,0 %** | Objectif 70 % conservé ; pas d’omit produit pour passer |

**Décision :** le plancher immédiat est **`fail_under = 50.0`**. L’objectif **70 %** reste la cible progressive. La remontée se fait par **tests métier** et ratchet manuel, pas par baisse de seuil ni omits produit.

**Validation CLI :**

- sans flag → exit 0 (rapport, affichage vs 70 %)
- `--fail-under 50` → exit 1 si global &lt; 50
- `--require-critical` → exit 1 si critiques sous seuil (non activé en CI tant que non promu)

## Jalons

### Couverture globale (ratchet)

1. Plancher 50 % (étape actuelle)
2. Relever à 55 / 60 / 65 quand la mesure CI le permet durablement
3. Atteindre l’objectif 70 %

### Modules critiques

1. Rapport honnête (étape actuelle)
2. Vague tests priorité risque : db / tracking-GPS / auth / bookings / payments / dispatch
3. Activer `--require-critical` sur allowlist puis sur l’ensemble des `CRITICAL_MODULES`
