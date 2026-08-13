# Gate — Staging observe GPS (PR #92)

Référence ops **figée**. Architecture autorisée pour un staging observe contrôlé ; **rien d’autre**.

Environnement isolé à construire/exécuter : [`gps-staging-environment.md`](./gps-staging-environment.md) (`docker-compose.staging.yml`, projet `atmrstg`). Pas un overlay de la production.

```text
HEAD                           = 26338ec0e0f124bac7b253b067970e08530aec3f
PR                             = #92 (draft, ouverte, non mergée)

PYTEST                         = GREEN ✅
LINT                           = GREEN ✅
ARCHITECTURE REVIEW            = GREEN ✅
MOBILE TRACKING                = GREEN ✅
E2E / MIGRATIONS               = GREEN ✅
CODEQL / BANDIT / SEMGREP      = GREEN ✅
GITGUARDIAN                    = RED ❌  (historique PR / commit 8398a4b)

STAGING OBSERVE                = autorisé ✅
READY_TO_MERGE_MAIN            = NO ❌
READY_FOR_MOBILE_CANARY        = NO ❌
READY_FOR_PRODUCTION           = NO ❌
READY_FOR_FANOUT               = NO ❌
READY_FOR_ENFORCE / STRICT     = NO ❌
```

Ne pas lire « sécurité verte » : CodeQL / Bandit / Semgrep sont verts, **GitGuardian reste rouge**.

---

## Objectif

Observer le comportement du firewall **sans modifier** l’éligibilité live / canonical.

`TRACKING_MISSION_FIREWALL_MODE=observe` :

- produit une décision (`would_block`, `reason`) + métriques ;
- **ne coupe pas** `live_eligible` ni `canonical_eligible`.

Ce n’est **pas** un changement de variable sur l’image actuelle de `main`. `MODE` n’existe pas sur `main`.

---

## Conditions de l’étape

- Branche `26338ec0` **déployée** sur staging (image de cette PR).
- `TRACKING_MISSION_FIREWALL_MODE=observe` sur **l’API**.
- `TRACKING_MISSION_FIREWALL_MODE=observe` sur **le consumer Kafka**.
- Aucun `enforce_mission` / `strict`.
- Aucun fanout supplémentaire.
- Aucun canary mobile.
- Aucun merge `main`.

Toujours-on dans cette image (indépendant du MODE, vs `main`) :

1. P1 — `sanitize_fanout_mission_id` : fanout `mission_id` = SINGLE, sinon `None`.
2. P7 watchdog — mismatch canonical / authoritative → unhealthy / kick possible.
3. Lectures Booking / resolver **par point GPS**.

---

## Rollback

```text
ROLLBACK = IMAGE PRÉCÉDENTE
MODE=off ≠ rollback
```

`MODE=off` n’annule ni P1 ni le watchdog P7.

---

## Suivi staging (4 signaux)

Capturer un **baseline avant déploiement**, puis comparer après.

### 1. Firewall — `would_block` n’est pas un échec

`would_block > 0` est **attendu** : c’est ce que observe doit révéler.

Suivre :

```text
taux would_block / total
répartition par reason
répartition par transport
évolution dans le temps
```

Métrique : `tracking_mission_firewall_total` (labels `mode`, `reason`, `would_block`, `enforced`, `transport`).

En observe : `enforced="0"` même si `would_block="1"`.

Exemples :

- ~2 % de `stale_mission` légitimes → découverte utile, pas un échec.
- ~60 % de `AMBIGUOUS` → probablement **NO-GO** pour la suite.

Ne pas juger observe uniquement par « aucun incident visible ». L’objectif est aussi de **produire assez de cas `would_block`** pour montrer que le resolver / firewall classe correctement les situations réelles. Quasi aucun événement observable = **preuve insuffisante** pour `enforce` (ce n’est pas un GO).

### 2. Watchdog

Kicks sans anomalie vs baseline. Un spike de kicks mismatch / AMBIGUOUS non expliqué par le métier → NO-GO.

### 3. `mission_id` fanout

Cohérent avec P1 : SINGLE → id authoritative ; AMBIGUOUS / NONE → `None`. Pas de pick silencieux d’une mission ambiguë.

### 4. Pression DB / Kafka — avant / après déploiement

```text
avant/après déploiement :
- p50 / p95 / p99 latence resolver / Booking
- pool DB utilisé / saturation / waits
- débit consumer
- consumer lag
- taux d'erreur / timeout DB
```

Si le taux `would_block` est lisible mais que le consumer accumule du lag ou que le pool sature → **NO-GO** pour la suite.

---

## Sortie observe

GO (vers une étape **ultérieure**, jamais directement prod / enforce depuis ce document) uniquement si **les quatre** sont vrais :

1. `would_block` expliqué, compatible avec le métier, **et** volume suffisant pour être probant.
2. Watchdog sans kicks anormaux vs baseline.
3. `mission_id` fanout cohérent.
4. Aucune dégradation DB / Kafka significative vs baseline.

Sinon : rester en observe, ou rollback image précédente.

Interdit tant que cette sortie n’est pas GO **et** qu’une revue dédiée ne l’autorise pas :

- merge `main`
- `enforce_mission` / `strict`
- canary mobile
- fanout supplémentaire
- production
