# Modèle GPS figé — POSITION vs PRÉSENCE (mission active)

```text
STATUT     = FIGÉ (invariant produit / technique)
DATE       = 2026-08-16
SCOPE      = mission active (IN_PROGRESS / mission_live)
PATCH      = hors scope de ce document (modèle seulement)
```

Liens : [pipeline](gps-tracking-pipeline.md) · [UI présence P0-F](gps-p0f-ui-fleet-presence.md) · [D5-B delivery](_release_exec_p0d_2026-08-16/d5_task_chain/D5_TASK_CHAIN_RCA.md)

---

## Invariant

**Un chauffeur immobile n’est pas un chauffeur stale.**

Si une mission est en cours et que le téléphone continue à fournir des fixes récents — **même lat/lon identiques**, vitesse 0, distance 0 — LIRIE doit considérer le chauffeur comme **connecté** et la position comme **fraîche**.

```text
MISSION EN COURS
+
fix GPS reçu récemment
+
timestamp Location récent
=
CHAUFFEUR CONNECTÉ ✅
```

Même si :

```text
lat/lon identiques pendant 5, 10 ou 30 minutes
vitesse = 0
distance parcourue = 0
```

→ signal de **présence** obligatoire.

---

## Deux mécanismes séparés

| Axe | Source de vérité | Signifie |
|-----|------------------|----------|
| **POSITION** | lat/lon du dernier fix valide | Où est le véhicule sur la carte |
| **PRÉSENCE GPS** | timestamp du dernier fix valide **reçu** | Le téléphone livre encore des fixes |

```text
stale = now - location.timestamp > seuil
```

**Interdit :**

```text
stale = position identique à la précédente
driver_connected = last_coordinates_changed_recently
```

**Correct (carte / flotte) :**

```text
driver_connected =
  mission_active
  AND last_valid_fix_age < seuil
```

Exemple :

```text
19:00:00  46.2044 / 6.1432
19:00:20  46.2044 / 6.1432
19:00:40  46.2044 / 6.1432
19:01:00  46.2044 / 6.1432
```

À 19:01 le chauffeur doit apparaître :

```text
GPS connecté ✅
Dernière position : maintenant
État : immobile
```

et **non** :

```text
GPS stale ❌
dernière connexion : 19:00
```

---

## Règle d’ingestion (mission active)

Pendant `IN_PROGRESS` / `mission_live`, un nouveau fix valide **doit pouvoir rafraîchir la présence** même s’il est géographiquement identique au précédent.

Déduplication volume **autorisée** pour l’écriture spatiale complète, **jamais** au détriment du heartbeat :

```text
fix reçu
↓
timestamp frais
↓
update last_seen / last_location_at  TOUJOURS

coordonnées changées ?
├─ oui → nouvelle position complète (lat/lon + meta)
└─ non → heartbeat présence / refresh freshness
         (même sans nouvelle ligne « déplacement »)
```

Cible opérationnelle : heartbeat backend **~20–30 s** en mission active, y compris à l’arrêt, pour distinguer :

```text
stationné + téléphone vivant ✅
aucune donnée depuis N minutes ❌
```

Ne **jamais** recycler une ancienne position comme preuve de connexion actuelle.

---

## Alignement UI existant (P0-F)

La machine `resolveDriverLocationPresence` vieillit déjà sur **l’âge** (`recorded_at` / `timestamp` / `last_seen_seconds`), pas sur le delta lat/lon — seuils `live ≤ 30 s` / `recent ≤ 120 s`.

Ce document fige le même principe **amont** (delivery native → enqueue → persist → projection).

---

## Écart observé (D5-B) — ne pas confondre avec le modèle

Sur Prod 126 (stationnaire / appart) :

```text
FLP voit des fixes
→ FusedLocation bloque delivery (too close / too fast)
→ LocationTaskConsumer : Location unavailable
→ 0× background-location-task Finished
→ 0 heartbeat position côté app
→ présence UI devient stale (âge du dernier fix réel)
```

```text
POSITION immobile
≠
GPS indisponible
≠
chauffeur déconnecté
```

**Interdit** : « corriger » en prolongeant artificiellement `connected=true` / P0-F UI sur l’ancienne position — cela **masquerait D5-B**.

```text
coords identiques + fix récent     → connecté ✅
coords identiques + aucun nouveau fix → stale ❌
```

Mock téléports (D5 mock) : Fused **peut** changer de coords, Expo **ne livre toujours rien** → affaiblit « Android ne livre rien seulement parce que le téléphone ne bouge pas ». Frontière = **filtrage / delivery** vers la request Expo (`too close` / `too fast` / `unavailable`).

```text
MODÈLE FIGÉ              = CLOSED / CORRECT ✅
"stale = immobile"       = RULED OUT ✅
D5-B FLP→Expo delivery   = LEADING ★
MOCK as workaround       = RULED OUT ✅
PATCH UI fake-connected  = NO-GO
PATCH runtime            = NO-GO
```

Cible : **nouveau heartbeat/fix effectivement livré**, même à coords identiques — pas un `connected` cosmétique.

---

## Hors scope immédiat

- Patch mobile / backend / fake `connected` UI
- Rouvrir A/B/ledger

**Suite discriminant** : A/B stationnaire Prod126 vs Dev Client — [d5_ab_stationary/D5_AB_STATIONARY.md](_release_exec_p0d_2026-08-16/d5_ab_stationary/D5_AB_STATIONARY.md).

