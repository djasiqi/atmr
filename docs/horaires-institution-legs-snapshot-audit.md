# STOP GATE audit — `legs_snapshot`

## Contexte

`legs_snapshot()` dans `backend/services/institutions/transport_request_legs_service.py` produit des snapshots `before` / `after` lors d'une réorganisation multi-étapes (`reorganize_multi_stop_legs`).

## Usages identifiés

| Usage | Fichier | Rôle |
|-------|---------|------|
| Comparaison `before != after` | `transport_request_legs_service.py` | Détecte un changement de parcours |
| Stockage timeline | `TransportTimelineEvent.payload` | `before_legs`, `after_legs` — preuve d'audit immuable |
| Libellé timeline backend | `transport_timeline_service.py` | Nombre d'étapes uniquement (`len(after_legs)`) |
| Libellé timeline frontend | `institutionTimelineDisplay.js` | « Parcours modifié » — pas d'affichage de `scheduled_time` |

## Format actuel

```python
"scheduled_time": leg.scheduled_time.isoformat()  # timestamptz → peut produire +00:00
```

## Risque d'une normalisation vers ISO naïf

`2026-06-16T12:30:00+00:00` et `2026-06-16T12:30:00` n'ont pas la même sémantique pour une preuve historique. Modifier le format sans décision explicite peut rompre l'intégrité d'interprétation de l'audit existant.

## Décision (Phase 1)

**Ne pas modifier `legs_snapshot` dans P1-A / P1-B / P1-C.**

- L'affichage utilisateur ne consomme pas les `scheduled_time` des snapshots.
- Les horaires mission visibles passent par `mission_scheduled_to_api_iso` sur les endpoints API principaux.

## Si correction future validée

1. Normaliser via `mission_scheduled_to_api_iso` **uniquement pour les nouveaux** événements timeline.
2. Conserver l'historique existant tel quel (format `+00:00`).
3. Documenter la rupture de format dans les release notes.
4. Ajouter un test unitaire sur `legs_snapshot` avant merge.

## Questions résolues

| Question | Réponse |
|----------|---------|
| Reconstruction historique ? | Non — comparaison + preuve d'audit uniquement |
| Affichage frontend des horaires snapshot ? | Non |
| Migration historique ? | Non recommandée |
