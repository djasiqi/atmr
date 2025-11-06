# Vérification des logs - Chauffeur préféré

## Messages clés à rechercher dans les logs backend

### 1. Configuration initiale (data.py)

```
[Dispatch] 🔍 Drivers disponibles (...) (vérification preferred_driver_id)
[Dispatch] 🔍 Valeur brute preferred_driver_id: 2 (type: int)
[Dispatch] 🔍 preferred_driver_id converti: 2 (type: int), driver_ids: [3, 2, 4, 1]
[Dispatch] 🎯 Chauffeur préféré CONFIGURÉ: ID=2 (type: int) - sera priorisé avec bonus +3.0
```

### 2. Entrée dans heuristics.assign() (heuristics.py)

```
[HEURISTIC] 🎯 assign() entry: preferred_driver_id=2, bookings=17, drivers=4
[HEURISTIC] 🎯 Chauffeur préféré 2 dans drivers disponibles: True
[HEURISTIC] 🎯 Chauffeur préféré détecté dans le problème: 2
```

### 3. Application du bonus (heuristics.py)

```
[HEURISTIC] 🎯 Bonus préférence FORT appliqué pour chauffeur #2 (+3.0) booking_id=XXXX
```

### 4. Sélection du chauffeur préféré (heuristics.py)

```
[HEURISTIC] ✅ Booking #XXXX → Chauffeur préféré #2 (score: X.XX, reason: preferred_bonus)
```

### 5. Fallback closest_feasible (heuristics.py)

```
[FALLBACK] 🎯 Chauffeur préféré détecté: 2 - bonus +3.0 sera appliqué
```

## Commandes pour vérifier les logs

### Option 1: Docker logs (Windows PowerShell)

```powershell
# Filtrer les logs du celery-worker pour le chauffeur préféré
docker logs celery-worker --tail 1000 2>&1 | Select-String -Pattern "preferred_driver|🎯|Chauffeur préféré|assign\(\) entry" | Select-Object -Last 50

# Filtrer pour un dispatch_run_id spécifique (334)
docker logs celery-worker --tail 2000 2>&1 | Select-String -Pattern "dispatch_run_id=334|DispatchRun id=334" -Context 0,30 | Select-Object -Last 100

# Voir tous les logs récents avec préféré
docker logs celery-worker --since 10m 2>&1 | Select-String -Pattern "preferred" -Context 2,2
```

### Option 2: Docker logs (Linux/Mac)

```bash
# Filtrer les logs du celery-worker
docker logs celery-worker --tail 1000 2>&1 | grep -i "preferred_driver\|🎯\|Chauffeur préféré" | tail -50

# Filtrer pour un dispatch_run_id spécifique
docker logs celery-worker --tail 2000 2>&1 | grep -A 30 -B 5 "dispatch_run_id=334\|DispatchRun id=334" | tail -100
```

### Option 3: Logs en temps réel

```powershell
# Suivre les logs en temps réel
docker logs -f celery-worker 2>&1 | Select-String -Pattern "preferred_driver|🎯"
```

## Checklist de vérification

- [ ] ✅ `preferred_driver_id=2` est détecté dans les overrides
- [ ] ✅ Le chauffeur #2 est dans la liste des drivers disponibles
- [ ] ✅ Le bonus +3.0 est appliqué dans `_score_driver_for_booking()`
- [ ] ✅ Le chauffeur préféré est sélectionné pour au moins une course
- [ ] ✅ Les logs montrent "✅ Booking → Chauffeur préféré"
- [ ] ✅ Si fallback utilisé, le préféré est aussi détecté dans `closest_feasible()`

## Problèmes possibles

1. **preferred_driver_id non dans overrides**: Vérifier que `overrides.preferred_driver_id = 2` est envoyé depuis le frontend
2. **Chauffeur #2 non disponible**: Vérifier `is_active=True` et `is_available=True` en DB
3. **Chauffeur #2 infaisable (TW)**: Vérifier les fenêtres de travail du chauffeur
4. **Bonus non appliqué**: Vérifier que `preferred_driver_id` est bien passé à `_score_driver_for_booking()`

## Requête SQL pour vérifier les assignations

```sql
-- Vérifier les assignations pour le dispatch_run_id 334
SELECT
    a.id,
    a.booking_id,
    a.driver_id,
    a.status,
    b.scheduled_time,
    dr.id as dispatch_run_id,
    dr.status as dispatch_run_status
FROM assignments a
JOIN bookings b ON a.booking_id = b.id
JOIN dispatch_runs dr ON a.dispatch_run_id = dr.id
WHERE dr.id = 334
ORDER BY a.driver_id, b.scheduled_time;

-- Compter les assignations par chauffeur
SELECT
    a.driver_id,
    COUNT(*) as nb_assignations
FROM assignments a
JOIN dispatch_runs dr ON a.dispatch_run_id = dr.id
WHERE dr.id = 334
GROUP BY a.driver_id
ORDER BY nb_assignations DESC;
```
