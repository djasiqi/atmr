# 🗺️ Correction affichage nom chauffeur sur la carte

**Date**: 2026-01-13  
**Version**: 1.0  
**Fichiers concernés**: 
- `backend/application/drivers/get_company_drivers_live_locations.py`
- `backend/routes/driver.py`

---

## 🎯 Problème

La carte de suivi des chauffeurs affichait **"Chauffeur 6855"** au lieu du **vrai nom du chauffeur** (prénom + nom).

### Cause

Les données retournées par le backend ne contenaient pas les champs `first_name` et `last_name` :

**Avant** :
```json
{
  "items": [
    {
      "driver_id": 6855,
      "lat": 46.2044,
      "lon": 6.1432,
      "speed": 10.5,
      "heading": 45.0
    }
  ]
}
```

Le frontend mobile utilisait alors un **fallback** :
```typescript
const markerName = nameParts.length > 0 
  ? nameParts.join(" ") 
  : `Chauffeur ${item.driver_id}`; // ❌ Fallback utilisé
```

---

## ✅ Solution implémentée

### 1️⃣ API HTTP : `/api/v1/driver/company/{company_id}/live-locations`

**Fichier** : `backend/application/drivers/get_company_drivers_live_locations.py`

Ajout de `first_name` et `last_name` dans la réponse :

```python
def execute(self, *, company_id: int) -> GetCompanyDriversLiveLocationsResult:
    items: list[dict[str, Any]] = []
    drivers = self._driver_repo.find_models_by_company_id(company_id=company_id)
    for d in drivers:
        rec = self._get_last_location(int(d.id))
        if not rec:
            continue
        
        # ✅ Extraire first_name et last_name depuis d.user
        first_name = None
        last_name = None
        if hasattr(d, 'user') and d.user is not None:
            first_name = getattr(d.user, 'first_name', None)
            last_name = getattr(d.user, 'last_name', None)
        
        items.append({
            "driver_id": int(d.id),
            "first_name": first_name,  # ✅ Nouveau
            "last_name": last_name,    # ✅ Nouveau
            **rec
        })
    return GetCompanyDriversLiveLocationsResult(
        response={"items": items}, status_code=200
    )
```

**Nouvelle réponse** :
```json
{
  "items": [
    {
      "driver_id": 6855,
      "first_name": "Jean",
      "last_name": "Dupont",
      "lat": 46.2044,
      "lon": 6.1432,
      "speed": 10.5,
      "heading": 45.0
    }
  ]
}
```

---

### 2️⃣ Socket.IO : Événement `driver_location_update`

**Fichier** : `backend/routes/driver.py` (ligne ~835-860)

Ajout de `first_name` et `last_name` dans le payload Socket.IO :

```python
# 5) Diffusion temps réel à la room entreprise
try:
    room = f"company_{driver.company_id}"
    
    # ✅ Extraire first_name et last_name depuis driver.user
    first_name = None
    last_name = None
    if hasattr(driver, 'user') and driver.user is not None:
        first_name = getattr(driver.user, 'first_name', None)
        last_name = getattr(driver.user, 'last_name', None)
    
    socketio.emit(
        "driver_location_update",
        {
            "driver_id": driver.id,
            "company_id": driver.company_id,
            "lat": lat,
            "lon": lon,
            "speed": speed,
            "heading": heading,
            "accuracy": accuracy,
            "ts": ts,
            "source": source,
            "first_name": first_name,  # ✅ Nouveau
            "last_name": last_name,    # ✅ Nouveau
        },
        to=room,
    )
```

---

## 🔄 Flux de données complet

### Avant la correction

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│   Driver    │────▶│   Backend    │────▶│   Frontend   │
│   Mobile    │     │   Flask API  │     │  (Web/Mob)   │
└─────────────┘     └──────────────┘     └──────────────┘
                           │
                           ▼
                    {
                      driver_id: 6855,
                      lat: 46.2044,
                      lon: 6.1432
                    }
                           │
                           ▼
                    ❌ "Chauffeur 6855"
```

### Après la correction

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│   Driver    │────▶│   Backend    │────▶│   Frontend   │
│   Mobile    │     │   Flask API  │     │  (Web/Mob)   │
└─────────────┘     └──────────────┘     └──────────────┘
                           │
                           ▼
                    {
                      driver_id: 6855,
                      first_name: "Jean",
                      last_name: "Dupont",
                      lat: 46.2044,
                      lon: 6.1432
                    }
                           │
                           ▼
                    ✅ "Jean Dupont"
```

---

## 📊 Exemple concret

### Chauffeur avec nom complet

**Données DB** :
- Driver ID: 6855
- User: { first_name: "Jean", last_name: "Dupont" }

**Affichage carte** :
```
📍 Jean Dupont
```

---

### Chauffeur avec prénom uniquement

**Données DB** :
- Driver ID: 6856
- User: { first_name: "Marie", last_name: null }

**Affichage carte** :
```
📍 Marie
```

---

### Chauffeur sans nom (fallback)

**Données DB** :
- Driver ID: 6857
- User: { first_name: null, last_name: null }

**Affichage carte** :
```
📍 Chauffeur 6857
```

Le fallback fonctionne toujours si les données sont manquantes.

---

## 🧪 Tests de validation

### Test 1 : API HTTP
```bash
# Appeler l'endpoint
curl -X GET "http://localhost:5000/api/v1/driver/company/1/live-locations" \
  -H "Authorization: Bearer {token}"

# Vérifier la réponse
{
  "items": [
    {
      "driver_id": 6855,
      "first_name": "Jean",      # ✅ Présent
      "last_name": "Dupont",      # ✅ Présent
      "lat": 46.2044,
      "lon": 6.1432
    }
  ]
}
```

### Test 2 : Socket.IO
```javascript
// Écouter l'événement
socket.on('driver_location_update', (data) => {
  console.log(data);
  // {
  //   driver_id: 6855,
  //   first_name: "Jean",    // ✅ Présent
  //   last_name: "Dupont",   // ✅ Présent
  //   lat: 46.2044,
  //   lon: 6.1432
  // }
});
```

### Test 3 : Affichage frontend
```typescript
// Mobile: useEnterpriseDriverTracking.ts
const markerName = nameParts.length > 0 
  ? nameParts.join(" ")         // ✅ "Jean Dupont"
  : `Chauffeur ${item.driver_id}`; // Fallback si nécessaire
```

---

## 🔧 Déploiement

### 1. Backend

```bash
# Redémarrer le backend
docker compose -f docker-compose.production.yml restart backend

# Vérifier les logs
docker compose -f docker-compose.production.yml logs -f backend
```

### 2. Frontend/Mobile

Aucune modification nécessaire côté frontend/mobile ! Le code existant utilise déjà `first_name` et `last_name` s'ils sont présents.

---

## ⚠️ Cas particuliers

### Driver sans user associé

Si un driver n'a pas de `user` associé (données corrompues) :
- `first_name` = `null`
- `last_name` = `null`
- Affichage = "Chauffeur {driver_id}" (fallback)

### User sans nom

Si un user n'a pas de `first_name`/`last_name` :
- `first_name` = `null`
- `last_name` = `null`
- Affichage = "Chauffeur {driver_id}" (fallback)

Le système est **robuste** et ne plante pas.

---

## 📌 Points importants

### ✅ Avantages de la solution

1. **Rétrocompatible** : Le fallback fonctionne toujours
2. **Robuste** : Gère les cas où les données sont manquantes
3. **Performant** : Pas de requête DB supplémentaire (relation déjà chargée)
4. **Cohérent** : Même logique pour HTTP et Socket.IO

### ⚠️ Limitations

1. **Eager loading** : La relation `driver.user` doit être chargée
   - Actuellement OK car `DriverRepository` charge les relations par défaut
   
2. **Nom d'utilisateur** : Si `first_name` et `last_name` sont vides mais `username` existe
   - On pourrait ajouter un fallback sur `username`
   - À implémenter si nécessaire

---

## 🔮 Évolutions possibles

### 1. Fallback sur username
```python
first_name = getattr(d.user, 'first_name', None)
last_name = getattr(d.user, 'last_name', None)
username = getattr(d.user, 'username', None)

# Si pas de nom, utiliser username
if not first_name and not last_name and username:
    first_name = username
```

### 2. Photo de profil
```python
items.append({
    "driver_id": int(d.id),
    "first_name": first_name,
    "last_name": last_name,
    "avatar_url": getattr(d.user, 'avatar_url', None),  # Nouveau
    **rec
})
```

### 3. Statut du chauffeur
```python
items.append({
    "driver_id": int(d.id),
    "first_name": first_name,
    "last_name": last_name,
    "is_available": getattr(d, 'is_available', None),  # Nouveau
    "current_booking_id": getattr(d, 'current_booking_id', None),  # Nouveau
    **rec
})
```

---

**Version**: 1.0  
**Dernière mise à jour**: 2026-01-13  
**Auteur**: Assistant IA  
**Status**: ✅ Implémenté et prêt pour test
