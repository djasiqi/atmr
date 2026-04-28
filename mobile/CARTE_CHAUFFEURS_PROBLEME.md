# 🗺️ Problème : Carte Chauffeurs Vide en Production

## 🎯 Symptôme

Sur l'application mobile entreprise (onglet Dashboard), la section **"Carte chauffeurs"** s'affiche mais **aucun chauffeur n'apparaît** sur la carte.

---

## 🔍 Diagnostic

### ✅ Ce qui fonctionne

1. **Composant `EnterpriseDriversMap`** : Fonctionne correctement

   - Affiche la carte Google Maps
   - Peut afficher des marqueurs quand ils existent
   - Gère le zoom et la centrage automatique

2. **Hook `useEnterpriseDriverTracking`** : Fonctionne correctement

   - Appelle l'API `/api/v1/driver/company/<id>/live-locations`
   - Récupère les données via HTTP et WebSocket
   - Polling automatique toutes les 10-30 secondes

3. **Backend API** : Fonctionne correctement
   - Endpoint `GET /api/v1/driver/company/<id>/live-locations` existe
   - Retourne `[]` (tableau vide) car aucune position enregistrée
   - Endpoint `PUT /api/v1/driver/me/location` prêt à recevoir des positions

### ❌ Ce qui manque

**Les chauffeurs n'envoient JAMAIS leur position GPS** car :

- Pas d'application mobile chauffeur avec tracking GPS actif
- Ou l'application existe mais le tracking n'est pas activé
- Ou l'application existe mais n'appelle pas l'endpoint d'envoi de position

---

## 🔧 Solutions

### Solution 1 : Application Mobile Chauffeur (Recommandé)

Créer ou activer une application mobile pour les chauffeurs qui :

1. **Demande les permissions GPS**
2. **Envoie la position toutes les 10-30 secondes** à :

   ```
   PUT /api/v1/driver/me/location

   Body:
   {
     "latitude": 46.2044,
     "longitude": 6.1432,
     "heading": 90,
     "speed": 15.5,
     "accuracy": 10.0
   }
   ```

3. **Fonctionne en arrière-plan** (même quand l'app est fermée)

**Code d'exemple** : Voir `mobile/TRACKING_GPS_ACTIVATION.md`

---

### Solution 2 : Test Manuel (Développement)

Pour **tester** que la carte fonctionne, envoyer manuellement une position via cURL :

```bash
# 1. Se connecter en tant que chauffeur
curl -X POST https://api.lirie.ch/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "chauffeur@example.com",
    "password": "password"
  }'

# Copier le token reçu

# 2. Envoyer une position GPS
curl -X PUT https://api.lirie.ch/api/v1/driver/me/location \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN_HERE" \
  -d '{
    "latitude": 46.2044,
    "longitude": 6.1432
  }'

# 3. Vérifier dans l'app entreprise
# → Le chauffeur devrait apparaître sur la carte
```

---

### Solution 3 : Widget Web pour Chauffeurs (Temporaire)

Créer une page web simple que les chauffeurs ouvrent sur leur téléphone :

```html
<!DOCTYPE html>
<html>
  <head>
    <title>Tracking GPS Chauffeur</title>
  </head>
  <body>
    <h1>Tracking GPS Actif</h1>
    <p id="status">Initialisation...</p>
    <button id="start">Démarrer le tracking</button>
    <button id="stop">Arrêter le tracking</button>

    <script>
      const API_URL = "https://api.lirie.ch";
      let token = localStorage.getItem("driver_token");
      let intervalId = null;

      // Fonction d'envoi de position
      async function sendLocation(latitude, longitude) {
        try {
          const response = await fetch(`${API_URL}/api/v1/driver/me/location`, {
            method: "PUT",
            headers: {
              "Content-Type": "application/json",
              Authorization: `Bearer ${token}`,
            },
            body: JSON.stringify({ latitude, longitude }),
          });

          if (response.ok) {
            document.getElementById(
              "status"
            ).textContent = `Position envoyée: ${latitude.toFixed(
              4
            )}, ${longitude.toFixed(4)}`;
          } else {
            document.getElementById(
              "status"
            ).textContent = `Erreur: ${response.status}`;
          }
        } catch (error) {
          document.getElementById(
            "status"
          ).textContent = `Erreur: ${error.message}`;
        }
      }

      // Démarrer le tracking
      document.getElementById("start").addEventListener("click", () => {
        if (intervalId) return;

        navigator.geolocation.getCurrentPosition(
          (position) => {
            sendLocation(position.coords.latitude, position.coords.longitude);

            // Envoyer toutes les 10 secondes
            intervalId = setInterval(() => {
              navigator.geolocation.getCurrentPosition((pos) =>
                sendLocation(pos.coords.latitude, pos.coords.longitude)
              );
            }, 10000);

            document.getElementById("status").textContent = "Tracking actif";
          },
          (error) => {
            document.getElementById(
              "status"
            ).textContent = `Erreur GPS: ${error.message}`;
          }
        );
      });

      // Arrêter le tracking
      document.getElementById("stop").addEventListener("click", () => {
        if (intervalId) {
          clearInterval(intervalId);
          intervalId = null;
          document.getElementById("status").textContent = "Tracking arrêté";
        }
      });
    </script>
  </body>
</html>
```

---

## 📊 Vérification

### Backend (vérifier que l'API reçoit les positions)

```bash
# Se connecter au serveur
# ssh (après export SERVER_HOST — voir docs/deployment-ssh.md)
ssh deploy@$SERVER_HOST

# Voir les logs de réception de positions
cd /srv/atmr
docker compose -f docker-compose.production.yml logs backend --follow | grep "location"
```

### Frontend (vérifier que l'app reçoit les positions)

Ouvrir l'app mobile entreprise et regarder les logs :

```
[useEnterpriseDriverTracking] Fetched 1 driver locations
[EnterpriseDriversMap] Rendering 1 markers
```

---

## 🎯 Checklist

- [ ] **Les chauffeurs ont une application** pour envoyer leur position
- [ ] **Les permissions GPS sont accordées** sur les téléphones des chauffeurs
- [ ] **Le tracking est activé** dans l'application chauffeur
- [ ] **Les positions sont envoyées** à l'API (`PUT /driver/me/location`)
- [ ] **L'API reçoit et stocke** les positions (logs backend)
- [ ] **L'app entreprise récupère** les positions (logs frontend)
- [ ] **La carte affiche** les marqueurs des chauffeurs

---

## 📝 Prochaines Étapes

1. **Identifier l'application chauffeur** :

   - Existe-t-elle déjà ?
   - Est-elle déployée sur les téléphones ?
   - Le tracking est-il activé ?

2. **Si pas d'application** :

   - Créer une application React Native simple
   - Avec tracking GPS en arrière-plan
   - Publication sur App Store / Play Store

3. **Tester** :
   - Envoyer manuellement une position (cURL)
   - Vérifier qu'elle apparaît sur la carte entreprise
   - Activer le tracking sur un téléphone de test

---

## 📚 Documentation Complète

Voir `mobile/TRACKING_GPS_ACTIVATION.md` pour :

- Code complet d'une application chauffeur
- Configuration du tracking en arrière-plan
- Gestion des permissions GPS
- Tests et débogage

---

## ✅ RÉSOLUTION DU PROBLÈME (2026-01-13)

### 🔍 Cause racine identifiée

Le hook `useEnterpriseDriverTracking` attendait des propriétés **`latitude`/`longitude`** alors que le backend envoie **`lat`/`lon`**.

**Backend envoie** (Socket.IO & HTTP):
```json
{
  "driver_id": 123,
  "lat": 46.2044,
  "lon": 6.1432,
  "ts": "2026-01-13T10:30:00Z"
}
```

**Mobile attendait**:
```typescript
const latitude = toNumber(payload.latitude); // ❌ undefined
const longitude = toNumber(payload.longitude); // ❌ undefined
```

### 🔧 Correction appliquée

**Fichier**: `mobile/operations-app/hooks/useEnterpriseDriverTracking.ts`

**Modification**:
```typescript
// ✅ Accepte maintenant les deux formats
const latitude = toNumber(payload.latitude ?? payload.lat);
const longitude = toNumber(payload.longitude ?? payload.lon);
```

Cette modification a été appliquée à:
1. ✅ Handler Socket.IO (`handleDriverLocation`)
2. ✅ Fetcher HTTP (`fetchLocationsViaHTTP`)

### 📋 Validations effectuées

1. ✅ **Backend vérifié**: Émet bien `lat`/`lon` (pas `latitude`/`longitude`)
2. ✅ **Redis vérifié**: Stocke bien `lat`/`lon` dans `driver:{id}:loc`
3. ✅ **Frontend web vérifié**: Accepte déjà les deux formats (compatible)
4. ✅ **Mobile corrigé**: Accepte maintenant les deux formats
5. ✅ **Types TypeScript mis à jour**: `frontend/src/types/socketEvents.ts`
6. ✅ **Documentation créée**: `docs/GPS_TRACKING_COHERENCE.md`

### 🎯 Prochaines étapes

1. **Rebuild mobile app** avec la correction
2. **Test en production** avec un chauffeur qui envoie sa position
3. **Vérifier** que les marqueurs apparaissent sur la carte entreprise

### 📚 Documentation technique complète

Voir `docs/GPS_TRACKING_COHERENCE.md` pour la documentation complète de la chaîne GPS:
- Formats d'entrée/sortie
- Cohérence entre backend/frontend
- Points d'attention pour développeurs
- Tests de validation
