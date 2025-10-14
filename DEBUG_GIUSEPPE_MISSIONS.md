# 🔍 Debug : Giuseppe voit les missions de Yannis

## 🎯 Problème

Giuseppe (chauffeur) voit les courses assignées à Yannis dans son app mobile, alors qu'il devrait voir uniquement ses propres courses.

## ✅ Vérifications Effectuées

### 1. **Backend Endpoint `/driver/me/bookings`**

**Filtrage** : ✅ **CORRECT**

```python
# backend/routes/driver.py (ligne 263)
bookings = (
    Booking.query
    .filter(Booking.driver_id == driver.id)  # ✅ Filtre par driver.id
    .filter(Booking.scheduled_time >= now)
    .filter(status_pred)
    .order_by(Booking.scheduled_time.asc())
    .all()
)
```

Le backend **filtre correctement** par `driver.id`.

### 2. **Mobile App Service `getAssignedTrips()`**

**Appel API** : ✅ **CORRECT**

```typescript
// mobile/driver-app/services/api.ts
export const getAssignedTrips = async (): Promise<Booking[]> => {
  const response = await api.get<Booking[]>("/driver/me/bookings");
  return response.data;
};
```

L'app mobile appelle le bon endpoint.

---

## 🔍 Diagnostic

Si Giuseppe voit les missions de Yannis, il y a **3 possibilités** :

### **Possibilité 1 : Giuseppe est connecté avec le compte de Yannis**

**Test** :

1. Ouvrez l'app mobile de Giuseppe
2. Allez dans "Profil"
3. **Vérifiez le nom affiché** en haut : doit être "Giuseppe [Nom]" et **PAS "Yannis Labrot"**

**Si c'est "Yannis"** → Giuseppe s'est connecté avec les credentials de Yannis !

**Solution** :

- Déconnecter Giuseppe
- Reconnecter avec ses propres credentials

---

### **Possibilité 2 : Token JWT partagé entre les deux appareils**

Si Giuseppe et Yannis utilisent **le même token** (stocké dans AsyncStorage), ils verront les mêmes données.

**Test** :

1. Vérifier dans les logs Docker quand Giuseppe charge ses missions :
   ```bash
   docker logs --tail 50 atmr-api-1 | grep "Driver.*loading bookings"
   ```

**Attendu** :

```
📱 [Driver Bookings] Driver Giuseppe [Nom] (ID: 3) loading bookings
Found 0 bookings for driver Giuseppe (ID: 3)
```

**Si on voit** :

```
📱 [Driver Bookings] Driver Yannis Labrot (ID: 2) loading bookings
```

→ Giuseppe utilise le token de Yannis !

---

### **Possibilité 3 : Bug de cache côté mobile**

L'app mobile cache les missions dans `AsyncStorage`. Si Giuseppe a ouvert l'app alors qu'elle était connectée à Yannis, le cache peut persister.

**Solution** :

1. Déconnecter Giuseppe (bouton "Se déconnecter")
2. Fermer complètement l'app (swipe up)
3. Rouvrir l'app
4. Reconnecter avec les credentials de Giuseppe

---

## 🧪 Test Immédiat

### **Étape 1 : Vérifier l'identité dans l'app de Giuseppe**

1. Ouvrez l'app sur le téléphone de Giuseppe
2. Allez dans **"Profil"** (dernier onglet)
3. **Regardez le nom** affiché en haut

**Attendu** : "Giuseppe [Son Nom]"  
**Si vous voyez** : "Yannis Labrot" → **PROBLÈME DE CONNEXION**

---

### **Étape 2 : Vérifier les logs backend**

**Demandez à Giuseppe de** :

1. Ouvrir l'onglet **"Mission"**
2. Faire un "Pull to refresh" (tirer vers le bas)

**Puis regardez les logs** :

```bash
docker logs --tail 20 atmr-api-1 | grep "Driver.*loading bookings"
```

**Attendu** :

```
📱 [Driver Bookings] Driver Giuseppe Rossi (ID: 3) loading bookings
Found 0 bookings for driver Giuseppe (ID: 3)
```

**Si vous voyez** :

```
📱 [Driver Bookings] Driver Yannis Labrot (ID: 2) loading bookings
Found 2 bookings for driver Yannis (ID: 2)
```

→ **Giuseppe utilise le token de Yannis !**

---

## ✅ Solution

### **Si Giuseppe est connecté avec le compte de Yannis** :

1. **Dans l'app de Giuseppe** :

   - Aller dans "Profil"
   - Cliquer sur "Se déconnecter"
   - **Fermer complètement l'app** (swipe up dans le gestionnaire d'apps)

2. **Rouvrir l'app**

3. **Se connecter avec les credentials de Giuseppe** :

   - Email : `giuseppe@[...]`
   - Mot de passe : `[son mot de passe]`

4. **Vérifier** :
   - Profil affiche "Giuseppe"
   - Mission affiche 0 courses (ou uniquement ses courses)

---

### **Si le problème persiste** :

Vérifiez dans la base de données :

```sql
-- Quelles courses sont assignées à Giuseppe ?
SELECT id, customer_name, driver_id, status
FROM booking
WHERE driver_id = (SELECT id FROM driver WHERE user_id = (SELECT id FROM "user" WHERE first_name = 'Giuseppe'));

-- Quelles courses sont assignées à Yannis ?
SELECT id, customer_name, driver_id, status
FROM booking
WHERE driver_id = (SELECT id FROM driver WHERE user_id = (SELECT id FROM "user" WHERE first_name = 'Yannis'));
```

---

## 📊 Résumé

**État actuel** :

- ✅ Backend filtre correctement par `driver_id`
- ✅ Logs ajoutés pour tracer les appels
- ❓ Giuseppe voit les courses de Yannis → **Problème d'authentification probable**

**Action immédiate** :

1. Vérifier le nom dans le profil de Giuseppe
2. Si c'est "Yannis" → Déconnecter et reconnecter Giuseppe
3. Vérifier les logs Docker pour confirmer

---

**Pouvez-vous vérifier le nom affiché dans le profil de Giuseppe et me dire ce que vous voyez ?**

Et ensuite, faites un pull-to-refresh dans l'onglet "Mission" et envoyez-moi les logs :

```bash
docker logs --tail 30 atmr-api-1 | grep "Driver.*loading"
```
