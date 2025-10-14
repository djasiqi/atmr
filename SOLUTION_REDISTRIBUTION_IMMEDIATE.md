# 🚨 Solution Immédiate : Redistribuer les Courses de Yannis

## 🎯 Situation

- **Yannis** a 2 courses assignées (#24, #25) avec **618 min de retard total**
- Le **dispatch automatique** réassigne toujours à Yannis
- **D'autres chauffeurs sont disponibles** (Giuseppe, Khalid, etc.)

---

## ✅ Solution Manuelle IMMÉDIATE

### **Étape 1 : Désassigner les Courses de Yannis**

**Via le Frontend Web** :

#### **Pour la Course #24** :

1. Aller dans **"Réservations"** ou la page **"Dispatch & Planification"**
2. Trouver la course **#24** (Claude Pittet, 13:00)
3. Dans la liste, cliquer sur les **"..."** ou **"Actions"**
4. Sélectionner **"Désassigner"** ou **"Changer le chauffeur"**
5. Choisir **"Aucun chauffeur"** (désassigner)
6. Sauvegarder

#### **Pour la Course #25** :

1. Même procédure
2. Désassigner également

---

### **Étape 2 : Relancer le Dispatch**

Une fois les 2 courses **désassignées** :

1. Aller dans **"Dispatch & Planification"**
2. Cliquer sur **"🚀 Lancer Dispatch Automatique"**
3. Le système va maintenant **répartir** les 2 courses sur **2 chauffeurs différents**

---

## 🔧 Solution Technique (Alternative)

### **Via l'API (Plus Rapide)**

Si vous avez accès à la base de données ou à l'API :

```python
# Désassigner les courses de Yannis
from models import Booking, BookingStatus
from ext import db

# Course #24
booking24 = Booking.query.get(24)
booking24.driver_id = None
booking24.status = BookingStatus.PENDING

# Course #25
booking25 = Booking.query.get(25)
booking25.driver_id = None
booking25.status = BookingStatus.PENDING

db.session.commit()
```

**Puis relancer le dispatch** depuis le frontend.

---

## 🚀 Solution Automatique (À Implémenter)

Pour que le dispatch **désassigne automatiquement** les courses en retard critique avant de relancer :

### **Modifier le Dispatch Engine**

Ajouter une option `force_reassign_delayed=True` qui :

1. **Avant le dispatch** :

   - Détecte les courses avec retard > 30 min
   - Les désassigne automatiquement
   - Marque comme "PENDING" ou "URGENT"

2. **Pendant le dispatch** :
   - Traite ces courses comme nouvelles
   - Réassigne selon l'algorithme optimal
   - Évite de réassigner au même chauffeur

---

## 📊 Pourquoi le Dispatch Réassigne à Yannis ?

Le dispatch considère que :

- ✅ Yannis est **déjà assigné** → pas de coût de réassignation
- ✅ Yannis connaît déjà le trajet
- ✅ Pas de coordination avec un nouveau chauffeur

**Pour forcer la redistribution**, il faut **d'abord désassigner**.

---

## 🎯 ACTION IMMÉDIATE

### **Méthode Rapide (Frontend)** :

1. **Ouvrir** : `http://localhost:3000/dashboard/company/{id}/dispatch`
2. Dans la section **"📋 Courses du Jour"**, pour chaque course :
   - Clic droit ou bouton d'action
   - **"Désassigner"** ou **"Changer chauffeur"** → Choisir "Aucun"
3. **Relancer** le dispatch automatique
4. **Vérifier** : Les courses sont maintenant assignées à **2 chauffeurs différents**

---

### **Méthode Alternative (Base de Données)** :

Si vous avez accès direct via Docker :

```bash
docker exec atmr-postgres-1 psql -U [user] -d atmr_db -c "UPDATE booking SET driver_id = NULL, status = 'pending' WHERE id IN (24, 25);"
```

**Puis relancer** le dispatch depuis le frontend.

---

## 📝 Vérification Finale

Après redistribution, vous devriez voir :

### **Frontend Web** :

```
📋 Courses du Jour

#24 - Claude Pittet - 13:00 - Giuseppe Bekasy ✅
#25 - Claude Pittet - 18:00 - Khalid Alaoui ✅
```

### **App Mobile Yannis** :

- 0 missions (courses désassignées)

### **Apps Mobiles des Nouveaux Chauffeurs** :

- Nouvelles missions apparaissent
- Notifications reçues

---

**Désassignez manuellement les 2 courses de Yannis depuis le frontend web, puis relancez le dispatch !** 🚀

---

**Date** : 10 octobre 2025, 20:35  
**Urgence** : 🔴 CRITIQUE  
**Action** : Désassigner puis redistribuer
