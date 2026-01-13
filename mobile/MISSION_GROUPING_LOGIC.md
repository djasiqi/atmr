# 📋 Logique de groupement et affichage des missions chauffeur

**Date**: 2026-01-13  
**Version**: 2.0  
**Fichiers concernés**:

- `mobile/operations-app/app/(tabs)/mission.tsx`
- `mobile/operations-app/utils/missionGrouping.ts`

---

## 🎯 Objectif

Afficher au chauffeur **uniquement la prochaine mission OU le prochain groupe de missions**, au lieu de toutes les missions de la journée.

### Avantages

✅ Interface simplifiée et claire pour le chauffeur  
✅ Réduit la confusion (pas besoin de scroller parmi 20 courses)  
✅ Progression séquentielle : une mission complétée → la suivante s'affiche  
✅ Gestion intelligente des courses groupées

---

## 📊 Logique d'affichage

### Règle principale

**Afficher UNIQUEMENT** :

1. Les missions **en cours** (`in_progress`) ou **en route** (`en_route`)
2. OU la **prochaine mission assignée** (la plus proche dans le temps)
3. OU le **prochain groupe de missions** (si plusieurs courses peuvent être groupées)

### Critères de période

- **Avant 19h00** : Afficher les missions d'aujourd'hui uniquement
- **Après 19h00** : Afficher aussi les missions du lendemain (si déjà attribuées)

---

## 🔄 Critères de groupement des missions

Deux missions sont **groupées** si :

1. ✅ **Même adresse de pickup** (normalisée)

   - Comparaison insensible à la casse
   - Ignore les espaces et ponctuation
   - Compare les 50 premiers caractères

2. ✅ **Écart de temps ≤ 5 minutes**
   - Course A: 10h00, Course B: 10h05 → GROUPÉES
   - Course A: 10h00, Course B: 10h06 → NON GROUPÉES

---

## 📝 Exemples concrets

### Exemple 1 : Courses séquentielles (non groupées)

**Situation** :

- Course A : départ 10h00, Hôpital Cantonal
- Course B : départ 11h00, Gare Cornavin
- Course C : départ 14h00, Aéroport

**Heure actuelle : 08h00**

**Affichage** :

```
✅ Mission 1 : Course A (10h00) - Hôpital Cantonal
```

**Après complétion de A** :

```
✅ Mission 1 : Course B (11h00) - Gare Cornavin
```

**Après complétion de B** :

```
✅ Mission 1 : Course C (14h00) - Aéroport
```

---

### Exemple 2 : Courses groupées (même horaire)

**Situation** :

- Course A : départ 10h00, Hôpital Cantonal
- Course B : départ 10h00, Hôpital Cantonal
- Course C : départ 14h00, Aéroport

**Heure actuelle : 08h00**

**Affichage** :

```
📦 Courses groupées (2) - Hôpital Cantonal

✅ Mission 1 : Course A (10h00)
✅ Mission 2 : Course B (10h00)
```

**Après complétion de A et B** :

```
✅ Mission 1 : Course C (14h00) - Aéroport
```

---

### Exemple 3 : Courses groupées (horaires proches)

**Situation** :

- Course A : départ 10h00, Hôpital Cantonal
- Course B : départ 10h05, Hôpital Cantonal (même lieu, +5 min)
- Course C : départ 11h00, Gare Cornavin

**Heure actuelle : 08h00**

**Affichage** :

```
📦 Courses groupées (2) - Hôpital Cantonal

✅ Mission 1 : Course A (10h00)
✅ Mission 2 : Course B (10h05)
```

**Après complétion de A et B** :

```
✅ Mission 1 : Course C (11h00) - Gare Cornavin
```

---

### Exemple 4 : Horaires proches MAIS lieux différents

**Situation** :

- Course A : départ 10h00, Hôpital Cantonal
- Course B : départ 10h05, Gare Cornavin (lieu différent)

**Heure actuelle : 08h00**

**Affichage** :

```
✅ Mission 1 : Course A (10h00) - Hôpital Cantonal
```

**Raison** : Lieux différents → pas de groupement, même si les horaires sont proches

---

### Exemple 5 : Affichage après 19h00

**Situation** :

- Aujourd'hui (13 jan) : Course A terminée, plus de courses
- Demain (14 jan) : Course B à 08h00, Course C à 10h00

**Heure actuelle : 19h30**

**Affichage** :

```
✅ Mission 1 : Course B (demain 08h00)
```

**Raison** : Après 19h00 → affiche les courses du lendemain

---

## 🔄 Cycle de vie d'une mission

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Mission ASSIGNED (assignée)                              │
│    → Visible si c'est la prochaine mission ou dans un groupe│
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Mission EN_ROUTE (chauffeur en route vers pickup)        │
│    → Toujours visible (priorité absolue)                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Mission IN_PROGRESS (client à bord, vers dropoff)        │
│    → Toujours visible (priorité absolue)                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Mission COMPLETED ou RETURN_COMPLETED                     │
│    → Retirée de la liste                                    │
│    → La prochaine mission (ou groupe) s'affiche             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Fonctions clés

### `filterActiveMissions(missions: Booking[])`

**Rôle** : Filtre les missions à considérer (aujourd'hui ou demain si après 19h)

**Logique** :

```typescript
- Toujours inclure: in_progress, en_route
- Avant 19h: missions assignées d'aujourd'hui
- Après 19h: missions assignées d'aujourd'hui + demain
```

**Fichier** : `mobile/operations-app/utils/missionGrouping.ts`

---

### `filterNextMissionsOnly(missions: Booking[])`

**Rôle** : Ne garde QUE le prochain groupe de missions

**Logique** :

```typescript
1. Si missions en cours (in_progress/en_route) → les retourner
2. Sinon :
   - Prendre la mission assignée la plus proche (première dans le temps)
   - Chercher toutes les missions avec :
     * Même pickup (normalisé)
     * Écart ≤ 5 minutes
   - Retourner ce groupe
```

**Fichier** : `mobile/operations-app/utils/missionGrouping.ts`

---

### `organizeMissionsForDisplay(missions: Booking[])`

**Rôle** : Organise les missions pour l'affichage avec indicateurs de groupe

**Sortie** :

```typescript
DisplayMission[] = [
  {
    mission: Booking,
    missionNumber: number, // 1, 2, 3...
    groupInfo: {
      isGrouped: boolean,
      groupId: string,
      groupSize: number,
      // ...
    }
  }
]
```

**Fichier** : `mobile/operations-app/utils/missionGrouping.ts`

---

## 🧪 Tests de validation

### Test 1 : Affichage initial

```
Données :
- Course A: 10h00
- Course B: 11h00

Heure actuelle : 08h00

Résultat attendu : Course A uniquement
```

### Test 2 : Groupement par horaire

```
Données :
- Course A: 10h00, Hôpital
- Course B: 10h00, Hôpital

Heure actuelle : 08h00

Résultat attendu : Course A + Course B (groupe de 2)
```

### Test 3 : Groupement par proximité temporelle

```
Données :
- Course A: 10h00, Hôpital
- Course B: 10h03, Hôpital
- Course C: 10h10, Hôpital (>5min, pas groupée)

Heure actuelle : 08h00

Résultat attendu : Course A + Course B (groupe de 2)
```

### Test 4 : Progression après complétion

```
Données initiales :
- Course A: 10h00 (visible)
- Course B: 11h00 (masquée)

Action : Terminer Course A

Résultat attendu : Course B devient visible
```

### Test 5 : Affichage après 19h00

```
Données :
- Aujourd'hui: Aucune course
- Demain: Course A à 08h00

Heure actuelle : 19h30

Résultat attendu : Course A visible
```

### Test 6 : Mission en cours (priorité)

```
Données :
- Course A: 10h00, status = "in_progress"
- Course B: 11h00, status = "assigned"
- Course C: 12h00, status = "assigned"

Heure actuelle : 10h30

Résultat attendu : Course A uniquement (en cours)
```

---

## ⚙️ Configuration

### Paramètres ajustables

```typescript
// Intervalle de groupement (actuellement 5 minutes)
const GROUPING_WINDOW_MS = 5 * 60 * 1000;

// Heure de basculement vers demain (actuellement 19h00)
const SHOW_TOMORROW_AFTER_HOUR = 19;
```

**Fichier** : `mobile/operations-app/utils/missionGrouping.ts`

Pour modifier ces valeurs, éditer les constantes dans le code.

---

## 📌 Points importants

### ✅ Comportements garantis

1. **Jamais de liste vide** si des missions sont assignées

   - Au minimum, la prochaine mission est toujours affichée

2. **Missions en cours toujours visibles**

   - `in_progress` et `en_route` ont la priorité absolue
   - Même si d'autres missions sont assignées plus tôt

3. **Progression automatique**

   - Après complétion → prochaine mission s'affiche automatiquement
   - Via Socket.IO (`booking_updated` event)

4. **Synchronisation temps réel**
   - Socket.IO pour mises à jour instantanées
   - Polling de secours toutes les 60s si socket déconnecté

### ⚠️ Limitations actuelles

1. **Distance géographique**

   - Actuellement basé uniquement sur l'adresse exacte (normalisée)
   - Pas de calcul de distance routière
   - **Évolution possible** : Intégrer OSRM pour distance réelle

2. **Fenêtre de groupement fixe**
   - Actuellement 5 minutes
   - Pas d'adaptation dynamique basée sur la distance

---

## 🔮 Évolutions futures possibles

### 1. Distance routière réelle

```typescript
// Utiliser OSRM pour calculer la distance réelle
const distance = await calculateRouteDistance(pickupA, pickupB);
if (distance <= 1000 && timeDiff <= 5 * 60 * 1000) {
  // Grouper
}
```

### 2. Groupement intelligent par itinéraire

```typescript
// Grouper si le détour est acceptable (ex: +10% de distance)
const directDistance = calculateDistance(pickupA, dropoffA);
const withDetour = calculateDistance(pickupA, pickupB, dropoffB, dropoffA);
if (withDetour <= directDistance * 1.1) {
  // Grouper (détour acceptable)
}
```

### 3. Prévisualisation de la prochaine mission

```typescript
// Afficher la prochaine mission en grisé / preview
- Mission actuelle (normal)
- Prochaine mission (preview grisé)
- Reste masqué
```

---

## 📞 Support

Pour toute question ou bug concernant le groupement des missions :

1. Consulter ce document
2. Vérifier les tests dans le code
3. Examiner les logs console avec tag `[Mission]`

---

**Version**: 2.0  
**Dernière mise à jour**: 2026-01-13  
**Auteur**: Assistant IA  
**Status**: ✅ Implémenté
