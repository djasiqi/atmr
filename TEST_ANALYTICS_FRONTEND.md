# 🧪 Guide de Test - Analytics Frontend

**Date** : 14 octobre 2025  
**Objectif** : Valider que le dashboard Analytics fonctionne correctement

---

## ✅ Problème Corrigé

### Erreur Initiale
```
GET http://localhost:3000/api/analytics/dashboard/undefined?period=30d 404 (NOT FOUND)
```

### Cause
Le composant utilisait `useParams()` au lieu de `useCompanyData()` comme les autres pages.

### Solution Appliquée ✅
- Utilisation de `useCompanyData()` pour récupérer le company
- Extraction du `public_id` depuis `company?.public_id`
- Ajout de vérification avant le chargement

---

## 🚀 Comment Tester Maintenant

### Étape 1 : Redémarrer le Frontend (si nécessaire)

```bash
cd frontend
npm start
```

### Étape 2 : Naviguer vers Analytics

1. Ouvrez votre navigateur : `http://localhost:3000`
2. Connectez-vous avec votre compte company
3. Dans le menu de gauche, cliquez sur **📊 Analytics**

### Étape 3 : Vérifier le Chargement

**Si vous voyez** :
```
"Aucune donnée disponible pour le moment.
Lancez des dispatches pour commencer à collecter des métriques."
```

✅ **C'est normal !** Vous devez d'abord lancer un dispatch.

---

## 📊 Générer des Données de Test

### Option 1 : Lancer un Dispatch (Recommandé)

1. Allez dans **Dispatch & Planification**
2. Sélectionnez une date (aujourd'hui ou demain)
3. Cliquez **Lancer Dispatch**
4. Attendez la fin (1-2 minutes)
5. ✅ Les métriques sont collectées automatiquement !
6. Retournez dans **Analytics**
7. Rafraîchissez (F5) ou changez de période

### Option 2 : Utiliser des Données Existantes

Si vous avez déjà lancé des dispatches avant l'installation :

1. Les anciennes données ne sont pas encore dans `dispatch_metrics`
2. Il faut lancer au moins 1 nouveau dispatch
3. Les métriques commenceront à s'accumuler

---

## 🔍 Validation Point par Point

### ✅ Checklist Frontend

#### Navigation
- [ ] Le lien **📊 Analytics** apparaît dans le menu
- [ ] Cliquer dessus charge la page Analytics
- [ ] L'URL est `/dashboard/company/<public_id>/analytics`

#### Chargement
- [ ] Un spinner apparaît pendant le chargement
- [ ] Pas d'erreur dans la console (F12)
- [ ] La requête API retourne 200 OK

#### Affichage (avec données)
- [ ] 4 KPI cards s'affichent en haut
- [ ] Les valeurs sont correctes (> 0 si vous avez des données)
- [ ] 4 graphiques s'affichent en dessous
- [ ] Les insights apparaissent (si disponibles)

#### Interactivité
- [ ] Les boutons de période fonctionnent (7j, 30j, 90j)
- [ ] Changer de période recharge les données
- [ ] Le bouton "Exporter CSV" télécharge un fichier
- [ ] Le bouton "Exporter JSON" fonctionne

#### Responsive
- [ ] La page s'affiche correctement sur desktop
- [ ] Les graphiques sont responsive
- [ ] Le layout s'adapte sur mobile/tablette

---

## 🐛 Dépannage

### Problème : "Aucune donnée disponible"

**Solution** :
1. Lancez au moins 1 dispatch
2. Attendez 30 secondes
3. Rafraîchissez la page Analytics (F5)
4. Vérifiez la console (F12) pour les erreurs

### Problème : Erreur 404 ou 401

**Vérifications** :
```javascript
// Dans la console DevTools (F12)
localStorage.getItem('token')  // Doit retourner un token
```

**Si pas de token** : Reconnectez-vous.

### Problème : Graphiques vides

**Vérifications** :
1. Ouvrez la console (F12)
2. Onglet Network
3. Cherchez la requête `/api/analytics/dashboard`
4. Vérifiez la réponse JSON

**Attendu** :
```json
{
  "success": true,
  "data": {
    "trends": [...]  // Doit contenir des données
  }
}
```

**Si `trends` est vide** : Lancez un dispatch d'abord !

### Problème : Console Warnings

**Warnings React normaux** (ignorables) :
- "Each child in a list should have a unique key" (si présent, je corrigerai)
- "Can't perform a React state update on an unmounted component"

**Erreurs critiques** (à corriger) :
- "Cannot read property of undefined"
- "Failed to fetch"

---

## 📈 Ce Que Vous Devriez Voir (Avec Données)

### En Haut : KPI Cards

```
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ 📦           │ │ ✅           │ │ ⏱️           │ │ ⭐           │
│ Total Courses│ │ Taux à       │ │ Retard moyen │ │ Score Qualité│
│              │ │ l'heure      │ │              │ │              │
│     450      │ │   87.2%      │ │   8.5 min    │ │   84/100     │
│ Sur la       │ │ ✨ Excellent │ │ 👍 Acceptable│ │ ✅ Bon       │
│ période      │ │              │ │              │ │              │
└──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
```

### Milieu : Insights

```
💡 Insights & Recommandations

┌────────────────────────────────────────────────────┐
│ ✅ Excellente ponctualité                          │
│ Votre taux de ponctualité (87.2%) est excellent ! │
│ Continuez ainsi.                                   │
└────────────────────────────────────────────────────┘
```

### Bas : Graphiques

```
┌──────────────────┐ ┌──────────────────┐
│ 📦 Volume        │ │ ✅ Ponctualité   │
│ [BarChart]       │ │ [AreaChart]      │
│                  │ │                  │
└──────────────────┘ └──────────────────┘

┌──────────────────┐ ┌──────────────────┐
│ ⏱️ Retards       │ │ ⭐ Qualité       │
│ [LineChart]      │ │ [AreaChart]      │
│                  │ │                  │
└──────────────────┘ └──────────────────┘
```

---

## ✅ Tests Backend Requis

Avant de tester le frontend, assurez-vous que :

### Test 1 : Tables Créées

```bash
docker compose exec db psql -U user -d atmr_db -c "\dt dispatch_metrics"
docker compose exec db psql -U user -d atmr_db -c "\dt daily_stats"
```

**Attendu** : Les 2 tables existent ✅

### Test 2 : Métriques Collectées

```bash
docker compose exec db psql -U user -d atmr_db -c "SELECT COUNT(*) FROM dispatch_metrics;"
```

**Si 0** : Lancez un dispatch d'abord !

### Test 3 : API Répond

```bash
# Remplacez YOUR_TOKEN et COMPANY_ID
curl -X GET \
  "http://localhost:5000/api/analytics/dashboard/COMPANY_ID?period=7d" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Attendu** : JSON avec `success: true`

---

## 🎯 Scénario de Test Complet

### Scénario : Première Utilisation

**Étape 1** : Page vide (normal)
```
📊 Analytics
Aucune donnée disponible pour le moment.
Lancez des dispatches pour commencer à collecter des métriques.
```

**Étape 2** : Lancer un dispatch
1. Allez dans Dispatch & Planification
2. Lancez un dispatch pour aujourd'hui
3. Attendez la fin

**Étape 3** : Retour sur Analytics
1. Cliquez sur Analytics dans le menu
2. La page devrait maintenant afficher les données !

**Étape 4** : Vérifier les KPIs
- Total courses : Devrait être > 0
- Taux à l'heure : Devrait être entre 0-100%
- Retard moyen : Devrait être affiché
- Score qualité : Devrait être calculé

**Étape 5** : Vérifier les graphiques
- 1 point sur chaque graphique (1 jour de données)
- Au fil des jours, les courbes se dessineront

**Étape 6** : Tester l'export
- Cliquez "Exporter CSV"
- Un fichier doit se télécharger
- Ouvrez-le : 1 ligne de données

---

## 💡 Conseils

### Pour des Graphiques Plus Intéressants

**Jour 1** : 1 point (peu intéressant)  
**Jour 7** : 7 points (tendances visibles) ✅  
**Jour 30** : 30 points (patterns clairs) ✅✅  

**Recommandation** : Attendez au moins 7 jours pour des insights pertinents.

### Pendant Ce Temps

Vous pouvez :
- Vérifier que la collecte fonctionne (DB)
- Tester l'export CSV/JSON
- Explorer l'API
- Lire la documentation

---

## 🎨 Cohérence Visuelle Validée

✅ **Couleurs** : Identiques aux autres pages (teal #0f766e)  
✅ **Layout** : Conteneur blanc, header, sidebar cohérents  
✅ **Typography** : Tailles et poids de police harmonisés  
✅ **Spacing** : Marges et padding uniformes  
✅ **Shadows** : Ombres douces cohérentes  
✅ **Hover** : Effets subtils identiques  
✅ **Responsive** : Adaptatif comme le reste  

---

## 🎉 Résultat Attendu

Une fois que vous aurez quelques jours de données, votre dashboard Analytics ressemblera à ça :

```
📊 Analytics & Performance

KPIs :
Total Courses: 450    Taux à l'heure: 87%    Retard: 8.5min    Qualité: 84/100

Insights :
✅ Excellente ponctualité (87.2%)
⚠️ Mardi a plus de retards (ajoutez du buffer)
📊 Volume élevé (activité soutenue)

Graphiques :
[Courbes montrant l'évolution sur 30 jours]

[📥 Exporter CSV] [📄 Exporter JSON]
```

---

## ✅ Validation Finale

Si vous voyez tout ça, **la Phase 1 est 100% fonctionnelle** ! 🎊

**Bravo !** Vous avez maintenant :
- ✅ Un système de collecte automatique
- ✅ Un dashboard Analytics professionnel
- ✅ Des rapports automatiques prêts
- ✅ Une base pour l'amélioration continue

---

**Prochaine étape** : Laissez collecter des données pendant 1 semaine, puis profitez des insights ! 📈

