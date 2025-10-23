# 🚀 GUIDE UTILISATION RAPIDE - SYSTÈME RL DISPATCH OPTIMISÉ

## ⚡ DÉMARRAGE EN 5 MINUTES

**Audience** : Dispatchers, Managers  
**Prérequis** : Accès compte Company ATMR  
**Date** : 21 octobre 2025

---

## 📋 ÉTAPE 1 : LANCER UN DISPATCH (2 min)

### **Accès** :

```
URL: /dashboard/company/{public_id}/dispatch
```

### **Actions** :

1. **Sélectionner date** : Choisir jour à dispatcher
2. **Options** :
   - ✅ Chauffeurs réguliers prioritaires (recommandé)
   - ✅ Autoriser chauffeurs d'urgence (selon besoin)
3. **Cliquer** : "🚀 Lancer Dispatch"

**Résultat** : Dispatch exécuté en 5-10 secondes

---

## 👁️ ÉTAPE 2 : VOIR SUGGESTIONS RL (1 min)

### **Après dispatch** :

- Panel **"🧠 Mode Semi-Auto - Assistant IA MDI"** s'affiche
- Liste 10-20 suggestions RL
- Auto-refresh toutes les 30 secondes

### **Interpréter suggestion** :

```
┌─────────────────────────────────────┐
│ 🤖 Booking #1234      🟢 85%       │
│                                     │
│ 👤 Driver A  →  👤 Driver B        │
│ (actuel)         (suggéré)         │
│                  📍 2.5 km         │
│                                     │
│ Gain: +12 min                      │
│                                     │
│ [✅ Appliquer]  [👍] [👎]         │
└─────────────────────────────────────┘
```

**Code couleur confiance** :

- 🟢 Vert (>90%) : Très fiable
- 🟡 Jaune (75-90%) : Fiable
- 🟠 Orange (50-75%) : Moyen
- 🔴 Rouge (<50%) : Prudence

---

## 👍 ÉTAPE 3 : DONNER FEEDBACK (30 sec)

### **3 options** :

1. **✅ Appliquer** :

   - Réassigne immédiatement le driver
   - Feedback "applied" enregistré automatiquement
   - Contribue à l'amélioration du modèle

2. **👍 Bonne suggestion** :

   - Vous ne l'appliquez pas maintenant
   - Mais vous validez que c'est une bonne idée
   - Feedback positif enregistré

3. **👎 Mauvaise suggestion** :
   - Vous rejetez la suggestion
   - Optionnel : Donner raison (ex: "Driver trop loin")
   - Feedback négatif enregistré

**Pourquoi c'est important** :

- Chaque feedback améliore le modèle IA
- Dimanche 3h : Ré-entraînement automatique
- Confiance augmente au fil du temps

---

## 📊 ÉTAPE 4 : CONSULTER MÉTRIQUES (2 min)

### **Accès** :

```
URL: /dashboard/company/{public_id}/dispatch/rl-metrics
```

### **Que voir** :

1. **KPIs** (4 cards en haut) :

   - Total suggestions générées
   - Confiance moyenne (%)
   - Taux application (%)
   - Précision gain (%)

2. **Graphiques** :

   - **LineChart** : Évolution confiance par jour
   - **PieChart** : DQN vs Heuristique

3. **Alertes** (si présentes) :

   - 🚨 Rouge : Action urgente requise
   - ⚠️ Orange : Attention nécessaire
   - ✅ Vert : Tout va bien

4. **Stats détaillées** :
   - Suggestions appliquées/rejetées
   - Gains temps (estimé vs réel)
   - Performance modèle

### **Sélecteur période** :

- **7 jours** : Vue court terme
- **30 jours** : Vue moyen terme (défaut)
- **90 jours** : Vue long terme

---

## ⚙️ ÉTAPE 5 : PERSONNALISER (AVANCÉ) (5 min)

### **Accès** :

Dans page Dispatch, cliquer **"⚙️ Avancé"**

### **Paramètres disponibles** :

#### **🎯 Heuristique** (5 params)

**Quand modifier** : Vous voulez favoriser certains critères

**Exemples** :

- Proximité importante → `proximity: 0.5`
- Équité stricte → `driver_load_balance: 0.9`

---

#### **🔧 Solver** (3 params)

**Quand modifier** : Journée compliquée

**Exemples** :

- Beaucoup de courses → `time_limit_sec: 120`
- Chauffeurs surchargés → `max_bookings_per_driver: 8`

---

#### **⏱️ Temps Service** (3 params)

**Quand modifier** : Retards fréquents

**Exemples** :

- Plus de marge → `min_transition_margin_min: 20`
- Clients lents → `pickup_service_min: 10`

---

#### **👥 Pooling** (4 params)

**Quand modifier** : Optimiser regroupements

**Exemples** :

- Désactiver pooling → `enabled: false`
- - de regroupements → `time_tolerance_min: 15`

---

#### **⚖️ Équité** (3 params)

**Quand modifier** : Répartition inégale

**Exemples** :

- Équité forte → `fairness_weight: 0.8`
- Sur 2 semaines → `fairness_window_days: 14`

---

### **Appliquer overrides** :

1. Ajuster paramètres
2. Cliquer "✅ Appliquer ces paramètres"
3. Modal se ferme
4. Bouton devient "⚙️ Paramètres ✓" (vert)
5. Lancer dispatch → Overrides appliqués

### **Reset** :

Cliquer "🔄 Réinitialiser" → Valeurs par défaut

---

## 🔄 CYCLE D'AMÉLIORATION CONTINUE

### **Votre rôle** :

```
Lundi-Dimanche:
  └→ Donner feedbacks sur suggestions
     ├→ 👍 Bonnes suggestions
     ├→ ✅ Appliquer meilleures
     └→ 👎 Rejeter mauvaises

Dimanche 3h:
  └→ Système ré-entraîne automatiquement
     └→ Apprend de vos feedbacks

Lundi 8h:
  └→ Rapport hebdomadaire disponible
     └→ Voir améliorations

Semaine suivante:
  └→ Suggestions plus précises !
     └→ Cercle vertueux 🔄
```

---

## 💡 BONNES PRATIQUES

### **Feedbacks** :

1. **Donnez feedback sur au moins 5-10 suggestions/jour**

   - Plus de feedbacks = modèle meilleur
   - Minimum 50 feedbacks/semaine pour ré-entraînement

2. **Soyez honnête** :

   - 👍 si vraiment bon
   - 👎 si vraiment mauvais
   - Qualité > Quantité

3. **Ajoutez raisons** (rejet) :
   - Aide modèle à comprendre
   - Ex: "Driver trop loin", "Client préfère autre driver"

---

### **Dashboard** :

1. **Consultez quotidiennement** :

   - Matin : Vérifier alertes
   - Fin journée : Vérifier stats

2. **Surveillez alertes** :

   - 🚨 Rouge : Action immédiate
   - ⚠️ Orange : Surveillance accrue

3. **Analysez trends** :
   - Confiance augmente ? ✅ Bon
   - Confiance baisse ? ⚠️ Problème

---

### **Overrides** :

1. **Utilisez avec modération** :

   - Valeurs par défaut sont optimales
   - Changez seulement si besoin spécifique

2. **Testez progressivement** :

   - 1 paramètre à la fois
   - Observez impact
   - Ajustez si besoin

3. **Documentez** :
   - Pourquoi override ?
   - Quel résultat ?
   - À garder ou non ?

---

## ⚠️ RÉSOLUTION PROBLÈMES

### **Problème 1 : Pas de suggestions**

**Cause possible** :

- Aucune assignation active
- Date incorrecte

**Solution** :

1. Vérifier date sélectionnée
2. Vérifier qu'il y a des courses assignées
3. Rafraîchir page

---

### **Problème 2 : Confiance très faible (<50%)**

**Cause possible** :

- Modèle nouveau / pas assez de données
- Situation inhabituelle

**Solution** :

1. Donnez + de feedbacks
2. Attendez ré-entraînement dimanche
3. Consultez dashboard alertes

---

### **Problème 3 : Alertes rouges dashboard**

**Exemple** : "🚨 Taux fallback élevé (25%)"

**Signification** : Modèle DQN échoue souvent, utilise heuristique

**Solution** :

1. Vérifier logs backend
2. Contacter développeur si persiste
3. Continuer feedbacks (aide modèle)

---

## 📞 SUPPORT

### **Questions** :

- 📧 Email technique : [À définir]
- 📖 Documentation : Ce dossier
- 🔧 Issues : [À définir]

### **Ressources** :

- [SYNTHESE_EXECUTIVE.md](./SYNTHESE_EXECUTIVE.md) - Vue d'ensemble
- [REPONSES_QUESTIONS_DETAILLEES.md](./REPONSES_QUESTIONS_DETAILLEES.md) - FAQ complète
- [SUCCES_COMPLET_PHASES_1_2_3.md](./SUCCES_COMPLET_PHASES_1_2_3.md) - Rapport technique

---

## ✅ CHECKLIST QUOTIDIENNE

### **Matin** (5 min) :

- [ ] Consulter dashboard `/rl-metrics`
- [ ] Vérifier alertes
- [ ] Noter anomalies

### **Utilisation** (tout au long de la journée) :

- [ ] Lancer dispatch pour journée
- [ ] Voir suggestions RL
- [ ] Donner 5-10 feedbacks
- [ ] Appliquer bonnes suggestions

### **Fin de journée** (5 min) :

- [ ] Vérifier dashboard à nouveau
- [ ] Confiance moyenne du jour ?
- [ ] Taux application acceptable ?
- [ ] Préparer amélioration demain

---

## 🎯 OBJECTIFS UTILISATEURS

### **Semaine 1** :

- Familiarisation avec dashboard
- Premiers feedbacks (objectif 20+)
- Comprendre code couleur confiance

### **Semaine 2-4** :

- Feedbacks réguliers (50+ /semaine)
- Utilisation active suggestions
- Voir première amélioration modèle

### **Mois 2-3** :

- Confiance moyenne >80%
- Taux application >60%
- Précision >90%
- Satisfaction 4/5

---

## 🏆 SUCCÈS = UTILISATION ACTIVE

**Le système s'améliore UNIQUEMENT avec votre participation** !

- Plus de feedbacks = Modèle meilleur
- Meilleur modèle = Suggestions précises
- Suggestions précises = Moins de travail
- Moins de travail = Plus de temps
- **Gagnant-gagnant** ! 🎊

---

**Bonne utilisation du nouveau système !** 🚀

---

**Auteur** : Assistant IA  
**Date** : 21 octobre 2025  
**Version** : 1.0  
**Type** : Guide Utilisateur
