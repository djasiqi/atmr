# 📖 GUIDE COMPLET - PARAMÈTRES AVANCÉS DU DISPATCH

## 🎯 Vue d'ensemble

Les paramètres avancés vous permettent de **personnaliser finement** le comportement de l'algorithme de dispatch selon vos besoins spécifiques. Chaque paramètre influence la décision d'assignation des courses aux chauffeurs.

---

## 1️⃣ POIDS HEURISTIQUE (Heuristic Weights)

Ces paramètres définissent l'**importance relative** de chaque critère dans le calcul du "score" d'une assignation. L'algorithme additionne tous ces scores pour choisir le meilleur chauffeur.

### 📏 **Proximité** (proximity)

- **Valeur par défaut** : `0.2` (20%)
- **Plage** : `0.0` à `1.0`

**À quoi ça sert ?**
Mesure l'importance de la **distance géographique** entre le chauffeur et le lieu de pickup du client.

**Impact concret** :

- **Valeur élevée (0.5-1.0)** → Le système privilégie les chauffeurs **les plus proches** du client
  - ✅ Réduit temps d'attente client
  - ✅ Économise carburant
  - ❌ Peut créer déséquilibre de charge
- **Valeur faible (0.0-0.3)** → La distance n'est pas prioritaire
  - ✅ Permet meilleure optimisation globale
  - ❌ Clients peuvent attendre plus longtemps

**Exemple** :

```
🚗 Chauffeur A : 2 km du client → Score proximité = 0.9 × 0.2 = 0.18
🚙 Chauffeur B : 8 km du client → Score proximité = 0.4 × 0.2 = 0.08
```

**Quand l'ajuster ?**

- **Urgences médicales** : Augmenter à `0.7-0.9` (rapidité cruciale)
- **Transport de groupe** : Baisser à `0.1-0.2` (optimisation globale)
- **Zone rurale dense** : Augmenter à `0.4-0.6` (distances importantes)

---

### ⚖️ **Équilibre charge** (driver_load_balance)

- **Valeur par défaut** : `0.7` (70%)
- **Plage** : `0.0` à `1.0`

**À quoi ça sert ?**
Mesure l'importance de la **répartition équilibrée** des courses entre tous les chauffeurs disponibles.

**Impact concret** :

- **Valeur élevée (0.6-1.0)** → Équilibre strict entre chauffeurs
  - ✅ Chauffeurs reçoivent nombre similaire de courses
  - ✅ Évite surcharge/sous-utilisation
  - ❌ Peut augmenter distance totale parcourue
- **Valeur faible (0.0-0.3)** → Priorité à l'efficacité
  - ✅ Optimisation géographique maximale
  - ❌ Certains chauffeurs surchargés, autres inactifs

**Exemple** :

```
Jour J :
🚗 Chauffeur A : 8 courses déjà assignées → Score charge = 0.2 × 0.7 = 0.14
🚙 Chauffeur B : 2 courses déjà assignées → Score charge = 0.9 × 0.7 = 0.63
→ Chauffeur B sera favorisé pour équilibrer
```

**Quand l'ajuster ?**

- **Petite équipe (3-5 chauffeurs)** : Augmenter à `0.8-1.0` (équité importante)
- **Grande équipe (15+ chauffeurs)** : Baisser à `0.4-0.6` (optimisation prioritaire)
- **Contrats horaires** : Augmenter à `0.9` (répartition stricte)

---

### 🏆 **Priorité** (priority)

- **Valeur par défaut** : `0.06` (6%)
- **Plage** : `0.0` à `1.0`

**À quoi ça sert ?**
Mesure l'importance des **courses prioritaires** (médicales, VIP, urgences).

**Impact concret** :

- **Valeur élevée (0.3-1.0)** → Les courses urgentes/VIP sont **toujours** assignées en premier
  - ✅ Garantit service premium
  - ✅ Respect protocoles médicaux
  - ❌ Peut créer retards sur courses normales
- **Valeur faible (0.0-0.1)** → Toutes les courses sont traitées également
  - ✅ Optimisation globale
  - ❌ Urgences peuvent être retardées

**Exemple** :

```
Course normale :    is_priority=False → Score priorité = 0.0 × 0.06 = 0.00
Course médicale :   is_priority=True  → Score priorité = 1.0 × 0.06 = 0.06
Course VIP client : is_priority=True  → Score priorité = 1.0 × 0.06 = 0.06
```

**Quand l'ajuster ?**

- **Transport médical** : Augmenter à `0.5-0.9` (sécurité avant tout)
- **VIP/corporate** : Augmenter à `0.3-0.5` (service premium)
- **Transport scolaire** : Baisser à `0.02` (toutes courses égales)

---

## 2️⃣ OPTIMISEUR OR-TOOLS (Solver Settings)

Ces paramètres contrôlent le comportement du **solveur d'optimisation** Google OR-Tools, qui calcule la solution mathématiquement optimale.

### ⏱️ **Temps limite** (time_limit_sec)

- **Valeur par défaut** : `60` secondes
- **Plage** : `10` à `300` secondes

**À quoi ça sert ?**
Définit le **temps maximal** que le solveur peut utiliser pour trouver la meilleure solution.

**Impact concret** :

- **Valeur élevée (120-300s)** → Solution plus optimale
  - ✅ Meilleur résultat mathématique
  - ✅ Économies maximales
  - ❌ Dispatch prend plus de temps
- **Valeur faible (10-30s)** → Solution rapide mais moins optimale
  - ✅ Dispatch quasi-instantané
  - ❌ Peut manquer optimisations
  - ❌ Coûts potentiellement supérieurs

**Exemple** :

```
Avec 20 courses à dispatcher :
- 10s  → Solution à 85% d'optimalité (bonne mais perfectible)
- 60s  → Solution à 95% d'optimalité (très bonne)
- 180s → Solution à 98% d'optimalité (quasi-parfaite)
```

**Quand l'ajuster ?**

- **Dispatch temps réel** : Baisser à `20-30s` (rapidité critique)
- **Dispatch planifié (J-1)** : Augmenter à `120-180s` (qualité prioritaire)
- **Petite flotte (<10 courses)** : Baisser à `30s` (suffit largement)
- **Grande flotte (50+ courses)** : Augmenter à `180-300s` (complexité élevée)

---

### 🚗 **Courses max par chauffeur** (max_bookings_per_driver)

- **Valeur par défaut** : `6` courses
- **Plage** : `1` à `12` courses

**À quoi ça sert ?**
Limite le **nombre maximal de courses** qu'un seul chauffeur peut recevoir dans une journée.

**Impact concret** :

- **Valeur élevée (8-12)** → Chauffeurs peuvent enchaîner beaucoup de courses
  - ✅ Utilisation maximale de la flotte
  - ✅ Moins de chauffeurs nécessaires
  - ❌ Risque fatigue/retards
  - ❌ Pression sur chauffeurs
- **Valeur faible (3-5)** → Charge de travail limitée
  - ✅ Chauffeurs moins stressés
  - ✅ Respect temps de pause
  - ❌ Plus de chauffeurs nécessaires
  - ❌ Coût masse salariale

**Exemple** :

```
15 courses à dispatcher :
- Limite 6 → Besoin minimum de 3 chauffeurs (5+5+5)
- Limite 3 → Besoin minimum de 5 chauffeurs (3+3+3+3+3)
```

**Quand l'ajuster ?**

- **Durée moyenne longue (>45min)** : Baisser à `3-4` (éviter fatigue)
- **Courtes distances urbaines** : Augmenter à `8-10` (rotation rapide)
- **Règles syndicales/légales** : Ajuster selon contrats
- **Période de pointe** : Augmenter temporairement à `8-10`

---

### 💰 **Pénalité non-assigné** (unassigned_penalty_base)

- **Valeur par défaut** : `10000`
- **Plage** : `1000` à `50000`

**À quoi ça sert ?**
Définit le "coût virtuel" attribué à une course **non assignée** dans le calcul d'optimisation. Plus cette valeur est élevée, plus le système **évite absolument** de laisser des courses sans chauffeur.

**Impact concret** :

- **Valeur élevée (20000-50000)** → Le système **DOIT** assigner toutes les courses
  - ✅ Zéro course non-assignée (sauf impossible)
  - ✅ Satisfaction client maximale
  - ❌ Peut créer assignations sous-optimales
  - ❌ Chauffeurs surchargés
- **Valeur faible (1000-5000)** → Accepte de laisser courses difficiles non-assignées
  - ✅ Solution plus équilibrée
  - ✅ Meilleure qualité d'assignations
  - ❌ Courses peuvent rester orphelines

**Exemple** :

```
Scénario : 10 courses, 3 chauffeurs disponibles

Pénalité 5000 :
→ 2 courses non-assignées (trop loin, pas de chauffeur optimal)
→ 8 courses bien assignées, chauffeurs détendus

Pénalité 30000 :
→ 0 courses non-assignées (toutes forcées)
→ 10 courses assignées, mais 2 chauffeurs surchargés
```

**Quand l'ajuster ?**

- **Engagement client 100%** : Augmenter à `30000-50000` (aucun refus)
- **Optimisation qualité** : Baisser à `5000-8000` (mieux vaut bien faire)
- **Période test** : Baisser à `3000` (voir limites système)
- **Flotte insuffisante** : Baisser temporairement (évite blocages)

---

## 3️⃣ TEMPS DE SERVICE (Service Times)

Ces paramètres définissent les **durées moyennes** des opérations, essentielles pour le calcul des horaires.

### 📥 **Pickup** (pickup_service_min)

- **Valeur par défaut** : `5` minutes
- **Plage** : `1` à `30` minutes

**À quoi ça sert ?**
Temps moyen pour **embarquer un client** (salutations, aide montée, installation, vérification).

**Impact concret** :

```
Pickup prévu 14:00
Service 5 min
→ Départ réel : 14:05

Si sous-estimé (3 min) mais réalité (7 min) :
→ Retards en cascade toute la journée
```

**Quand l'ajuster ?**

- **Clients autonomes** : `3-4 min`
- **PMR (fauteuil roulant)** : `10-15 min` (installation équipement)
- **Personnes âgées** : `7-10 min` (aide, patience)
- **Transport médical** : `8-12 min` (vérifications sécurité)

---

### 📤 **Dropoff** (dropoff_service_min)

- **Valeur par défaut** : `10` minutes
- **Plage** : `1` à `30` minutes

**À quoi ça sert ?**
Temps moyen pour **déposer un client** (aide descente, accompagnement entrée, paiement si applicable).

**Impact concret** :

```
Arrivée 15:00
Service 10 min
→ Chauffeur libre : 15:10 pour prochaine course
```

**Quand l'ajuster ?**

- **Dropoff simple (domicile)** : `5-7 min`
- **Hôpital/clinique** : `12-20 min` (attente parking, accompagnement)
- **Aéroport** : `15-25 min` (trafic, déchargement bagages)
- **Personne âgée** : `10-15 min` (accompagnement sécurisé)

---

### ⏳ **Marge transition** (min_transition_margin_min)

- **Valeur par défaut** : `15` minutes
- **Plage** : `5` à `60` minutes

**À quoi ça sert ?**
**Temps minimum** requis entre le dropoff d'une course et le pickup de la suivante. Inclut : trajet + imprévus + pause éventuelle.

**Impact concret** :

```
Course 1 : Dropoff 14:00
Marge : 15 min
→ Prochaine course possible : 14:15 minimum

Si marge trop courte (5 min) :
→ Retards permanents (trafic, imprévus)
→ Chauffeurs stressés
```

**Quand l'ajuster ?**

- **Zone urbaine dense** : Augmenter à `20-25 min` (trafic imprévisible)
- **Zone rurale fluide** : Baisser à `10-12 min` (circulation fluide)
- **Période de pointe** : Augmenter à `25-30 min` (embouteillages)
- **Nuit/weekend** : Baisser à `10 min` (routes dégagées)

---

## 4️⃣ REGROUPEMENT DE COURSES (Ride-Pooling)

Permet de **combiner plusieurs clients** dans un même véhicule pour optimiser coûts et écologie.

### ✅ **Activer le regroupement** (enabled)

- **Valeur par défaut** : `true` (activé)

**À quoi ça sert ?**
Active/désactive la fonctionnalité de **partage de course** (plusieurs clients, un chauffeur).

**Impact concret** :

- **Activé** → Économies 30-40%, écologie
  - ✅ Moins de véhicules nécessaires
  - ✅ Réduction CO2
  - ❌ Temps trajet légèrement allongé
- **Désactivé** → Service individuel premium
  - ✅ Trajet direct pour chaque client
  - ❌ Coût plus élevé
  - ❌ Plus de véhicules requis

**Quand le désactiver ?**

- Transport médical sensible
- Clients VIP/corporate
- Pandémie (distanciation)

---

### ⏰ **Tolérance temporelle** (time_tolerance_min)

- **Valeur par défaut** : `10` minutes
- **Plage** : `5` à `30` minutes

**À quoi ça sert ?**
**Écart maximal** autorisé entre les heures de pickup de deux clients regroupés.

**Exemple** :

```
Tolérance 10 min :

Client A : Pickup 14:00
Client B : Pickup 14:08 ✅ (écart 8 min, OK)
Client C : Pickup 14:15 ❌ (écart 15 min, trop tard)

→ A et B peuvent être groupés, pas C
```

**Quand l'ajuster ?**

- **Service express** : Baisser à `5 min` (attente minimale)
- **Économie maximale** : Augmenter à `20-30 min` (plus de possibilités)
- **Horaires scolaires** : Baisser à `5 min` (ponctualité stricte)

---

### 📍 **Distance pickup max** (pickup_distance_m)

- **Valeur par défaut** : `500` mètres
- **Plage** : `100` à `2000` mètres

**À quoi ça sert ?**
**Distance géographique maximale** entre les lieux de pickup de deux clients pour être regroupés.

**Exemple** :

```
Distance max 500m :

Client A : 123 Rue Principale
Client B : 150 Rue Principale (200m) ✅ Regroupement possible
Client C : Quartier voisin (1.2 km) ❌ Trop éloigné

→ Chauffeur fait pickup A + B, pas C
```

**Quand l'ajuster ?**

- **Zone urbaine dense** : Baisser à `300m` (circulation lente)
- **Zone rurale/autoroute** : Augmenter à `1000-2000m` (déplacements rapides)
- **Parking limité** : Augmenter à `800m` (éviter allers-retours)

---

### 🔀 **Détour max** (max_detour_min)

- **Valeur par défaut** : `15` minutes
- **Plage** : `5` à `30` minutes

**À quoi ça sert ?**
**Allongement maximal** du temps de trajet d'un client dû au détour pour déposer un autre client.

**Exemple** :

```
Client A seul : 20 min de trajet direct
Client A + B groupés : 32 min (détour pour B)
Détour : 12 min ✅ (< 15 min, OK)

Client A + C groupés : 38 min
Détour : 18 min ❌ (> 15 min, refuse groupement)
```

**Quand l'ajuster ?**

- **Service premium** : Baisser à `5-10 min` (confort client)
- **Transport économique** : Augmenter à `20-25 min` (max économies)
- **Personnes âgées** : Baisser à `10 min` (fatigue limitée)

---

## 5️⃣ ÉQUITÉ CHAUFFEURS (Driver Fairness)

Assure une **répartition juste** du nombre de courses et revenus entre chauffeurs.

### ✅ **Activer l'équité** (enable_fairness)

- **Valeur par défaut** : `true` (activé)

**À quoi ça sert ?**
Active le **système d'équilibrage** qui suit l'historique des courses de chaque chauffeur.

**Impact concret** :

```
Semaine passée :
🚗 Chauffeur A : 45 courses (beaucoup)
🚙 Chauffeur B : 20 courses (peu)

Avec équité activée :
→ Chauffeur B sera favorisé cette semaine pour compenser

Sans équité :
→ Seule l'efficacité compte, A peut continuer à dominer
```

**Quand le désactiver ?**

- Tests/développement
- Flotte mixte (temps plein + temps partiel)
- Commission au volume (compétition voulue)

---

### 📅 **Fenêtre d'équité** (fairness_window_days)

- **Valeur par défaut** : `7` jours
- **Plage** : `1` à `30` jours

**À quoi ça sert ?**
**Période historique** utilisée pour calculer si un chauffeur est en retard ou en avance sur ses collègues.

**Exemple** :

```
Fenêtre 7 jours (semaine glissante) :

Lundi 21/10 : Système regarde 14/10 → 21/10
🚗 A : 30 courses sur 7 jours
🚙 B : 25 courses sur 7 jours
→ B sera favorisé aujourd'hui

Fenêtre 1 jour (quotidien) :
→ Regarde seulement hier
→ Rééquilibrage très rapide
```

**Quand l'ajuster ?**

- **Contrat temps partiel** : Augmenter à `14-30 jours` (vue long terme)
- **Rotation rapide** : Baisser à `3-5 jours` (réactivité)
- **Équipe stable** : `7 jours` (standard)
- **Saisonniers** : Baisser à `2-3 jours` (éviter biais)

---

### ⚖️ **Poids équité** (fairness_weight)

- **Valeur par défaut** : `0.3` (30%)
- **Plage** : `0.0` à `1.0`

**À quoi ça sert ?**
**Importance** du critère d'équité dans le score global d'assignation (vs proximité, charge, etc.).

**Impact concret** :

```
Poids 0.8 (élevé) :
→ Équité domine toutes les autres considérations
→ Distribution 100% égalitaire, même si sous-optimal géographiquement

Poids 0.2 (faible) :
→ Équité compte peu
→ Optimisation géographique prioritaire
```

**Quand l'ajuster ?**

- **Syndicat/contrat strict** : Augmenter à `0.6-0.9` (égalité absolue)
- **Startup/flexibilité** : Baisser à `0.1-0.2` (performance first)
- **Mix optimal** : `0.3-0.4` (compromis équitable)

---

## 🎯 SCÉNARIOS D'UTILISATION PRATIQUES

### 🚑 **Scénario 1 : Transport médical urgences**

```yaml
heuristic:
  proximity: 0.8 # Distance critique
  driver_load_balance: 0.4 # Moins important
  priority: 0.9 # Urgences TOUJOURS en premier

solver:
  time_limit_sec: 30 # Rapidité essentielle
  max_bookings_per_driver: 4 # Éviter surcharge
  unassigned_penalty_base: 50000 # Zéro refus

service_times:
  pickup_service_min: 8 # Vérifications sécurité
  dropoff_service_min: 15 # Accompagnement hôpital
  min_transition_margin_min: 20 # Imprévus fréquents

pooling:
  enabled: false # Aucun partage (hygiène)

fairness:
  enable_fairness: true
  fairness_window_days: 7
  fairness_weight: 0.5 # Important (stress)
```

---

### 🏢 **Scénario 2 : Navettes corporate VIP**

```yaml
heuristic:
  proximity: 0.5 # Équilibre
  driver_load_balance: 0.3 # Moins critique
  priority: 0.7 # VIP prioritaires

solver:
  time_limit_sec: 90 # Qualité maximale
  max_bookings_per_driver: 8
  unassigned_penalty_base: 40000 # Service premium

service_times:
  pickup_service_min: 3 # Clients autonomes
  dropoff_service_min: 5
  min_transition_margin_min: 10 # Ponctualité

pooling:
  enabled: false # Service individuel

fairness:
  enable_fairness: true
  fairness_window_days: 14 # Vue long terme
  fairness_weight: 0.4
```

---

### 🌳 **Scénario 3 : Covoiturage écologique**

```yaml
heuristic:
  proximity: 0.3 # Moins important
  driver_load_balance: 0.8 # Utilisation max
  priority: 0.05 # Égalité

solver:
  time_limit_sec: 120 # Optimisation poussée
  max_bookings_per_driver: 10 # Max économies
  unassigned_penalty_base: 5000 # Accepte limites

service_times:
  pickup_service_min: 4
  dropoff_service_min: 6
  min_transition_margin_min: 12

pooling:
  enabled: true # ✅ Cœur du service
  time_tolerance_min: 20 # Flexibilité
  pickup_distance_m: 1000 # Large zone
  max_detour_min: 20 # Acceptable

fairness:
  enable_fairness: true
  fairness_window_days: 7
  fairness_weight: 0.6 # Important (motivation)
```

---

### 👴 **Scénario 4 : Transport personnes âgées**

```yaml
heuristic:
  proximity: 0.6 # Réduire attente
  driver_load_balance: 0.5
  priority: 0.4 # Médicales prioritaires

solver:
  time_limit_sec: 60
  max_bookings_per_driver: 5 # Éviter fatigue chauffeur
  unassigned_penalty_base: 35000

service_times:
  pickup_service_min: 10 # Aide montée lente
  dropoff_service_min: 12 # Accompagnement sécurisé
  min_transition_margin_min: 20 # Imprévus fréquents

pooling:
  enabled: true
  time_tolerance_min: 15 # Un peu de patience
  pickup_distance_m: 300 # Courtes distances
  max_detour_min: 10 # Confort limité

fairness:
  enable_fairness: true
  fairness_window_days: 7
  fairness_weight: 0.4
```

---

## 📊 TABLEAU DE DÉCISION RAPIDE

| Besoin                     | Paramètre clé                                    | Valeur          |
| -------------------------- | ------------------------------------------------ | --------------- |
| **Rapidité maximale**      | `time_limit_sec`                                 | 20-30s          |
| **Qualité optimale**       | `time_limit_sec`                                 | 120-180s        |
| **Aucun refus**            | `unassigned_penalty_base`                        | 40000-50000     |
| **Équité stricte**         | `fairness_weight`                                | 0.6-0.9         |
| **Économies max**          | `pooling.enabled` + `pooling.time_tolerance_min` | true + 20-30    |
| **Service premium**        | `pooling.enabled` + `priority`                   | false + 0.7-0.9 |
| **Chauffeurs détendus**    | `max_bookings_per_driver`                        | 3-5             |
| **Utilisation max flotte** | `max_bookings_per_driver`                        | 8-12            |
| **Zone urbaine dense**     | `min_transition_margin_min`                      | 20-30           |
| **Zone rurale**            | `min_transition_margin_min`                      | 10-15           |

---

## ⚠️ PIÈGES À ÉVITER

### ❌ **Piège 1 : Temps de service sous-estimés**

```
pickup_service_min: 2 → Réalité : 7 min
→ Résultat : Retards en cascade, stress, clients mécontents
```

**Solution** : Toujours prévoir 20% de marge (si réel = 5 min, configurer 6 min)

---

### ❌ **Piège 2 : Pénalité trop faible**

```
unassigned_penalty_base: 2000
→ Résultat : 30% des courses non-assignées (algorithme "abandonne")
```

**Solution** : Minimum 8000-10000 pour forcer assignations

---

### ❌ **Piège 3 : Pooling trop agressif**

```
time_tolerance_min: 30 + max_detour_min: 25
→ Résultat : Clients attendent 30 min, trajets 2× plus longs
```

**Solution** : Commencer conservateur (10/15), ajuster progressivement

---

### ❌ **Piège 4 : Tous les poids à 1.0**

```
proximity: 1.0 + driver_load_balance: 1.0 + priority: 1.0
→ Résultat : Système confus, comportement erratique
```

**Solution** : Total des poids ≈ 1.0 (ex: 0.3 + 0.6 + 0.1)

---

## 💡 CONSEILS FINAUX

1. **Commencer par défaut** → Tester 1 semaine → Ajuster petit à petit
2. **Un paramètre à la fois** → Isoler les effets
3. **Documenter changements** → Savoir ce qui marche
4. **A/B testing** → Comparer anciennes vs nouvelles valeurs
5. **Écouter chauffeurs** → Ils connaissent le terrain

---

## 🔗 RESSOURCES

- **Documentation backend** : `backend/services/unified_dispatch/merge_overrides.py`
- **Tests dispatch** : `/dashboard/company/{id}/dispatch`
- **Métriques qualité** : `/dashboard/company/{id}/dispatch/rl-metrics`

---

**📌 Note** : Ces paramètres s'appliquent **uniquement au prochain dispatch**. Pour les sauvegarder de manière permanente, utilisez la page "Configuration Dispatch" dans les paramètres de l'entreprise.
