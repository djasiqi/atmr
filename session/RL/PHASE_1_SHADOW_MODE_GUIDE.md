# 🔍 PHASE 1 : SHADOW MODE - GUIDE COMPLET

**Date :** 21 Octobre 2025  
**Durée estimée :** 1 semaine  
**Statut :** ✅ **PRÊT POUR DÉPLOIEMENT**

---

## 🎯 OBJECTIFS PHASE 1

```yaml
Objectif principal: → Valider le modèle DQN en production SANS impact utilisateurs

Approche: → DQN prédit en parallèle du système actuel
  → Enregistrement de toutes les prédictions
  → Comparaison systématique avec décisions réelles
  → Monitoring 24/7 sans intervention

Durée: → 1 semaine minimum
  → Extensible selon résultats

Critère de succès: → Taux d'accord >75% avec système actuel
  → Pas de bugs ou erreurs critiques
  → Performance acceptable (latence <100ms)
```

---

## 📁 COMPOSANTS DÉVELOPPÉS

### 1. Shadow Mode Manager (`services/rl/shadow_mode_manager.py`)

**Fonctionnalités :**

- ✅ Chargement automatique du modèle DQN
- ✅ Prédictions en parallèle (non-bloquantes)
- ✅ Logging de toutes les prédictions
- ✅ Comparaison avec décisions réelles
- ✅ Calcul de métriques de confiance
- ✅ Génération de rapports quotidiens

**Méthodes principales :**

```python
# Prédiction shadow (aucun impact sur le système réel)
prediction = shadow_manager.predict_driver_assignment(
    booking=booking,
    available_drivers=drivers,
    current_assignments=assignments
)

# Comparaison avec décision réelle
comparison = shadow_manager.compare_with_actual_decision(
    prediction=prediction,
    actual_driver_id=assigned_driver.id,
    outcome_metrics=metrics
)

# Statistiques en temps réel
stats = shadow_manager.get_stats()

# Rapport quotidien
report = shadow_manager.generate_daily_report()
```

### 2. Routes API (`routes/shadow_mode_routes.py`)

**Endpoints disponibles :**

```yaml
GET /api/shadow-mode/status:
  → Statut actuel du shadow mode
  → Auth: Admin only

GET /api/shadow-mode/stats:
  → Statistiques détaillées
  → Auth: Admin only

GET /api/shadow-mode/report/<date>:
  → Rapport quotidien pour une date
  → Auth: Admin only

GET /api/shadow-mode/predictions:
  → Liste des prédictions récentes
  → Auth: Admin only

GET /api/shadow-mode/comparisons:
  → Comparaisons DQN vs Réel
  → Filtrage par accord/désaccord
  → Auth: Admin only

POST /api/shadow-mode/reload-model:
  → Recharger le modèle (après réentraînement)
  → Auth: Admin only
```

### 3. Script d'Analyse (`scripts/rl/shadow_mode_analysis.py`)

**Fonctionnalités :**

- ✅ Analyse multi-jours
- ✅ Calcul des taux d'accord
- ✅ Distribution des actions
- ✅ Corrélation confiance/accord
- ✅ Génération de graphiques
- ✅ Rapport JSON complet
- ✅ Recommandations automatiques

**Usage :**

```bash
python scripts/rl/shadow_mode_analysis.py \
  --start-date 20251021 \
  --end-date 20251027 \
  --log-dir data/rl/shadow_mode \
  --output-dir data/rl/shadow_mode/analysis
```

---

## 🚀 DÉPLOIEMENT PHASE 1

### Étape 1 : Préparation

```bash
# 1. Vérifier que le meilleur modèle est en place
ls -lh backend/data/rl/models/dqn_best.pth

# 2. Créer les répertoires nécessaires
mkdir -p backend/data/rl/shadow_mode
mkdir -p backend/data/rl/shadow_mode/analysis

# 3. Vérifier les permissions
chmod 755 backend/data/rl/shadow_mode
```

### Étape 2 : Intégration dans le code de dispatch

**Modifier `routes/dispatch_routes.py` :**

```python
from services.rl.shadow_mode_manager import ShadowModeManager

# Instance globale (ou injection de dépendance)
shadow_manager = ShadowModeManager(
    model_path="data/rl/models/dqn_best.pth",
    log_dir="data/rl/shadow_mode",
    enable_logging=True
)

@dispatch_bp.route('/assign-booking/<int:booking_id>', methods=['POST'])
@jwt_required()
def assign_booking(booking_id):
    """Assigner un booking à un driver (avec shadow mode)."""
    try:
        booking = Booking.query.get_or_404(booking_id)
        available_drivers = get_available_drivers(booking.company_id)

        # ✅ SHADOW MODE: Prédiction DQN (NON-BLOQUANTE)
        try:
            shadow_prediction = shadow_manager.predict_driver_assignment(
                booking=booking,
                available_drivers=available_drivers,
                current_assignments=get_current_assignments()
            )
        except Exception as e:
            logger.warning(f"Shadow mode error: {e}")
            shadow_prediction = None

        # ✅ SYSTÈME ACTUEL: Logique normale (INCHANGÉE)
        assigned_driver = assign_driver_logic(booking, available_drivers)

        # Sauvegarder l'assignation
        booking.driver_id = assigned_driver.id
        db.session.commit()

        # ✅ SHADOW MODE: Comparaison avec décision réelle
        if shadow_prediction:
            try:
                shadow_manager.compare_with_actual_decision(
                    prediction=shadow_prediction,
                    actual_driver_id=assigned_driver.id,
                    outcome_metrics={
                        'distance': calculate_distance(booking, assigned_driver),
                        'estimated_time': estimate_pickup_time(booking, assigned_driver)
                    }
                )
            except Exception as e:
                logger.warning(f"Shadow comparison error: {e}")

        return jsonify({"success": True, "driver_id": assigned_driver.id}), 200

    except Exception as e:
        logger.error(f"Assignment error: {e}")
        return jsonify({"error": str(e)}), 500
```

**Points clés de l'intégration :**

- ✅ Prédictions shadow dans un `try/except` (non-bloquant)
- ✅ Aucun impact sur la logique actuelle
- ✅ Comparaison automatique après décision réelle
- ✅ Logging détaillé pour debugging

### Étape 3 : Enregistrer les routes API

**Dans `app.py` ou `routes_api.py` :**

```python
from routes.shadow_mode_routes import shadow_mode_bp

# Enregistrer le blueprint
app.register_blueprint(shadow_mode_bp)
```

### Étape 4 : Démarrage

```bash
# 1. Redémarrer l'API backend
docker-compose restart api

# 2. Vérifier que le shadow mode est actif
curl -X GET http://localhost:5000/api/shadow-mode/status \
  -H "Authorization: Bearer <admin_token>"

# Réponse attendue:
# {
#   "status": "active",
#   "model_loaded": true,
#   "stats": {
#     "predictions_count": 0,
#     "comparisons_count": 0,
#     "agreement_rate": 0.0
#   }
# }
```

---

## 📊 MONITORING QUOTIDIEN

### Routine Matin (09h00)

```bash
# 1. Récupérer les stats d'hier
curl -X GET "http://localhost:5000/api/shadow-mode/report/$(date -d 'yesterday' +%Y%m%d)" \
  -H "Authorization: Bearer <admin_token>" \
  | jq '.'

# 2. Générer l'analyse complète
docker-compose exec api python scripts/rl/shadow_mode_analysis.py \
  --start-date $(date -d 'yesterday' +%Y%m%d) \
  --end-date $(date -d 'yesterday' +%Y%m%d)

# 3. Vérifier les graphiques générés
ls -lh backend/data/rl/shadow_mode/analysis/*.png
```

### Routine Soir (18h00)

```bash
# 1. Stats en temps réel
curl -X GET "http://localhost:5000/api/shadow-mode/stats" \
  -H "Authorization: Bearer <admin_token>"

# 2. Dernières prédictions
curl -X GET "http://localhost:5000/api/shadow-mode/predictions?limit=10" \
  -H "Authorization: Bearer <admin_token>" \
  | jq '.predictions[] | {booking_id, action_type, confidence}'

# 3. Désaccords récents (pour investigation)
curl -X GET "http://localhost:5000/api/shadow-mode/comparisons?agreement=false&limit=10" \
  -H "Authorization: Bearer <admin_token>"
```

### Analyse Hebdomadaire (Vendredi)

```bash
# Rapport complet de la semaine
docker-compose exec api python scripts/rl/shadow_mode_analysis.py \
  --start-date $(date -d '7 days ago' +%Y%m%d) \
  --end-date $(date +%Y%m%d) \
  --output-dir data/rl/shadow_mode/analysis/week_$(date +%U)

# Visualiser les graphiques
open backend/data/rl/shadow_mode/analysis/week_*/agreement_rate_daily.png
open backend/data/rl/shadow_mode/analysis/week_*/confidence_vs_agreement.png
```

---

## 📈 MÉTRIQUES À SURVEILLER

### Métriques Critiques

```yaml
Taux d'accord global:
  → Objectif: >75%
  → Seuil minimum: >60%
  → Action si <60%: Investigation immédiate

Taux d'accord (haute confiance):
  → Objectif: >90%
  → Confiance >0.8
  → Devrait être très élevé

Performance:
  → Latence shadow prediction: <100ms
  → Pas d'impact sur latence totale
  → Monitoring CPU/RAM

Stabilité:
  → Zéro erreur critique
  → Logs d'erreur shadow < 1%
  → Modèle chargé en permanence
```

### Métriques Secondaires

```yaml
Distribution des actions:
  → Comparer DQN vs Actuel
  → Identifier patterns différents
  → Analyser les désaccords

Confiance moyenne:
  → Objectif: >0.7
  → Stable dans le temps
  → Cohérente avec accord

Volume:
  → >100 prédictions/jour minimum
  → Représentatif de la production
  → Couverture tous types de bookings
```

---

## 🔍 ANALYSES RECOMMANDÉES

### Analyse 1 : Taux d'accord par type de booking

```python
# Grouper par priorité, heure, distance, etc.
comparisons_df['booking_priority'] = comparisons_df['booking_id'].apply(
    lambda x: get_booking_priority(x)
)
agreement_by_priority = comparisons_df.groupby('booking_priority')['agreement'].mean()
```

### Analyse 2 : Désaccords à haute confiance

```python
# Cas où DQN est très confiant mais différent du système
high_conf_disagree = comparisons_df[
    (comparisons_df['confidence'] > 0.8) &
    (comparisons_df['agreement'] == False)
]

# Investigation manuelle de ces cas
for _, case in high_conf_disagree.iterrows():
    print(f"Booking {case['booking_id']}")
    print(f"  DQN predict : Driver {case['predicted_driver_id']}")
    print(f"  Actual      : Driver {case['actual_driver_id']}")
    print(f"  Confidence  : {case['confidence']:.2f}")
```

### Analyse 3 : Patterns temporels

```python
# Accord par heure de la journée
comparisons_df['hour'] = pd.to_datetime(
    comparisons_df['timestamp']
).dt.hour

agreement_by_hour = comparisons_df.groupby('hour')['agreement'].mean()

plt.plot(agreement_by_hour.index, agreement_by_hour.values)
plt.title('Taux d\'accord par heure')
plt.xlabel('Heure de la journée')
plt.ylabel('Taux d\'accord')
```

---

## ⚠️ PROBLÈMES POTENTIELS & SOLUTIONS

### Problème 1 : Taux d'accord <60%

**Causes possibles :**

- Modèle pas adapté aux données réelles
- Fonction de reward mal alignée
- Système actuel a évolué depuis le training

**Actions :**

1. Analyser les types de désaccords
2. Comparer outcomes (distance, délai) DQN vs Actuel
3. Si DQN meilleur: continuer Phase 1
4. Si Actuel meilleur: investiguer et potentiellement réentraîner

### Problème 2 : Erreurs fréquentes

**Causes possibles :**

- État incompatible avec le modèle
- Drivers/bookings avec features manquantes
- Problèmes de performance

**Actions :**

1. Vérifier les logs d'erreur détaillés
2. Améliorer la construction de l'état
3. Ajouter validation des données d'entrée
4. Désactiver temporairement si critique

### Problème 3 : Performance dégradée

**Causes possibles :**

- Prédiction DQN trop lente
- Trop de logging
- Modèle trop gros

**Actions :**

1. Profiler le code (cProfile)
2. Réduire verbosité du logging
3. Optimiser la construction de l'état
4. Envisager inférence batch

---

## ✅ CRITÈRES DE PASSAGE À PHASE 2

### Critères Techniques

```yaml
Taux d'accord: ✅ >75% global
  ✅ >85% sur bookings haute priorité
  ✅ >90% sur prédictions haute confiance

Stabilité: ✅ Zéro erreur critique pendant 7 jours
  ✅ Latence <100ms sur 99% des prédictions
  ✅ Pas d'impact sur performance système

Volume: ✅ >1000 prédictions sur 7 jours
  ✅ Couverture représentative
  ✅ Tous types de scénarios testés
```

### Critères Business

```yaml
Outcomes comparables ou meilleurs: ✅ Distance moyenne DQN ≤ Actuel +10%
  ✅ Délai pickup DQN ≤ Actuel +5%
  ✅ Satisfaction drivers stable

Confiance équipe: ✅ Admins confortables avec les prédictions
  ✅ Cas de désaccords bien compris
  ✅ Plan d'action clair si problème en Phase 2
```

### Décision Go/No-Go

**✅ GO vers Phase 2 si :**

- Tous les critères techniques ✅
- Au moins 2/3 critères business ✅
- Équipe confiante

**⏸️ PAUSE si :**

- Taux d'accord 60-75%
- Analyser 1 semaine de plus
- Comprendre les désaccords

**❌ NO-GO si :**

- Taux d'accord <60%
- Erreurs critiques fréquentes
- Performance dégradée
- → Retour au développement/réentraînement

---

## 📝 CHECKLIST PHASE 1

### Avant Déploiement

- [ ] Modèle `dqn_best.pth` vérifié et testé
- [ ] Code shadow mode intégré dans dispatch
- [ ] Routes API enregistrées et testées
- [ ] Répertoires de logs créés
- [ ] Scripts d'analyse testés
- [ ] Documentation lue et comprise par l'équipe
- [ ] Plan de monitoring défini
- [ ] Alertes configurées (optionnel)

### Pendant Phase 1 (Quotidien)

- [ ] Vérifier statut shadow mode (matin)
- [ ] Consulter rapport quotidien (matin)
- [ ] Vérifier stats temps réel (soir)
- [ ] Investiguer désaccords critiques
- [ ] Monitorer performance système
- [ ] Logger observations importantes

### Fin de Semaine 1

- [ ] Générer rapport hebdomadaire complet
- [ ] Analyser tous les graphiques
- [ ] Calculer métriques de décision
- [ ] Réunion équipe: GO/NO-GO Phase 2
- [ ] Documenter learnings
- [ ] Préparer Phase 2 si GO

---

## 🎯 RÉSUMÉ PHASE 1

```
╔═══════════════════════════════════════════════╗
║  🔍 PHASE 1 : SHADOW MODE                     ║
║                                               ║
║  ✅ Objectif: Validation production           ║
║  ⏱️  Durée: 1 semaine minimum                 ║
║  🎯 Critère: Taux d'accord >75%               ║
║  📊 Monitoring: 24/7 automatique              ║
║  💼 Impact: ZÉRO sur utilisateurs             ║
║                                               ║
║  🚀 PRÊT POUR DÉPLOIEMENT                     ║
╚═══════════════════════════════════════════════╝
```

---

_Phase 1 Shadow Mode - Guide créé le 21 octobre 2025_  
_Prêt pour déploiement production_ ✅  
_Prochaine étape : Intégration + Monitoring 1 semaine_ 🚀
