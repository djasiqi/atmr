# ✅ Implémentation : Sauvegarde permanente des paramètres avancés

**Date** : 21 octobre 2025, 19:00  
**Statut** : ✅ TERMINÉE

---

## 🎯 **OBJECTIF**

Implémenter une sauvegarde **permanente** des paramètres avancés de dispatch dans la base de données, au lieu d'utiliser le localStorage du navigateur.

---

## ✅ **ÉTAPE 1 : Backend API**

### Routes créées

**Fichier** : `backend/routes/dispatch_routes.py`

#### 1. GET `/dispatch/advanced_settings`

Récupère les paramètres sauvegardés pour l'entreprise.

**Response** :

```json
{
  "company_id": 1,
  "dispatch_overrides": {
    "heuristic": { "proximity_weight": 0.3, ... },
    "solver": { "time_limit": 60, ... },
    "emergency": { "allow_emergency": false, ... },
    ...
  }
}
```

#### 2. PUT `/dispatch/advanced_settings`

Sauvegarde les paramètres dans la DB.

**Body** :

```json
{
  "dispatch_overrides": {
    "allow_emergency": false,
    "emergency": { "emergency_per_stop_penalty": 800 },
    "heuristic": { "load_balance_weight": 0.9 },
    ...
  }
}
```

**Response** :

```json
{
  "company_id": 1,
  "dispatch_overrides": { ... },
  "message": "Paramètres avancés sauvegardés avec succès"
}
```

#### 3. DELETE `/dispatch/advanced_settings`

Réinitialise aux valeurs par défaut.

**Response** :

```json
{
  "company_id": 1,
  "message": "Paramètres avancés réinitialisés aux valeurs par défaut"
}
```

### Stockage

Les paramètres sont stockés dans `company.autonomous_config` (colonne JSON) sous la clé `dispatch_overrides` :

```python
# backend/models/company.py
autonomous_config = Column(
    Text,
    nullable=True,
    comment="Configuration JSON pour le dispatch autonome"
)

# Structure :
{
  "auto_dispatch": { ... },
  "rl_dispatch": { ... },
  "dispatch_overrides": {  # ← Nouvelle clé
    "allow_emergency": false,
    "heuristic": { ... },
    "solver": { ... },
    ...
  }
}
```

---

## ✅ **ÉTAPE 2 : Frontend - Page Settings**

### Nouvelle section dans Operations Tab

**Fichier** : `frontend/src/pages/company/Settings/tabs/OperationsTab.jsx`

**Emplacement** : Colonne gauche, sous "📍 Géolocalisation"

**Interface** :

```
⚙️ Configuration Dispatch Avancée
Personnalisez finement les paramètres de dispatch (heuristiques, solver, équité, chauffeurs d'urgence, etc.)

[⚙️ Configurer] [🔄 Réinitialiser]

💡 Aucune configuration personnalisée. Les valeurs par défaut seront utilisées.
    (ou)
✅ Paramètres personnalisés actifs
```

### Fonctionnalités

1. **Bouton "⚙️ Configurer"** : Ouvre le modal `AdvancedSettings`
2. **Bouton "🔄 Réinitialiser"** : Appelle `DELETE /dispatch/advanced_settings`
3. **Chargement automatique** : Au montage du composant, charge les paramètres depuis la DB
4. **Sauvegarde** : Enregistre dans la DB via `PUT /dispatch/advanced_settings`

### Code clé

```javascript
// Charger les paramètres depuis la DB
const loadAdvancedSettings = async () => {
  const { data } = await apiClient.get("/dispatch/advanced_settings");
  setAdvancedSettings(data.dispatch_overrides);
};

// Sauvegarder dans la DB
const saveAdvancedSettings = async (newSettings) => {
  await apiClient.put("/dispatch/advanced_settings", {
    dispatch_overrides: newSettings,
  });
  showSuccess("✅ Paramètres avancés sauvegardés avec succès !");
};

// Réinitialiser
const resetAdvancedSettings = async () => {
  await apiClient.delete("/dispatch/advanced_settings");
  setAdvancedSettings(null);
  showSuccess("✅ Paramètres réinitialisés aux valeurs par défaut");
};
```

---

## ✅ **ÉTAPE 3 : Modification AdvancedSettings**

### Ancien comportement (localStorage)

```javascript
// ❌ AVANT : Sauvegarde dans localStorage
const [overrides, setOverrides] = useState(() => {
  const saved = localStorage.getItem("atmr_dispatch_advanced_settings");
  return saved ? JSON.parse(saved) : initialSettings;
});

const handleApply = () => {
  localStorage.setItem(
    "atmr_dispatch_advanced_settings",
    JSON.stringify(overrides)
  );
  onApply(overrides);
};
```

### Nouveau comportement (DB via parent)

```javascript
// ✅ APRÈS : Le parent décide où sauvegarder
const [overrides, setOverrides] = useState(initialSettings);

const handleApply = () => {
  onApply(overrides); // Le parent sauvegarde en DB ou applique temporairement
};
```

**Fichier** : `frontend/src/pages/company/Dispatch/components/AdvancedSettings.jsx`

**Changements** :

- ✅ Suppression du `STORAGE_KEY` et de la logique localStorage
- ✅ Utilise uniquement `initialSettings` fourni par le parent
- ✅ Délègue la sauvegarde au composant parent via `onApply()`
- ✅ Message mis à jour : "Vous pouvez sauvegarder ces paramètres de manière permanente dans Paramètres → Opérations"

---

## ✅ **ÉTAPE 4 : Chargement automatique dans Dispatch**

### Chargement au montage

**Fichier** : `frontend/src/pages/company/Dispatch/UnifiedDispatchRefactored.jsx`

```javascript
// Charger les paramètres avancés depuis la DB au montage
useEffect(() => {
  const loadAdvancedSettings = async () => {
    try {
      const { data } = await apiClient.get("/dispatch/advanced_settings");
      if (data.dispatch_overrides) {
        setOverrides(data.dispatch_overrides);
        console.log(
          "🔄 [Dispatch] Paramètres avancés chargés depuis la DB:",
          data.dispatch_overrides
        );
      } else {
        console.log(
          "📌 [Dispatch] Aucun paramètre avancé configuré (utilise valeurs par défaut)"
        );
      }
    } catch (err) {
      console.error("[Dispatch] Erreur chargement paramètres avancés:", err);
    }
  };

  loadAdvancedSettings();
}, []);
```

### Application temporaire vs permanente

**Deux façons d'utiliser les paramètres avancés** :

1. **Sauvegarde permanente** (Settings → Opérations)

   - Stocké en DB
   - Chargé automatiquement à chaque ouverture de la page Dispatch
   - Appliqué à **tous les dispatchs futurs**

2. **Application temporaire** (Page Dispatch)
   - Non sauvegardé en DB
   - Appliqué uniquement au **prochain dispatch**
   - Utile pour tester des paramètres sans les sauvegarder

---

## 📊 **AVANTAGES**

### Avant (localStorage)

- ❌ Données perdues si l'utilisateur change de navigateur
- ❌ Pas de synchronisation entre appareils
- ❌ Pas de sauvegarde côté serveur
- ❌ Aucune auditabilité

### Après (DB)

- ✅ **Persistance réelle** : Les données sont sauvegardées côté serveur
- ✅ **Synchronisation multi-appareils** : Même config sur desktop/mobile/tablette
- ✅ **Par entreprise** : Chaque company a ses propres paramètres
- ✅ **Auditabilité** : Logs des modifications
- ✅ **Backup** : Inclus dans les sauvegardes DB
- ✅ **Partage** : Tous les utilisateurs de la même entreprise voient la même config

---

## 🧪 **TESTS RECOMMANDÉS**

### 1. Test de sauvegarde permanente

1. Aller dans **Paramètres → Opérations**
2. Cliquer sur **⚙️ Configurer**
3. Modifier les paramètres (ex: décocher "Autoriser chauffeurs d'urgence")
4. Cliquer sur **✅ Appliquer ces paramètres**
5. **Vérifier** : Message de succès affiché
6. Rafraîchir la page (F5)
7. **Vérifier** : Le badge "✅ Paramètres personnalisés actifs" est visible
8. Aller dans **Dispatch**
9. **Vérifier** : Les paramètres sont chargés automatiquement (voir console)
10. Lancer un dispatch
11. **Vérifier** : Les paramètres sont bien appliqués (ex: Khalid non utilisé)

### 2. Test de réinitialisation

1. Dans **Paramètres → Opérations**
2. Cliquer sur **🔄 Réinitialiser**
3. Confirmer
4. **Vérifier** : Message "Paramètres réinitialisés aux valeurs par défaut"
5. **Vérifier** : Le badge "✅ Paramètres personnalisés actifs" disparaît
6. **Vérifier** : Le message "💡 Aucune configuration personnalisée" apparaît
7. Lancer un dispatch
8. **Vérifier** : Les valeurs par défaut sont utilisées

### 3. Test d'application temporaire

1. Dans **Dispatch**, cliquer sur **⚙️ Paramètres Avancés**
2. Modifier les paramètres
3. Cliquer sur **✅ Appliquer ces paramètres**
4. **Vérifier** : Message "appliqués temporairement ! Pour une sauvegarde permanente..."
5. Lancer un dispatch
6. **Vérifier** : Les paramètres temporaires sont appliqués
7. Rafraîchir la page
8. **Vérifier** : Les paramètres DB (ou valeurs par défaut) sont rechargés

### 4. Test multi-navigateur

1. Sauvegarder des paramètres dans Chrome
2. Ouvrir Firefox
3. Se connecter avec le même compte
4. Aller dans **Dispatch**
5. **Vérifier** : Les mêmes paramètres sont chargés (synchronisation DB)

---

## 📝 **FICHIERS MODIFIÉS**

1. **Backend**

   - `backend/routes/dispatch_routes.py` : Nouvelles routes API (GET/PUT/DELETE)

2. **Frontend**
   - `frontend/src/pages/company/Settings/tabs/OperationsTab.jsx` : Nouvelle section "Configuration Dispatch Avancée"
   - `frontend/src/pages/company/Dispatch/components/AdvancedSettings.jsx` : Suppression localStorage, délégation au parent
   - `frontend/src/pages/company/Dispatch/UnifiedDispatchRefactored.jsx` : Chargement automatique depuis DB

---

## 🔗 **RÉFÉRENCES**

- [Model Company](../../backend/models/company.py) - Champ `autonomous_config`
- [Dispatch Routes](../../backend/routes/dispatch_routes.py) - Routes API
- [Guide Paramètres Avancés](./GUIDE_PARAMETRES_AVANCES.md) - Documentation des paramètres
- [Bug Fix allow_emergency](./FIX_ALLOW_EMERGENCY_IGNORE.md) - Correction du bug d'ignorance du paramètre

---

## ✅ **RÉSULTAT FINAL**

Les utilisateurs peuvent maintenant :

1. ✅ **Sauvegarder définitivement** leurs paramètres avancés dans Settings → Opérations
2. ✅ **Modifier facilement** via un bouton "✏️ Modifier les paramètres"
3. ✅ **Réinitialiser** aux valeurs par défaut en un clic
4. ✅ **Voir l'état** : Badge "✅ Paramètres personnalisés actifs" ou "💡 Aucune configuration"
5. ✅ **Application automatique** : Les paramètres sauvegardés sont chargés à chaque dispatch
6. ✅ **Test rapide** : Possibilité d'appliquer temporairement sans sauvegarder

**Plus besoin de réappliquer les paramètres à chaque fois !** 🎉
