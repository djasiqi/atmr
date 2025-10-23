# Plan d'Amélioration du Système de Dispatch

## Vue d'ensemble

Ce document présente un plan complet d'amélioration du système de dispatch, couvrant les aspects backend (moteur de dispatch, performance, observabilité) et frontend (architecture, UX, tests).

---

## Phase 1 : Corrections Critiques (Bugs)

### 1.1 Correction du bug de verrou Redis

**Problème** : Le verrou dans `_release_day_lock` peut échouer silencieusement si Redis est indisponible.

**Solution** :

- Ligne 89-93 de `backend/services/unified_dispatch/engine.py`
- Ajouter gestion d'erreur avec logging approprié
- Implémenter fallback gracieux

**Priorité** : ✅ **CRITIQUE - Complété**

### 1.2 Migration des verrous vers Redis distribué

**Problème** : Les verrous en mémoire (`threading.Lock`) ne fonctionnent pas dans un environnement distribué multi-processus.

**Solution** :

- Remplacer les verrous en mémoire par des verrous Redis (`SETNX`)
- Implémenter TTL pour éviter les blocages permanents
- Fichier : `backend/services/unified_dispatch/engine.py`

**Priorité** : ✅ **HAUTE - Complété**

---

## Phase 2 : Performance et Robustesse Backend

### 2.1 Anti-duplication des runs Celery

**Problème** : Plusieurs runs identiques peuvent être enfilés simultanément.

**Solution** :

- Utiliser Redis pour détecter les doublons via hash des paramètres
- Fichier : `backend/services/unified_dispatch/queue.py`
- Implémenter clé de déduplication : `dispatch:enqueued:{company_id}:{params_hash}`

**Priorité** : ✅ **HAUTE - Complété**

### 2.2 Cache Redis pour les appels OSRM

**Problème** : Les appels OSRM sont coûteux et répétitifs (matrice de distances journalière).

**Solution** :

- Implémenter cache Redis avec TTL de 24h
- Fichier : `backend/services/osrm_client.py`
- Fonctions : `get_distance_time_cached()`, `get_matrix_cached()`

**Priorité** : ✅ **HAUTE - Complété**

### 2.3 Circuit Breaker pour OSRM

**Problème** : Si OSRM tombe, les appels bloquent et font échouer tout le dispatch.

**Solution** :

- Implémenter pattern Circuit Breaker
- Fichier : `backend/services/osrm_client.py`
- États : CLOSED → OPEN → HALF_OPEN

**Priorité** : ✅ **MOYENNE - Complété**

---

## Phase 3 : Externalisation et Configuration

### 3.1 Externalisation des constantes heuristiques

**Problème** : Les constantes (`PICKUP_SERVICE_MIN`, `DROPOFF_SERVICE_MIN`, etc.) sont hardcodées.

**Solution** :

- Créer dataclasses dans `backend/services/unified_dispatch/settings.py`
- Classes : `ServiceTimesSettings`, `PoolingSettings`, `TimeSettings`, etc.
- Permettre configuration par entreprise via API

**Priorité** : ✅ **MOYENNE - Complété**

### 3.2 Validation Marshmallow pour l'API

**Problème** : Paramètres de dispatch non validés, risque d'erreurs silencieuses.

**Solution** :

- Créer schemas Marshmallow : `DispatchRunSchema`, `DispatchOverridesSchema`
- Fichier : `backend/routes/dispatch_routes.py`
- Validation stricte des types et valeurs

**Priorité** : ✅ **HAUTE - Complété**

---

## Phase 4 : Observabilité et Monitoring

### 4.1 Enrichissement des métriques de qualité

**Problème** : On sait combien de courses ne sont pas assignées, mais pas pourquoi.

**Solution** :

- Implémenter `_analyze_unassigned_reasons()` dans `engine.py`
- Raisons détaillées : `no_driver_available`, `capacity_exceeded`, `time_window_infeasible`, etc.
- Exposer via API `/company_dispatch/health`

**Priorité** : ✅ **HAUTE - Complété**

### 4.2 Dashboard de santé du dispatch

**Problème** : Pas de vue d'ensemble de la santé du système.

**Solution** :

- Créer endpoint `/company_dispatch/health`
- Métriques : taux d'assignation moyen, temps d'exécution, disponibilité OSRM
- Endpoint `/company_dispatch/health/trends` pour tendances

**Priorité** : ✅ **MOYENNE - Complété**

### 4.3 Métriques OSRM détaillées

**Problème** : Pas de visibilité sur les appels OSRM (latence, taux d'échec).

**Solution** :

- Logger : nombre d'appels, latence moyenne, taux de cache hit
- Inclure dans les métriques de dispatch
- Ajouter alertes si latence > seuil

**Priorité** : ✅ **MOYENNE - Complété**

---

## Phase 5 : Refactoring Frontend (Architecture)

### 5.1 Découpage en composants réutilisables

**Problème** : `UnifiedDispatch.jsx` est monolithique (1484 lignes).

**Solution** :

- Créer composants modulaires :
  - `DispatchHeader.jsx` : En-tête avec contrôles
  - `DispatchSummary.jsx` : Résumé statistiques
  - `ManualModePanel.jsx` : Interface mode manuel
  - `SemiAutoPanel.jsx` : Interface mode semi-auto
  - `FullyAutoPanel.jsx` : Interface mode automatique

**Priorité** : ✅ **HAUTE - Complété**

### 5.2 Création de hooks personnalisés

**Problème** : Logique métier mélangée avec présentation, états dispersés.

**Solution** :

- Créer hooks réutilisables :
  - `useDispatchData.js` : Chargement des données
  - `useDispatchMode.js` : Gestion du mode
  - `useLiveDelays.js` : Retards en temps réel
  - `useAssignmentActions.js` : Actions assignation/suppression

**Priorité** : ✅ **HAUTE - Complété**

### 5.3 Nettoyage du code mort

**Problème** : `console.log` en production, code commenté inutile.

**Solution** :

- Wrapper `console.log` avec `if (process.env.NODE_ENV === 'development')`
- Supprimer code commenté et variables inutilisées
- Préfixer variables inutilisées avec `_`

**Priorité** : ✅ **BASSE - Complété**

---

## Phase 6 : Tests et Documentation

### 6.1 Tests unitaires backend

**Problème** : Pas de tests pour les fonctions critiques.

**Solution** :

- Fichier : `backend/tests/test_heuristics.py`
- Tests : `_can_be_pooled()`, `_analyze_unassigned_reasons()`, poids heuristiques
- Utiliser pytest avec fixtures

**Priorité** : ✅ **HAUTE - Complété**

### 6.2 Tests d'intégration backend

**Problème** : Pas de tests end-to-end du dispatch.

**Solution** :

- Fichier : `backend/tests/test_dispatch_integration.py`
- Tests : run complet, gestion des verrous, mode heuristic_only, urgences
- Mock OSRM et Redis

**Priorité** : ✅ **HAUTE - Complété**

### 6.3 Documentation technique

**Problème** : Manque de documentation pour nouveaux développeurs.

**Solution** :

- Créer 3 documents :
  - `ARCHITECTURE.md` : Vue d'ensemble du système
  - `RUNBOOK.md` : Guide opérationnel (déploiement, monitoring, troubleshooting)
  - `TUNING.md` : Guide d'optimisation des paramètres

**Priorité** : ✅ **MOYENNE - Complété**

---

## Phase 7 : Améliorations Frontend Avancées (Nouvelle)

### 7.1 Gestion d'état globale (Context/Redux)

**Problème** :

- Les hooks rechargent les données indépendamment
- Pas de single source of truth
- Appels API redondants

**Solution** :

```javascript
// Créer Context Provider
<DispatchProvider>
  <UnifiedDispatchRefactored />
</DispatchProvider>;

// Hook unique pour accéder à l'état
const { state, actions } = useDispatchContext();
```

**Fichiers à créer** :

- `frontend/src/contexts/DispatchContext.jsx`
- `frontend/src/hooks/useDispatchContext.js`

**Impact** : Synchronisation automatique entre composants, moins d'appels API

**Priorité** : 🟡 **MOYENNE**

---

### 7.2 Optimisation des performances

**Problème** :

- Pas de mémoisation des calculs coûteux (tris, filtres)
- Pas de virtualisation pour grandes listes
- Re-renders inutiles

**Solution** :

```javascript
// Mémoisation avec useMemo
const sortedDispatches = useMemo(
  () => dispatches.sort((a, b) => a.scheduled_time - b.scheduled_time),
  [dispatches]
);

// Virtualisation avec react-window
<FixedSizeList height={600} itemCount={dispatches.length} itemSize={80}>
  {DispatchRow}
</FixedSizeList>;
```

**Fichiers concernés** :

- `frontend/src/pages/company/Dispatch/components/ManualModePanel.jsx`
- `frontend/src/pages/company/Dispatch/components/SemiAutoPanel.jsx`
- `frontend/src/pages/company/Dispatch/components/FullyAutoPanel.jsx`

**Dépendances à ajouter** :

```json
{
  "react-window": "^1.8.10"
}
```

**Impact** : Amélioration significative des performances avec 100+ réservations

**Priorité** : 🟡 **MOYENNE**

---

### 7.3 Tests unitaires et d'intégration frontend

**Problème** :

- ❌ Aucun test pour les hooks
- ❌ Aucun test pour les composants
- ❌ Pas de tests d'intégration
- Risque élevé de régressions

**Solution** :

```javascript
// Tests pour hooks
describe("useDispatchData", () => {
  it("should load dispatches on mount", async () => {
    const { result, waitForNextUpdate } = renderHook(() =>
      useDispatchData("2024-01-15", "manual")
    );

    await waitForNextUpdate();
    expect(result.current.dispatches).toHaveLength(5);
  });
});

// Tests pour composants
describe("DispatchHeader", () => {
  it("should trigger dispatch on button click", () => {
    const onRunDispatch = jest.fn();
    const { getByText } = render(
      <DispatchHeader onRunDispatch={onRunDispatch} />
    );

    fireEvent.click(getByText("Lancer le Dispatch"));
    expect(onRunDispatch).toHaveBeenCalled();
  });
});
```

**Fichiers à créer** :

- `frontend/src/hooks/__tests__/useDispatchData.test.js`
- `frontend/src/hooks/__tests__/useDispatchMode.test.js`
- `frontend/src/hooks/__tests__/useLiveDelays.test.js`
- `frontend/src/hooks/__tests__/useAssignmentActions.test.js`
- `frontend/src/pages/company/Dispatch/components/__tests__/DispatchHeader.test.jsx`
- `frontend/src/pages/company/Dispatch/components/__tests__/ManualModePanel.test.jsx`

**Dépendances à ajouter** :

```json
{
  "@testing-library/react": "^14.0.0",
  "@testing-library/react-hooks": "^8.0.1",
  "@testing-library/jest-dom": "^6.1.5"
}
```

**Impact** : Prévention des régressions, confiance dans les modifications

**Priorité** : 🔴 **HAUTE**

---

### 7.4 Mise à jour optimiste de l'UI

**Problème** :

- L'UI attend toujours la réponse serveur avant mise à jour
- Sensation de lenteur pour l'utilisateur
- Pas de feedback immédiat

**Solution** :

```javascript
const onAssignDriver = async (reservationId, driverId) => {
  // 1. Mise à jour immédiate de l'UI (optimiste)
  setDispatches((prev) =>
    prev.map((d) =>
      d.id === reservationId
        ? { ...d, driver_id: driverId, status: "assigned" }
        : d
    )
  );

  try {
    // 2. Appel API
    await handleAssignDriver(reservationId, driverId);
    // Succès : l'UI est déjà à jour
  } catch (error) {
    // 3. Rollback en cas d'erreur
    loadDispatches();
    toast.error("Erreur lors de l'assignation");
  }
};
```

**Fichiers concernés** :

- `frontend/src/hooks/useAssignmentActions.js`
- `frontend/src/pages/company/Dispatch/UnifiedDispatchRefactored.jsx`

**Impact** : UX beaucoup plus réactive et fluide

**Priorité** : 🟢 **BASSE**

---

### 7.5 Amélioration du feedback visuel et UX

**Problème** :

- Pas de skeleton loaders pendant le chargement (écran blanc)
- Notifications basiques avec `alert()` (non professionnel)
- Pas d'animations pour les transitions (UX rigide)

**Solution** :

#### Skeleton Loaders

```javascript
{
  loading ? (
    <DispatchTableSkeleton rows={10} />
  ) : (
    <DispatchTable data={dispatches} />
  );
}
```

#### Toast Notifications

```javascript
import toast from "react-hot-toast";

// Succès
toast.success("✅ Chauffeur assigné avec succès!");

// Erreur
toast.error("❌ Erreur lors de l'assignation");

// Loading
const toastId = toast.loading("Assignation en cours...");
// Plus tard...
toast.success("✅ Terminé!", { id: toastId });
```

#### Animations

```javascript
import { motion } from "framer-motion";

<motion.div
  initial={{ opacity: 0, y: 20 }}
  animate={{ opacity: 1, y: 0 }}
  exit={{ opacity: 0, y: -20 }}
  transition={{ duration: 0.3 }}
>
  <DispatchRow {...dispatch} />
</motion.div>;
```

**Fichiers à créer** :

- `frontend/src/components/SkeletonLoaders/DispatchTableSkeleton.jsx`
- `frontend/src/components/SkeletonLoaders/DispatchCardSkeleton.jsx`

**Fichiers à modifier** :

- Tous les composants de panneaux pour ajouter animations
- `frontend/src/App.jsx` pour ajouter `<Toaster />` global

**Dépendances à ajouter** :

```json
{
  "react-hot-toast": "^2.4.1",
  "framer-motion": "^10.16.16"
}
```

**Impact** : Expérience utilisateur moderne et professionnelle

**Priorité** : 🟡 **MOYENNE**

---

### 7.6 Gestion d'erreurs robuste

**Problème** :

- Erreurs affichées simplement dans un `<div>`
- Pas de retry automatique en cas d'échec réseau
- Erreurs React non capturées (crash de l'application)

**Solution** :

#### Error Boundary

```javascript
// frontend/src/components/ErrorBoundary.jsx
class ErrorBoundary extends React.Component {
  state = { hasError: false, error: null };

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error("Error caught by boundary:", error, errorInfo);
    // Optionnel : envoyer à un service de logging (Sentry, etc.)
  }

  render() {
    if (this.state.hasError) {
      return <ErrorFallback error={this.state.error} />;
    }
    return this.props.children;
  }
}

// Utilisation
<ErrorBoundary>
  <UnifiedDispatchRefactored />
</ErrorBoundary>;
```

#### Retry avec Exponential Backoff

```javascript
// frontend/src/utils/retry.js
export async function retryWithBackoff(fn, retries = 3, delay = 1000) {
  for (let i = 0; i < retries; i++) {
    try {
      return await fn();
    } catch (error) {
      if (i === retries - 1) throw error;

      const backoffDelay = delay * Math.pow(2, i);
      await new Promise((resolve) => setTimeout(resolve, backoffDelay));
    }
  }
}

// Utilisation dans hooks
const loadDispatches = async () => {
  try {
    const data = await retryWithBackoff(
      () => fetchAssignedReservations(date),
      3,
      1000
    );
    setDispatches(data);
  } catch (error) {
    setError("Échec du chargement après 3 tentatives");
  }
};
```

**Fichiers à créer** :

- `frontend/src/components/ErrorBoundary.jsx`
- `frontend/src/components/ErrorFallback.jsx`
- `frontend/src/utils/retry.js`

**Fichiers à modifier** :

- Tous les hooks pour intégrer retry
- `frontend/src/App.jsx` pour wrapper avec ErrorBoundary

**Impact** : Application beaucoup plus résiliente et fiable

**Priorité** : 🔴 **HAUTE**

---

### 7.7 Accessibilité (A11y)

**Problème** :

- Pas de support clavier complet (navigation par Tab, Enter, Esc)
- Pas d'attributs ARIA (mauvais pour lecteurs d'écran)
- Gestion du focus manquante pour modals
- Non conforme WCAG 2.1

**Solution** :

#### Attributs ARIA

```javascript
<button
  aria-label="Lancer le dispatch automatique"
  aria-busy={isDispatching}
  aria-describedby="dispatch-help-text"
  disabled={isDispatching}
>
  {isDispatching ? 'En cours...' : 'Lancer le Dispatch'}
</button>

<div id="dispatch-help-text" className="sr-only">
  Lance l'algorithme d'optimisation pour assigner les réservations aux chauffeurs
</div>
```

#### Gestion du Focus

```javascript
const assignModalRef = useRef();

useEffect(() => {
  if (selectedReservation) {
    // Focus automatique sur le modal
    assignModalRef.current?.focus();

    // Trap focus dans le modal
    const handleTab = (e) => {
      if (e.key === "Tab") {
        // Logique pour garder focus dans modal
      }
    };

    document.addEventListener("keydown", handleTab);
    return () => document.removeEventListener("keydown", handleTab);
  }
}, [selectedReservation]);
```

#### Support Clavier

```javascript
const handleKeyDown = (e, dispatch) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    handleSelectDispatch(dispatch);
  }
};

<div
  role="button"
  tabIndex={0}
  onKeyDown={(e) => handleKeyDown(e, dispatch)}
  onClick={() => handleSelectDispatch(dispatch)}
>
  {/* Contenu */}
</div>;
```

**Fichiers concernés** :

- Tous les composants interactifs (boutons, modals, tableaux)
- `frontend/src/pages/company/Dispatch/components/ManualModePanel.jsx`
- `frontend/src/pages/company/Dispatch/components/SemiAutoPanel.jsx`
- `frontend/src/pages/company/Dispatch/components/FullyAutoPanel.jsx`

**Checklist A11y** :

- [ ] Tous les boutons ont `aria-label` descriptif
- [ ] Éléments interactifs ont `tabIndex` approprié
- [ ] Support complet clavier (Tab, Enter, Esc, flèches)
- [ ] Gestion focus pour modals
- [ ] Contraste couleurs conforme WCAG AA (4.5:1)
- [ ] Messages d'erreur annoncés aux lecteurs d'écran (`aria-live`)

**Impact** : Application accessible à tous les utilisateurs, conforme standards

**Priorité** : 🟡 **MOYENNE**

---

## Résumé des Priorités

### 🔴 Priorité Critique (Immédiat)

- ✅ Correction bug verrou Redis (1.1) - **Complété**
- ✅ Migration verrous distribués (1.2) - **Complété**
- ✅ Anti-duplication Celery (2.1) - **Complété**
- ✅ Validation API Marshmallow (3.2) - **Complété**
- ✅ Métriques unassigned reasons (4.1) - **Complété**

### 🟡 Priorité Haute (1-2 semaines)

- ✅ Cache OSRM (2.2) - **Complété**
- ✅ Externalisation constantes (3.1) - **Complété**
- ✅ Découpage frontend (5.1) - **Complété**
- ✅ Hooks personnalisés (5.2) - **Complété**
- ✅ Tests unitaires backend (6.1) - **Complété**
- ✅ Tests intégration backend (6.2) - **Complété**
- ⏳ Tests unitaires frontend (7.3) - **En attente**
- ⏳ Gestion erreurs robuste (7.6) - **En attente**

### 🟢 Priorité Moyenne (2-4 semaines)

- ✅ Circuit Breaker OSRM (2.3) - **Complété**
- ✅ Dashboard santé (4.2) - **Complété**
- ✅ Métriques OSRM (4.3) - **Complété**
- ✅ Documentation technique (6.3) - **Complété**
- ⏳ Gestion état globale (7.1) - **En attente**
- ⏳ Optimisation performances (7.2) - **En attente**
- ⏳ Amélioration UX (7.5) - **En attente**
- ⏳ Accessibilité (7.7) - **En attente**

### 🔵 Priorité Basse (Long terme)

- ✅ Nettoyage code mort (5.3) - **Complété**
- ⏳ Mise à jour optimiste (7.4) - **En attente**

---

## Métriques de Succès

### Backend

- ✅ Taux d'assignation > 95% pour journées normales
- ✅ Temps d'exécution < 5s pour 50 courses
- ✅ Cache OSRM hit rate > 80%
- ✅ Zéro verrou bloqué (grâce TTL Redis)
- ✅ Couverture tests > 80%

### Frontend

- ✅ Réduction taille fichier principal : 1484 → 309 lignes (-79%)
- ⏳ Couverture tests > 70%
- ⏳ Lighthouse Performance Score > 90
- ⏳ Time to Interactive < 2s
- ⏳ Conformité WCAG 2.1 AA

---

## Dépendances à Ajouter

### Backend (déjà ajoutées)

```txt
# requirements.txt
marshmallow>=3.20.0
redis>=5.0.0
```

### Frontend (à ajouter pour Phase 7)

```json
{
  "dependencies": {
    "react-hot-toast": "^2.4.1",
    "framer-motion": "^10.16.16",
    "react-window": "^1.8.10"
  },
  "devDependencies": {
    "@testing-library/react": "^14.0.0",
    "@testing-library/react-hooks": "^8.0.1",
    "@testing-library/jest-dom": "^6.1.5"
  }
}
```

---

## Chronologie Estimée

| Phase | Description                     | Durée       | Statut        |
| ----- | ------------------------------- | ----------- | ------------- |
| 1     | Corrections critiques           | 2-3 jours   | ✅ Complété   |
| 2     | Performance backend             | 3-5 jours   | ✅ Complété   |
| 3     | Configuration                   | 2-3 jours   | ✅ Complété   |
| 4     | Observabilité                   | 3-4 jours   | ✅ Complété   |
| 5     | Refactoring frontend            | 5-7 jours   | ✅ Complété   |
| 6     | Tests & documentation           | 4-5 jours   | ✅ Complété   |
| 7     | Améliorations frontend avancées | 10-15 jours | ⏳ En attente |

**Total estimé Phase 7** : 2-3 semaines

---

## Notes d'Implémentation

### Phases 1-6 (Complétées)

- ✅ Tous les fichiers backend ont été mis à jour
- ✅ Refactoring frontend majeur effectué
- ✅ Tests backend créés et passants
- ✅ Documentation technique rédigée
- ✅ Tous les linters propres (Ruff, Pyright, ESLint)

### Phase 7 (En attente)

- Nécessite installation de dépendances npm
- Tests frontend sont prioritaires avant autres améliorations
- Gestion erreurs robuste critique pour production
- Autres améliorations peuvent être faites progressivement

---

## Contact & Support

Pour toute question sur ce plan d'amélioration :

- Consulter `ARCHITECTURE.md` pour vue d'ensemble
- Consulter `RUNBOOK.md` pour guide opérationnel
- Consulter `TUNING.md` pour optimisation paramètres
