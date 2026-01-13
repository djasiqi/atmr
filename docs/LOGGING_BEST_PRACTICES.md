# 📋 Bonnes Pratiques de Logging

## 🎯 Objectif

Éviter que les logs de debug ne s'affichent en production, tout en conservant les erreurs et warnings importants.

---

## 🔧 Utilisation

### Mobile (React Native)

#### Option 1 : Utiliser `__DEV__` (recommandé)

```typescript
// ✅ Bon - Log seulement en développement
if (__DEV__) {
  console.log('[Component] Debug info:', data);
}

// ❌ Mauvais - Log toujours affiché
console.log('[Component] Debug info:', data);
```

#### Option 2 : Utiliser le logger helper

```typescript
import { logger } from '@/utils/logger';

// Logs de debug (dev only)
logger.log('[Component] Debug info:', data);
logger.debug('[Component] Detailed info:', details);

// Warnings et erreurs (toujours affichés)
logger.warn('[Component] Something suspicious:', issue);
logger.error('[Component] Error occurred:', error);
```

### Frontend Web (React)

#### Option 1 : Utiliser `import.meta.env.DEV`

```javascript
// ✅ Bon - Log seulement en développement
if (import.meta.env.DEV) {
  console.log('[Component] Debug info:', data);
}
```

#### Option 2 : Utiliser le logger helper

```javascript
import { logger } from '@/utils/logger';

// Logs de debug (dev only)
logger.log('[Component] Debug info:', data);
logger.debug('[Component] Detailed info:', details);

// Warnings et erreurs (toujours affichés)
logger.warn('[Component] Something suspicious:', issue);
logger.error('[Component] Error occurred:', error);
```

---

## 📊 Types de Logs

### 🟢 `logger.log()` / `logger.debug()` - Dev only
**Quand l'utiliser** :
- Debug d'état React
- Trace de flux de données
- Vérification de props
- Informations temporaires de développement

**Exemples** :
```typescript
logger.log('[useAuth] Token changed:', token);
logger.debug('[API] Request payload:', payload);
```

### 🟡 `logger.warn()` - Toujours affiché
**Quand l'utiliser** :
- Utilisation déconseillée d'une fonctionnalité
- Données manquantes non-critiques
- Problèmes de performance potentiels
- Fallbacks activés

**Exemples** :
```typescript
logger.warn('[Component] Prop missing, using default:', defaultValue);
logger.warn('[API] Slow response time:', responseTime);
```

### 🔴 `logger.error()` - Toujours affiché
**Quand l'utiliser** :
- Erreurs d'API
- Exceptions capturées
- Erreurs de validation
- Problèmes critiques

**Exemples** :
```typescript
logger.error('[API] Request failed:', error);
logger.error('[Auth] Token validation failed');
```

---

## 🎯 Patterns Recommandés

### ✅ Bon

```typescript
// Debug conditionnel
if (__DEV__) {
  console.log('[Component] State updated:', state);
}

// Ou avec le helper
logger.debug('[Component] State updated:', state);

// Erreurs toujours loggées
try {
  await someOperation();
} catch (error) {
  logger.error('[Component] Operation failed:', error);
  // Gérer l'erreur
}
```

### ❌ Mauvais

```typescript
// Log non conditionnel en production
console.log('[Component] Debug info'); // ❌

// Erreurs silencieuses
try {
  await someOperation();
} catch (error) {
  // Rien... ❌
}

// Trop de détails en production
console.log('[API] Full response:', response); // ❌
```

---

## 🔍 Vérification

### Avant de Commiter

1. **Rechercher les `console.log` non conditionnels** :
```bash
# Mobile
grep -r "console\.log" mobile/operations-app --exclude-dir=node_modules

# Frontend
grep -r "console\.log" frontend/src
```

2. **Vérifier que les logs de debug utilisent** :
   - `if (__DEV__)` (mobile)
   - `if (import.meta.env.DEV)` (frontend)
   - Ou `logger.log()` / `logger.debug()`

3. **S'assurer que les erreurs utilisent** :
   - `logger.error()` ou `console.error()`

---

## 🚀 Migration des Logs Existants

### Script de recherche

```bash
# Trouver tous les console.log non conditionnels
rg "console\.log\(" --type typescript --type javascript \
  | grep -v "__DEV__" \
  | grep -v "import.meta.env.DEV"
```

### Remplacement manuel

Remplacer :
```typescript
console.log('[Component] Debug:', data);
```

Par :
```typescript
if (__DEV__) {
  console.log('[Component] Debug:', data);
}
```

Ou :
```typescript
logger.debug('[Component] Debug:', data);
```

---

## 📱 Configuration Build

### Mobile (EAS)

Les logs de debug seront automatiquement exclus en production grâce à `__DEV__`.

### Frontend (Vite)

Vite supprime automatiquement le code dans les blocs `if (import.meta.env.DEV)` lors du build de production.

---

## 🎯 Checklist de Review

- [ ] Pas de `console.log()` non conditionnel
- [ ] Les erreurs utilisent `logger.error()` ou `console.error()`
- [ ] Les warnings importants utilisent `logger.warn()`
- [ ] Les logs de debug sont dans `if (__DEV__)` ou utilisent `logger.debug()`
- [ ] Pas de données sensibles loggées (tokens, passwords, etc.)

---

## 📚 Ressources

- [React Native __DEV__](https://reactnative.dev/docs/performance#using-the-dev-variable)
- [Vite Environment Variables](https://vitejs.dev/guide/env-and-mode.html)
- [Console API MDN](https://developer.mozilla.org/en-US/docs/Web/API/Console)
