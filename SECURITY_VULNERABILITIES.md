# 🔒 Plan de Correction des Vulnérabilités Dependabot

## 📊 Résumé

- **4 Critiques** ⚠️⚠️⚠️
- **2 Élevées** ⚠️⚠️
- **8 Modérées** ⚠️
- **3 Faibles** ℹ️

---

## 🔴 CRITIQUES (À corriger immédiatement)

### 1. @react-native-community/cli - Injection de commande OS

**Fichier:** `mobile/operations-app/package.json` (ligne 74)
**Version actuelle:** `^19.1.0`
**Action:** Mettre à jour vers `^19.2.0` ou supérieur

```json
"@react-native-community/cli": "^19.2.0"
```

### 2. @react-native-community/cli-server-api - Injection de commande OS

**Fichier:** `mobile/operations-app/package-lock.json` (dépendance transitive)
**Action:** Sera corrigée automatiquement avec la mise à jour de `@react-native-community/cli`

### 3. form-data - Fonction random non sécurisée

**Fichier:** `mobile/client-app/package-lock.json` (dépendance transitive)
**Action:** Mettre à jour les dépendances qui utilisent `form-data` ou forcer la version sécurisée

```json
"overrides": {
  "form-data": "^4.0.1"
}
```

### 4. form-data (duplicate) - Fonction random non sécurisée

**Fichier:** `mobile/client-app/package-lock.json` (dépendance transitive)
**Action:** Même correction que #3

---

## 🟠 ÉLEVÉES (À corriger rapidement)

### 5. nth-check - Complexité d'expression régulière inefficace

**Fichier:** `frontend/package-lock.json` (dépendance transitive)
**Action:** Mettre à jour les dépendances ou forcer la version sécurisée

```json
"overrides": {
  "nth-check": "^3.0.0"
}
```

### 6. axios - DoS via manque de vérification de taille

**Fichier:** `mobile/operations-app/package.json` (ligne 27)
**Version actuelle:** `^1.8.4`
**Action:** Mettre à jour vers `^1.8.7` ou supérieur

```json
"axios": "^1.8.7"
```

---

## 🟡 MODÉRÉES (À planifier)

### 7. webpack-dev-server - Vol de code source

**Fichier:** `frontend/package-lock.json` (dépendance transitive via react-scripts)
**Action:** Mettre à jour `react-scripts` vers `^5.0.2` ou supérieur

```json
"react-scripts": "^5.0.2"
```

### 8. webpack-dev-server (duplicate) - Vol de code source

**Fichier:** `frontend/package-lock.json`
**Action:** Même correction que #7

### 9. @sentry/browser - Pollution de prototype

**Fichier:** `mobile/operations-app/package-lock.json` (dépendance transitive)
**Action:** Mettre à jour `@sentry/react-native` vers `~6.15.0` ou supérieur

```json
"@sentry/react-native": "~6.15.0"
```

### 10. tar - Condition de course exposant mémoire

**Fichier:** `mobile/operations-app/package-lock.json` (dépendance transitive)
**Action:** Forcer la version sécurisée

```json
"overrides": {
  "tar": "^7.4.3"
}
```

### 11. postcss - Erreur de parsing

**Fichier:** `frontend/package-lock.json` (dépendance transitive)
**Action:** Mettre à jour les dépendances ou forcer la version

```json
"overrides": {
  "postcss": "^8.4.49"
}
```

### 12. js-yaml - Pollution de prototype (3 occurrences)

**Fichiers:**

- `frontend/package-lock.json`
- `mobile/client-app/package-lock.json`
- `mobile/operations-app/package-lock.json`

**Action:** Forcer la version sécurisée dans tous les projets

```json
"overrides": {
  "js-yaml": "^4.1.0"
}
```

---

## 🔵 FAIBLES (À planifier)

### 13. on-headers - Manipulation de headers HTTP

**Fichier:** `mobile/client-app/package-lock.json` (dépendance transitive)
**Action:** Forcer la version sécurisée

```json
"overrides": {
  "on-headers": "^1.1.0"
}
```

### 14. brace-expansion - ReDoS (2 occurrences)

**Fichier:** `mobile/client-app/package-lock.json` (dépendance transitive)
**Action:** Forcer la version sécurisée

```json
"overrides": {
  "brace-expansion": "^2.0.2"
}
```

---

## 🛠️ Actions Recommandées

### Étape 1: Corrections Critiques (Immédiat)

1. Mettre à jour `@react-native-community/cli` dans `mobile/operations-app/package.json`
2. Mettre à jour `axios` dans `mobile/operations-app/package.json`
3. Ajouter `overrides` pour `form-data` dans `mobile/client-app/package.json`

### Étape 2: Corrections Élevées (Cette semaine)

1. Ajouter `overrides` pour `nth-check` dans `frontend/package.json`

### Étape 3: Corrections Modérées (Ce mois)

1. Mettre à jour `react-scripts` dans `frontend/package.json`
2. Mettre à jour `@sentry/react-native` dans `mobile/operations-app/package.json`
3. Ajouter `overrides` pour `tar`, `postcss`, `js-yaml` dans tous les projets concernés

### Étape 4: Corrections Faibles (Prochain sprint)

1. Ajouter `overrides` pour `on-headers` et `brace-expansion`

---

## 📝 Notes Importantes

- Les `overrides` dans `package.json` forcent npm à utiliser des versions spécifiques pour les dépendances transitives
- Après chaque modification, exécuter `npm install` puis `npm audit fix`
- Tester l'application après chaque correction
- Vérifier que les nouvelles versions sont compatibles avec votre code

---

## 🔗 Ressources

- [GitHub Dependabot Alerts](https://github.com/djasiqi/atmr/security/dependabot)
- [npm overrides documentation](https://docs.npmjs.com/cli/v9/configuring-npm/package-json#overrides)
