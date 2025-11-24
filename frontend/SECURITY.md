# Sécurité - Vulnérabilité glob CLI

## Vulnérabilité GHSA-ww39-953v-wcq6 (Command Injection)

### Description

La vulnérabilité affecte la CLI de `glob` (pas l'API) via l'option `-c/--cmd` qui permet l'injection de commandes arbitraires lors du traitement de fichiers avec des noms malveillants.

### Versions affectées

- `glob >= 10.2.0, < 10.5.0`
- `glob >= 11.0.0, < 11.1.0`

### Versions corrigées

- `glob >= 10.5.0`
- `glob >= 11.1.0`

### Solution appliquée

#### 1. Override dans package.json

Un override a été ajouté dans `package.json` pour forcer toutes les dépendances transitives à utiliser `glob >= 11.1.0` :

```json
"overrides": {
  "glob": "^11.1.0"
}
```

Cette configuration force `react-scripts` et toutes ses dépendances transitives à utiliser la version corrigée.

#### 2. Configuration .npmrc

Le fichier `.npmrc` a été configuré pour s'assurer que npm respecte les overrides.

### Mise à jour des dépendances

Pour appliquer la correction, vous avez deux options :

#### Option 1 : Script automatisé (recommandé)

```bash
cd frontend
npm run fix-glob
```

Ce script :

- Vérifie que l'override est configuré
- Installe les dépendances avec l'override
- Vérifie que toutes les versions de glob sont sécurisées
- Exécute un audit de sécurité

#### Option 2 : Installation manuelle

```bash
cd frontend
npm install
```

Cela mettra à jour `package-lock.json` avec la version corrigée de `glob`.

### Vérification

Pour vérifier que la vulnérabilité est corrigée :

```bash
npm audit
npm list glob
```

La commande `npm list glob` doit afficher `glob@11.1.0` ou supérieur pour toutes les instances.

### Références

- [GitHub Security Advisory](https://github.com/isaacs/node-glob/security/advisories/GHSA-ww39-953v-wcq6)
- [npm glob package](https://www.npmjs.com/package/glob)

### Notes importantes

- Cette vulnérabilité affecte uniquement la CLI de `glob`, pas l'API utilisée par les dépendances
- L'override force toutes les dépendances transitives (`react-scripts`, etc.) à utiliser la version corrigée
- La mise à jour de `react-scripts` vers une version plus récente pourrait résoudre le problème à la source, mais l'override reste une solution de sécurité efficace
