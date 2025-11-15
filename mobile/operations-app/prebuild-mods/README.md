# Prebuild Mods

Ce dossier contient les mods Prebuild personnalisés pour le projet.

## withAndroidBackButtonMod

Ce mod préserve le comportement personnalisé du bouton retour dans `MainActivity.kt`. Il ajoute automatiquement la méthode `invokeDefaultOnBackPressed()` qui gère correctement le comportement du bouton retour sur Android S et versions antérieures.

## Utilisation

Le mod est automatiquement appliqué via `app.config.js`. Pour régénérer les dossiers natifs avec Prebuild :

```bash
npm run prebuild
```

OU

```bash
npx expo prebuild --clean
```

Cela régénérera les dossiers `android/` et `ios/` à partir de `app.config.js`, et le mod appliquera automatiquement les personnalisations nécessaires.

