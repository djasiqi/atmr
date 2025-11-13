# Welcome to your Expo app 👋

This is an [Expo](https://expo.dev) project created with [`create-expo-app`](https://www.npmjs.com/package/create-expo-app).

## Get started

1. Install dependencies

   ```bash
   npm install
   ```

2. Configure environment variables
   - Copie `env.example` en `.env.development` pour ton environnement local, puis adapte les valeurs (URL backend local, clés Google, etc.).
   - Pour la production, crée `.env.production` avec les URL et clés de prod (ou configure-les via EAS/GitHub Actions en secrets).
   - Les fichiers `.env*` sont ignorés par Git afin d’éviter toute fuite de secrets.
   - Tu peux utiliser `npm run env:generate -- production` (ou `-- development`) pour générer un fichier à partir de `env.example` avec des invites interactives.

3. Start the app

   ```bash
    npx expo start
   ```

## Production (EAS / Play Store)

- Crée les secrets GitHub `EXPO_TOKEN`, `MOBILE_API_URL`, `MOBILE_BACKEND_PORT`, `MOBILE_SOCKET_HOST`, `MOBILE_SOCKET_PORT`, `EXPO_PUBLIC_GOOGLE_API_KEY`, `EXPO_PUBLIC_ANDROID_MAPS_API_KEY`. Ils sont injectés dans le workflow `mobile-android-build.yml`.
- Pour une build locale de prod, copie `env.example` en `.env.production` et renseigne les valeurs prod avant d’exécuter `NODE_ENV=production expo run:android`.
- Sur EAS, configure les mêmes variables via `eas secret:create --scope project` (ou via l’interface web) pour que les builds cloud aient accès aux secrets.
- Vérifie que `GOOGLE_SERVICES_JSON` et la clé de service Play Store sont bien fournis dans GitHub Secrets avant d’autoriser la soumission automatique vers le Play Store.

In the output, you'll find options to open the app in a

- [development build](https://docs.expo.dev/develop/development-builds/introduction/)
- [Android emulator](https://docs.expo.dev/workflow/android-studio-emulator/)
- [iOS simulator](https://docs.expo.dev/workflow/ios-simulator/)
- [Expo Go](https://expo.dev/go), a limited sandbox for trying out app development with Expo

You can start developing by editing the files inside the **app** directory. This project uses [file-based routing](https://docs.expo.dev/router/introduction).

## Get a fresh project

When you're ready, run:

```bash
npm run reset-project
```

This command will move the starter code to the **app-example** directory and create a blank **app** directory where you can start developing.

## Learn more

To learn more about developing your project with Expo, look at the following resources:

- [Expo documentation](https://docs.expo.dev/): Learn fundamentals, or go into advanced topics with our [guides](https://docs.expo.dev/guides).
- [Learn Expo tutorial](https://docs.expo.dev/tutorial/introduction/): Follow a step-by-step tutorial where you'll create a project that runs on Android, iOS, and the web.

## Join the community

Join our community of developers creating universal apps.

- [Expo on GitHub](https://github.com/expo/expo): View our open source platform and contribute.
- [Discord community](https://chat.expo.dev): Chat with Expo users and ask questions.
