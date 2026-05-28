// https://docs.expo.dev/guides/using-eslint/
const { defineConfig } = require('eslint/config');
const expoConfig = require("eslint-config-expo/flat");

module.exports = defineConfig([
  expoConfig,
  {
    ignores: ["dist/*"],
  },
  {
    files: ["app/**/*.{ts,tsx}"],
    rules: {
      "no-restricted-imports": [
        "error",
        {
          paths: [
            {
              name: "react-native",
              importNames: ["Dimensions", "useWindowDimensions"],
              message: "Use useAppViewport() from src/design/responsive instead.",
            },
            {
              name: "react-native-safe-area-context",
              importNames: ["useSafeAreaInsets"],
              message: "Use Screen/useAppViewport() so safe area behavior remains centralized.",
            },
          ],
        },
      ],
    },
  },
  {
    files: ["src/features/company/components/maps/**/*.{ts,tsx}"],
    ignores: [
      "src/features/company/components/maps/resolveMetroAssetSource.ts",
      "src/features/company/components/maps/fleetLirieDriverMarkerAssets.ts",
      "src/features/company/components/maps/fleetNativeMarkerImage.ts",
    ],
    rules: {
      "no-restricted-imports": [
        "error",
        {
          paths: [
            {
              name: "./resolveMetroAssetSource",
              message:
                "Résolution asset uniquement dans fleetNativeMarkerImage / fleetLirieDriverMarkerAssets.",
            },
          ],
        },
      ],
    },
  },
]);
