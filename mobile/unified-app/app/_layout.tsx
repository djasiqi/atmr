import { Philosopher_700Bold } from "@expo-google-fonts/philosopher";
import { useFonts } from "expo-font";
import { Stack } from "expo-router";
import * as SplashScreen from "expo-splash-screen";
import { StatusBar } from "expo-status-bar";
import { useEffect } from "react";
import { Platform } from "react-native";
import { SafeAreaProvider } from "react-native-safe-area-context";
import { QueryProvider } from "../src/core/QueryProvider";
import { BootSplashGate } from "../src/core/boot/BootSplashGate";
import { SessionProvider } from "../src/core/sessionProvider";
import { MonitoringProvider } from "../src/core/providers/MonitoringProvider";
import { NativeCapabilitiesProvider } from "../src/core/providers/NativeCapabilitiesProvider";
import { NotificationsProvider } from "../src/core/providers/NotificationsProvider";
import { ExternalIntentProvider } from "../src/core/providers/ExternalIntentProvider";

void SplashScreen.preventAutoHideAsync().catch(() => {
  // ignore if splash is already controlled elsewhere
});

if (Platform.OS !== "web" && !__DEV__) {
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    require("../tasks/locationTask");
  } catch {
    // task unavailable in environments without native TaskManager support
  }
}

export default function RootLayout() {
  const [fontsLoaded] = useFonts({
    Philosopher_700Bold,
  });

  useEffect(() => {
    if (fontsLoaded) {
      void SplashScreen.hideAsync().catch(() => {
        // no-op
      });
    }
  }, [fontsLoaded]);

  useEffect(() => {
    if (Platform.OS !== "web" || typeof document === "undefined") {
      return;
    }

    const html = document.documentElement;
    const body = document.body;
    const root =
      document.getElementById("root") ??
      (body.firstElementChild instanceof HTMLElement ? body.firstElementChild : null);

    const previous = {
      htmlBackground: html.style.background,
      htmlHeight: html.style.height,
      htmlOverflow: html.style.overflow,
      bodyBackground: body.style.background,
      bodyHeight: body.style.height,
      bodyMinHeight: body.style.minHeight,
      bodyMargin: body.style.margin,
      bodyPadding: body.style.padding,
      bodyOverflow: body.style.overflow,
      rootBackground: root?.style.background ?? "",
      rootHeight: root?.style.height ?? "",
      rootMinHeight: root?.style.minHeight ?? "",
      rootOverflow: root?.style.overflow ?? "",
    };

    html.style.background = "#061A18";
    html.style.height = "100%";
    html.style.overflow = "hidden";

    body.style.background = "#061A18";
    body.style.height = "100%";
    body.style.minHeight = "100%";
    body.style.margin = "0";
    body.style.padding = "0";
    body.style.overflow = "hidden";

    if (root) {
      root.style.background = "#061A18";
      root.style.height = "100%";
      root.style.minHeight = "100%";
      root.style.overflow = "hidden";
    }

    return () => {
      html.style.background = previous.htmlBackground;
      html.style.height = previous.htmlHeight;
      html.style.overflow = previous.htmlOverflow;

      body.style.background = previous.bodyBackground;
      body.style.height = previous.bodyHeight;
      body.style.minHeight = previous.bodyMinHeight;
      body.style.margin = previous.bodyMargin;
      body.style.padding = previous.bodyPadding;
      body.style.overflow = previous.bodyOverflow;

      if (root) {
        root.style.background = previous.rootBackground;
        root.style.height = previous.rootHeight;
        root.style.minHeight = previous.rootMinHeight;
        root.style.overflow = previous.rootOverflow;
      }
    };
  }, []);

  if (!fontsLoaded) {
    return null;
  }

  return (
    <SafeAreaProvider>
      <QueryProvider>
        <MonitoringProvider>
          <NativeCapabilitiesProvider>
            <SessionProvider>
              <BootSplashGate>
                <ExternalIntentProvider>
                  <NotificationsProvider>
                    <Stack screenOptions={{ headerShown: false }}>
                      <Stack.Screen name="index" />
                      <Stack.Screen name="(public)" />
                      <Stack.Screen name="(app)" />
                      <Stack.Screen name="quick-action" />
                      <Stack.Screen name="payment-return" />
                      <Stack.Screen name="guest-payment-return" />
                    </Stack>
                    <StatusBar style="auto" />
                  </NotificationsProvider>
                </ExternalIntentProvider>
              </BootSplashGate>
            </SessionProvider>
          </NativeCapabilitiesProvider>
        </MonitoringProvider>
      </QueryProvider>
    </SafeAreaProvider>
  );
}
