import { Redirect } from "expo-router";
import { Pressable, StyleSheet, Text, View } from "react-native";
import { useSession } from "../src/core/sessionProvider";
import { resolveInitialRoute } from "../src/core/navigation/resolveInitialRoute";
import { brandSurfaceSoft, ResponsiveContainer, Screen } from "../src/design/responsive";
import { FONT_SIZE } from "../src/design/responsive/typographyTokens";
import { BootBrandSurface } from "../src/core/boot/BootBrandSurface";
import { canEnterFromLocalSession } from "../src/core/auth/canEnterFromLocalSession";
import { shouldShowColdBootBrandSurface } from "../src/core/auth/authGuardDecision";
// eslint-disable-next-line @typescript-eslint/no-require-imports
const ReactRuntime: any = require("react");

function BootRedirect({ href }: { href: string }) {
  return (
    <BootBrandSurface>
      <Redirect href={href as any} />
    </BootBrandSurface>
  );
}

export default function IndexScreen() {
  const {
    status,
    bootstrap,
    error,
    bootstrapSession,
    mobileSessionStatus,
    autoBootstrapAllowed,
    activeContext,
  } = useSession();

  ReactRuntime.useEffect(() => {
    if (status === "idle") {
      void bootstrapSession({ trigger: "cold_start_auto" });
    }
  }, [status, bootstrapSession]);

  // Post-logout : pas d'auto-bootstrap — orienter vers login.
  if (
    status === "idle" &&
    !autoBootstrapAllowed &&
    (mobileSessionStatus === "anonymous" || mobileSessionStatus === "logging_out")
  ) {
    return <BootRedirect href={"/(public)/login"} />;
  }

  if (mobileSessionStatus === "revoked") {
    return <BootRedirect href={"/(public)/login"} />;
  }

  if (status === "error") {
    return (
      <Screen scroll backgroundColor={brandSurfaceSoft} contentContainerStyle={styles.centerContent}>
        <ResponsiveContainer>
          <View style={styles.panel}>
            <Text style={styles.title}>Échec du démarrage</Text>
            <Text style={styles.body}>
              {error ?? "Impossible de charger la session depuis le backend."}
            </Text>
            <Pressable
              onPress={() => void bootstrapSession({ trigger: "manual_retry" })}
              style={styles.retry}
            >
              <Text style={styles.retryText}>Réessayer</Text>
            </Pressable>
          </View>
        </ResponsiveContainer>
      </Screen>
    );
  }

  // P0-5 final : warm recovery / session ready → destination app, jamais BootBrand plein écran
  const warmOrReady =
    status === "ready" ||
    mobileSessionStatus === "authenticated_online" ||
    mobileSessionStatus === "authenticated_offline" ||
    mobileSessionStatus === "auth_recovering" ||
    canEnterFromLocalSession({ bootstrap, activeContext });

  if (bootstrap && warmOrReady && mobileSessionStatus !== "logging_out") {
    const href = resolveInitialRoute(bootstrap, null, mobileSessionStatus);
    if (href && !href.startsWith("/(public)")) {
      return <BootRedirect href={href} />;
    }
    // AuthGuard garde l'arbre monté ; Index ne doit pas démonter via BootBrand
    if (!shouldShowColdBootBrandSurface({ status, mobileSessionStatus })) {
      return <BootRedirect href={"/(app)/context-selector"} />;
    }
  }

  if (
    shouldShowColdBootBrandSurface({ status, mobileSessionStatus }) ||
    !bootstrap ||
    status === "bootstrapping" ||
    mobileSessionStatus === "logging_out"
  ) {
    return <BootBrandSurface />;
  }

  const href = resolveInitialRoute(bootstrap, null, mobileSessionStatus);
  if (!href) return <BootBrandSurface />;
  return <BootRedirect href={href} />;
}

const styles = StyleSheet.create({
  centerContent: {
    flexGrow: 1,
    justifyContent: "center",
    paddingVertical: 32,
  },
  panel: {
    gap: 14,
    alignItems: "stretch",
  },
  title: {
    fontSize: FONT_SIZE.px20,
    fontWeight: "700",
    color: "#163A34",
    textAlign: "center",
  },
  body: {
    fontSize: FONT_SIZE.px15,
    lineHeight: 22,
    color: "#475569",
    textAlign: "center",
  },
  retry: {
    alignSelf: "center",
    paddingHorizontal: 20,
    paddingVertical: 12,
    borderRadius: 12,
    borderWidth: 1.5,
    borderColor: "#0A8F7A",
    backgroundColor: "#FFFFFF",
  },
  retryText: {
    fontSize: FONT_SIZE.px16,
    fontWeight: "600",
    color: "#0A8F7A",
  },
});
