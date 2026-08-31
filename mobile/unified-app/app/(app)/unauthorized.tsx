import { Redirect } from "expo-router";
import { StyleSheet, Text, View } from "react-native";
import { brandSurfaceSoft, ResponsiveContainer, Screen } from "../../src/design/responsive";
import { FONT_SIZE } from "../../src/design/responsive/typographyTokens";
import { resolveUnauthorizedRecoveryRedirect } from "../../src/core/guardDecisions";
import { useSession } from "../../src/core/sessionProvider";

export default function UnauthorizedScreen() {
  const { activeContext } = useSession();
  const recovery = resolveUnauthorizedRecoveryRedirect(activeContext);
  if (recovery) {
    return <Redirect href={recovery as any} />;
  }

  return (
    <Screen scroll backgroundColor={brandSurfaceSoft} contentContainerStyle={styles.scroll}>
      <ResponsiveContainer>
        <View style={styles.block}>
          <Text style={styles.title}>Accès refusé</Text>
          <Text style={styles.body}>Permission insuffisante pour cette action.</Text>
        </View>
      </ResponsiveContainer>
    </Screen>
  );
}

const styles = StyleSheet.create({
  scroll: {
    flexGrow: 1,
    justifyContent: "center",
    paddingVertical: 32,
  },
  block: {
    gap: 12,
  },
  title: {
    fontSize: FONT_SIZE.px22,
    fontWeight: "700",
    color: "#163A34",
  },
  body: {
    fontSize: FONT_SIZE.px15,
    lineHeight: 22,
    color: "#5F7369",
  },
});
