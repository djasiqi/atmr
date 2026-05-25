import { StyleSheet, Text, View } from "react-native";
import { brandSurfaceSoft, ResponsiveContainer, Screen } from "../../src/design/responsive";
import { FONT_SIZE } from "../../src/design/responsive/typographyTokens";

export default function OnboardingScreen() {
  return (
    <Screen scroll backgroundColor={brandSurfaceSoft} contentContainerStyle={styles.scroll}>
      <ResponsiveContainer>
        <View style={styles.block}>
          <Text style={styles.title}>Onboarding requis</Text>
          <Text style={styles.body}>
            Complétez le parcours d&apos;introduction avant l&apos;accès aux espaces métiers.
          </Text>
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
