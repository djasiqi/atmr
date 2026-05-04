import { Pressable, StyleSheet, Text, View } from "react-native";
import * as Linking from "expo-linking";
import { ResponsiveContainer, Screen } from "../../src/design/responsive";

export default function ContactScreen() {
  return (
    <Screen scroll backgroundColor="#F7FBFA" contentContainerStyle={styles.scroll}>
      <ResponsiveContainer>
        <View style={styles.block}>
          <Text style={styles.title}>Contact support</Text>
          <Text style={styles.intro}>
            Notre équipe répond pour les problèmes de connexion, d&apos;activation de compte et de
            réservation.
          </Text>

          <Text style={styles.sectionTitle}>Établissements</Text>
          <Text style={styles.body}>
            Demandez un accès institution pour la planification patient et le suivi des transports.
          </Text>

          <Text style={styles.sectionTitle}>Transporteurs</Text>
          <Text style={styles.body}>
            Rejoignez Lirie pour recevoir des missions et connecter votre flotte.
          </Text>

          <Text style={styles.body}>E-mail : support@lirie.ch</Text>
          <Text style={styles.body}>Téléphone : +41 22 000 00 00 (08:00–18:00)</Text>

          <Pressable
            onPress={() => void Linking.openURL("mailto:support@lirie.ch")}
            style={styles.action}
          >
            <Text style={styles.actionText}>Envoyer un e-mail</Text>
          </Pressable>
          <Pressable
            onPress={() => void Linking.openURL("tel:+41220000000")}
            style={styles.action}
          >
            <Text style={styles.actionText}>Appeler le support</Text>
          </Pressable>

          <Text style={styles.footnote}>
            Si vous n&apos;avez pas accès à l&apos;application, les flux d&apos;activation et de mot de
            passe restent disponibles depuis le web.
          </Text>
        </View>
      </ResponsiveContainer>
    </Screen>
  );
}

const styles = StyleSheet.create({
  scroll: {
    flexGrow: 1,
    paddingVertical: 24,
  },
  block: {
    gap: 12,
  },
  title: {
    fontSize: 24,
    fontWeight: "700",
    color: "#163A34",
  },
  intro: {
    fontSize: 15,
    lineHeight: 22,
    color: "#5F7369",
    marginBottom: 4,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: "700",
    color: "#163A34",
    marginTop: 8,
  },
  body: {
    fontSize: 15,
    lineHeight: 22,
    color: "#475569",
  },
  footnote: {
    fontSize: 13,
    lineHeight: 19,
    color: "#64748b",
    marginTop: 8,
  },
  action: {
    alignSelf: "flex-start",
    paddingVertical: 4,
  },
  actionText: {
    color: "#0A8F7A",
    fontWeight: "600",
    fontSize: 15,
    textDecorationLine: "underline",
  },
});
