import React from "react";
import { View, Text, TouchableOpacity, Linking } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { router } from "expo-router";
import Ionicons from "react-native-vector-icons/Ionicons";
import { getLoginStyles } from "@/styles/loginStyles";

const SUPPORT_EMAIL = "info@lirie.ch";

export default function EnterpriseForgotPasswordScreen() {
  const { styles, palette } = React.useMemo(
    () => getLoginStyles("enterprise"),
    []
  );

  const openEmail = () => {
    Linking.openURL(`mailto:${SUPPORT_EMAIL}`);
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <View style={styles.container}>
        <TouchableOpacity
          onPress={() => router.back()}
          style={{
            flexDirection: "row",
            alignItems: "center",
            gap: 6,
            marginBottom: 24,
          }}
        >
          <Ionicons name="arrow-back" size={22} color={palette.secondary} />
          <Text style={{ fontSize: 15, color: palette.secondary }}>
            Retour
          </Text>
        </TouchableOpacity>

        <View style={styles.card}>
          <View
            style={{
              width: 48,
              height: 48,
              borderRadius: 24,
              backgroundColor: "rgba(10,122,77,0.2)",
              alignItems: "center",
              justifyContent: "center",
              marginBottom: 16,
            }}
          >
            <Ionicons name="key-outline" size={24} color={palette.accent} />
          </View>
          <Text
            style={{
              fontSize: 20,
              fontWeight: "600",
              color: palette.text,
              marginBottom: 12,
            }}
          >
            Mot de passe oublié
          </Text>
          <Text
            style={{
              fontSize: 15,
              color: palette.secondary,
              lineHeight: 22,
              marginBottom: 20,
            }}
          >
            Pour réinitialiser le mot de passe de votre compte entreprise,
            contactez le support à l’adresse suivante :
          </Text>
          <TouchableOpacity
            onPress={openEmail}
            style={{ marginBottom: 20 }}
            activeOpacity={0.8}
          >
            <Text
              style={{
                fontSize: 16,
                color: palette.accent,
                fontWeight: "600",
              }}
            >
              {SUPPORT_EMAIL}
            </Text>
          </TouchableOpacity>
          <Text
            style={{
              fontSize: 14,
              color: palette.secondary,
              lineHeight: 20,
              opacity: 0.9,
            }}
          >
            Le support vous communiquera la procédure de réinitialisation.
          </Text>

          <TouchableOpacity
            style={[
              styles.primaryButton,
              { marginTop: 28 },
            ]}
            onPress={() => router.back()}
            activeOpacity={0.9}
          >
            <Text style={styles.primaryButtonText}>Retour à la connexion</Text>
          </TouchableOpacity>
        </View>
      </View>
    </SafeAreaView>
  );
}
