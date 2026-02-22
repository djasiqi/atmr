import React, { useState } from "react";
import { View, Text, TouchableOpacity, StyleSheet } from "react-native";

export function PushFailureBanner() {
  const [dismissed, setDismissed] = useState(false);

  if (dismissed) return null;

  return (
    <View style={styles.banner}>
      <View style={styles.row}>
        <Text style={styles.text}>
          Notifications indisponibles — vous ne recevrez pas d'alertes push.
        </Text>
        <TouchableOpacity
          onPress={() => setDismissed(true)}
          hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
        >
          <Text style={styles.dismiss}>OK</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  banner: {
    backgroundColor: "#FEF9C3",
    paddingVertical: 8,
    paddingHorizontal: 16,
    borderBottomWidth: 1,
    borderBottomColor: "#FDE68A",
  },
  row: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },
  text: {
    color: "#92400E",
    fontSize: 12,
    fontWeight: "500",
    flex: 1,
    marginRight: 8,
  },
  dismiss: {
    color: "#B45309",
    fontSize: 12,
    fontWeight: "600",
  },
});
