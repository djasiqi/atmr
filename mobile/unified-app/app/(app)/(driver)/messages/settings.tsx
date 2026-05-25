import { useState } from "react";
import { StyleSheet, Switch, View } from "react-native";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { DriverContextGuard, PermissionGuard } from "../../../../src/core/guards";
import { Screen, AppText, useAppViewport } from "../../../../src/design/responsive";
import { D } from "../../../../src/features/driver/theme/driverDashboardTheme";

const KEYS = {
  push: "driver-msg-settings-push",
  sound: "driver-msg-settings-sound",
  autoPhoto: "driver-msg-settings-auto-photo",
  autoDoc: "driver-msg-settings-auto-doc",
  compact: "driver-msg-settings-compact",
} as const;

export default function DriverMessagesSettingsScreen() {
  const { horizontalPadding } = useAppViewport();
  const [push, setPush] = useState(true);
  const [sound, setSound] = useState(true);
  const [autoPhoto, setAutoPhoto] = useState(true);
  const [autoDoc, setAutoDoc] = useState(false);
  const [compact, setCompact] = useState(true);

  const persist = async (key: string, value: boolean) => {
    await AsyncStorage.setItem(key, value ? "1" : "0");
  };

  return (
    <DriverContextGuard>
      <PermissionGuard permission="chat:read">
        <Screen scroll backgroundColor={D.pageBg}>
          <View style={{ paddingHorizontal: horizontalPadding, gap: 12 }}>
            <SettingRow
              label="Notifications push"
              value={push}
              onChange={(v) => {
                setPush(v);
                void persist(KEYS.push, v);
              }}
            />
            <SettingRow
              label="Notifications sonores"
              value={sound}
              onChange={(v) => {
                setSound(v);
                void persist(KEYS.sound, v);
              }}
            />
            <SettingRow
              label="Téléchargement auto photos"
              value={autoPhoto}
              onChange={(v) => {
                setAutoPhoto(v);
                void persist(KEYS.autoPhoto, v);
              }}
            />
            <SettingRow
              label="Téléchargement auto documents"
              value={autoDoc}
              onChange={(v) => {
                setAutoDoc(v);
                void persist(KEYS.autoDoc, v);
              }}
            />
            <SettingRow
              label="Mode lecture rapide (compact)"
              value={compact}
              onChange={(v) => {
                setCompact(v);
                void persist(KEYS.compact, v);
              }}
            />
          </View>
        </Screen>
      </PermissionGuard>
    </DriverContextGuard>
  );
}

function SettingRow({
  label,
  value,
  onChange,
}: {
  label: string;
  value: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <View style={styles.row}>
      <AppText variant="body" style={styles.label}>
        {label}
      </AppText>
      <Switch value={value} onValueChange={onChange} trackColor={{ true: "#0A8F7A" }} />
    </View>
  );
}

const styles = StyleSheet.create({
  row: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    backgroundColor: "#fff",
    borderRadius: 12,
    padding: 14,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: "#e5e7eb",
  },
  label: { flex: 1, paddingRight: 12 },
});
