import React from "react";
import { View, Text, TouchableOpacity } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { styles, palette } from "@/styles/missionHeaderStyles";
import { useSocketStatus } from "@/hooks/useSocketStatus";

type Props = {
  driverName: string;
  date?: string;
  missionCount?: number;
};

function getGreetingPrefix(): string {
  const h = new Date().getHours();
  if (h < 12) return "Bonjour";
  if (h < 18) return "Bon après-midi";
  return "Bonsoir";
}

function formatDateFr(): string {
  const now = new Date();
  const days = ["Dimanche", "Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi"];
  const months = [
    "janvier", "février", "mars", "avril", "mai", "juin",
    "juillet", "août", "septembre", "octobre", "novembre", "décembre",
  ];
  return `${days[now.getDay()]} ${now.getDate()} ${months[now.getMonth()]}`;
}

const MissionHeader: React.FC<Props> = ({ driverName, missionCount = 0 }) => {
  const { connected, reconnecting } = useSocketStatus();
  const [showStatus, setShowStatus] = React.useState(false);

  const statusColor = reconnecting
    ? palette.reconnecting
    : connected
      ? palette.connected
      : palette.disconnected;

  const statusBg = reconnecting
    ? "rgba(245,158,11,0.1)"
    : connected
      ? "rgba(22,163,74,0.1)"
      : "rgba(239,68,68,0.1)";

  const statusText = reconnecting
    ? "Reconnexion"
    : connected
      ? "En ligne"
      : "Hors ligne";

  return (
    <View style={styles.container}>
      <View style={styles.topRow}>
        <View style={{ flex: 1 }}>
          <Text style={styles.greeting}>
            {getGreetingPrefix()},{" "}
            <Text style={styles.greetingName}>{driverName || "Chauffeur"}</Text>
          </Text>
          <Text style={styles.dateText}>{formatDateFr()}</Text>
        </View>

        <TouchableOpacity
          style={[styles.statusPill, { backgroundColor: statusBg }]}
          onPress={() => setShowStatus(!showStatus)}
          activeOpacity={0.7}
        >
          <View style={[styles.statusDot, { backgroundColor: statusColor }]} />
          <Text style={[styles.statusLabel, { color: statusColor }]}>{statusText}</Text>
        </TouchableOpacity>
      </View>

      <View style={styles.summaryRow}>
        <View style={styles.summaryCard}>
          <View style={[styles.summaryIconWrap, { backgroundColor: palette.brandLight }]}>
            <Ionicons name="car-outline" size={14} color={palette.brand} />
          </View>
          <View>
            <Text style={styles.summaryValue}>{missionCount}</Text>
            <Text style={styles.summaryLabel}>
              {missionCount <= 1 ? "Course" : "Courses"}
            </Text>
          </View>
        </View>

        <View style={styles.summaryCard}>
          <View style={[styles.summaryIconWrap, { backgroundColor: palette.brandLight }]}>
            <Ionicons name="time-outline" size={14} color={palette.brand} />
          </View>
          <View>
            <Text style={styles.summaryValue}>
              {new Date().toLocaleTimeString("fr-FR", { hour: "2-digit", minute: "2-digit" })}
            </Text>
            <Text style={styles.summaryLabel}>Heure</Text>
          </View>
        </View>
      </View>
    </View>
  );
};

export default MissionHeader;
