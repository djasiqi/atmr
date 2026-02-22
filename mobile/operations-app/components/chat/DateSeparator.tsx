import React from "react";
import { View, Text, StyleSheet } from "react-native";

const TXT_SEC = "#6b7280";
const BORDER = "#e5e7eb";

interface Props {
  date: string;
}

export default function DateSeparator({ date }: Props) {
  const formatDate = (dateString: string): string => {
    try {
      const d = new Date(dateString);
      const now = new Date();
      const isToday =
        d.getDate() === now.getDate() &&
        d.getMonth() === now.getMonth() &&
        d.getFullYear() === now.getFullYear();

      const yesterday = new Date(now);
      yesterday.setDate(yesterday.getDate() - 1);
      const isYesterday =
        d.getDate() === yesterday.getDate() &&
        d.getMonth() === yesterday.getMonth() &&
        d.getFullYear() === yesterday.getFullYear();

      if (isToday) return "Aujourd'hui";
      if (isYesterday) return "Hier";

      const day = d.getDate().toString().padStart(2, "0");
      const month = (d.getMonth() + 1).toString().padStart(2, "0");
      const year = d.getFullYear();
      return `${day}.${month}.${year}`;
    } catch {
      return dateString;
    }
  };

  return (
    <View style={st.container}>
      <View style={st.line} />
      <View style={st.pill}>
        <Text style={st.text}>{formatDate(date)}</Text>
      </View>
      <View style={st.line} />
    </View>
  );
}

const st = StyleSheet.create({
  container: { flexDirection: "row", alignItems: "center", marginVertical: 14, paddingHorizontal: 12 },
  line: { flex: 1, height: 1, backgroundColor: BORDER },
  pill: {
    backgroundColor: "#f4f4f5",
    paddingVertical: 3,
    paddingHorizontal: 12,
    borderRadius: 10,
    marginHorizontal: 10,
  },
  text: { fontSize: 11, color: TXT_SEC, fontWeight: "600", letterSpacing: 0.1 },
});
