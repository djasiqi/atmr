import { ReactNode } from "react";
import { Platform, StyleSheet, View } from "react-native";
import { AppText } from "../../../design/ui/AppText";

const BORDER = "rgba(145, 165, 157, 0.45)";

const sheetShadow = Platform.select({
  web: { boxShadow: "0 2px 10px rgba(22, 58, 52, 0.06)" },
  default: {
    shadowColor: "#163A34",
    shadowOpacity: 0.06,
    shadowRadius: 6,
    shadowOffset: { width: 0, height: 2 },
    elevation: 2,
  },
});

type EnterpriseSheetProps = {
  title: string;
  subtitle?: string;
  children: ReactNode;
};

export function EnterpriseSheet({ title, subtitle, children }: EnterpriseSheetProps) {
  return (
    <View style={s.root}>
      <AppText variant="sectionTitle">{title}</AppText>
      {subtitle ? (
        <AppText variant="caption" style={s.subtitleSpacing}>
          {subtitle}
        </AppText>
      ) : null}
      <View style={s.body}>{children}</View>
    </View>
  );
}

const s = StyleSheet.create({
  root: {
    borderWidth: 1,
    borderColor: BORDER,
    borderRadius: 12,
    paddingHorizontal: 10,
    paddingVertical: 9,
    gap: 4,
    backgroundColor: "#FFFFFF",
    ...sheetShadow,
  },
  subtitleSpacing: { marginTop: 1 },
  body: { gap: 6, marginTop: 2 },
});
