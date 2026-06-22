import { Pressable, StyleSheet, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { AppText } from "../../../../design/ui/AppText";
import { FONT_SIZE } from "../../../../design/responsive/typographyTokens";
import type { InstitutionOfferListPreview } from "../../utils/institutionOfferDisplay";
import type { InstitutionOfferSegment } from "../../utils/institutionOfferResponse";
import { E } from "../../theme/enterpriseOpsTheme";
import { createShadow } from "../../../../styles/shadowStyles";

const cardShadow = createShadow({
  shadowColor: "#000000",
  shadowOffset: { width: 0, height: 2 },
  shadowOpacity: 0.04,
  shadowRadius: 8,
  elevation: 2,
});

type InstitutionOfferListCardProps = {
  preview: InstitutionOfferListPreview;
  segment: InstitutionOfferSegment;
  onPress: () => void;
};

export function InstitutionOfferListCard({
  preview,
  segment,
  onPress,
}: InstitutionOfferListCardProps) {
  const urgent = segment === "urgent";

  return (
    <Pressable
      onPress={onPress}
      style={({ pressed }) => [
        s.card,
        urgent && s.cardUrgent,
        pressed && s.pressed,
      ]}
      accessibilityRole="button"
      accessibilityLabel={`${preview.title}, ${preview.schedule}, ${preview.route}`}
    >
      <View style={s.summaryRow}>
        <View style={s.timeCol}>
          {preview.primaryTime ? (
            <AppText style={s.timeMain}>{preview.primaryTime}</AppText>
          ) : (
            <Ionicons name="time-outline" size={18} color={E.TEXT_MUTED} />
          )}
          {preview.scheduleDate ? (
            <AppText style={s.timeDate} numberOfLines={1}>
              {preview.scheduleDate}
            </AppText>
          ) : null}
        </View>

        <View style={s.mainCol}>
          <AppText style={s.patient} numberOfLines={1}>
            {preview.title}
          </AppText>
          {preview.institutionLabel ? (
            <AppText style={s.institution} numberOfLines={1}>
              {preview.institutionLabel}
            </AppText>
          ) : null}
          {preview.scheduleExtras ? (
            <AppText style={s.scheduleExtras} numberOfLines={2}>
              {preview.scheduleExtras}
            </AppText>
          ) : !preview.primaryTime && preview.scheduleDetail ? (
            <AppText style={s.scheduleExtras} numberOfLines={2}>
              {preview.scheduleDetail}
            </AppText>
          ) : null}
        </View>

        <View style={s.chevronWrap}>
          <Ionicons name="chevron-forward" size={18} color={E.TEXT_MUTED} />
        </View>
      </View>

      <View style={s.routeRow}>
        <Ionicons name="navigate-outline" size={14} color={E.BRAND} style={s.routeIcon} />
        <AppText style={s.routeText} numberOfLines={2}>
          {preview.route}
        </AppText>
      </View>

      {preview.tripBadge ? (
        <View style={s.tripBadge}>
          <AppText style={s.tripBadgeText} numberOfLines={1}>
            {preview.tripBadge}
          </AppText>
        </View>
      ) : null}
    </Pressable>
  );
}

const s = StyleSheet.create({
  card: {
    borderWidth: 1,
    borderColor: "rgba(148, 163, 184, 0.22)",
    borderRadius: 14,
    paddingHorizontal: 12,
    paddingVertical: 11,
    marginBottom: 10,
    backgroundColor: E.CARD,
    ...cardShadow,
  },
  cardUrgent: {
    borderLeftWidth: 3,
    borderLeftColor: E.URGENT,
  },
  pressed: { opacity: 0.88 },
  summaryRow: {
    flexDirection: "row",
    alignItems: "flex-start",
    gap: 10,
  },
  timeCol: {
    width: 54,
    minHeight: 36,
    justifyContent: "center",
    gap: 2,
  },
  timeMain: {
    color: E.BRAND,
    fontWeight: "700",
    fontSize: FONT_SIZE.px15,
    letterSpacing: 0.2,
    lineHeight: 18,
  },
  timeDate: {
    color: E.TEXT_SEC,
    fontSize: FONT_SIZE.px11,
    lineHeight: 14,
    fontWeight: "600",
  },
  mainCol: {
    flex: 1,
    minWidth: 0,
    gap: 1,
    paddingTop: 1,
  },
  patient: {
    color: E.TEXT,
    fontWeight: "600",
    fontSize: FONT_SIZE.px14,
    lineHeight: 18,
  },
  institution: {
    color: E.TEXT_SEC,
    fontSize: FONT_SIZE.px11,
    lineHeight: 15,
  },
  scheduleExtras: {
    color: E.TEXT,
    fontSize: FONT_SIZE.px12,
    lineHeight: 16,
    fontWeight: "600",
    marginTop: 2,
  },
  chevronWrap: {
    width: 20,
    alignItems: "center",
    justifyContent: "center",
    paddingTop: 4,
  },
  routeRow: {
    flexDirection: "row",
    alignItems: "flex-start",
    gap: 6,
    marginTop: 10,
    paddingTop: 10,
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: "rgba(148, 163, 184, 0.2)",
  },
  routeIcon: {
    marginTop: 2,
  },
  routeText: {
    flex: 1,
    color: E.TEXT_SEC,
    fontSize: FONT_SIZE.px13,
    lineHeight: 18,
  },
  tripBadge: {
    alignSelf: "flex-start",
    marginTop: 8,
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 6,
    backgroundColor: "rgba(0, 121, 107, 0.08)",
  },
  tripBadgeText: {
    color: E.BRAND_DARK,
    fontSize: FONT_SIZE.px10,
    fontWeight: "700",
    letterSpacing: 0.3,
    textTransform: "uppercase",
  },
});
