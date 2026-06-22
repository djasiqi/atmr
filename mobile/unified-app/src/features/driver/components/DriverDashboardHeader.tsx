import { useEffect, useMemo, useState } from "react";
import { Image, Pressable, StyleSheet, Text, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import * as Location from "expo-location";
import { AppText } from "../../../design/ui/AppText";
import { useSession } from "../../../core/sessionProvider";
import { getDriverProfile } from "../api";
import { readDriverProfileCache } from "../services/driverProfileCache";
import { useNotifications, useTrackingState } from "../hooks";
import { D } from "../theme/driverDashboardTheme";
import { FONT_SIZE } from "../../../design/responsive/typographyTokens";

const C = {
  text: D.text,
  textSub: D.textSub,
  textMuted: D.textMuted,
  brand: D.brand,
  available: D.available,
  iconMuted: "rgba(30, 41, 59, 0.52)",
  iconPressedBg: "rgba(145, 165, 157, 0.14)",
} as const;

const SWISS_TZ = "Europe/Zurich";

type Props = {
  isAvailable: boolean;
  onToggleAvailability?: () => void;
  availabilityPending?: boolean;
};

function formatSyncTime(ts: number | undefined): string {
  if (ts == null || !Number.isFinite(ts)) return "—";
  return new Date(ts).toLocaleTimeString("fr-CH", {
    timeZone: SWISS_TZ,
    hour: "2-digit",
    minute: "2-digit",
  });
}

function formatDriverName(raw: string | null | undefined): string {
  const t = (raw ?? "").trim();
  if (!t) return "Chauffeur";
  const parts = t.split(/\s+/);
  if (parts.length === 1) return parts[0]!;
  const family = parts[parts.length - 1]!.toUpperCase();
  return `${parts.slice(0, -1).join(" ")} ${family}`;
}

export function DriverDashboardHeader({
  isAvailable,
  onToggleAvailability,
  availabilityPending = false,
}: Props) {
  const { bootstrap } = useSession();
  const tracking = useTrackingState();
  const { unreadCount } = useNotifications();
  const [photoUrl, setPhotoUrl] = useState<string | null>(null);
  const [profileName, setProfileName] = useState<string | null>(null);
  const [gpsEnabled, setGpsEnabled] = useState(true);

  const user = bootstrap?.user ?? null;
  const displayName = formatDriverName(profileName ?? user?.full_name ?? user?.username);

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      const cached = await readDriverProfileCache({ allowStale: true });
      if (cancelled) return;
      if (cached.profile) {
        const p = cached.profile;
        const fullName =
          typeof p.full_name === "string" && p.full_name.length > 0
            ? p.full_name
            : [p.first_name, p.last_name]
                .filter((v) => typeof v === "string" && v.length > 0)
                .join(" ");
        if (fullName.length > 0) setProfileName(fullName);
        if (typeof p.photo_url === "string" && p.photo_url.length > 0) {
          setPhotoUrl(p.photo_url);
        }
      }
      try {
        const profile = await getDriverProfile();
        if (cancelled) return;
        const fullName =
          typeof profile.full_name === "string" && profile.full_name.length > 0
            ? profile.full_name
            : [profile.first_name, profile.last_name]
                .filter((v) => typeof v === "string" && v.length > 0)
                .join(" ");
        if (fullName.length > 0) setProfileName(fullName);
        if (typeof profile.photo_url === "string" && profile.photo_url.length > 0) {
          setPhotoUrl(profile.photo_url);
        }
      } catch {
        /* cache / session suffisent */
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let mounted = true;
    void Location.hasServicesEnabledAsync()
      .then((enabled) => {
        if (mounted) setGpsEnabled(enabled);
      })
      .catch(() => {
        if (mounted) setGpsEnabled(false);
      });
    return () => {
      mounted = false;
    };
  }, [tracking.isTracking, tracking.lastUpdate]);

  const syncLabel = useMemo(() => {
    if (!gpsEnabled) return "GPS indisponible";
    if (!tracking.isTracking) return "GPS prêt";
    const lastSendTime = formatSyncTime(tracking.lastUpdate);
    const lastAckTime = formatSyncTime(tracking.lastAckAt);
    if (tracking.queueDepth > 0) {
      return `GPS connecté • Dernier envoi ${lastSendTime}`;
    }
    if (tracking.lastAckAt != null && tracking.queueDepth === 0) {
      return `GPS connecté • Confirmé backend ${lastAckTime}`;
    }
    return `GPS connecté • Dernier envoi ${lastSendTime}`;
  }, [
    gpsEnabled,
    tracking.isTracking,
    tracking.lastUpdate,
    tracking.lastAckAt,
    tracking.queueDepth,
  ]);

  const initials = useMemo(() => {
    const parts = displayName.split(/\s+/).filter(Boolean);
    if (parts.length === 0) return "?";
    if (parts.length === 1) return parts[0]!.slice(0, 2).toUpperCase();
    return `${parts[0]![0] ?? ""}${parts[parts.length - 1]![0] ?? ""}`.toUpperCase();
  }, [displayName]);

  return (
    <View style={styles.wrap}>
      <View style={styles.leftCol}>
        <View style={styles.avatarCircle} accessibilityElementsHidden>
          {photoUrl ? (
            <Image
              source={{ uri: photoUrl }}
              style={styles.avatarImage}
              resizeMode="cover"
              accessibilityIgnoresInvertColors
            />
          ) : (
            <View style={styles.avatarFallback}>
              <Text style={styles.avatarInitials} includeFontPadding={false}>
                {initials}
              </Text>
            </View>
          )}
        </View>
        <View style={styles.identityCol}>
          <AppText variant="sectionTitle" style={styles.name} numberOfLines={1}>
            {displayName}
          </AppText>
          <Pressable
            onPress={onToggleAvailability}
            disabled={!onToggleAvailability || availabilityPending}
            accessibilityRole="button"
            accessibilityLabel={
              availabilityPending
                ? "Mise à jour de la disponibilité"
                : isAvailable
                  ? "Disponible. Appuyer pour passer indisponible."
                  : "Indisponible. Appuyer pour passer disponible."
            }
            android_ripple={{ color: "rgba(0, 0, 0, 0.06)", borderless: true }}
            style={({ pressed }) => [
              styles.statusChip,
              pressed && styles.statusChipPressed,
              availabilityPending && styles.statusChipPending,
            ]}
          >
            <View
              style={[
                styles.statusDot,
                { backgroundColor: isAvailable ? C.available : C.textMuted },
              ]}
            />
            <AppText
              variant="caption"
              style={[styles.statusLabel, isAvailable && styles.statusLabelAvailable]}
            >
              {availabilityPending
                ? "Mise à jour…"
                : isAvailable
                  ? "Disponible"
                  : "Indisponible"}
            </AppText>
          </Pressable>
          <AppText variant="caption" style={styles.syncLine} numberOfLines={1}>
            {syncLabel}
          </AppText>
        </View>
      </View>

      <View style={styles.actionsCol}>
        <Pressable
          accessibilityRole="button"
          accessibilityLabel={`Notifications${unreadCount > 0 ? `, ${unreadCount} non lues` : ""}`}
          style={({ pressed }) => [styles.iconBtn, pressed && styles.iconBtnPressed]}
        >
          <Ionicons name="notifications-outline" size={16} color={C.iconMuted} />
          {unreadCount > 0 ? <View style={styles.notifDot} accessibilityElementsHidden /> : null}
        </Pressable>
      </View>
    </View>
  );
}

const AVATAR_SIZE = 46;

const styles = StyleSheet.create({
  wrap: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    gap: 10,
  },
  leftCol: {
    flex: 1,
    minWidth: 0,
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
  },
  /** Cercle strict — pas d'elevation ici (sinon artefact octogonal sur Android). */
  avatarCircle: {
    width: AVATAR_SIZE,
    height: AVATAR_SIZE,
    borderRadius: AVATAR_SIZE / 2,
    overflow: "hidden",
    borderWidth: 1.5,
    borderColor: "rgba(0, 121, 107, 0.16)",
    backgroundColor: "rgba(0, 121, 107, 0.1)",
  },
  avatarImage: {
    width: AVATAR_SIZE,
    height: AVATAR_SIZE,
  },
  avatarFallback: {
    flex: 1,
    width: "100%",
    height: "100%",
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "rgba(0, 121, 107, 0.12)",
  },
  avatarInitials: {
    color: C.brand,
    fontWeight: "800",
    fontSize: FONT_SIZE.px15,
    lineHeight: 15,
    letterSpacing: 0.4,
    textAlign: "center",
  },
  identityCol: {
    flex: 1,
    minWidth: 0,
    gap: 0,
  },
  name: {
    color: C.text,
    fontSize: FONT_SIZE.px17,
    fontWeight: "700",
    lineHeight: 20,
    letterSpacing: -0.12,
  },
  statusChip: {
    flexDirection: "row",
    alignItems: "center",
    gap: 5,
    alignSelf: "flex-start",
    marginTop: 2,
    paddingVertical: 2,
    paddingHorizontal: 0,
    borderWidth: 0,
    borderColor: "transparent",
    backgroundColor: "transparent",
  },
  statusChipPressed: {
    opacity: 0.72,
  },
  statusChipPending: {
    opacity: 0.65,
  },
  statusDot: {
    width: 7,
    height: 7,
    borderRadius: 3.5,
  },
  statusLabel: {
    color: C.textSub,
    fontWeight: "700",
    fontSize: FONT_SIZE.px12,
    lineHeight: 14,
    backgroundColor: "transparent",
  },
  statusLabelAvailable: {
    color: C.available,
  },
  syncLine: {
    color: C.textMuted,
    fontSize: FONT_SIZE.px10,
    lineHeight: 12,
    marginTop: 1,
    letterSpacing: 0.05,
  },
  actionsCol: {
    flexDirection: "row",
    alignItems: "center",
    flexShrink: 0,
  },
  iconBtn: {
    width: 32,
    height: 32,
    borderRadius: 16,
    alignItems: "center",
    justifyContent: "center",
    position: "relative",
  },
  iconBtnPressed: {
    backgroundColor: C.iconPressedBg,
  },
  notifDot: {
    position: "absolute",
    top: 7,
    right: 7,
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: C.brand,
  },
});
