import { Platform, Pressable, StyleSheet, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { AppText } from "../../../design/ui/AppText";
import { createShadow } from "../../../styles/shadowStyles";
import { E } from "../../company/theme/enterpriseOpsTheme";
import { resolveDriverStatusForUx, getDriverStatusUx } from "../statusDictionary";
import type { DriverMission, DriverMissionStatus, DriverTransitionStatus } from "../types";
import { getClientBirthDateDisplay } from "../domain/missionDisplay";
import {
  getCallablePhoneFromMission,
  openNavigation,
  safeCall,
} from "../utils/missionContact";
import {
  getDropoffHints,
  getPickupHints,
  type HintItem,
  type MissionHintLike,
} from "../domain/missionHints";
import { useSession } from "../../../core/sessionProvider";
import { FONT_SIZE } from "../../../design/responsive/typographyTokens";

type Props = {
  mission: DriverMission;
  /** Si défini et permission OK : actions « operations-app » (appel, GPS, transitions). */
  onMissionTransition?: (target: DriverTransitionStatus) => void;
  /**
   * Si défini et permission OK : bouton « Libérer » visible sur `ASSIGNED || EN_ROUTE`
   * (parité `operations-app/MissionCard.tsx` lignes 1042–1061). Sémantique distincte
   * d’« Annuler » : la mission est rendue au pool dispatch, sans facturation client.
   */
  onMissionRelease?: () => void;
  pending?: boolean;
};

const C = {
  text: E.TEXT,
  textSub: E.TEXT_SEC,
  textMuted: E.TEXT_MUTED,
  border: E.BORDER,
  cardBg: E.CARD,
  brand: E.BRAND,
  brandSoft: "rgba(0, 121, 107, 0.08)",
  statusDotWell: "#E2E8F0",
  /**
   * Bouton « Libérer » : slate-500 Tailwind (`#64748B`) — variante modernisée
   * du `secondaryAction #6c757d` d'`operations-app`, mieux harmonisée avec
   * `enterpriseOpsTheme.TEXT_MUTED`.
   */
  releaseBg: "#64748B",
} as const;

const cardShadow = createShadow({
  shadowColor: "#000000",
  shadowOffset: { width: 0, height: 2 },
  shadowOpacity: 0.04,
  shadowRadius: 8,
  elevation: 2,
});

/** Ordre du flux principal (évite d’afficher FAILED comme action principale). */
const FORWARD_TRANSITION_PRIORITY: DriverTransitionStatus[] = [
  "EN_ROUTE",
  "ARRIVED",
  "IN_PROGRESS",
  "COMPLETED",
];

/**
 * Adresse complète : on laisse `numberOfLines={2}` faire le clamp visuel
 * (snapshot web utilise `-webkit-line-clamp: 2`). Pas de troncation côté JS
 * pour préserver le contexte (ville, pays) si l'utilisateur agrandit la carte.
 */
function conciseAddressLine(s: string | null | undefined): string {
  const t = s?.trim() ?? "";
  return t || "—";
}

/** Nom affiché en en-tête (priorité alignée `operations-app/MissionCard.tsx`). */
function getMissionClientDisplayName(mission: DriverMission): string {
  const direct = typeof mission.client_name === "string" ? mission.client_name.trim() : "";
  if (direct.length > 0) return formatClientHeaderName(direct);
  const nest = mission.client as { full_name?: unknown } | null | undefined;
  const full =
    nest?.full_name != null && String(nest.full_name).trim().length > 0
      ? String(nest.full_name).trim()
      : "";
  if (full.length > 0) return formatClientHeaderName(full);
  return `Mission #${mission.id}`;
}

/**
 * Convention typographique FR : prénom(s) inchangé(s), nom de famille en majuscules.
 * Heuristique : dernier segment après espaces = nom (ex. « Marie-Claire Dupont » → segment unique si pas d'espace interne).
 */
function formatClientHeaderName(raw: string): string {
  const t = raw.trim().replace(/\s+/g, " ");
  if (!t) return t;
  if (/^mission\s*#\s*\d+$/i.test(t)) return t;
  const parts = t.split(" ");
  if (parts.length === 1) return parts[0]!;
  const family = parts[parts.length - 1]!.toUpperCase();
  return `${parts.slice(0, -1).join(" ")} ${family}`;
}

const SWISS_TZ = "Europe/Zurich";

/**
 * Libellé court pour le badge statut — parité `operations-app/MissionCard.tsx:172-188`
 * (`formatStatus`). Évite la répétition du mot « Mission » déjà induit par le contexte
 * de la carte.
 */
function getBadgeStatusLabel(statusKey: DriverMissionStatus): string {
  switch (statusKey) {
    case "ASSIGNED":
      return "Assignée";
    case "EN_ROUTE":
      return "En route";
    case "ARRIVED":
      return "Arrivé";
    case "IN_PROGRESS":
      return "En cours";
    case "COMPLETED":
      return "Terminée";
    case "CANCELLED":
      return "Annulée";
    case "REASSIGNED":
      return "Réassignée";
    case "NO_SHOW":
      return "Absent";
    case "FAILED":
      return "Échec";
    default:
      return "À venir";
  }
}

/**
 * Date+heure planifiée mission → ex. `jeu. 07.05.2026, 23:18` (fr-CH, Europe/Zurich).
 * Aligné `app/(app)/(driver)/index.tsx:formatNextCourseWhen` pour cohérence inter-écrans.
 */
function getScheduledWhenDisplay(mission: DriverMission): string | null {
  const raw =
    typeof mission.scheduled_time === "string" && mission.scheduled_time.length > 0
      ? mission.scheduled_time
      : typeof mission.scheduled_at === "string"
        ? (mission.scheduled_at as string)
        : null;
  if (!raw) return null;
  const d = new Date(raw);
  if (!Number.isFinite(d.getTime())) return null;
  return d.toLocaleString("fr-CH", {
    timeZone: SWISS_TZ,
    weekday: "short",
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

/**
 * Civilité affichée au-dessus du nom client (parité `operations-app/MissionCard.tsx:32-38`).
 * Renvoie `"MADAME"` / `"MONSIEUR"` (majuscules) ou `null` si genre inconnu / AUTRE.
 * Accepte les variantes FR (`HOMME`/`FEMME`) et EN (`MALE`/`FEMALE`) pour tolérance backend.
 */
function getClientCivilityLabel(mission: DriverMission): string | null {
  const nest = mission.client as { gender?: unknown } | null | undefined;
  const raw = nest?.gender;
  if (raw == null) return null;
  const g = String(raw).trim().toUpperCase();
  if (!g) return null;
  if (g === "FEMME" || g === "FEMALE" || g === "F") return "MADAME";
  if (g === "HOMME" || g === "MALE" || g === "M") return "MONSIEUR";
  return null;
}

function transitionLabel(target: DriverTransitionStatus): string {
  switch (target) {
    case "EN_ROUTE":
      return "En route";
    case "ARRIVED":
      return "Arrivé";
    case "IN_PROGRESS":
      return "À bord";
    case "COMPLETED":
      return "Terminer";
    case "CANCELLED":
      return "Annuler";
    case "FAILED":
      return "Échec";
    default:
      return target;
  }
}

function transitionIcon(target: DriverTransitionStatus): keyof typeof Ionicons.glyphMap {
  switch (target) {
    case "EN_ROUTE":
      return "walk-outline";
    case "ARRIVED":
      return "flag-outline";
    case "IN_PROGRESS":
      return "person-outline";
    case "COMPLETED":
      return "checkmark-done-outline";
    case "CANCELLED":
      return "close-circle-outline";
    default:
      return "alert-circle-outline";
  }
}

function navigationDestination(mission: DriverMission, statusKey: DriverMissionStatus): string {
  const pickup = String(mission.pickup_location ?? "").trim();
  const dropoff = String(mission.dropoff_location ?? "").trim();
  if (statusKey === "IN_PROGRESS") return dropoff || pickup;
  return pickup || dropoff;
}

export function MissionCard({
  mission,
  onMissionTransition,
  onMissionRelease,
  pending = false,
}: Props) {
  const { can } = useSession();
  const statusUx = getDriverStatusUx(mission.status);
  const statusKey = resolveDriverStatusForUx(mission.status);
  const pickup = (mission.pickup_location as string | null | undefined) ?? null;
  const dropoff = (mission.dropoff_location as string | null | undefined) ?? null;

  const phone = getCallablePhoneFromMission(mission);
  const dest = navigationDestination(mission, statusKey);

  const canMutate = can("mission:update_status") && typeof onMissionTransition === "function";
  const showTransitionActions = canMutate && !statusUx.terminal;
  const showCall = Boolean(phone) && !statusUx.terminal;

  const forwardTransition = FORWARD_TRANSITION_PRIORITY.find((t) =>
    statusUx.nextTransitions.includes(t)
  );
  const canCancel = Boolean(showTransitionActions && statusUx.nextTransitions.includes("CANCELLED"));
  /**
   * « Libérer » : visible uniquement sur ASSIGNED ou EN_ROUTE — parité stricte
   * `operations-app/MissionCard.tsx:1043`. Côté API : `CANCELLED` + `reason: "RELEASE"`,
   * donc même permission `mission:update_status`.
   */
  const canRelease = Boolean(
    can("mission:update_status") &&
    typeof onMissionRelease === "function" &&
    (statusKey === "ASSIGNED" || statusKey === "EN_ROUTE")
  );

  /**
   * GPS (navigation) : disponible dès la prise en charge — le chauffeur peut naviguer
   * vers la prise en charge avant même de confirmer « En route ».
   * Aucune permission requise car n’altère pas l’état mission.
   */
  const showGps = !statusUx.terminal && dest.length > 0;

  const showActionsSection = showCall || showGps || showTransitionActions || canRelease;

  /**
   * Hints contextuels affichés sous le `routeBlock`. Choix dynamique :
   * - `IN_PROGRESS` (client à bord) → infos pour l'**arrivée** (destination)
   * - sinon (ASSIGNED / EN_ROUTE / ARRIVED) → infos pour la **prise en charge** (pickup)
   * Cachés sur les statuts terminaux (COMPLETED / CANCELLED / NO_SHOW / FAILED).
   */
  const hints: HintItem[] = !statusUx.terminal
    ? statusKey === "IN_PROGRESS"
      ? getDropoffHints(mission as unknown as MissionHintLike)
      : getPickupHints(mission as unknown as MissionHintLike)
    : [];
  const hintsContextLabel =
    statusKey === "IN_PROGRESS" ? "À l'arrivée" : "Pour la prise en charge";

  const openTel = () => {
    if (!phone) return;
    void safeCall(phone);
  };

  /**
   * Lance la navigation : tente l'app native (geo:/maps:) puis bascule sur
   * Google Maps web — aligné `operations-app/services/deepLinks.ts:openNavigation`.
   */
  const openGps = () => {
    if (!dest) return;
    void openNavigation(dest);
  };

  const clientTitle = getMissionClientDisplayName(mission);
  const civilityLabel = getClientCivilityLabel(mission);
  const birthDateDisplay = getClientBirthDateDisplay(mission);
  const scheduledWhen = getScheduledWhenDisplay(mission);
  const missionRefFallback = `Mission #${mission.id}`;
  /**
   * Sous-titre prioritaire : date de naissance client (parité snapshot
   * `operations-app/MissionCard.tsx:304-315`). Repli `Mission #id` si absente
   * et seulement si on n'a pas déjà ça en titre.
   */
  const showBirthDateRow = birthDateDisplay != null;
  const showMissionIdRef = !showBirthDateRow && clientTitle !== missionRefFallback;

  return (
    <View
      style={styles.card}
      accessibilityLabel={`Mission ${mission.id}${
        civilityLabel ? `, ${civilityLabel.toLowerCase()}` : ""
      }, ${clientTitle}${birthDateDisplay ? `, né(e) le ${birthDateDisplay}` : ""}`}
    >
      <View style={styles.headerRow}>
        <View style={styles.iconWrap} accessibilityElementsHidden>
          <Ionicons name="person-outline" size={16} color={C.brand} />
        </View>
        <View style={styles.headerTextCol}>
          {civilityLabel ? (
            <AppText variant="caption" style={styles.civilityLabel} numberOfLines={1}>
              {civilityLabel}
            </AppText>
          ) : null}
          <AppText variant="sectionTitle" style={styles.title} numberOfLines={2}>
            {clientTitle}
          </AppText>
          {showBirthDateRow ? (
            <AppText variant="caption" style={styles.birthDateText} numberOfLines={1}>
              {birthDateDisplay}
            </AppText>
          ) : null}
          {showMissionIdRef ? (
            <AppText variant="caption" style={styles.missionIdRef} numberOfLines={1}>
              {missionRefFallback}
            </AppText>
          ) : null}
        </View>
        {/**
         * Colonne droite : badge statut mission — parité `operations-app/MissionCard.tsx:326-328`
         * (`statusBadgeContainer` + `statusBadgeText`). Caché si terminal pour éviter
         * un badge "Terminée" persistant sur des cartes archivées.
         */}
        <View style={styles.headerBadgesCol} accessibilityElementsHidden>
          <View style={styles.statusBadge}>
            <AppText
              variant="caption"
              style={styles.statusBadgeText}
              numberOfLines={1}
              adjustsFontSizeToFit
              minimumFontScale={0.85}
              ellipsizeMode="tail"
            >
              {getBadgeStatusLabel(statusKey)}
            </AppText>
          </View>
        </View>
      </View>

      <View style={styles.body}>
        <View style={styles.statusRow}>
          <View style={styles.statusDotWell} accessibilityElementsHidden>
            <View style={styles.statusDotInner} />
          </View>
          <AppText variant="caption" style={styles.statusLabel} numberOfLines={1}>
            Prévu
          </AppText>
          <AppText variant="caption" style={styles.statusValue} numberOfLines={2}>
            {scheduledWhen ?? "Non planifiée"}
          </AppText>
        </View>

        <View style={styles.routeBlock}>
          <AppText variant="caption" style={styles.addressLine} numberOfLines={2}>
            <AppText variant="caption" style={styles.addressKey}>
              Départ :{" "}
            </AppText>
            {conciseAddressLine(pickup)}
          </AppText>
          <AppText variant="caption" style={styles.addressLine} numberOfLines={2}>
            <AppText variant="caption" style={styles.addressKey}>
              Arrivée :{" "}
            </AppText>
            {conciseAddressLine(dropoff)}
          </AppText>
        </View>

        {hints.length > 0 ? (
          <View
            style={styles.hintsBlock}
            accessibilityLabel={`Informations ${hintsContextLabel.toLowerCase()}`}
          >
            <AppText variant="caption" style={styles.hintsHeader} numberOfLines={1}>
              {hintsContextLabel}
            </AppText>
            <View style={styles.hintsList}>
              {hints.map((hint, idx) => (
                <View key={`${hint.label}-${idx}`} style={styles.hintRow}>
                  <View style={styles.hintIconWrap} accessibilityElementsHidden>
                    <Ionicons name={hint.icon} size={13} color={C.brand} />
                  </View>
                  <AppText variant="caption" style={styles.hintText} numberOfLines={2}>
                    <AppText variant="caption" style={styles.hintKey}>
                      {hint.label} :{" "}
                    </AppText>
                    {hint.value}
                  </AppText>
                </View>
              ))}
            </View>
          </View>
        ) : null}
      </View>

      {showActionsSection ? (
        <View style={styles.actionsBlock}>
          <View style={styles.actionsPrimary}>
            {showCall && phone ? (
              <Pressable
                onPress={openTel}
                disabled={pending}
                style={({ pressed }) => [
                  styles.actionPill,
                  styles.actionPillFlex,
                  styles.actionPillBrand,
                  pending && styles.disabledOpacity,
                  pressed && styles.pressed,
                ]}
                accessibilityRole="button"
                accessibilityLabel="Appeler le client"
              >
                <Ionicons name="call-outline" size={13} color="#FFFFFF" />
                <AppText variant="caption" style={styles.actionPillLabel}>
                  Appeler
                </AppText>
              </Pressable>
            ) : null}

            {showGps ? (
              <Pressable
                onPress={openGps}
                disabled={pending}
                style={({ pressed }) => [
                  styles.actionPill,
                  styles.actionPillFlex,
                  styles.actionPillBrand,
                  pending && styles.disabledOpacity,
                  pressed && styles.pressed,
                ]}
                accessibilityRole="button"
                accessibilityLabel="Ouvrir la navigation vers l’étape en cours"
              >
                <Ionicons name="navigate-outline" size={13} color="#FFFFFF" />
                <AppText variant="caption" style={styles.actionPillLabel}>
                  GPS
                </AppText>
              </Pressable>
            ) : null}

            {showTransitionActions && forwardTransition ? (
              <Pressable
                onPress={() => onMissionTransition?.(forwardTransition)}
                disabled={pending}
                style={({ pressed }) => [
                  styles.actionPill,
                  styles.actionPillFlex,
                  styles.actionPillBrand,
                  pending && styles.disabledOpacity,
                  pressed && styles.pressed,
                ]}
                accessibilityRole="button"
                accessibilityLabel={transitionLabel(forwardTransition)}
              >
                <Ionicons name={transitionIcon(forwardTransition)} size={13} color="#FFFFFF" />
                <AppText variant="caption" style={styles.actionPillLabel}>
                  {transitionLabel(forwardTransition)}
                </AppText>
              </Pressable>
            ) : null}
          </View>

          {canRelease && canCancel ? (
            <View style={styles.actionsSecondary}>
              <Pressable
                onPress={() => onMissionRelease?.()}
                disabled={pending}
                style={({ pressed }) => [
                  styles.actionPill,
                  styles.actionPillSecondary,
                  styles.actionPillRelease,
                  pending && styles.disabledOpacity,
                  pressed && styles.pressed,
                ]}
                accessibilityRole="button"
                accessibilityLabel="Libérer la mission (sera réassignée)"
              >
                <Ionicons name="refresh-outline" size={13} color="#FFFFFF" />
                <AppText variant="caption" style={styles.actionPillLabel}>
                  Libérer
                </AppText>
              </Pressable>
              <Pressable
                onPress={() => onMissionTransition?.("CANCELLED")}
                disabled={pending}
                style={({ pressed }) => [
                  styles.actionPill,
                  styles.actionPillSecondary,
                  styles.actionPillDanger,
                  pending && styles.disabledOpacity,
                  pressed && styles.pressed,
                ]}
                accessibilityRole="button"
                accessibilityLabel="Annuler la mission avec justification"
              >
                <Ionicons name="close-circle-outline" size={13} color="#FFFFFF" />
                <AppText variant="caption" style={styles.actionPillLabel}>
                  Annuler
                </AppText>
              </Pressable>
            </View>
          ) : canRelease ? (
            <Pressable
              onPress={() => onMissionRelease?.()}
              disabled={pending}
              style={({ pressed }) => [
                styles.actionPill,
                styles.actionPillRelease,
                styles.actionPillSingleRow,
                styles.actionPillSelfEnd,
                pending && styles.disabledOpacity,
                pressed && styles.pressed,
              ]}
              accessibilityRole="button"
              accessibilityLabel="Libérer la mission (sera réassignée)"
            >
              <Ionicons name="refresh-outline" size={13} color="#FFFFFF" />
              <AppText variant="caption" style={styles.actionPillLabel}>
                Libérer
              </AppText>
            </Pressable>
          ) : canCancel ? (
            <Pressable
              onPress={() => onMissionTransition?.("CANCELLED")}
              disabled={pending}
              style={({ pressed }) => [
                styles.actionPill,
                styles.actionPillDanger,
                styles.actionPillSingleRow,
                styles.actionPillSelfEnd,
                pending && styles.disabledOpacity,
                pressed && styles.pressed,
              ]}
              accessibilityRole="button"
              accessibilityLabel="Annuler la mission avec justification"
            >
              <Ionicons name="close-circle-outline" size={13} color="#FFFFFF" />
              <AppText variant="caption" style={styles.actionPillLabel}>
                Annuler
              </AppText>
            </Pressable>
          ) : null}
        </View>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    alignSelf: "stretch",
    backgroundColor: C.cardBg,
    borderWidth: 1,
    borderColor: C.border,
    borderRadius: 16,
    padding: 16,
    gap: 12,
    ...cardShadow,
  },
  headerRow: {
    flexDirection: "row",
    alignItems: "flex-start",
    gap: 8,
  },
  headerTextCol: {
    flex: 1,
    minWidth: 0,
    gap: 2,
  },
  /**
   * Colonne droite du header — badge statut mission. `flexShrink: 0` pour
   * garantir que le badge ne se compresse pas, `marginLeft: auto` pour qu'il
   * se cale à droite même si la colonne texte est étroite.
   */
  headerBadgesCol: {
    flexShrink: 0,
    marginLeft: 4,
    alignItems: "flex-end",
  },
  /**
   * Badge statut — taille fixe 80x28 (parité visuelle avec les autres badges).
   * Fond brand 8 % + bordure brand 15 % conservés. `paddingHorizontal: 6` laisse
   * de la marge pour les statuts longs (« Mission assignee »).
   */
  statusBadge: {
    width: 80,
    height: 28,
    backgroundColor: "rgba(0, 121, 107, 0.08)",
    borderWidth: 1,
    borderColor: "rgba(0, 121, 107, 0.15)",
    paddingHorizontal: 6,
    borderRadius: 8,
    alignItems: "center",
    justifyContent: "center",
  },
  statusBadgeText: {
    color: C.brand,
    fontSize: FONT_SIZE.px11,
    fontWeight: "700",
    letterSpacing: 0.2,
    lineHeight: 14,
    textAlign: "center",
  },
  iconWrap: {
    width: 28,
    height: 28,
    borderRadius: 8,
    backgroundColor: C.brandSoft,
    alignItems: "center",
    justifyContent: "center",
  },
  title: {
    color: C.text,
    fontSize: FONT_SIZE.px16,
    fontWeight: "700",
    lineHeight: 22,
  },
  missionIdRef: {
    color: C.textMuted,
    fontSize: FONT_SIZE.px12,
    fontWeight: "600",
    lineHeight: 16,
    marginTop: 2,
  },
  /**
   * Civilité au-dessus du nom client (parité `operations-app:clientCivility`) :
   * petit eyebrow texte 11px, MAJUSCULES, lettrage espacé pour hiérarchie claire.
   */
  civilityLabel: {
    color: C.textMuted,
    fontSize: FONT_SIZE.px11,
    fontWeight: "700",
    letterSpacing: 0.6,
    lineHeight: 14,
    textTransform: "uppercase",
    marginBottom: 2,
  },
  /** Date de naissance client — sans icône, 13px secondaire, gap col 2. */
  birthDateText: {
    color: C.textSub,
    fontSize: FONT_SIZE.px13,
    fontWeight: "500",
    lineHeight: 16,
    marginTop: 2,
  },
  body: {
    gap: 10,
  },
  /** Snapshot : pastille statut sur fond gris (`r-backgroundColor-1h78ys6`). */
  statusRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    flexWrap: "wrap",
  },
  statusDotWell: {
    width: 10,
    height: 10,
    borderRadius: 5,
    backgroundColor: C.statusDotWell,
    alignItems: "center",
    justifyContent: "center",
  },
  statusDotInner: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: C.brand,
  },
  statusLabel: {
    color: C.textMuted,
    fontWeight: "700",
    letterSpacing: 0.5,
    textTransform: "uppercase",
    fontSize: FONT_SIZE.px12,
    lineHeight: 16,
  },
  /** Valeur statut 13px comme le snapshot web. */
  statusValue: {
    color: C.text,
    fontWeight: "600",
    flex: 1,
    minWidth: "45%",
    fontSize: FONT_SIZE.px13,
    lineHeight: 16,
    ...(Platform.OS === "android" ? { includeFontPadding: false } : {}),
  },
  routeBlock: {
    gap: 8,
  },
  addressLine: {
    color: C.textSub,
    fontSize: FONT_SIZE.px13,
    lineHeight: 18,
    fontWeight: "500",
  },
  addressKey: {
    color: C.textMuted,
    fontWeight: "700",
    fontSize: FONT_SIZE.px13,
    lineHeight: 16,
  },
  /**
   * Bloc hints — visible sous le routeBlock, contenu dynamique selon le statut
   * (parité `operations-app/src/domain/missionHints.ts` + UX dispatch).
   */
  hintsBlock: {
    marginTop: 8,
    paddingTop: 8,
    paddingHorizontal: 10,
    paddingBottom: 8,
    backgroundColor: C.brandSoft,
    borderRadius: 10,
    gap: 6,
  },
  hintsHeader: {
    color: C.brand,
    fontSize: FONT_SIZE.px11,
    fontWeight: "700",
    letterSpacing: 0.4,
    lineHeight: 14,
    textTransform: "uppercase",
  },
  hintsList: {
    gap: 4,
  },
  hintRow: {
    flexDirection: "row",
    alignItems: "flex-start",
    gap: 6,
  },
  hintIconWrap: {
    width: 18,
    height: 18,
    alignItems: "center",
    justifyContent: "center",
    paddingTop: 1,
  },
  hintText: {
    color: C.textSub,
    fontSize: FONT_SIZE.px13,
    lineHeight: 18,
    fontWeight: "500",
    flex: 1,
    minWidth: 0,
  },
  hintKey: {
    color: C.text,
    fontWeight: "700",
    fontSize: FONT_SIZE.px13,
    lineHeight: 18,
  },
  /**
   * Réf. snapshot web compact (`r-gap-f4gmv6 r-paddingTop-ttdzmv r-paddingBottom-xd6kpl`) :
   * version resserrée — paddingTop 10, gap inter-rangées 10, paddingBottom 4.
   */
  actionsBlock: {
    gap: 10,
    paddingTop: 10,
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: C.border,
    marginTop: 4,
    paddingBottom: 4,
  },
  actionsPrimary: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 6,
    alignItems: "stretch",
  },
  /** Pills compactes premium : lisibles, tactiles, mais plus sobres. */
  actionPill: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 6,
    minHeight: 34,
    paddingHorizontal: 11,
    paddingVertical: 7,
    borderRadius: 12,
  },
  /**
   * Étirement uniforme des 3 pills primaires (Appeler / GPS / Forward) :
   * parité `operations-app/styles/missionCardStyles.ts:actionItemEnhanced`
   * (`flex: 1, flexBasis: 0`). `minWidth: 80` (compact) — l'icône + le label
   * « En route » tiennent sans tronquage avant qu'un wrap ne s'enclenche.
   */
  actionPillFlex: {
    flexGrow: 1,
    flexShrink: 1,
    flexBasis: 0,
    minWidth: 80,
  },
  actionPillBrand: {
    backgroundColor: "#0A6A61",
    borderWidth: 1,
    borderColor: "#095B53",
  },
  /**
   * Rangée secondaire compacte — Libérer + Annuler étirés sur **toute la largeur**
   * de la carte (parité visuelle stricte avec `actionsPrimary` : les 2 rangées
   * couvrent désormais la même largeur).
   */
  actionsSecondary: {
    flexDirection: "row",
    alignItems: "stretch",
    gap: 6,
  },
  /**
   * Pills danger : `flex: 1, flexBasis: 0` (sans `maxWidth`) → chaque bouton
   * occupe ~50 % de la largeur disponible, identique au comportement des
   * 3 pills primaires (parité avec rangée 1).
   */
  actionPillSecondary: {
    flexGrow: 1,
    flexShrink: 1,
    flexBasis: 0,
  },
  /**
   * Cas single button (Annuler ou Libérer seul) : aligné à droite directement,
   * comme `r-alignSelf-173mn98` d'un snapshot antérieur. Largeur naturelle.
   */
  actionPillSelfEnd: {
    alignSelf: "stretch",
  },
  /** Bouton seul: occupe toute la largeur pour éviter un CTA "perdu" dans un coin. */
  actionPillSingleRow: {
    width: "100%",
    justifyContent: "center",
  },
  /** Bouton « Libérer » : variante slate-500 (cf. `C.releaseBg`). */
  actionPillRelease: {
    backgroundColor: "#5C6B7D",
    borderWidth: 1,
    borderColor: "#4B5868",
  },
  actionPillDanger: {
    backgroundColor: "#B42318",
    borderWidth: 1,
    borderColor: "#9A1F16",
  },
  actionPillLabel: {
    color: "#FFFFFF",
    fontWeight: "700",
    fontSize: FONT_SIZE.px11_5,
    lineHeight: 14,
    letterSpacing: 0.15,
  },
  disabledOpacity: { opacity: 0.55 },
  pressed: { opacity: 0.9, transform: [{ scale: 0.985 }] },
});
