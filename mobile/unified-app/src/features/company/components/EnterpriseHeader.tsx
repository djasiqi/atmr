import type { ReactNode } from "react";
import type { ViewStyle } from "react-native";
import { Platform, Pressable, StyleSheet, Text, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import dayjs from "dayjs";
import "dayjs/locale/fr";
import { E } from "../theme/enterpriseOpsTheme";
import { realtimeStatusA11yLabel } from "../utils/companyRealtimeFluxStatus";
import { CompanyInboxButton } from "./CompanyInboxButton";
import { createShadow } from "../../../styles/shadowStyles";
import { useAppViewport } from "../../../design/responsive/useAppViewport";
import { useAccessibilityScale } from "../../../design/responsive/useAccessibilityScale";
import { FONT_SIZE } from "../../../design/responsive/typographyTokens";

dayjs.locale("fr");

/**
 * Ardoise unique pour date, libellé mode et ligne méta (réf. `color-121wj93` / `1s7ct43`).
 * On n’utilise pas `AppText` ici : le variant `caption` injecte `brandTextMuted` (#5F7369) et des tailles
 * responsives, ce qui cassait l’homogénéité avec la pilule date en `Text`.
 */
const HEADER_SLATE_FG = E.TEXT_SEC;
/** Réf. `borderBottomColor-1a8zoe3` — séparation bandeau / contenu. */
const STRIP_BORDER_SLATE = "rgba(148, 163, 184, 0.28)";

const stripShadow = createShadow({
  shadowColor: "rgba(15, 54, 43, 0.06)",
  shadowOffset: { width: 0, height: 2 },
  shadowOpacity: 1,
  shadowRadius: 8,
  elevation: 2,
});

/** Réf. `boxShadow-1brrdx6` (web) — même esprit que les cartes dispatch (`companyShellCardShadow`). */
const stripShadowResolved: ViewStyle =
  Platform.OS === "web"
    ? ({ boxShadow: "0 2px 10px rgba(22, 58, 52, 0.06)" } as ViewStyle)
    : stripShadow;

const pressableTransitionWeb: ViewStyle =
  Platform.OS === "web" ? ({ transition: "opacity 0.25s ease" } as ViewStyle) : {};

/** Même ombre que les tuiles KPI dashboard (`kpiStat` / `dashboardSurfaceShadow`). */
const headerKpiTileShadowNative = createShadow({
  shadowColor: "#000000",
  shadowOffset: { width: 0, height: 2 },
  shadowOpacity: 0.04,
  shadowRadius: 8,
  elevation: 2,
});
const headerKpiTileShadow: ViewStyle =
  Platform.OS === "web"
    ? ({ boxShadow: "0 2px 8px rgba(0, 0, 0, 0.04)" } as ViewStyle)
    : headerKpiTileShadowNative;

/** Hauteur de ligne unique pour la rangée (titre, pilule date, chip mode) — évite le décalage baseline. */
const HEADER_ROW_LINE_HEIGHT = 20;
/** Rangée : pilule date + chip mode — minHeight (grandit avec Larger Text). */
const HEADER_ROW_MIN_HEIGHT = 39;

export type EnterpriseHeaderProps = {
  date: string;
  mode: string | null;
  realtimeStatus: string;
  /**
   * `full` : ligne méta complète (journée ISO, mode, réseau).
   * `networkOnly` : uniquement le statut réseau/temps réel (ex. tableau de bord).
   */
  metaDetail?: "full" | "networkOnly";
  /** Icône cloche / autre, à droite (ex. boîte entreprise). */
  trailing?: ReactNode;
  /** Pill statut temps réel (ex. Hors ligne / Connecté) rendu dans la rangée header. */
  liveStatusPill?: ReactNode;
  /** Titre court optionnel (ex. « Dispatch ») — masqué si absent. */
  title?: string;
  /** Jour précédent / suivant ; sans callback, les chevrons sont désactivés (sauf si `onOpenDatePicker` masque les chevrons). */
  onShiftDay?: (deltaDays: number) => void;
  /** Ouvre la feuille « Choisir une date » (pilule date entière). */
  onOpenDatePicker?: () => void;
  /** Ouvre la feuille « Mode de dispatch » (chip mode). */
  onOpenModePicker?: () => void;
  /** Affiche le chip mode de dispatch (désactivé sur certaines pages, ex. Courses). */
  showModeChip?: boolean;
  /**
   * Injecté par `Screen` (sticky) : hauteur de la zone sous la barre de statut / encoche.
   * Le fond de la bande (`E.CARD`) remplit cette zone ; le contenu reste en dessous.
   */
  topSafeAreaPx?: number;
  /** `floating` : bandeau glass au-dessus de la carte cockpit (sans fond pleine largeur). */
  variant?: "sticky" | "floating";
};

function formatMode(mode: string | null): string {
  if (mode === "manual") return "Manuel";
  if (mode === "semi_auto") return "Semi-auto";
  if (mode === "fully_auto") return "Plein auto";
  if (mode == null || mode === "") return "—";
  return mode;
}

/** Réf. dashboard / Courses web : `mar. 5 mai` (`r-color-121wj93`, `r-fontSize-1b43r93`). */
function formatDayLineHeader(isoDate: string): string {
  const d = dayjs(isoDate);
  if (!d.isValid()) return isoDate;
  return d.format("ddd D MMM");
}

export function EnterpriseHeader({
  date,
  mode,
  realtimeStatus,
  metaDetail = "full",
  trailing,
  liveStatusPill,
  title,
  onShiftDay,
  onOpenDatePicker,
  onOpenModePicker,
  showModeChip = true,
  topSafeAreaPx: topSafeAreaPxProp,
  variant = "sticky",
}: EnterpriseHeaderProps) {
  const { horizontalPadding, safeLeft, safeRight, topInset } = useAppViewport();
  const { chromeMaxFontMultiplier } = useAccessibilityScale();
  const topSafe = topSafeAreaPxProp !== undefined ? topSafeAreaPxProp : topInset;
  const stripContentPadLeft = Math.max(horizontalPadding, safeLeft);
  const stripContentPadRight = Math.max(horizontalPadding, safeRight);

  const dayLine = formatDayLineHeader(date);
  const isFloating = variant === "floating";
  const sheetDate = typeof onOpenDatePicker === "function";
  const canShift = typeof onShiftDay === "function" && !sheetDate;

  /** Sans statut réseau : la pastille du chip mode porte l’état WS. */
  const fullMetaLine = `Journée ${date} · ${formatMode(mode)}`;
  const socketA11y = realtimeStatusA11yLabel(realtimeStatus);

  const iconColor = sheetDate || canShift ? E.TEXT_SEC : E.TEXT_MUTED;
  const chevronLeftSize = 18;
  const chevronRightSize = 16;
  const headerIconSize = 16;
  const headerChevronDownSize = 14;

  const dateBlock = sheetDate ? (
    <Pressable
      onPress={onOpenDatePicker}
      style={({ pressed }) => [
        s.datePillWrap,
        s.datePillWrapSheet,
        isFloating ? s.datePillWrapFloating : null,
        headerKpiTileShadow,
        pressableTransitionWeb,
        pressed && s.pressed,
        Platform.OS === "web" ? s.datePillWeb : null,
      ]}
      accessibilityRole="button"
      accessibilityLabel={`Choisir une date. Jour affiché : ${dayLine}`}
    >
      <View style={s.headerIconWell} accessibilityElementsHidden>
        <Ionicons name="calendar-outline" size={headerIconSize} color={E.TEXT_SEC} accessible={false} />
      </View>
      <Text
        maxFontSizeMultiplier={chromeMaxFontMultiplier}
        style={[s.datePillText, s.datePillTextSheet, s.datePillTextEmphasis]}
        numberOfLines={1}
        ellipsizeMode="tail"
        accessible={false}
      >
        {dayLine}
      </Text>
      <Ionicons
        name="chevron-down-outline"
        size={headerChevronDownSize}
        color={E.TEXT_MUTED}
        style={s.datePillTrailingChevron}
        accessible={false}
      />
    </Pressable>
  ) : (
    <View
      style={[s.datePillWrap, headerKpiTileShadow]}
      accessibilityRole="summary"
      accessibilityLabel={`Jour : ${dayLine}`}
    >
      <Pressable
        onPress={() => onShiftDay?.(-1)}
        disabled={!canShift}
        style={({ pressed }) => [s.chevronHit, !canShift && s.chevDisabled, pressed && canShift && s.pressed]}
        accessibilityLabel="Jour précédent"
        accessibilityRole="button"
        hitSlop={8}
      >
        <Ionicons name="caret-back" size={chevronLeftSize} color={iconColor} />
      </Pressable>
      <Text
        maxFontSizeMultiplier={chromeMaxFontMultiplier}
        style={[s.datePillText, s.datePillTextEmphasis]}
        numberOfLines={1}
      >
        {dayLine}
      </Text>
      <Pressable
        onPress={() => onShiftDay?.(1)}
        disabled={!canShift}
        style={({ pressed }) => [s.chevronHit, !canShift && s.chevDisabled, pressed && canShift && s.pressed]}
        accessibilityLabel="Jour suivant"
        accessibilityRole="button"
        hitSlop={8}
      >
        <Ionicons name="caret-forward" size={chevronRightSize} color={iconColor} />
      </Pressable>
    </View>
  );

  const modeBlock =
    typeof onOpenModePicker === "function" ? (
      <Pressable
        onPress={onOpenModePicker}
        style={({ pressed }) => [
          s.modeChip,
          isFloating ? s.modeChipFloating : null,
          headerKpiTileShadow,
          pressableTransitionWeb,
          pressed && s.pressed,
          Platform.OS === "web" ? s.modeChipWeb : null,
        ]}
        accessibilityRole="button"
        accessibilityLabel={`Mode de dispatch. Actuel : ${formatMode(mode)}. ${socketA11y}. Appuyer pour choisir Manuel, Semi-auto ou Auto.`}
      >
        <Text
          maxFontSizeMultiplier={chromeMaxFontMultiplier}
          style={[s.modeChipText, s.modeChipTextEmphasis]}
          numberOfLines={1}
          ellipsizeMode="tail"
        >
          {formatMode(mode)}
        </Text>
      </Pressable>
    ) : (
      <View
        style={[s.modeChip, headerKpiTileShadow]}
        accessibilityLabel={`Mode ${formatMode(mode)}. ${socketA11y}`}
      >
        <Text
          maxFontSizeMultiplier={chromeMaxFontMultiplier}
          style={[s.modeChipText, s.modeChipTextEmphasis]}
          numberOfLines={1}
          ellipsizeMode="tail"
        >
          {formatMode(mode)}
        </Text>
      </View>
    );

  const resolvedPadLeft = isFloating ? Math.max(4, safeLeft) : stripContentPadLeft;
  const resolvedPadRight = isFloating ? Math.max(6, safeRight + 2) : stripContentPadRight;

  return (
    <View
      style={[
        s.stripBase,
        isFloating ? s.stripFloating : s.strip,
        { paddingTop: isFloating ? 0 : topSafe },
      ]}
    >
      <View
        style={[
          s.stripInner,
          isFloating ? s.stripInnerFloating : null,
          { paddingLeft: resolvedPadLeft, paddingRight: resolvedPadRight },
        ]}
      >
        {isFloating ? (
          <View style={[s.row, s.rowFloating]}>
            <View style={[s.rightCluster, s.rightClusterFloating, s.floatingUnifiedControls]}>
              <View style={s.floatingDateSlot}>{dateBlock}</View>
              {liveStatusPill ? <View style={s.floatingLiveSlot}>{liveStatusPill}</View> : null}
              {showModeChip ? <View style={s.floatingModeSlot}>{modeBlock}</View> : null}
              <View
                style={[
                  s.inboxWell,
                  headerKpiTileShadow,
                  Platform.OS === "web" ? s.inboxWellWeb : null,
                  s.floatingTrailingSlot,
                ]}
              >
                <CompanyInboxButton />
              </View>
              {trailing != null ? <View style={s.floatingTrailingSlot}>{trailing}</View> : null}
            </View>
          </View>
        ) : (
          <View style={s.row}>
            <View style={s.leftCluster}>
              {title ? (
                <View style={s.titleColumn}>
                  <Text
                    maxFontSizeMultiplier={chromeMaxFontMultiplier}
                    style={s.inlineTitle}
                    numberOfLines={1}
                    accessibilityRole="header"
                    {...(Platform.OS === "android" ? { includeFontPadding: false } : {})}
                  >
                    {title}
                  </Text>
                </View>
              ) : null}
              {dateBlock}
            </View>
            <View style={s.rightCluster}>
              {liveStatusPill ? <View style={s.rightLeadGap}>{liveStatusPill}</View> : null}
              {showModeChip ? (
                <View style={[liveStatusPill ? s.rightItemGap : s.rightLeadGap]}>{modeBlock}</View>
              ) : null}
              <View
                style={[
                  s.inboxWell,
                  headerKpiTileShadow,
                  Platform.OS === "web" ? s.inboxWellWeb : null,
                  s.rightItemGap,
                ]}
              >
                <CompanyInboxButton />
              </View>
              {trailing != null ? <View style={s.rightItemGap}>{trailing}</View> : null}
            </View>
          </View>
        )}
        {metaDetail === "full" ? (
          <Text maxFontSizeMultiplier={chromeMaxFontMultiplier} style={s.metaLine} numberOfLines={2}>
            {fullMetaLine}
          </Text>
        ) : null}
      </View>
    </View>
  );
}

const s = StyleSheet.create({
  stripBase: {
    alignSelf: "stretch",
    width: "100%",
  },
  /**
   * Fond + ombre + bordure en pleine largeur.
   * `paddingTop` dynamique (safe area) est appliqué en inline : le fond blanc monte sous la barre système.
   */
  strip: {
    backgroundColor: E.CARD,
    paddingBottom: 16,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: STRIP_BORDER_SLATE,
    ...stripShadowResolved,
  },
  stripFloating: {
    backgroundColor: "transparent",
    paddingBottom: 0,
    borderBottomWidth: 0,
    shadowOpacity: 0,
    elevation: 0,
    position: "relative",
    zIndex: 1,
    ...Platform.select({
      web: { boxShadow: "none" } as ViewStyle,
      default: {},
    }),
  },
  /** Même décalage vertical qu’avant sous la zone encoche (réf. ~14px `paddingTop-5t7p9m`). */
  stripInner: {
    width: "100%",
    paddingTop: 14,
  },
  stripInnerFloating: {
    paddingTop: 8,
  },
  row: {
    flexDirection: "row",
    flexWrap: "nowrap",
    alignItems: "center",
    justifyContent: "space-between",
    width: "100%",
    minHeight: HEADER_ROW_MIN_HEIGHT,
  },
  rowFloating: {
    justifyContent: "flex-start",
    width: "100%",
    minWidth: 0,
  },
  /** Bloc gauche : titre optionnel + pilule date (réf. Courses / justify-between). */
  leftCluster: {
    flexDirection: "row",
    flexWrap: "nowrap",
    alignItems: "center",
    gap: 10,
    flex: 1,
    minWidth: 0,
  },
  /** Bloc droit : chip mode + cloche + trailing. */
  rightCluster: {
    flexDirection: "row",
    flexWrap: "nowrap",
    alignItems: "center",
    gap: 0,
    flexShrink: 1,
    minWidth: 0,
  },
  rightClusterFloating: {
    marginLeft: 0,
    paddingLeft: 0,
    flex: 1,
    minWidth: 0,
    flexWrap: "nowrap",
  },
  floatingUnifiedControls: {
    flex: 1,
    minWidth: 0,
    flexDirection: "row",
    flexWrap: "nowrap",
    alignItems: "center",
    justifyContent: "flex-start",
    overflow: "hidden",
  },
  rightLeadGap: { marginLeft: 6 },
  rightItemGap: { marginLeft: 6 },
  rightItemGapFloating: { marginLeft: 12 },
  floatingDateSlot: {
    flexShrink: 1,
    flexGrow: 0,
    minWidth: 72,
    maxWidth: 148,
  },
  floatingLiveSlot: {
    flexGrow: 1,
    flexShrink: 1,
    flexBasis: 0,
    minWidth: 108,
    marginLeft: 8,
    overflow: "hidden",
    justifyContent: "center",
  },
  floatingModeSlot: {
    flexShrink: 0,
    flexGrow: 0,
    marginLeft: 12,
  },
  floatingTrailingSlot: {
    flexShrink: 0,
    flexGrow: 0,
    marginLeft: 8,
  },
  /** Aligne la cloche sur la hauteur du chip mode (minHeight 39). Jamais wrap. */
  inboxWell: {
    flexDirection: "row",
    flexShrink: 0,
    minHeight: 39,
    minWidth: 44,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: E.BORDER,
    backgroundColor: E.CARD,
    paddingHorizontal: 10,
    paddingVertical: 6,
    justifyContent: "center",
    alignItems: "center",
    overflow: "visible",
  },
  inboxWellWeb: {
    cursor: "pointer" as const,
    ...(Platform.OS === "web" ? ({ WebkitTapHighlightColor: "transparent" } as const) : {}),
  },
  /** Colonne titre : même axe vertical que la pilule date / chip (rétrécit avant la pilule). */
  titleColumn: {
    flexShrink: 1,
    minWidth: 0,
    justifyContent: "center",
  },
  inlineTitle: {
    color: E.TEXT,
    fontSize: FONT_SIZE.px16,
    fontWeight: "600" as const,
    letterSpacing: 0.15,
    lineHeight: HEADER_ROW_LINE_HEIGHT,
    flexShrink: 1,
    minWidth: 0,
  },
  /**
   * Pilule date : même enveloppe que `kpiStat` (fond carte, bordure marque, rayon 16, ombre).
   */
  datePillWrap: {
    flexDirection: "row",
    alignItems: "center",
    flexShrink: 1,
    minWidth: 0,
    minHeight: 39,
    gap: 8,
    borderWidth: 1,
    borderColor: E.BORDER,
    borderRadius: 16,
    backgroundColor: E.CARD,
    paddingHorizontal: 12,
    paddingVertical: 6,
  },
  /** Variante feuille date : minHeight 39px comme le chip mode (grandit avec Larger Text). */
  datePillWrapSheet: {
    minWidth: 0,
    minHeight: 39,
    flexShrink: 1,
    paddingHorizontal: 10,
    paddingVertical: 6,
    gap: 8,
    justifyContent: "center",
    overflow: "hidden",
  },
  datePillWrapFloating: {
    minWidth: 0,
    maxWidth: 158,
    paddingHorizontal: 9,
    gap: 6,
  },
  /** Texte dense dans la pilule date (cap chrome). */
  datePillTextSheet: {
    fontSize: FONT_SIZE.px14,
    lineHeight: 18,
    letterSpacing: 0.08,
    flexShrink: 1,
    minWidth: 0,
  },
  /** Texte principal : lisible sans extrabold. */
  datePillTextEmphasis: {
    color: E.TEXT,
    fontWeight: "600" as const,
  },
  datePillTrailingChevron: { opacity: 0.72 },
  /** Réf. `kpiIconWrap` : 28×28, rayon 8, fond vert léger. */
  headerIconWell: {
    width: 28,
    height: 28,
    borderRadius: 8,
    backgroundColor: "rgba(0, 121, 107, 0.08)",
    alignItems: "center",
    justifyContent: "center",
  },
  /** Chevron cliquable sans zone paddée supplémentaire (hitSlop gère la cible). */
  chevronHit: {
    justifyContent: "center",
    alignItems: "center",
    padding: 0,
  },
  chevDisabled: { opacity: 0.35 },
  pressed: { opacity: 0.82 },
  datePillText: {
    flexShrink: 1,
    minWidth: 0,
    fontSize: FONT_SIZE.px15,
    fontWeight: "600" as const,
    textTransform: "capitalize" as const,
    letterSpacing: 0.2,
    lineHeight: HEADER_ROW_LINE_HEIGHT,
    pointerEvents: "none",
    ...(Platform.OS === "android" ? { includeFontPadding: false } : {}),
  },
  /** Web : curseur bouton sur la pilule date cliquable. */
  datePillWeb: {
    cursor: "pointer" as const,
    ...(Platform.OS === "web" ? ({ WebkitTapHighlightColor: "transparent" } as const) : {}),
  },
  modeChipWeb: {
    cursor: "pointer" as const,
    ...(Platform.OS === "web" ? ({ WebkitTapHighlightColor: "transparent" } as const) : {}),
  },
  /** Chip mode : aligné visuellement sur les tuiles KPI (`kpiStat`). */
  modeChip: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 8,
    minHeight: 39,
    paddingVertical: 6,
    paddingHorizontal: 12,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: E.BORDER,
    backgroundColor: E.CARD,
    flexShrink: 1,
    minWidth: 0,
    maxWidth: 220,
    overflow: "hidden",
  },
  modeChipFloating: {
    maxWidth: 120,
    minWidth: 0,
    flexShrink: 1,
    paddingHorizontal: 10,
  },
  modeChipText: {
    fontSize: FONT_SIZE.px14,
    fontWeight: "500" as const,
    flexShrink: 1,
    minWidth: 0,
    letterSpacing: 0.06,
    lineHeight: 17,
    ...(Platform.OS === "android" ? { includeFontPadding: false } : {}),
  },
  modeChipTextEmphasis: {
    color: E.TEXT,
    fontWeight: "600" as const,
  },
  metaLine: {
    marginTop: 9,
    color: HEADER_SLATE_FG,
    fontSize: FONT_SIZE.px12,
    fontWeight: "500" as const,
    letterSpacing: 0.1,
    lineHeight: 16,
    ...(Platform.OS === "android" ? { includeFontPadding: false } : {}),
  },
});
