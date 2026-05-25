import { useMemo } from "react";
import { Platform } from "react-native";
import { useAccessibilityScale } from "./useAccessibilityScale";
import type { AppViewport } from "./useAppViewport";
import { useAppViewport } from "./useAppViewport";

const CTA_BUTTON_MIN_LANDING = 54;
/** Aligné sur les lignes d’autosuggestion après hausse de la barre d’adresse (~50px). */
const SUGGESTION_ROW_HEIGHT = 46;

/** Web : ligne champ compacte (optique ~30px zone utile + bordure) vs ~38px avant. */
const FIELD_SHELL_MIN_WEB = 32;
/** Web : boutons formulaires alignés visuellement sur les champs (≥30, confort clic). */
const FORM_BUTTON_MIN_WEB = 36;

/** Borne l’amplification des mesures en px (espacements, rayons) tout en laissant les Text scaler via maxFontSizeMultiplier. */
const EFFECTIVE_FONT_SCALE_CAP = 1.25;

type FieldMetrics = {
  fieldShellMinHeight: number;
  fieldTextInputMinHeight: number;
  fieldTextInputPaddingV: number;
  formButtonMinHeight: number;
};

/** Hauteurs champs / boutons formulaires — isolé pour éviter toute référence orpheline au build. */
export function computeFieldMetrics(
  os: typeof Platform.OS,
  minTouchHeight: number,
  isLargeText: boolean,
  fontScale: number
): FieldMetrics {
  const fieldShellMinHeight =
    os === "web"
      ? isLargeText
        ? Math.max(40, Math.round(FIELD_SHELL_MIN_WEB * Math.min(fontScale, 1.25)))
        : FIELD_SHELL_MIN_WEB
      : os === "ios"
        ? Math.max(minTouchHeight - 10, 40)
        : Math.max(minTouchHeight - 6, 40);

  const fieldTextInputMinHeight =
    os === "web" ? Math.max(28, fieldShellMinHeight - 2) : fieldShellMinHeight - 2;

  const fieldTextInputPaddingV = os === "web" ? 5 : 10;

  const formButtonMinHeight =
    os === "web"
      ? isLargeText
        ? Math.max(40, Math.round(FORM_BUTTON_MIN_WEB * Math.min(fontScale, 1.2)))
        : FORM_BUTTON_MIN_WEB
      : minTouchHeight;

  return {
    fieldShellMinHeight,
    fieldTextInputMinHeight,
    fieldTextInputPaddingV,
    formButtonMinHeight,
  };
}

export type PublicLandingTokens = {
  horizontalPadding: number;
  columnPadding: number;
  contentMaxWidth: number;
  topPadding: number;
  bottomPadding: number;
  logoHeight: number;
  logoWidth: number;
  titleFontSize: number;
  titleLineHeight: number;
  titleMaxWidth: number;
  titleMarginTop: number;
  cardMarginTop: number;
  cardMaxWidth: number;
  cardPadding: number;
  cardRadius: number;
  cardLabelSize: number;
  cardLabelOpacity: number;
  cardValueSize: number;
  cardValueWeight: "400" | "500" | "600" | "700";
  cardLineGap: number;
  cardBlockGap: number;
  ctaHeight: number;
  ctaRadius: number;
  ctaFontSize: number;
  microProofFontSize: number;
  microProofLineHeight: number;
  secondaryFontSize: number;
  hintFontSize: number;
  hintLineHeight: number;
  spaceCardToCta: number;
  spaceCtaToProof: number;
  spaceProofToSecondary: number;
  stackSecondaryLinks: boolean;
  suggestionListMaxHeight: number;
  flexSpacerMinHeight: number;
};

export type ResponsiveTokens = {
  /** Échelle police système brute (PixelRatio.getFontScale). */
  fontScale: number;
  effectiveFontScale: number;
  /** Padding additionnel bas pour ScrollView quand la police système est grande. */
  scrollExtraBottomPadding: number;
  /** minHeight touches — boutons / rangées sensibles. */
  minTouchHeight: number;
  /** Hauteur min. conteneur champ (bordure) — `AppInput` / rangées type login. */
  fieldShellMinHeight: number;
  /** Hauteur min. `TextInput` interne — cohérent avec `fieldShellMinHeight`. */
  fieldTextInputMinHeight: number;
  /** Padding vertical interne du `TextInput` dans `AppInput`. */
  fieldTextInputPaddingV: number;
  /** Hauteur min. boutons formulaires (`AppButton`) — web aligné champs. */
  formButtonMinHeight: number;
  spacingXs: number;
  spacingSm: number;
  spacingMd: number;
  spacingLg: number;
  radiusSm: number;
  radiusMd: number;
  bodyLineHeightRatio: number;
  headingLineHeightRatio: number;
  /** Espacement vertical typique entre blocs principaux d’une page. */
  pageGap: number;
  /** Espacement entre sections (titres de bloc, groupes de champs). */
  sectionGap: number;
  /** Espacement label / champ / message d’aide / erreur. */
  fieldGap: number;
  /** Padding interne des cartes (aligné sur la logique `landing.cardPadding`). */
  cardPadding: number;
  /** Taille corps / inputs formulaires — indépendante de `landing.cardValueSize`. */
  bodyFontSize: number;
  /** Taille libellé boutons primaires/secondaires — indépendante de `landing.ctaFontSize`. */
  buttonFontSize: number;
  /** Rayon grand (cartes modales, sheets). */
  radiusLg: number;
  /** Padding bas minimum quand le clavier ouvre une ScrollView publique. */
  keyboardScrollPaddingMin: number;
  /** Marge ajoutée à la hauteur clavier pour le padding bas scroll. */
  keyboardScrollPaddingExtra: number;
  /** Hauteur max d'une liste déroulante / picker inline (autocomplete, pays, etc.). */
  dropdownListMaxHeight: number;
  /** Ratio max d'une bottom sheet / modale par rapport à `usableHeight`. */
  modalSheetMaxHeightRatio: number;
  /** Plafond absolu d'une bottom sheet en px. */
  modalSheetMaxHeightCap: number;
  landing: PublicLandingTokens;
};

/** Exposé pour tests unitaires (landing + seuils a11y). */
export function computePublicLanding(
  viewport: AppViewport,
  fontScale: number,
  isLargeText: boolean,
  isVeryLargeText: boolean,
  effectiveFontScale: number
): PublicLandingTokens {
  const { width, usableHeight, shortest, topInset, bottomInset, isTiny, isCompact, isTablet, contentWidth } =
    viewport;

  const isWideWeb = Platform.OS === "web" && width >= 1100;
  const isComfortableWeb = Platform.OS === "web" && width >= 900;
  const isShortViewport = usableHeight < 640;

  const columnPadding = isTiny ? 8 : isCompact ? 10 : 12;

  const cardTargetMax = isTablet
    ? isWideWeb
      ? 540
      : 500
    : isTiny
      ? 340
      : isCompact
        ? 380
        : 440;

  /** Pas de Math.max(288, …) : la colonne suit contentWidth (borne écran). */
  const contentMaxWidth = Math.min(cardTargetMax, contentWidth);

  const titleFontSize = isTablet
    ? isWideWeb
      ? 56
      : 52
    : isTiny
      ? 28
      : isCompact
        ? 32
        : isShortViewport
          ? 34
          : 42;

  const titleLineHeight = Math.round(
    titleFontSize * (isTablet ? 1.08 : isLargeText ? 1.22 : 1.12)
  );

  const logoScale = isTablet ? 1.15 : isTiny ? 0.82 : isCompact ? 0.9 : 1;
  const baseLogoHeight = isShortViewport ? 20 : isCompact ? 22 : 26;
  const baseLogoWidth = isShortViewport ? 136 : isCompact ? 148 : 168;

  const decorativeTopBoost = isLargeText ? 4 * (effectiveFontScale - 1) : 0;

  return {
    horizontalPadding: viewport.horizontalPadding,
    columnPadding,
    contentMaxWidth,
    topPadding: Math.max(
      topInset + (isShortViewport ? 10 : isCompact ? 14 : 18) + decorativeTopBoost,
      isTablet ? 56 : isTiny ? 28 : 40
    ),
    bottomPadding: Math.max(
      bottomInset + (isShortViewport ? 14 : 20) + (fontScale > 1 ? 8 * (fontScale - 1) : 0),
      isTiny ? 16 : 28
    ),
    logoHeight: Math.round(baseLogoHeight * logoScale),
    logoWidth: Math.round(baseLogoWidth * logoScale),
    titleFontSize,
    titleLineHeight,
    titleMaxWidth: isTablet ? Math.min(440, contentMaxWidth) : Math.min(320, contentMaxWidth),
    titleMarginTop: isShortViewport ? 12 : isCompact ? 16 : isTablet ? 28 : 24,
    cardMarginTop: isShortViewport ? 14 : isCompact ? 18 : isTablet ? 32 : 26,
    cardMaxWidth: contentMaxWidth,
    cardPadding: isTablet ? 24 : isTiny ? 14 : isCompact ? 20 : 24,
    cardRadius: isTablet ? 26 : isTiny ? 20 : 26,
    cardLabelSize: isTiny ? 12 : isCompact ? 12.5 : 13,
    cardLabelOpacity: 1,
    cardValueSize: isTiny ? 15 : isCompact ? 16 : isComfortableWeb ? 17 : 16,
    cardValueWeight: "500",
    cardLineGap: isTiny ? 5 : 6,
    cardBlockGap: isTiny ? 10 : isCompact ? 11 : 12,
    ctaHeight: Math.max(CTA_BUTTON_MIN_LANDING, Math.round(54 * Math.min(fontScale, 1.35))),
    ctaRadius: 14,
    ctaFontSize: isTiny ? 12 : isCompact ? 12.5 : isTablet ? 14 : 13,
    microProofFontSize: isTiny ? 12 : isCompact ? 12.5 : isTablet ? 14 : 13,
    microProofLineHeight: isTiny ? 16 : 16,
    secondaryFontSize: isTiny ? 12 : isCompact ? 12.5 : isTablet ? 14 : 13,
    hintFontSize: isTiny ? 12 : 13,
    hintLineHeight: 16,
    spaceCardToCta: isShortViewport
      ? 16
      : isCompact
        ? 22
        : isTablet
          ? 36
          : 28 - (isLargeText ? 2 : 0),
    spaceCtaToProof: isTiny ? 8 : isCompact ? 10 : 14,
    spaceProofToSecondary: isTiny ? 10 : isCompact ? 12 : 18,
    stackSecondaryLinks: isTiny || width < 380 || isVeryLargeText,
    suggestionListMaxHeight: SUGGESTION_ROW_HEIGHT * (isTablet ? 6 : 5) + 4,
    flexSpacerMinHeight: isShortViewport ? 12 : isLargeText ? 16 : 20,
  };
}

export function useResponsiveTokens(): ResponsiveTokens {
  const viewport = useAppViewport();
  const { fontScale, isLargeText, isVeryLargeText } = useAccessibilityScale();

  return useMemo(() => {
    const effectiveFontScale = Math.min(fontScale, EFFECTIVE_FONT_SCALE_CAP);
    const scrollExtraBottomPadding =
      viewport.bottomInset + 12 + (fontScale > 1 ? 8 * (fontScale - 1) : 0);

    const landing = computePublicLanding(
      viewport,
      fontScale,
      isLargeText,
      isVeryLargeText,
      effectiveFontScale
    );

    const baseTouch = 44;
    const minTouchHeight = Math.round(baseTouch * (isLargeText ? Math.min(fontScale, 1.35) : 1));

    const {
      fieldShellMinHeight,
      fieldTextInputMinHeight,
      fieldTextInputPaddingV,
      formButtonMinHeight,
    } = computeFieldMetrics(Platform.OS, minTouchHeight, isLargeText, fontScale);

    const spacingXs = Math.round(4 * effectiveFontScale);
    const spacingSm = Math.round(8 * effectiveFontScale);
    const spacingMd = Math.round(16 * effectiveFontScale);
    const spacingLg = Math.round(24 * effectiveFontScale);
    const radiusSm = Math.round(8 * effectiveFontScale);
    const radiusMd = Math.round(12 * effectiveFontScale);
    const radiusLg = Math.round(16 * effectiveFontScale);

    const { isTiny, isCompact, isTablet } = viewport;
    const bodyFontSize = isTablet ? 17 : isTiny ? 14 : isCompact ? 15 : 16;
    const buttonFontSize = isTablet ? 14 : isTiny ? 12 : isCompact ? 12.5 : 13;

    const bodyLineHeightRatio = isLargeText ? 1.3 : 1.25;
    const headingLineHeightRatio = isLargeText ? 1.28 : 1.22;

    const pageGap = spacingLg;
    const sectionGap = spacingMd;
    const fieldGap = spacingSm;
    const cardPadding = landing.cardPadding;

    const keyboardScrollPaddingMin = Math.round(260 * (isLargeText ? 1.1 : 1));
    const keyboardScrollPaddingExtra = Math.round(48 * (isLargeText ? 1.05 : 1));
    const dropdownListMaxHeight = Math.min(
      Math.round(viewport.usableHeight * (viewport.isTablet ? 0.45 : 0.4)),
      viewport.isTablet ? 360 : 280
    );
    const modalSheetMaxHeightRatio = viewport.isTablet ? 0.85 : 0.88;
    const modalSheetMaxHeightCap = viewport.isTablet ? 720 : 620;

    return {
      fontScale,
      effectiveFontScale,
      scrollExtraBottomPadding,
      minTouchHeight,
      fieldShellMinHeight,
      fieldTextInputMinHeight,
      fieldTextInputPaddingV,
      formButtonMinHeight,
      spacingXs,
      spacingSm,
      spacingMd,
      spacingLg,
      radiusSm,
      radiusMd,
      radiusLg,
      bodyLineHeightRatio,
      headingLineHeightRatio,
      pageGap,
      sectionGap,
      fieldGap,
      cardPadding,
      bodyFontSize,
      buttonFontSize,
      keyboardScrollPaddingMin,
      keyboardScrollPaddingExtra,
      dropdownListMaxHeight,
      modalSheetMaxHeightRatio,
      modalSheetMaxHeightCap,
      landing,
    };
  }, [fontScale, isLargeText, isVeryLargeText, viewport]);
}
