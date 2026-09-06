import { useCallback, useEffect, useState, type ReactNode } from "react";
import { Linking, Modal, Pressable, ScrollView, StyleSheet, View } from "react-native";
import { semanticDanger, semanticWarning } from "../../../design/responsive/colors";
import { AppText } from "../../../design/ui/AppText";
import { FONT_SIZE } from "../../../design/responsive/typographyTokens";
import { requestNotificationDisclosure } from "../../../core/notifications/pushRegistrationState";
import { setDriverTrackingReadinessPanelVisible } from "../services/driverDisclosureOrchestrator";
import {
  openOemBatterySettings,
  requestIgnoreBatteryOptimizations,
} from "../services/batteryOptimization";
import { useDriverStatusIssues } from "../hooks/useDriverStatusIssues";
import { DriverTrackingReadinessGate } from "./DriverTrackingReadinessGate";
import {
  DRIVER_DASHBOARD_STATUS_LINE_HEIGHT,
  type DriverStatusActionKind,
  type DriverStatusIssue,
} from "./driverHubStatusModel";
import { D } from "../theme/driverDashboardTheme";

type Props = {
  trackingNeedsAttention: boolean;
  trackingOnboarded: boolean | null;
  onTrackingReadyChange: (ready: boolean) => void;
  onDismissTracking: () => void;
  /** Ligne GPS / « Localisation en cours… » — remplacée à l’identique par l’alerte. */
  idleLabel?: ReactNode;
};

export function DriverStatusArea({
  trackingNeedsAttention,
  trackingOnboarded,
  onTrackingReadyChange,
  onDismissTracking,
  idleLabel,
}: Props) {
  const { issues, view, refreshBattery } = useDriverStatusIssues({
    hideTrackingPrepDuplicates: trackingNeedsAttention,
    trackingNeedsAttention,
  });
  const [sheetOpen, setSheetOpen] = useState(false);
  const [trackingOpen, setTrackingOpen] = useState(false);

  useEffect(() => {
    setDriverTrackingReadinessPanelVisible(trackingOpen);
    return () => setDriverTrackingReadinessPanelVisible(false);
  }, [trackingOpen]);

  const runAction = useCallback(
    (kind: DriverStatusActionKind) => {
      if (kind === "disclosure") {
        requestNotificationDisclosure();
        return;
      }
      if (kind === "settings") {
        void Linking.openSettings();
        return;
      }
      if (kind === "battery") {
        void requestIgnoreBatteryOptimizations().then(() => refreshBattery());
        return;
      }
      if (kind === "oem") {
        void openOemBatterySettings();
        return;
      }
      if (kind === "tracking") {
        setTrackingOpen(true);
      }
    },
    [refreshBattery]
  );

  const onPressArea = () => {
    if (view.mode === "empty") return;
    if (view.mode === "single") {
      if (view.issue.actionKind === "tracking" || !view.issue.actionKind) {
        if (view.issue.actionKind === "tracking") {
          setTrackingOpen(true);
          return;
        }
        setSheetOpen(true);
        return;
      }
      runAction(view.issue.actionKind);
      return;
    }
    setSheetOpen(true);
  };

  return (
    <View
      style={s.slot}
      accessibilityRole="summary"
      onLayout={(event) => {
        if (!__DEV__) return;
        const { height } = event.nativeEvent.layout;
        console.log(
          `[driver-shell-layout] statusArea height=${Math.round(height)} mode=${view.mode}`
        );
      }}
    >
      {view.mode === "empty" ? (
        idleLabel ?? null
      ) : (
        <Pressable
          onPress={onPressArea}
          style={[
            s.chip,
            view.mode === "single" && view.issue.tone === "error" ? s.chipError : s.chipWarn,
          ]}
          accessibilityRole="button"
          accessibilityLabel={
            view.mode === "single" ? `${view.issue.title}. ${view.issue.message}` : view.label
          }
        >
          <AppText
            variant="caption"
            numberOfLines={1}
            style={[
              s.chipText,
              view.mode === "single" && view.issue.tone === "error" ? s.chipTextError : s.chipTextWarn,
            ]}
          >
            {view.mode === "single"
              ? `${view.issue.title}${view.issue.actionLabel ? ` · ${view.issue.actionLabel}` : ""}`
              : view.label}
          </AppText>
        </Pressable>
      )}

      <Modal visible={sheetOpen} transparent animationType="fade" onRequestClose={() => setSheetOpen(false)}>
        <View style={s.modalRoot}>
          <Pressable style={s.backdrop} onPress={() => setSheetOpen(false)} accessibilityLabel="Fermer" />
          <View style={s.sheet}>
            <AppText variant="sectionTitle" style={s.sheetTitle}>
              À vérifier
            </AppText>
            <ScrollView style={s.sheetScroll}>
              {issues.map((issue) => (
                <IssueRow key={issue.id} issue={issue} onAction={runAction} />
              ))}
            </ScrollView>
            <Pressable onPress={() => setSheetOpen(false)} style={s.sheetClose}>
              <AppText variant="label" style={s.sheetCloseLabel}>
                Fermer
              </AppText>
            </Pressable>
          </View>
        </View>
      </Modal>

      <Modal
        visible={trackingOpen}
        transparent
        animationType="fade"
        onRequestClose={() => setTrackingOpen(false)}
      >
        <View style={s.modalRoot}>
          <Pressable
            style={s.backdrop}
            onPress={() => setTrackingOpen(false)}
            accessibilityLabel="Fermer"
          />
          <View style={s.sheet}>
            <DriverTrackingReadinessGate
              mode={trackingOnboarded ? "needs_attention" : "onboarding"}
              onReadyChange={(ready) => {
                onTrackingReadyChange(ready);
                if (ready) setTrackingOpen(false);
              }}
              onDismiss={() => {
                onDismissTracking();
                setTrackingOpen(false);
              }}
            />
          </View>
        </View>
      </Modal>
    </View>
  );
}

function IssueRow({
  issue,
  onAction,
}: {
  issue: DriverStatusIssue;
  onAction: (kind: DriverStatusActionKind) => void;
}) {
  const tokens = issue.tone === "error" ? semanticDanger : semanticWarning;
  return (
    <Pressable
      onPress={() => (issue.actionKind ? onAction(issue.actionKind) : undefined)}
      disabled={!issue.actionKind}
      style={[s.issueRow, { backgroundColor: tokens.bg }]}
    >
      <AppText variant="body" style={[s.issueTitle, { color: tokens.fg }]}>
        {issue.title}
      </AppText>
      <AppText variant="caption" style={{ color: tokens.fg }}>
        {issue.message}
        {issue.actionLabel ? ` ${issue.actionLabel}` : ""}
      </AppText>
    </Pressable>
  );
}

const s = StyleSheet.create({
  slot: {
    minHeight: DRIVER_DASHBOARD_STATUS_LINE_HEIGHT,
    justifyContent: "center",
  },
  chip: {
    alignSelf: "flex-start",
    minHeight: DRIVER_DASHBOARD_STATUS_LINE_HEIGHT,
    borderRadius: 4,
    paddingHorizontal: 0,
    paddingVertical: 0,
    justifyContent: "center",
  },
  chipWarn: { backgroundColor: "transparent" },
  chipError: { backgroundColor: "transparent" },
  chipText: {
    fontWeight: "700",
    fontSize: FONT_SIZE.px10,
    lineHeight: DRIVER_DASHBOARD_STATUS_LINE_HEIGHT,
  },
  chipTextWarn: { color: semanticWarning.fg },
  chipTextError: { color: semanticDanger.fg },
  modalRoot: {
    flex: 1,
    justifyContent: "flex-end",
  },
  backdrop: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: "rgba(15, 23, 42, 0.4)",
  },
  sheet: {
    backgroundColor: D.cardBg,
    borderTopLeftRadius: 16,
    borderTopRightRadius: 16,
    paddingHorizontal: 16,
    paddingTop: 14,
    paddingBottom: 16,
    maxHeight: "72%",
  },
  sheetTitle: {
    color: D.text,
    fontWeight: "700",
    marginBottom: 10,
  },
  sheetScroll: { maxHeight: 360 },
  sheetClose: {
    marginTop: 12,
    minHeight: 44,
    alignItems: "center",
    justifyContent: "center",
  },
  sheetCloseLabel: { color: D.brand, fontWeight: "700" },
  issueRow: {
    borderRadius: 8,
    paddingHorizontal: 10,
    paddingVertical: 8,
    marginBottom: 6,
    gap: 2,
  },
  issueTitle: { fontWeight: "700" },
});
