import { useEffect, useMemo, useState } from "react";
import { ActivityIndicator, StyleSheet, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { Modal, AppButton } from "../../../../design/responsive";
import { AppText } from "../../../../design/ui/AppText";
import type { InstitutionRequestOffer } from "../../api/institutionOffersApi";
import { fetchOfferTravelEstimate } from "../../api/institutionOffersApi";
import { TimeDatePicker } from "../rides/TimeDatePicker";
import { E } from "../../theme/enterpriseOpsTheme";
import { buildInstitutionScheduleLabel } from "../../utils/institutionOfferDisplay";
import {
  computeDefaultProposedDate,
  formatOutboundRouteLabel,
  formatProposedPickupIso,
  isoFromDatetimeLocalValue,
} from "../../utils/institutionOfferProposeTime";

type PlanOfferTimeModalProps = {
  visible: boolean;
  offer: InstitutionRequestOffer | null | undefined;
  pending?: boolean;
  onClose: () => void;
  onConfirm: (offerId: number, proposedPickupIso: string) => void;
};

/** Planifier = acceptation avec définition du pickup (pas de validation institution). */
export function PlanOfferTimeModal({
  visible,
  offer,
  pending = false,
  onClose,
  onConfirm,
}: PlanOfferTimeModalProps) {
  const req = offer?.transport_request;
  const [travelMinutes, setTravelMinutes] = useState<number | null>(null);
  const [travelLoading, setTravelLoading] = useState(false);
  const [proposedLocal, setProposedLocal] = useState("");
  const [openPickerSignal, setOpenPickerSignal] = useState(0);

  useEffect(() => {
    if (!visible || !offer?.id) return;
    let cancelled = false;
    setTravelLoading(true);
    setTravelMinutes(null);
    setProposedLocal("");
    setOpenPickerSignal(0);

    const applyDefault = (minutes: number | null) => {
      const defaultDate = computeDefaultProposedDate(req, minutes);
      setProposedLocal(defaultDate ? formatProposedPickupIso(defaultDate) : "");
      setOpenPickerSignal((n) => n + 1);
    };

    void fetchOfferTravelEstimate(offer.id)
      .then((payload) => {
        if (cancelled) return;
        const minutes =
          payload.travel_minutes != null && Number.isFinite(payload.travel_minutes)
            ? Number(payload.travel_minutes)
            : null;
        setTravelMinutes(minutes);
        applyDefault(minutes);
      })
      .catch(() => {
        if (cancelled) return;
        setTravelMinutes(null);
        applyDefault(null);
      })
      .finally(() => {
        if (!cancelled) setTravelLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [visible, offer?.id, req]);

  const scheduleLabel = buildInstitutionScheduleLabel(req);
  const routeLabel = formatOutboundRouteLabel(req);
  const fixedDateIso = useMemo(() => {
    const missionDate = req?.mission_date?.trim();
    if (missionDate && /^\d{4}-\d{2}-\d{2}$/.test(missionDate)) {
      return `${missionDate}T12:00:00`;
    }
    const scheduled = req?.scheduled_time?.trim();
    if (scheduled && scheduled.length >= 10) {
      return `${scheduled.slice(0, 10)}T12:00:00`;
    }
    return undefined;
  }, [req?.mission_date, req?.scheduled_time]);
  const travelLabel = travelLoading
    ? "calcul…"
    : travelMinutes != null
      ? `~${travelMinutes} min`
      : "non disponible";

  const handleSubmit = () => {
    if (!offer?.id || !proposedLocal.trim()) return;
    const iso =
      isoFromDatetimeLocalValue(proposedLocal) ??
      (proposedLocal.length >= 19 ? proposedLocal : null);
    if (!iso) return;
    onConfirm(offer.id, iso);
  };

  return (
    <Modal
      visible={visible}
      title="Planifier la prise en charge"
      subtitle="Heure opérationnelle de prise en charge"
      onClose={onClose}
      presentation="bottomSheet"
      sheetBodyMaxHeightRatio={0.72}
      footer={
        <View style={s.footerRow}>
          <AppButton
            title="Annuler"
            variant="secondary"
            onPress={onClose}
            disabled={pending}
            style={{ ...s.footerBtn, ...s.footerBtnSecondary }}
          />
          <AppButton
            title={pending ? "Envoi…" : "Confirmer"}
            variant="primary"
            onPress={handleSubmit}
            disabled={pending || !proposedLocal.trim() || travelLoading}
            loading={pending}
            style={s.footerBtn}
            leftIcon={
              <Ionicons
                name="checkmark-circle-outline"
                size={20}
                color={
                  pending || !proposedLocal.trim() || travelLoading
                    ? "rgba(255, 255, 255, 0.85)"
                    : "#fff"
                }
              />
            }
          />
        </View>
      }
    >
      <View style={s.body}>
        <View style={s.summary}>
          <AppText variant="body" style={s.summaryTitle}>
            {req?.institution_name ?? "Institution"}
          </AppText>
          <AppText variant="bodyMuted" style={s.summaryLine}>
            {`Horaire demandé : ${scheduleLabel}`}
          </AppText>
          <AppText variant="bodyMuted" style={s.summaryLine}>
            {routeLabel}
          </AppText>
          <AppText variant="bodyMuted" style={s.summaryLine}>
            {`Trajet estimé : ${travelLabel}`}
          </AppText>
        </View>

        {travelLoading && !proposedLocal ? (
          <ActivityIndicator color={E.BRAND} style={{ marginVertical: 12 }} />
        ) : (
          <TimeDatePicker
            value={proposedLocal}
            onChange={setProposedLocal}
            timeOnly
            emptyPreviewReferenceIso={fixedDateIso}
            label="Horaire proposé"
            modalTitle="Heure"
            emptyLabel="Choisir un horaire"
            accessibilityLabel="Choisir l'heure de prise en charge"
            openEditorSignal={openPickerSignal}
          />
        )}
      </View>
    </Modal>
  );
}

const s = StyleSheet.create({
  body: { gap: 12 },
  summary: {
    backgroundColor: "rgba(0, 121, 107, 0.06)",
    borderRadius: 10,
    padding: 12,
    gap: 4,
  },
  summaryTitle: { color: E.TEXT, fontWeight: "600" },
  summaryLine: { fontSize: 13, lineHeight: 18 },
  footerRow: {
    flexDirection: "row",
    gap: 10,
  },
  footerBtn: {
    flex: 1,
    minHeight: 48,
    borderRadius: 12,
  },
  footerBtnSecondary: {
    borderColor: "rgba(0, 121, 107, 0.28)",
    backgroundColor: "#fff",
  },
});

/** @deprecated Utiliser PlanOfferTimeModal */
export const ProposeOfferTimeModal = PlanOfferTimeModal;
