import type { ReactNode } from "react";
import { Pressable, StyleSheet, View } from "react-native";
import { AppInput } from "../../../../design/ui/AppInput";
import { AppText } from "../../../../design/ui/AppText";
import { FONT_SIZE } from "../../../../design/responsive/typographyTokens";
import { createShadow } from "../../../../styles/shadowStyles";
import {
  patchStepDetails,
  setDestinationKind,
  type CanonicalRouteDraft,
  type DestinationKind,
} from "../../utils/canonicalRouteBuilder";

type RouteClinicalDetailsProps = {
  draft: CanonicalRouteDraft;
  isMaterialDelivery: boolean;
  notesMedical: string;
  needsAssistance: boolean;
  requesterName: string;
  requesterPhone: string;
  wheelchairClient: boolean;
  wheelchairProvide: boolean;
  onChangeDraft: (draft: CanonicalRouteDraft) => void;
  onNotesMedical: (value: string) => void;
  onNeedsAssistance: (value: boolean) => void;
  onRequesterName: (value: string) => void;
  onRequesterPhone: (value: string) => void;
  onWheelchairClient: (value: boolean) => void;
  onWheelchairProvide: (value: boolean) => void;
};

const FIELD_SHELL = {
  height: 35,
  minHeight: 35,
  borderRadius: 8,
  borderColor: "#E2E8F0",
  backgroundColor: "#FFFFFF",
  paddingHorizontal: 10,
};

export function RouteClinicalDetails({
  draft,
  isMaterialDelivery,
  notesMedical,
  needsAssistance,
  requesterName,
  requesterPhone,
  wheelchairClient,
  wheelchairProvide,
  onChangeDraft,
  onNotesMedical,
  onNeedsAssistance,
  onRequesterName,
  onRequesterPhone,
  onWheelchairClient,
  onWheelchairProvide,
}: RouteClinicalDetailsProps) {
  const destinations = draft.routeSteps.flatMap((step, index) =>
    step.kind === "destination" ? [{ step, index }] : [],
  );

  return (
    <View style={s.wrap}>
      {draft.routeSteps[0] ? (
        <View style={[s.section, s.sectionFirst]}>
          <AppText variant="label" style={s.sectionTitle}>
            Départ
          </AppText>
          <DetailRow label="Accès">
            <AppInput
              value={draft.routeSteps[0].accessNotes}
              onChangeText={(value) =>
                onChangeDraft(patchStepDetails(draft, 0, { accessNotes: value }))
              }
              placeholder="Entrée, code, sonnette…"
              accessibilityLabel="Accès au départ"
              shellStyle={FIELD_SHELL}
            />
          </DetailRow>
        </View>
      ) : null}

      {destinations.map(({ step, index }, ordinal) => {
        const medical = !isMaterialDelivery && step.destinationKind === "medical";
        const needsServiceOrDoctor = medical && !step.service.trim() && !step.doctor.trim();
        return (
          <View key={`dest-${index}`} style={s.section}>
            <AppText variant="label" style={s.sectionTitle}>
              {`Destination ${ordinal + 1}`}
            </AppText>
            {!isMaterialDelivery ? (
              <View
                style={s.segment}
                accessibilityLabel={`Type de lieu, destination ${ordinal + 1}`}
              >
                <SegmentButton
                  label="Médical"
                  selected={step.destinationKind === "medical"}
                  onPress={() => onChangeDraft(setDestinationKind(draft, index, "medical"))}
                />
                <SegmentButton
                  label="Autre lieu"
                  selected={step.destinationKind === "other"}
                  onPress={() =>
                    onChangeDraft(setDestinationKind(draft, index, "other" satisfies DestinationKind))
                  }
                />
              </View>
            ) : null}
            {medical ? (
              <>
                <DetailRow label="Établissement *">
                  <AppInput
                    value={step.establishment}
                    onChangeText={(value) =>
                      onChangeDraft(patchStepDetails(draft, index, { establishment: value }))
                    }
                    placeholder="HUG, Clinique La Colline…"
                    accessibilityLabel="Établissement"
                    shellStyle={FIELD_SHELL}
                  />
                </DetailRow>
                <DetailRow label={needsServiceOrDoctor ? "Service *" : "Service"}>
                  <AppInput
                    value={step.service}
                    onChangeText={(value) =>
                      onChangeDraft(patchStepDetails(draft, index, { service: value }))
                    }
                    placeholder="Ex: Chirurgie, Urgences…"
                    accessibilityLabel="Service"
                    shellStyle={FIELD_SHELL}
                  />
                </DetailRow>
                <DetailRow label={needsServiceOrDoctor ? "Médecin *" : "Médecin"}>
                  <AppInput
                    value={step.doctor}
                    onChangeText={(value) =>
                      onChangeDraft(patchStepDetails(draft, index, { doctor: value }))
                    }
                    placeholder="Ex : Dr Dupont"
                    accessibilityLabel="Médecin"
                    shellStyle={FIELD_SHELL}
                  />
                </DetailRow>
              </>
            ) : null}
            <DetailRow label="Accès">
              <AppInput
                value={step.accessNotes}
                onChangeText={(value) =>
                  onChangeDraft(patchStepDetails(draft, index, { accessNotes: value }))
                }
                placeholder="Entrée, étage, secrétariat…"
                accessibilityLabel={`Accès destination ${ordinal + 1}`}
                shellStyle={FIELD_SHELL}
              />
            </DetailRow>
          </View>
        );
      })}

      <View style={s.section}>
        <AppText variant="label" style={s.sectionTitle}>
          Mobilité
        </AppText>
        <View style={s.segment} accessibilityLabel="Mobilité">
          <SegmentButton
            label="En chaise"
            selected={wheelchairClient}
            onPress={() => {
              if (wheelchairClient) onWheelchairClient(false);
              else {
                onWheelchairClient(true);
                onWheelchairProvide(false);
              }
            }}
          />
          <SegmentButton
            label="Fournir chaise"
            selected={wheelchairProvide}
            onPress={() => {
              if (wheelchairProvide) onWheelchairProvide(false);
              else {
                onWheelchairProvide(true);
                onWheelchairClient(false);
              }
            }}
          />
          <SegmentButton
            label="Assistance"
            selected={needsAssistance}
            onPress={() => onNeedsAssistance(!needsAssistance)}
          />
        </View>
        <AppText variant="label" style={s.notesLabel}>
          {needsAssistance ? "Notes *" : "Notes"}
        </AppText>
        <AppInput
          value={notesMedical}
          onChangeText={onNotesMedical}
          placeholder="Instructions particulières, bâtiment, étage…"
          accessibilityLabel="Notes"
          shellStyle={FIELD_SHELL}
        />
      </View>

      <View style={s.section}>
        <AppText variant="label" style={s.sectionTitle}>
          Contact
        </AppText>
        <DetailRow label="Nom">
          <AppInput
            value={requesterName}
            onChangeText={onRequesterName}
            placeholder="Nom du contact"
            accessibilityLabel="Nom du contact"
            shellStyle={FIELD_SHELL}
          />
        </DetailRow>
        <DetailRow label="Téléphone">
          <AppInput
            value={requesterPhone}
            onChangeText={onRequesterPhone}
            placeholder="Téléphone"
            accessibilityLabel="Téléphone du contact"
            keyboardType="phone-pad"
            shellStyle={FIELD_SHELL}
          />
        </DetailRow>
      </View>
    </View>
  );
}

function DetailRow({ label, children }: { label: string; children: ReactNode }) {
  return (
    <View style={s.row}>
      <AppText variant="label" style={s.rowLabel}>
        {label}
      </AppText>
      <View style={s.rowField}>{children}</View>
    </View>
  );
}

function SegmentButton({
  label,
  selected,
  onPress,
}: {
  label: string;
  selected: boolean;
  onPress: () => void;
}) {
  return (
    <Pressable
      onPress={onPress}
      style={[s.segmentBtn, selected ? s.segmentBtnOn : null]}
      accessibilityRole="button"
      accessibilityState={{ selected }}
    >
      <AppText variant="label" style={selected ? s.segmentLabelOn : s.segmentLabelOff}>
        {label}
      </AppText>
    </Pressable>
  );
}

const s = StyleSheet.create({
  wrap: { gap: 0 },
  section: {
    gap: 8,
    paddingVertical: 12,
    borderTopWidth: 1,
    borderTopColor: "#E2E8F0",
  },
  sectionFirst: {
    borderTopWidth: 0,
    paddingTop: 0,
  },
  sectionTitle: {
    marginBottom: 0,
    fontSize: FONT_SIZE.px12,
    fontWeight: "700",
    letterSpacing: 0.4,
    textTransform: "uppercase",
    color: "#64748B",
  },
  row: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    minHeight: 35,
  },
  rowLabel: {
    flexShrink: 0,
    fontSize: FONT_SIZE.px12,
    fontWeight: "600",
    color: "#334155",
  },
  rowField: { flex: 1, minWidth: 0 },
  segment: {
    flexDirection: "row",
    alignItems: "stretch",
    height: 30,
    borderRadius: 8,
    backgroundColor: "#F1F5F9",
    overflow: "hidden",
  },
  segmentBtn: {
    flex: 1,
    height: 30,
    alignItems: "center",
    justifyContent: "center",
    paddingHorizontal: 6,
    borderRadius: 8,
    backgroundColor: "transparent",
  },
  segmentBtnOn: {
    backgroundColor: "#FFFFFF",
    ...createShadow({
      shadowColor: "#0F172A",
      shadowOffset: { width: 0, height: 1 },
      shadowOpacity: 0.08,
      shadowRadius: 2,
      elevation: 1,
    }),
  },
  segmentLabelOn: {
    color: "#0F766E",
    fontSize: FONT_SIZE.px12,
    lineHeight: 16,
    fontWeight: "600",
  },
  segmentLabelOff: {
    color: "#475569",
    fontSize: FONT_SIZE.px12,
    lineHeight: 16,
    fontWeight: "600",
  },
  notesLabel: {
    marginTop: 4,
    fontSize: FONT_SIZE.px12,
    fontWeight: "700",
    letterSpacing: 0.4,
    textTransform: "uppercase",
    color: "#64748B",
  },
});
