import { useCallback, useEffect, useState } from "react";
import { Alert, View } from "react-native";
import { useActiveCompanyContextId } from "../../hooks";
import { getDispatchApiErrorMessage, scheduleCompanyRide } from "../../api/companyApi";
import { normalizeScheduledTimeIso } from "../../useRideForms";
import { scheduledTimeToFormNaiveIso } from "../../utils/companyDateUtils";
import { TimeDatePicker } from "./TimeDatePicker";

type RideScheduleModalProps = {
  visible: boolean;
  missionId: number | null;
  initialScheduledAt?: string | null;
  onClose: () => void;
  onSaved?: () => void;
};

export function RideScheduleModal({
  visible,
  missionId,
  initialScheduledAt,
  onClose,
  onSaved,
}: RideScheduleModalProps) {
  const contextId = useActiveCompanyContextId();
  const [scheduledAt, setScheduledAt] = useState("");
  const [openEditorSignal, setOpenEditorSignal] = useState(0);

  useEffect(() => {
    if (!visible || missionId == null) return;
    setScheduledAt(scheduledTimeToFormNaiveIso(initialScheduledAt ?? ""));
    setOpenEditorSignal((n) => n + 1);
  }, [visible, missionId, initialScheduledAt]);

  const submitSchedule = useCallback(
    async (finalValue: string) => {
      if (!contextId || missionId == null) {
        onClose();
        return;
      }
      const pickupAt = normalizeScheduledTimeIso(finalValue);
      if (!pickupAt) {
        Alert.alert("Planification", "Choisissez une date et une heure de départ.");
        setOpenEditorSignal((n) => n + 1);
        return;
      }
      try {
        await scheduleCompanyRide({
          contextId,
          missionId,
          payload: {
            pickup_at: pickupAt,
            timezone: "Europe/Zurich",
            note: "Planification depuis liste courses",
            force_recompute: true,
          },
        });
        onSaved?.();
        onClose();
      } catch (e) {
        Alert.alert("Planification", getDispatchApiErrorMessage(e, "Planification impossible."));
        setOpenEditorSignal((n) => n + 1);
      }
    },
    [contextId, missionId, onClose, onSaved]
  );

  if (!visible || missionId == null) return null;

  return (
    <View pointerEvents="box-none" style={{ position: "absolute", width: 0, height: 0, overflow: "hidden" }}>
      <TimeDatePicker
        standaloneEditor
        openEditorSignal={openEditorSignal}
        value={scheduledAt}
        onChange={setScheduledAt}
        label=""
        modalTitle="Date & heure de départ"
        emptyLabel="Non défini"
        required={false}
        onEditorConfirm={(value) => void submitSchedule(value)}
        onEditorDismiss={onClose}
      />
    </View>
  );
}
