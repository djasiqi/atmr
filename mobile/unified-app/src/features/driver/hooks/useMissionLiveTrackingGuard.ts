import { useEffect, useState } from "react";

import {
  cancelMissionLiveTrackingDisclosure,
  continueMissionLiveTrackingDisclosure,
  guardMissionLiveTransition,
  getMissionLiveTrackingDisclosureSnapshot,
  openMissionLiveTrackingSettings,
  subscribeMissionLiveTrackingDisclosure,
} from "../services/missionLiveTrackingDisclosureBridge";

/**
 * Garde transition mission live — modale centralisée dans le layout (P2).
 */
export function useMissionLiveTrackingGuard() {
  const [snapshot, setSnapshot] = useState(getMissionLiveTrackingDisclosureSnapshot());

  useEffect(() => {
    return subscribeMissionLiveTrackingDisclosure(() => {
      setSnapshot(getMissionLiveTrackingDisclosureSnapshot());
    });
  }, []);

  return {
    guardTransition: guardMissionLiveTransition,
    disclosureVisible: snapshot.visible,
    disclosurePending: snapshot.pending,
    showOpenSettings: snapshot.showOpenSettings,
    onDisclosureCancel: cancelMissionLiveTrackingDisclosure,
    onDisclosureContinue: continueMissionLiveTrackingDisclosure,
    onDisclosureOpenSettings: openMissionLiveTrackingSettings,
  };
}
