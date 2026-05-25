import type { DriverMission } from "../types";
import { extractMissionMapCoordInput } from "../domain/missionMapCoordUtils";
import { GoogleMapsMissionRoute } from "./maps/GoogleMapsMissionRoute.web";

type Props = {
  mission: DriverMission;
  height?: number;
  showRecenterControl?: boolean;
  etaMinutes?: number | null;
};

export function MissionMap({ mission, height, etaMinutes }: Props) {
  const mapInput = extractMissionMapCoordInput(mission);
  return (
    <GoogleMapsMissionRoute
      {...mapInput}
      missionStatus={mission.status}
      height={height}
      etaMinutes={etaMinutes}
    />
  );
}
