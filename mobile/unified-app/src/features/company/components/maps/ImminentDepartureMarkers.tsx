import type { ReactElement } from "react";
import { Circle } from "react-native-maps";
import type { ImminentDeparture } from "../../dashboard/cockpit/imminentDepartures";

const RISK_COLOR: Record<ImminentDeparture["risk"], string> = {
  normal: "rgba(0, 121, 107, 0.85)",
  warning: "rgba(245, 158, 11, 0.9)",
  critical: "rgba(239, 68, 68, 0.92)",
};

type Props = {
  departures: ImminentDeparture[];
};

/** Cercles hôtes uniquement : pas de fragment ni d'enfant `null` sous MapView. */
export function buildImminentDepartureCircleElements(
  departures: ImminentDeparture[]
): ReactElement[] {
  const elements: ReactElement[] = [];
  for (const dep of departures) {
    if (dep.pickupLat == null || dep.pickupLon == null) continue;
    if (!Number.isFinite(dep.pickupLat) || !Number.isFinite(dep.pickupLon)) continue;
    const color = RISK_COLOR[dep.risk];
    elements.push(
      <Circle
        key={`imminent-${dep.missionId}`}
        center={{ latitude: dep.pickupLat, longitude: dep.pickupLon }}
        radius={dep.risk === "critical" ? 48 : 36}
        fillColor={color.replace("0.9", "0.18").replace("0.85", "0.16").replace("0.92", "0.2")}
        strokeColor={color.replace("0.92", "0.55").replace("0.9", "0.5").replace("0.85", "0.45")}
        strokeWidth={1}
        zIndex={8}
      />
    );
  }
  return elements;
}

/** Points légers pour départs imminents — pas de gros pins Google. */
export function ImminentDepartureMarkers({ departures }: Props) {
  return <>{buildImminentDepartureCircleElements(departures)}</>;
}
