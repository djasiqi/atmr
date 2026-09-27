import { buildMissionTimelineView } from "./companyRideDetailPresentation";
import { unwrapCompanyRideDetail } from "./companyRideDetailPayload";

describe("unwrapCompanyRideDetail", () => {
  const routeSteps = [
    { kind: "pickup", location: "A" },
    { kind: "destination", location: "B" },
    { kind: "destination", location: "C" },
    { kind: "return", location: "A" },
  ];

  it("conserve les champs mission qui sont dans summary", () => {
    const mission = unwrapCompanyRideDetail(
      {
        suggestions: [],
        summary: {
          id: "46779",
          route: { pickup_address: "B", dropoff_address: "C" },
          mission_anchor_booking_id: 46778,
          route_steps_source: "canonical",
          mission_segment_index: 2,
          mission_segment_count: 3,
          route_steps: routeSteps,
          notes_medical: "Note d'ancre",
        },
      },
      46779
    );
    expect(mission?.route).toEqual({ pickup_address: "B", dropoff_address: "C" });
    expect(mission?.mission_segment_index).toBe(2);
    expect(mission?.mission_anchor_booking_id).toBe(46778);
    const view = buildMissionTimelineView(mission as Record<string, unknown>);
    expect(view?.segmentLabel).toBe("Trajet 2 / 3");
    expect(view?.currentSegment?.fromTitle).toBe("Destination 1");
    expect(view?.currentSegment?.toTitle).toBe("Destination 2");
  });

  it("ignore une mission posée hors de summary", () => {
    const mission = unwrapCompanyRideDetail(
      {
        mission_anchor_booking_id: 46778,
        route_steps: routeSteps,
        summary: {
          id: "46779",
          route: { pickup_address: "B", dropoff_address: "C" },
        },
      },
      46779
    );
    expect(mission?.mission_anchor_booking_id).toBeUndefined();
    expect(buildMissionTimelineView(mission as Record<string, unknown>)).toBeNull();
  });
});
