import type { CompanyDispatchMission } from "../api/contracts";
import type { CompanyInboxNotification } from "../api/companyInboxApi";
import {
  buildOperationalLiveFeed,
  pickDelayedMissionForFeed,
  shouldSkipAlertLineForFeed,
  shouldSkipAlertLineForSticky,
} from "./companyDashboardLiveFeed";

function mission(partial: Partial<CompanyDispatchMission> & { mission_id: number }): CompanyDispatchMission {
  return {
    mission_id: partial.mission_id,
    status: partial.status ?? "assigned",
    scheduled_at: partial.scheduled_at ?? "2026-05-16T14:30:00+02:00",
    client_name: partial.client_name ?? "Dupont",
    pickup_label: partial.pickup_label ?? "HUG",
    dropoff_label: partial.dropoff_label ?? "Aéroport",
    driver_name: partial.driver_name ?? null,
    driver_id: partial.driver_id ?? null,
    assignment_pickup_delay_minutes: partial.assignment_pickup_delay_minutes ?? null,
  } as CompanyDispatchMission;
}

describe("companyDashboardLiveFeed", () => {
  const selectedDateIso = "2026-05-16";
  const nowMs = Date.parse("2026-05-16T15:00:00+02:00");

  it("affiche un retard avec horaire prévu, pas une heure courante", () => {
    const items = buildOperationalLiveFeed({
      missions: [
        mission({
          mission_id: 1,
          status: "assigned",
          scheduled_at: "2026-05-16T14:00:00+02:00",
          assignment_pickup_delay_minutes: 18,
        }),
      ],
      drivers: [],
      alertTexts: [],
      selectedDateIso,
      nowMs,
    });

    expect(items[0]?.kind).toBe("mission_delayed");
    expect(items[0]?.timeCaption).toBe("Prévu 14:00");
    expect(items[0]?.timeKind).toBe("scheduled");
    expect(items[0]?.message).toMatch(/Retard/);
  });

  it("marque les courses en cours comme prévues à l'horaire planifié", () => {
    const items = buildOperationalLiveFeed({
      missions: [
        mission({
          mission_id: 2,
          status: "in_progress",
          scheduled_at: "2026-05-16T16:15:00+02:00",
          driver_name: "Martin",
        }),
      ],
      drivers: [],
      alertTexts: [],
      selectedDateIso,
      nowMs,
    });

    const active = items.find((i) => i.kind === "mission_active");
    expect(active?.timeCaption).toBe("Prévu 16:15");
    expect(active?.timeKind).toBe("scheduled");
    expect(active?.message).toContain("Martin");
  });

  it("utilise À l'instant pour la disponibilité chauffeur", () => {
    const items = buildOperationalLiveFeed({
      missions: [],
      drivers: [{ driver_id: 9, driver_name: "Lea", mission_id: null, location_status: "online" } as never],
      alertTexts: [],
      selectedDateIso,
      nowMs,
    });

    const row = items.find((i) => i.kind === "driver_available");
    expect(row?.timeCaption).toBe("À l'instant");
    expect(row?.timeKind).toBe("instant");
  });

  it("intègre les notifications inbox du jour avec horodatage reçu", () => {
    const notification: CompanyInboxNotification = {
      id: 42,
      event_type: "booking.delayed",
      title: "Retard signalé",
      message: "Course #12 en retard",
      is_read: false,
      created_at: "2026-05-16T14:58:00+02:00",
    };

    const items = buildOperationalLiveFeed({
      missions: [],
      drivers: [],
      alertTexts: [],
      inboxNotifications: [notification],
      selectedDateIso,
      nowMs,
    });

    const inbox = items.find((i) => i.kind === "inbox_event");
    expect(inbox?.message).toBe("Retard signalé");
    expect(inbox?.timeCaption).toBe("Il y a 2 min");
  });

  it("ignore l'alerte synthétique retard si une ligne mission retard existe", () => {
    expect(shouldSkipAlertLineForFeed("2 courses en retard", true)).toBe(true);
    expect(shouldSkipAlertLineForFeed("Connexion temps réel instable", true)).toBe(false);
  });

  it("exclut toujours la synthèse retard du sticky", () => {
    expect(
      shouldSkipAlertLineForSticky({
        id: "delayed",
        text: "1 retard(s) signalé(s) sur le réseau.",
      })
    ).toBe(true);
    expect(
      shouldSkipAlertLineForSticky({
        id: "network",
        text: "Connexion temps réel instable",
      })
    ).toBe(false);
    expect(
      shouldSkipAlertLineForSticky({
        id: "pending_past",
        text: "Des courses en attente ont un horaire dépassé.",
      })
    ).toBe(false);
  });

  it("pickDelayedMissionForFeed choisit la course la plus ancienne en retard", () => {
    const picked = pickDelayedMissionForFeed(
      [
        mission({
          mission_id: 10,
          scheduled_at: "2026-05-16T13:00:00+02:00",
          assignment_pickup_delay_minutes: 5,
        }),
        mission({
          mission_id: 11,
          scheduled_at: "2026-05-16T12:00:00+02:00",
          assignment_pickup_delay_minutes: 8,
        }),
      ],
      nowMs
    );
    expect(picked?.mission_id).toBe(11);
  });
});
