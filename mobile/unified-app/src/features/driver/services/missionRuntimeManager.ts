import { DriverSocketEvent, DriverTransitionStatus } from "../types";

type TransitionEntry = {
  targetStatus: DriverTransitionStatus;
  startedAt: number;
};

type MissionVersion = {
  updatedAtMs: number;
  eventSequence: number;
};

type ShouldApplyResult = {
  apply: boolean;
  reason: "ok" | "duplicate_event_id" | "sequence_old" | "updated_at_old";
  gapDetected: boolean;
};

class MissionRuntimeManager {
  private readonly inFlightTransitions = new Map<number, TransitionEntry>();
  private readonly missionVersions = new Map<number, MissionVersion>();
  private readonly seenEventIds: string[] = [];
  private readonly seenEventIdSet = new Set<string>();
  private readonly maxSeenEventIds = 3000;

  private parseIso(value: string | null | undefined): number {
    if (!value) return Number.NaN;
    const parsed = Date.parse(value);
    return Number.isFinite(parsed) ? parsed : Number.NaN;
  }

  private rememberEventId(eventId: string) {
    if (this.seenEventIdSet.has(eventId)) return;
    this.seenEventIdSet.add(eventId);
    this.seenEventIds.push(eventId);
    if (this.seenEventIds.length > this.maxSeenEventIds) {
      const removed = this.seenEventIds.shift();
      if (removed) {
        this.seenEventIdSet.delete(removed);
      }
    }
  }

  beginTransition(missionId: number, targetStatus: DriverTransitionStatus): boolean {
    if (this.inFlightTransitions.has(missionId)) {
      return false;
    }
    this.inFlightTransitions.set(missionId, { targetStatus, startedAt: Date.now() });
    return true;
  }

  completeTransition(missionId: number) {
    this.inFlightTransitions.delete(missionId);
  }

  hasTransitionInFlight(missionId: number): boolean {
    return this.inFlightTransitions.has(missionId);
  }

  registerSnapshot(missionId: number, updatedAt: string | null | undefined) {
    const updatedAtMs = this.parseIso(updatedAt);
    if (!Number.isFinite(updatedAtMs)) return;
    const current = this.missionVersions.get(missionId);
    this.missionVersions.set(missionId, {
      updatedAtMs: Math.max(current?.updatedAtMs ?? updatedAtMs, updatedAtMs),
      eventSequence: current?.eventSequence ?? -1,
    });
  }

  shouldApplyRealtimeEvent(event: DriverSocketEvent): ShouldApplyResult {
    if (typeof event.event_id === "string" && this.seenEventIdSet.has(event.event_id)) {
      return { apply: false, reason: "duplicate_event_id", gapDetected: false };
    }
    const current = this.missionVersions.get(event.mission_id);
    const eventSequence = typeof event.event_sequence === "number" ? event.event_sequence : null;
    const eventUpdatedAtMs = this.parseIso(event.updated_at);
    const sequenceOld = eventSequence !== null && current && eventSequence <= current.eventSequence;
    if (sequenceOld) {
      return { apply: false, reason: "sequence_old", gapDetected: false };
    }
    if (current && Number.isFinite(eventUpdatedAtMs) && eventUpdatedAtMs < current.updatedAtMs) {
      return { apply: false, reason: "updated_at_old", gapDetected: false };
    }
    const gapDetected =
      Boolean(current) &&
      eventSequence !== null &&
      current.eventSequence >= 0 &&
      eventSequence > current.eventSequence + 1;
    return { apply: true, reason: "ok", gapDetected };
  }

  registerRealtimeEvent(event: DriverSocketEvent) {
    if (typeof event.event_id === "string") {
      this.rememberEventId(event.event_id);
    }
    const current = this.missionVersions.get(event.mission_id);
    const nextSequence =
      typeof event.event_sequence === "number"
        ? Math.max(current?.eventSequence ?? -1, event.event_sequence)
        : current?.eventSequence ?? -1;
    const nextUpdatedAt = this.parseIso(event.updated_at);
    this.missionVersions.set(event.mission_id, {
      eventSequence: nextSequence,
      updatedAtMs: Number.isFinite(nextUpdatedAt)
        ? Math.max(current?.updatedAtMs ?? nextUpdatedAt, nextUpdatedAt)
        : current?.updatedAtMs ?? Number.NaN,
    });
  }

  snapshot() {
    return {
      inFlightTransitions: this.inFlightTransitions.size,
      knownMissions: this.missionVersions.size,
      seenEventIds: this.seenEventIds.length,
    };
  }

  resetForTests() {
    this.inFlightTransitions.clear();
    this.missionVersions.clear();
    this.seenEventIds.length = 0;
    this.seenEventIdSet.clear();
  }
}

export const missionRuntimeManager = new MissionRuntimeManager();
