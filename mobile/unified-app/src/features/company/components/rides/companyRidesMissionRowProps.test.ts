import { describe, expect, it } from "@jest/globals";
import type { CompanyDispatchMission } from "../../api/contracts";
import {
  COMPANY_RIDES_LIST_VIRTUALIZATION,
  areCompanyRidesMissionRowPropsEqual,
  missionListKeyExtractor,
  resolveRideCardDelayMinutes,
  type CompanyRidesMissionRowProps,
} from "./companyRidesMissionRowProps";

function mission(id: number): CompanyDispatchMission {
  return { mission_id: id, status: "assigned", client_name: `Client ${id}` };
}

const stableHandlers: Pick<
  CompanyRidesMissionRowProps,
  | "onToggleExpand"
  | "onOpenAssign"
  | "onGoDetails"
  | "onEdit"
  | "onSchedule"
  | "onTransfer"
  | "onCancel"
  | "onMarkUrgent"
> = {
  onToggleExpand: () => undefined,
  onOpenAssign: () => undefined,
  onGoDetails: () => undefined,
  onEdit: () => undefined,
  onSchedule: () => undefined,
  onTransfer: () => undefined,
  onCancel: () => undefined,
  onMarkUrgent: () => undefined,
};

function row(
  overrides: Partial<CompanyRidesMissionRowProps> & Pick<CompanyRidesMissionRowProps, "mission">
): CompanyRidesMissionRowProps {
  return {
    isExpanded: false,
    isActionPending: false,
    delayMinutes: null,
    pickupEtaIso: null,
    contextId: "company:1",
    canAssignRide: true,
    canEditRide: true,
    canTransferRide: true,
    canUrgentRide: true,
    canCancelRide: true,
    canScheduleRide: true,
    ...stableHandlers,
    ...overrides,
  };
}

describe("areCompanyRidesMissionRowPropsEqual", () => {
  it("ne considère pas les cartes sœurs quand seule l’expansion change", () => {
    const sonia = mission(45711);
    const other = mission(45710);
    const collapsedSonia = row({ mission: sonia });
    const expandedSonia = row({ mission: sonia, isExpanded: true });
    const otherBefore = row({ mission: other });
    const otherAfter = row({ mission: other });

    expect(areCompanyRidesMissionRowPropsEqual(collapsedSonia, expandedSonia)).toBe(false);
    expect(areCompanyRidesMissionRowPropsEqual(otherBefore, otherAfter)).toBe(true);
  });

  it("ne change pas une carte si la mission et l’état visuel sont stables", () => {
    const item = mission(1);
    const first = row({ mission: item });
    const second = row({ mission: item });
    expect(areCompanyRidesMissionRowPropsEqual(first, second)).toBe(true);
  });
});

describe("missionListKeyExtractor", () => {
  it("est stable par mission_id", () => {
    expect(missionListKeyExtractor(mission(45711))).toBe("45711");
  });
});

describe("COMPANY_RIDES_LIST_VIRTUALIZATION", () => {
  it("reste bornée et sans getItemLayout", () => {
    expect(COMPANY_RIDES_LIST_VIRTUALIZATION.initialNumToRender).toBeLessThanOrEqual(16);
    expect(COMPANY_RIDES_LIST_VIRTUALIZATION.windowSize).toBeLessThanOrEqual(10);
    expect(COMPANY_RIDES_LIST_VIRTUALIZATION.removeClippedSubviews).toBe(true);
  });
});

describe("resolveRideCardDelayMinutes", () => {
  it("préfère l’ETA fetchée à l’assignment", () => {
    const item = mission(12);
    item.assignment_pickup_delay_minutes = 4;
    const delays = new Map<number, number>([[12, 9]]);
    expect(resolveRideCardDelayMinutes(item, false, delays)).toBe(4);
    expect(resolveRideCardDelayMinutes(item, true, delays)).toBe(9);
  });
});
