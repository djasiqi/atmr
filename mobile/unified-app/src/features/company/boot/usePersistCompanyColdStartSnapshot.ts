import { useEffect, useRef } from "react";
import type {
  CompanyDispatchMissionListResponse,
  CompanyDispatchRealtimeDashboard,
  CompanyDriverLiveLocation,
  CompanyDriverLiveLocationResponse,
} from "../api/contracts";
import type { CompanyDispatchStatusResponse } from "../api/companyApi";
import { persistCompanyColdStartSnapshot } from "./companyColdStartSnapshot";

const PERSIST_DEBOUNCE_MS = 750;

type PersistArgs = {
  contextId: string | null;
  date: string;
  missions?: CompanyDispatchMissionListResponse | null;
  dashboard?: CompanyDispatchRealtimeDashboard | null;
  drivers?: CompanyDriverLiveLocation[];
  driversRefreshedAt?: string | null;
  dispatchStatus?: CompanyDispatchStatusResponse | null;
};

export function usePersistCompanyColdStartSnapshot(args: PersistArgs): void {
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (!args.contextId) return;
    const hasSlice = Boolean(args.missions || args.dashboard || args.drivers?.length || args.dispatchStatus);
    if (!hasSlice) return;

    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => {
      const roster: CompanyDriverLiveLocationResponse | null =
        args.drivers && args.drivers.length > 0
          ? {
              context_id: args.contextId as string,
              locations: args.drivers,
              refreshed_at: args.driversRefreshedAt ?? new Date().toISOString(),
            }
          : null;
      void persistCompanyColdStartSnapshot({
        contextId: args.contextId as string,
        date: args.date,
        missions: args.missions,
        dashboard: args.dashboard,
        roster,
        dispatchStatus: args.dispatchStatus,
      }).catch(() => undefined);
    }, PERSIST_DEBOUNCE_MS);

    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [
    args.contextId,
    args.date,
    args.dashboard,
    args.dispatchStatus,
    args.drivers,
    args.driversRefreshedAt,
    args.missions,
  ]);
}
