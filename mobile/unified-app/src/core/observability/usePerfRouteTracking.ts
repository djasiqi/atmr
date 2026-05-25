import { useEffect } from "react";
import { usePathname } from "expo-router";
import { setPerfRole, setPerfScreen, type PerfRole } from "./perfActiveContext";

export function usePerfRouteTracking(role: PerfRole): void {
  const pathname = usePathname();
  useEffect(() => {
    setPerfRole(role);
    setPerfScreen(pathname && pathname.length > 0 ? pathname : `${role}.root`);
  }, [pathname, role]);
}
