import { useEffect, type ReactNode } from "react";
import "./perfReport";
import "./perfStabilizationSnapshot";
import {
  flushPerfInstrumentationAggregates,
  startPerfInstrumentationAggregates,
  stopPerfInstrumentationAggregates,
} from "./perfInstrumentation";
import { isPerfInstrumentationActive } from "./perfInstrumentationTier";
import { startJsLongTaskMonitor, stopJsLongTaskMonitor } from "./jsLongTaskMonitor";
import { startPerfMemoryMonitor, stopPerfMemoryMonitor } from "./perfMemoryMonitor";

type Props = { children: ReactNode };

export function PerfInstrumentationProvider({ children }: Props) {
  useEffect(() => {
    if (!isPerfInstrumentationActive()) return undefined;
    startPerfInstrumentationAggregates();
    startJsLongTaskMonitor();
    startPerfMemoryMonitor();
    return () => {
      flushPerfInstrumentationAggregates();
      stopPerfInstrumentationAggregates();
      stopJsLongTaskMonitor();
      stopPerfMemoryMonitor();
    };
  }, []);

  return children;
}
