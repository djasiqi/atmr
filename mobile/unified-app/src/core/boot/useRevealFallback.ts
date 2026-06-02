import { useCallback, useRef } from "react";
import {
  reportBootFallback,
  type BootFallbackName,
} from "../observability/bootDiagnostics";

export function useRevealFallback(args: {
  enabled: boolean;
  timeoutMs: number;
  name: BootFallbackName;
  reveal: () => void;
  extra?: Record<string, unknown>;
}): {
  arm: () => void;
  settled: (finished: boolean) => void;
  disarm: () => void;
} {
  const { enabled, timeoutMs, name, reveal, extra } = args;
  const revealRef = useRef(reveal);
  revealRef.current = reveal;
  const extraRef = useRef(extra);
  extraRef.current = extra;

  const animationCompletedRef = useRef(false);
  const fallbackTriggeredRef = useRef(false);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const clearTimer = useCallback(() => {
    if (timerRef.current != null) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const disarm = useCallback(() => {
    clearTimer();
    animationCompletedRef.current = false;
    fallbackTriggeredRef.current = false;
  }, [clearTimer]);

  const arm = useCallback(() => {
    if (!enabled) {
      return;
    }
    disarm();
    timerRef.current = setTimeout(() => {
      if (animationCompletedRef.current || fallbackTriggeredRef.current) {
        return;
      }
      fallbackTriggeredRef.current = true;
      revealRef.current();
      reportBootFallback(name, extraRef.current);
    }, timeoutMs);
  }, [disarm, enabled, name, timeoutMs]);

  const settled = useCallback(
    (finished: boolean) => {
      if (finished) {
        animationCompletedRef.current = true;
        clearTimer();
      }
    },
    [clearTimer],
  );

  return { arm, settled, disarm };
}
