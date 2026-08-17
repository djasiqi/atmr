import { useCallback, useEffect, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { getDriverProfile, updateDriverAvailability } from "../api";
import { emitDriverTelemetry } from "../../../core/observability/driverTelemetry";
import {
  normalizeDriverProfilePayload,
  resolveDriverAvailabilityFromProfile,
} from "../domain/driverAvailability";
import {
  setDriverAvailabilityActive,
} from "../services/driverAvailabilityBridge";
import { readDriverProfileCache, writeDriverProfileCache } from "../services/driverProfileCache";

type Options = {
  onToggleSuccess?: (next: boolean) => void;
  onToggleError?: (message: string) => void;
};

export function useDriverAvailability(options?: Options) {
  const mutation = useMutation({
    mutationFn: (isAvailable: boolean) => updateDriverAvailability(isAvailable),
    onSuccess: (_, isAvailable) => {
      setDriverAvailabilityActive(isAvailable);
      emitDriverTelemetry("driver.availability.updated", {
        source: "driver.hooks.useDriverAvailability",
        available: isAvailable,
      });
    },
  });
  const [isAvailable, setIsAvailable] = useState<boolean | null>(null);
  const [unavailableConfirmOpen, setUnavailableConfirmOpen] = useState(false);

  const applyAvailability = useCallback((next: boolean) => {
    setIsAvailable(next);
    setDriverAvailabilityActive(next);
  }, []);

  useEffect(() => {
    let cancelled = false;

    const applyProfile = (profile: ReturnType<typeof normalizeDriverProfilePayload>) => {
      if (cancelled) return;
      const next = resolveDriverAvailabilityFromProfile(profile);
      if (next == null) return;
      applyAvailability(next);
    };

    void (async () => {
      const cached = await readDriverProfileCache({ allowStale: true });
      if (cached.profile) {
        applyProfile(cached.profile);
      }
      try {
        const profile = normalizeDriverProfilePayload(await getDriverProfile());
        applyProfile(profile);
      } catch {
        // Ne pas inventer AVAILABLE : conserver la valeur connue, sinon UNKNOWN.
        if (!cancelled) {
          setIsAvailable((current) => {
            if (current != null) {
              setDriverAvailabilityActive(current);
              return current;
            }
            setDriverAvailabilityActive(null);
            return null;
          });
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [applyAvailability]);

  const setAvailability = useCallback(
    (next: boolean) => {
      if (mutation.isPending) return;
      mutation.mutate(next, {
        onSuccess: async () => {
          applyAvailability(next);
          options?.onToggleSuccess?.(next);
          const cached = await readDriverProfileCache({ allowStale: true });
          if (cached.profile) {
            await writeDriverProfileCache({ ...cached.profile, is_available: next });
          }
        },
        onError: (error) => {
          options?.onToggleError?.(
            error instanceof Error ? error.message : "Impossible de mettre à jour la disponibilité."
          );
        },
      });
    },
    [applyAvailability, mutation, options]
  );

  const requestToggleAvailability = useCallback(() => {
    if (isAvailable == null || mutation.isPending) return;
    if (isAvailable) {
      setUnavailableConfirmOpen(true);
      return;
    }
    setAvailability(true);
  }, [isAvailable, mutation.isPending, setAvailability]);

  const confirmUnavailable = useCallback(() => {
    setAvailability(false);
    setUnavailableConfirmOpen(false);
  }, [setAvailability]);

  const cancelUnavailableConfirm = useCallback(() => {
    if (mutation.isPending) return;
    setUnavailableConfirmOpen(false);
  }, [mutation.isPending]);

  return {
    /** null = UNKNOWN (pas encore hydraté) — ne pas interpréter comme hors service. */
    isAvailable,
    availabilityLoading: isAvailable === null,
    availabilityPending: mutation.isPending,
    unavailableConfirmOpen,
    requestToggleAvailability,
    confirmUnavailable,
    cancelUnavailableConfirm,
    setAvailability,
    setIsAvailable: applyAvailability,
  };
}
