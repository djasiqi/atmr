// contexts/VersionContext.tsx
// Contexte React pour gérer le statut de mise à jour de l'application

import { getLogger } from "@/utils/logger";
import React, {
    createContext,
    ReactNode,
    useCallback,
    useContext,
    useEffect,
    useState,
} from "react";
import {
    UpdateStatus,
    VersionCheckResponse,
    checkVersion,
} from "@/services/versionService";

const log = getLogger("Version");

/** Comparaison semver simple (x.y.z), suffisante pour les logs de version store. */
function compareSemver(a: string, b: string): number {
  const pa = a.split(".").map((p) => parseInt(p.replace(/[^\d].*$/, ""), 10) || 0);
  const pb = b.split(".").map((p) => parseInt(p.replace(/[^\d].*$/, ""), 10) || 0);
  const n = Math.max(pa.length, pb.length);
  for (let i = 0; i < n; i++) {
    const da = pa[i] ?? 0;
    const db = pb[i] ?? 0;
    if (da !== db) return da > db ? 1 : -1;
  }
  return 0;
}

interface VersionContextType {
    versionInfo: VersionCheckResponse | null;
    status: UpdateStatus;
    isLoading: boolean;
    error: Error | null;
    refreshVersionCheck: () => Promise<void>;
}

const VersionContext = createContext<VersionContextType | undefined>(undefined);

export const VersionProvider = ({ children }: { children: ReactNode }) => {
    const [versionInfo, setVersionInfo] = useState<VersionCheckResponse | null>(
        null
    );
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<Error | null>(null);

    const refreshVersionCheck = useCallback(async () => {
        setIsLoading(true);
        setError(null);
        try {
            const result = await checkVersion();
            setVersionInfo(result);
            const vsLatest = compareSemver(
                result.current_version,
                result.latest_version
            );
            const vsMin = compareSemver(
                result.current_version,
                result.min_required_version
            );
            log.info("Version check status", {
                status: result.status,
                currentVersion: result.current_version,
                latestVersion: result.latest_version,
                minRequiredVersion: result.min_required_version,
                vsPublishedLatest:
                    vsLatest === 0 ? "same" : vsLatest > 0 ? "ahead" : "behind",
                vsMinRequired:
                    vsMin === 0 ? "same" : vsMin > 0 ? "above" : "below",
            });
        } catch (err) {
            const error =
                err instanceof Error ? err : new Error("Erreur vérification version");
            setError(error);
            log.error("Version check failed", { error });
        } finally {
            setIsLoading(false);
        }
    }, []);

    // Vérification automatique au montage du provider
    useEffect(() => {
        refreshVersionCheck();
    }, [refreshVersionCheck]);

    const status: UpdateStatus = versionInfo?.status || "OK";

    const value: VersionContextType = {
        versionInfo,
        status,
        isLoading,
        error,
        refreshVersionCheck,
    };

    return (
        <VersionContext.Provider value={value}>{children}</VersionContext.Provider>
    );
};

export const useVersion = (): VersionContextType => {
    const context = useContext(VersionContext);
    if (!context) {
        throw new Error("useVersion doit être utilisé au sein d'un VersionProvider");
    }
    return context;
};

