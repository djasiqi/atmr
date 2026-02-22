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

