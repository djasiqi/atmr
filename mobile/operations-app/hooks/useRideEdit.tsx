import { useState, useCallback } from "react";
import { Alert } from "react-native";
import { RideDetail, RideEditPayload } from "@/types/enterpriseDispatch";
import { updateRide, getDispatchRideDetails } from "@/services/enterpriseDispatch";
import { getLogger } from "@/utils/logger";

const log = getLogger("RideEdit");

export const useRideEdit = (onSuccess?: () => Promise<void>) => {
    const [loading, setLoading] = useState(false);
    const [rideDetail, setRideDetail] = useState<RideDetail | null>(null);
    const [loadingDetail, setLoadingDetail] = useState(false);

    const loadRideDetail = useCallback(async (rideId: string) => {
        setLoadingDetail(true);
        try {
            const detail = await getDispatchRideDetails(rideId);
            setRideDetail(detail);
            return detail;
        } catch (error: any) {
            const status = error?.response?.status;
            const errorData = error?.response?.data;
            
            // ✅ Ne pas afficher d'alerte pour les erreurs 500 - on utilisera les données de base
            // L'utilisateur peut quand même éditer avec les données disponibles
            if (status === 500) {
                log.warn("details 500, using base data", {});
                // Ne pas définir rideDetail, le modal utilisera les données de ride (RideSummary)
                return null;
            }
            
            // Pour les autres erreurs (404, 403, etc.), afficher un message
            const message =
                errorData?.error ??
                errorData?.message ??
                error?.message ??
                "Impossible de charger les détails de la course.";
            
            // Ne pas bloquer l'édition - l'utilisateur peut toujours modifier avec les données de base
            log.warn("load ride details failed", { message });
            return null;
        } finally {
            setLoadingDetail(false);
        }
    }, []);

    const update = useCallback(
        async (rideId: string, payload: RideEditPayload) => {
            setLoading(true);
            try {
                const updated = await updateRide(rideId, payload);
                setRideDetail(updated);
                if (onSuccess) {
                    await onSuccess();
                }
                Alert.alert("Succès", "La course a été mise à jour avec succès.");
                return updated;
            } catch (error: any) {
                const message =
                    error?.response?.data?.error ??
                    error?.message ??
                    "Impossible de mettre à jour la course.";
                Alert.alert("Erreur", message);
                throw error;
            } finally {
                setLoading(false);
            }
        },
        [onSuccess]
    );

    const clear = useCallback(() => {
        setRideDetail(null);
    }, []);

    return {
        rideDetail,
        loading,
        loadingDetail,
        loadRideDetail,
        update,
        clear,
    };
};

