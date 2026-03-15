import { useState, useCallback } from "react";
import { RideDetail, RideCreatePayload } from "@/types/enterpriseDispatch";
import { createRide } from "@/services/enterpriseDispatch";
import { useAppAlert } from "@/contexts/AppAlertContext";

export const useRideCreate = (onSuccess?: () => Promise<void>) => {
    const appAlert = useAppAlert();
    const [loading, setLoading] = useState(false);

    const create = useCallback(
        async (payload: RideCreatePayload) => {
            // client_id requis (alignement web — plus de client_name seul)
            if (!payload.client_id) {
                appAlert.showAlert("Erreur", "Un client existant doit être sélectionné.");
                return null;
            }

            if (!payload.pickup_address?.trim()) {
                appAlert.showAlert("Erreur", "L'adresse de départ est requise.");
                return null;
            }

            if (!payload.dropoff_address?.trim()) {
                appAlert.showAlert("Erreur", "L'adresse d'arrivée est requise.");
                return null;
            }

            if (!payload.scheduled_time) {
                appAlert.showAlert("Erreur", "La date et l'heure de départ sont requises.");
                return null;
            }

            setLoading(true);
            try {
                const created = await createRide(payload);
                appAlert.showAlert("Succès", "La course a été créée avec succès.");
                if (onSuccess) {
                    await onSuccess();
                }
                return created;
            } catch (error: any) {
                const message =
                    error?.response?.data?.error ??
                    error?.message ??
                    "Impossible de créer la course.";
                appAlert.showAlert("Erreur", message);
                throw error;
            } finally {
                setLoading(false);
            }
        },
        [onSuccess, appAlert]
    );

    return {
        loading,
        create,
    };
};

