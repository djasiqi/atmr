import { useState, useCallback } from "react";
import { Alert } from "react-native";
import { RideDetail, RideCreatePayload } from "@/types/enterpriseDispatch";
import { createRide } from "@/services/enterpriseDispatch";

export const useRideCreate = (onSuccess?: () => Promise<void>) => {
    const [loading, setLoading] = useState(false);

    const create = useCallback(
        async (payload: RideCreatePayload) => {
            // client_id requis (alignement web — plus de client_name seul)
            if (!payload.client_id) {
                Alert.alert("Erreur", "Un client existant doit être sélectionné.");
                return null;
            }

            if (!payload.pickup_address?.trim()) {
                Alert.alert("Erreur", "L'adresse de départ est requise.");
                return null;
            }

            if (!payload.dropoff_address?.trim()) {
                Alert.alert("Erreur", "L'adresse d'arrivée est requise.");
                return null;
            }

            if (!payload.scheduled_time) {
                Alert.alert("Erreur", "La date et l'heure de départ sont requises.");
                return null;
            }

            setLoading(true);
            try {
                const created = await createRide(payload);
                Alert.alert("Succès", "La course a été créée avec succès.");
                if (onSuccess) {
                    await onSuccess();
                }
                return created;
            } catch (error: any) {
                const message =
                    error?.response?.data?.error ??
                    error?.message ??
                    "Impossible de créer la course.";
                Alert.alert("Erreur", message);
                throw error;
            } finally {
                setLoading(false);
            }
        },
        [onSuccess]
    );

    return {
        loading,
        create,
    };
};

