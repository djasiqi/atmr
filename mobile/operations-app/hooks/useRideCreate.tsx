import { useState, useCallback } from "react";
import { Alert } from "react-native";
import { RideDetail, RideCreatePayload } from "@/types/enterpriseDispatch";
import { createRide } from "@/services/enterpriseDispatch";

export const useRideCreate = (onSuccess?: () => Promise<void>) => {
    const [loading, setLoading] = useState(false);

    const create = useCallback(
        async (payload: RideCreatePayload) => {
            // ✅ P1-4 Phase 3.3: Utiliser client_name au lieu de customer_name
            // Validation basique : client_id OU client_name requis
            if (!payload.client_id && !payload.client_name?.trim()) {
                Alert.alert("Erreur", "Un client doit être sélectionné ou un nom de client doit être fourni.");
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

