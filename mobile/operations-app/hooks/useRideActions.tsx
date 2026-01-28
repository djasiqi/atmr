import { useCallback, useState } from "react";
import { Alert, Platform } from "react-native";
import * as Crypto from "expo-crypto";
import { useAuth } from "@/hooks/useAuth";
import {
    markRideUrgent,
    assignRide,
    reassignRide,
    getDispatchRideDetails,
    getAvailableDrivers,
} from "@/services/enterpriseDispatch";
import { acceptTransfer, rejectTransfer } from "@/services/partnershipService";
import type {
    RideSummary,
    RideDetail,
    DriverSuggestion,
} from "@/types/enterpriseDispatch";
import { sendIngestEvent } from "@/src/config/telemetry";

// ✅ Helper pour afficher des alertes cross-platform (web + native)
const showAlert = (title: string, message: string, buttons: Array<{ text: string; style?: string; onPress?: () => void }>) => {
    if (Platform.OS === "web") {
        // Sur web, utiliser window.confirm
        const confirmed = window.confirm(`${title}\n\n${message}`);
        if (confirmed) {
            const confirmButton = buttons.find(b => b.style !== "cancel");
            confirmButton?.onPress?.();
        } else {
            const cancelButton = buttons.find(b => b.style === "cancel");
            cancelButton?.onPress?.();
        }
    } else {
        // Sur native, utiliser Alert.alert
        Alert.alert(title, message, buttons as any);
    }
};

/**
 * Hook partagé pour les actions sur les courses (assignation, urgent, etc.)
 * Évite la duplication entre dashboard et rides
 */
export function useRideActions(onSuccess?: () => void | Promise<void>) {
    const { enterpriseSession } = useAuth();
    const dispatchMode = (enterpriseSession?.company?.dispatchMode as "manual" | "semi_auto" | "fully_auto" | undefined) || "manual";
    const isManualMode = dispatchMode === "manual";

    const [assigning, setAssigning] = useState(false);
    const [markingUrgent, setMarkingUrgent] = useState<string | null>(null);
    const [assignModalVisible, setAssignModalVisible] = useState(false);
    const [selectedRide, setSelectedRide] = useState<RideSummary | null>(null);
    const [rideSuggestions, setRideSuggestions] = useState<DriverSuggestion[]>([]);
    const [loadingSuggestions, setLoadingSuggestions] = useState(false);
    const [allDrivers, setAllDrivers] = useState<DriverSuggestion[]>([]);
    const [loadingAllDrivers, setLoadingAllDrivers] = useState(false);

    // ✅ Marquer une course urgente (autorisé uniquement si pickup_at sentinelle 00:00)
    const handleMarkUrgent = useCallback(
        async (rideId: string, extraDelayMinutes = 15) => {
            if (markingUrgent === rideId) return;
            setMarkingUrgent(rideId);
            try {
                await markRideUrgent(rideId, {
                    extra_delay_minutes: extraDelayMinutes,
                    reason: "Action mobile: urgence",
                });
                Alert.alert(
                    "Urgence",
                    `La course a été marquée urgente avec un délai +${extraDelayMinutes} minutes.`
                );
                await onSuccess?.();
            } catch (error: any) {
                if (error?.response?.status === 409) {
                    Alert.alert("Course déjà planifiée", "Course déjà planifiée (urgent indisponible).");
                    return;
                }
                const message =
                    error?.response?.data?.error ??
                    error?.message ??
                    "Impossible de marquer la course urgente.";
                Alert.alert("Erreur", message);
            } finally {
                setMarkingUrgent(null);
            }
        },
        [markingUrgent, onSuccess]
    );

    // ✅ Ouvrir le modal d'assignation
    const handleOpenAssignModal = useCallback(
        async (ride: RideSummary) => {
            // ✅ P0-1: Utiliser la fonction de normalisation
            const { isCompletedStatus } = require("@/utils/bookingStatus");
            const isCompleted = isCompletedStatus(ride?.status);
            const statusLower = ride?.status?.toLowerCase() || "";
            const isInBoard = statusLower === "in_board";
            const isActionDisabled = isCompleted || isInBoard;

            if (isActionDisabled) {
                Alert.alert(
                    "Action impossible",
                    isCompleted
                        ? "La course est terminée. L'assignation n'est plus possible."
                        : "Le client est à bord. L'assignation n'est plus possible."
                );
                return;
            }

            setSelectedRide(ride);
            setAssignModalVisible(true);

            // ✅ En mode manuel, ne pas charger les suggestions, charger directement tous les chauffeurs
            if (isManualMode) {
                setLoadingSuggestions(false);
                setRideSuggestions([]);
                setLoadingAllDrivers(true);
                try {
                    const drivers = await getAvailableDrivers();
                    // Convertir en format DriverSuggestion pour compatibilité
                    const driverSuggestions: DriverSuggestion[] = drivers.map((driver) => ({
                        driver_id: driver.driver_id,
                        driver_name: driver.driver_name,
                        score: 0.0,
                        fairness_delta: null,
                        preferred_match: false,
                        is_emergency: driver.is_emergency,
                        reason: "Sélection manuelle",
                    }));
                    setAllDrivers(driverSuggestions);
                } catch (driversError: any) {
                    console.error("[useRideActions] Erreur chargement tous les chauffeurs:", driversError);
                } finally {
                    setLoadingAllDrivers(false);
                }
                return;
            }

            // ✅ En mode semi-auto ou fully-auto, charger les suggestions
            setLoadingSuggestions(true);
            setRideSuggestions([]);

            try {
                const details: RideDetail = await getDispatchRideDetails(ride.id);
                setRideSuggestions(details.suggestions || []);
            } catch (error: any) {
                // ✅ Ne pas fermer le modal en cas d'erreur, juste afficher un avertissement
                // Cela permet de réassigner même si les suggestions ne peuvent pas être chargées
                const message =
                    error?.response?.data?.error ??
                    error?.message ??
                    "Impossible de charger les suggestions de chauffeurs.";
                console.warn("[useRideActions] Erreur chargement suggestions:", message);
                // Ne pas fermer le modal - l'utilisateur peut toujours réassigner manuellement
                // Charger tous les chauffeurs disponibles en fallback
                setLoadingAllDrivers(true);
                try {
                    const drivers = await getAvailableDrivers();
                    // Convertir en format DriverSuggestion pour compatibilité
                    const driverSuggestions: DriverSuggestion[] = drivers.map((driver) => ({
                        driver_id: driver.driver_id,
                        driver_name: driver.driver_name,
                        score: 0.0,
                        fairness_delta: null,
                        preferred_match: false,
                        is_emergency: driver.is_emergency,
                        reason: "Sélection manuelle",
                    }));
                    setAllDrivers(driverSuggestions);
                } catch (driversError: any) {
                    console.error("[useRideActions] Erreur chargement tous les chauffeurs:", driversError);
                    // Ne pas bloquer, juste logger
                } finally {
                    setLoadingAllDrivers(false);
                }
                // Afficher un message informatif mais ne pas bloquer
                if (error?.response?.status !== 404) {
                    // Ne pas alerter pour les 404 (course introuvable), juste logger
                    Alert.alert(
                        "Avertissement",
                        `${message}\n\nVous pouvez toujours sélectionner un chauffeur manuellement.`
                    );
                }
            } finally {
                setLoadingSuggestions(false);
            }
        },
        []
    );

    // ✅ Fermer le modal d'assignation
    const handleCloseAssignModal = useCallback(() => {
        setAssignModalVisible(false);
        setSelectedRide(null);
        setRideSuggestions([]);
        setAllDrivers([]);
    }, []);

    // ✅ Assigner/réassigner un chauffeur
    const handleAssignDriver = useCallback(
        async (driverId: string, reason?: string) => {
            console.log("[useRideActions] handleAssignDriver appelé:", { driverId, selectedRide: selectedRide?.id });

            if (!selectedRide) {
                console.error("[useRideActions] Aucune course sélectionnée");
                Alert.alert("Erreur", "Aucune course sélectionnée.");
                return;
            }

            const isAssigned =
                selectedRide.status === "assigned" ||
                !!selectedRide.driver?.id;

            console.log("[useRideActions] Assignation:", {
                rideId: selectedRide.id,
                driverId,
                isAssigned,
                currentDriver: selectedRide.driver?.id
            });

            setAssigning(true);
            try {
                if (isAssigned) {
                    console.log("[useRideActions] Réassignation de la course");
                    await reassignRide(selectedRide.id, {
                        driver_id: driverId,
                        reason: reason ?? undefined,
                        allow_emergency: false,
                        respect_preferences: true,
                        idempotency_key: Crypto.randomUUID(),
                    });
                } else {
                    console.log("[useRideActions] Assignation de la course");
                    await assignRide(selectedRide.id, {
                        driver_id: driverId,
                        reason: reason ?? undefined,
                        allow_emergency: false,
                        respect_preferences: true,
                        idempotency_key: Crypto.randomUUID(),
                    });
                }
                console.log("[useRideActions] Assignation réussie");
                Alert.alert(
                    "Assignation effectuée",
                    "La course a été mise à jour avec succès."
                );
                handleCloseAssignModal();
                await onSuccess?.();
            } catch (error: any) {
                console.error("[useRideActions] Erreur lors de l'assignation:", error);
                console.error("[useRideActions] Erreur response:", error?.response?.data);
                console.error("[useRideActions] Erreur status:", error?.response?.status);

                const status = error?.response?.status;
                const errorData = error?.response?.data;

                // ✅ Gestion spécifique de l'erreur 409 (Conflit de planning)
                if (status === 409) {
                    const conflictMessage =
                        errorData?.message ??
                        errorData?.error ??
                        "Conflit de planning détecté";

                    // Extraire le numéro de course en conflit si présent dans le message
                    const conflictMatch = conflictMessage.match(/#(\d+)/);
                    const conflictRideId = conflictMatch ? conflictMatch[1] : null;
                    const timeMatch = conflictMessage.match(/à (\d{1,2}:\d{2})/);
                    const conflictTime = timeMatch ? timeMatch[1] : null;

                    let alertMessage = conflictMessage;
                    if (conflictRideId || conflictTime) {
                        alertMessage = `⚠️ ${conflictMessage}`;
                        if (conflictRideId) {
                            alertMessage += `\n\nCourse en conflit : #${conflictRideId}`;
                        }
                        if (conflictTime) {
                            alertMessage += `\nHeure : ${conflictTime}`;
                        }
                    }
                    alertMessage += "\n\nLe chauffeur a déjà une course assignée à la même heure. Veuillez choisir un autre chauffeur ou modifier l'heure de la course.";

                    Alert.alert(
                        "⚠️ Conflit de planning",
                        alertMessage,
                        [
                            { text: "OK", style: "default" },
                        ]
                    );
                } else if (status === 500) {
                    // ✅ Gestion spécifique des erreurs 500 (erreur serveur)
                    // En mode manuel, on peut toujours réassigner même si les suggestions ne sont pas actives
                    const errorMessage =
                        errorData?.error ??
                        errorData?.message ??
                        "Une erreur serveur est survenue lors de l'assignation.";

                    // Si la course est déjà assignée et qu'on essaie de réassigner, c'est probablement OK
                    // Le backend peut avoir eu un problème mais l'assignation peut quand même avoir fonctionné
                    if (isAssigned) {
                        Alert.alert(
                            "⚠️ Avertissement",
                            `${errorMessage}\n\nLa réassignation peut avoir été effectuée malgré l'erreur. Veuillez vérifier l'état de la course.`,
                            [
                                {
                                    text: "OK",
                                    onPress: async () => {
                                        // Rafraîchir les données pour voir l'état réel
                                        await onSuccess?.();
                                    },
                                },
                            ]
                        );
                    } else {
                        Alert.alert(
                            "Erreur serveur",
                            `${errorMessage}\n\nVeuillez réessayer. Si le problème persiste, contactez le support.`,
                            [
                                { text: "OK", style: "default" },
                            ]
                        );
                    }
                } else {
                    // Autres erreurs (400, 404, etc.)
                    const message =
                        errorData?.error ??
                        errorData?.message ??
                        error?.message ??
                        "Impossible de finaliser l'assignation.";
                    Alert.alert("Erreur", message);
                }
            } finally {
                setAssigning(false);
            }
        },
        [selectedRide, handleCloseAssignModal, onSuccess]
    );

    // ✅ Accepter un transfert de course
    const handleAcceptTransfer = useCallback(
        async (transferId: string) => {
            // #region agent log
            try {
                sendIngestEvent({ location: 'useRideActions.tsx:handleAcceptTransfer', message: 'handleAcceptTransfer called', data: { transferId: transferId }, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run11', hypothesisId: 'H1-H2' });
            } catch { }
            // #endregion
            showAlert(
                "Accepter le transfert",
                "Voulez-vous accepter cette course transférée ?",
                [
                    { text: "Annuler", style: "cancel" },
                    {
                        text: "Accepter",
                        style: "default",
                        onPress: async () => {
                            try {
                                // #region agent log
                                try {
                                    sendIngestEvent({ location: 'useRideActions.tsx:handleAcceptTransfer:onPress', message: 'Calling acceptTransfer', data: { transferId: transferId }, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run11', hypothesisId: 'H3' });
                                } catch { }
                                // #endregion
                                await acceptTransfer(transferId);
                                showAlert(
                                    "Transfert accepté",
                                    "La course vous a été transférée avec succès.",
                                    [{ text: "OK" }]
                                );
                                await onSuccess?.();
                            } catch (error: any) {
                                // #region agent log
                                try {
                                    sendIngestEvent({ location: 'useRideActions.tsx:handleAcceptTransfer:error', message: 'acceptTransfer error', data: { transferId: transferId, error: String(error), errorMessage: error?.message, errorResponse: error?.response?.data }, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run11', hypothesisId: 'H4' });
                                } catch { }
                                // #endregion
                                const message =
                                    error?.response?.data?.error ??
                                    error?.message ??
                                    "Impossible d'accepter le transfert.";
                                showAlert("Erreur", message, [{ text: "OK" }]);
                            }
                        },
                    },
                ]
            );
        },
        [onSuccess]
    );

    // ✅ Refuser un transfert de course
    const handleRejectTransfer = useCallback(
        async (transferId: string) => {
            // #region agent log
            try {
                sendIngestEvent({ location: 'useRideActions.tsx:handleRejectTransfer', message: 'handleRejectTransfer called', data: { transferId: transferId }, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run11', hypothesisId: 'H1-H2' });
            } catch { }
            // #endregion
            showAlert(
                "Refuser le transfert",
                "Voulez-vous refuser cette course transférée ?",
                [
                    { text: "Annuler", style: "cancel" },
                    {
                        text: "Refuser",
                        style: "destructive",
                        onPress: async () => {
                            try {
                                // #region agent log
                                try {
                                    sendIngestEvent({ location: 'useRideActions.tsx:handleRejectTransfer:onPress', message: 'Calling rejectTransfer', data: { transferId: transferId }, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run11', hypothesisId: 'H3' });
                                } catch { }
                                // #endregion
                                await rejectTransfer(transferId);
                                showAlert(
                                    "Transfert refusé",
                                    "La course a été retournée à l'entreprise émettrice.",
                                    [{ text: "OK" }]
                                );
                                await onSuccess?.();
                            } catch (error: any) {
                                // #region agent log
                                try {
                                    sendIngestEvent({ location: 'useRideActions.tsx:handleRejectTransfer:error', message: 'rejectTransfer error', data: { transferId: transferId, error: String(error), errorMessage: error?.message, errorResponse: error?.response?.data }, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run11', hypothesisId: 'H4' });
                                } catch { }
                                // #endregion
                                const message =
                                    error?.response?.data?.error ??
                                    error?.message ??
                                    "Impossible de refuser le transfert.";
                                showAlert("Erreur", message, [{ text: "OK" }]);
                            }
                        },
                    },
                ]
            );
        },
        [onSuccess]
    );

    return {
        // États
        assigning,
        markingUrgent,
        assignModalVisible,
        selectedRide,
        rideSuggestions,
        loadingSuggestions,
        allDrivers,
        loadingAllDrivers,
        // Actions
        handleMarkUrgent,
        handleOpenAssignModal,
        handleCloseAssignModal,
        handleAssignDriver,
        handleAcceptTransfer,
        handleRejectTransfer,
        // Setters pour contrôle externe si nécessaire
        setAssignModalVisible,
    };
}

