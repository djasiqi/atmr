// src/hooks/useDriver.js
import { useState, useEffect, useCallback } from "react";
import {
  fetchCompanyDriver,
  updateDriverStatus,
  deleteDriver,
} from "../services/companyService";

const useDriver = () => {
  // CORRECTION : On utilise "drivers" (pluriel) pour la liste
  const [drivers, setDrivers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const getDrivers = useCallback(async () => {
    try {
      setLoading(true);
      const data = await fetchCompanyDriver();
      // fetchCompanyDriver() renvoie déjà un tableau normalisé
      setDrivers(Array.isArray(data) ? data : []);
      setError(null);
    } catch (err) {
      console.error(err);
      setError("Erreur lors du chargement des chauffeurs.");
    } finally {
      setLoading(false);
    }
  }, []);

  // CORRECTION : On renomme la fonction pour plus de clarté
  const toggleDriverStatus = useCallback(async (driverId, newStatus) => {
    try {
      await updateDriverStatus(driverId, newStatus);
      // Met à jour l'état local pour un retour visuel immédiat
      setDrivers((prev) =>
        prev.map((d) =>
          d.id === driverId ? { ...d, is_active: newStatus } : d
        )
      );
    } catch (err) {
      console.error("Erreur lors de la mise à jour du statut :", err);
    }
  }, []);

  const deleteDriverById = useCallback(async (driverId) => {
    try {
      await deleteDriver(driverId);
      // Met à jour l'état local
      setDrivers((prev) => prev.filter((d) => d.id !== driverId));
    } catch (err) {
      console.error("Erreur lors de la suppression :", err);
    }
  }, []);

  useEffect(() => {
    getDrivers();
  }, [getDrivers]);

  return {
    drivers, // <-- renvoie "drivers"
    loading,
    error,
    refreshDrivers: getDrivers, // <-- alias pour rafraîchir
    toggleDriverStatus, // <-- nom de fonction clair
    deleteDriverById,
  };
};

export default useDriver;
