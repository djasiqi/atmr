import { useState, useCallback, useEffect } from "react";
import {
  fetchCompanyReservations,
  fetchCompanyDriver,
  fetchCompanyInfo
} from "../services/companyService";
import { getAccessToken } from "./useAuthToken";

const useCompanyData = () => {
  const [reservations, setReservations] = useState([]);
  const [driver, setDriver] = useState([]);
  const [loadingReservations, setLoadingReservations] = useState(true);
  const [loadingDriver, setLoadingDriver] = useState(true);
  const [error, setError] = useState(null);
  const [company, setCompany] = useState(null);

  const loadCompany = useCallback(async () => {
    try {
      const token = getAccessToken();
      if (!token) return;

      const data = await fetchCompanyInfo();
      setCompany(data);
      console.log("✅ Company chargée :", data);
    } catch (err) {
      console.error("❌ Erreur lors du chargement de l'entreprise :", err);
      setError("Erreur lors du chargement de l'entreprise.");
    }
  }, []);

  const loadReservations = useCallback(async () => {
    try {
      setLoadingReservations(true);
      const data = await fetchCompanyReservations();
      // Le service renvoie déjà un ARRAY normalisé
      setReservations(Array.isArray(data) ? data : (data?.reservations ?? []));
      setError(null); // Réinitialiser l'erreur en cas de succès
    } catch (err) {
      // Gérer spécifiquement les erreurs de timeout
      if (err.code === 'ECONNABORTED' || err.message?.includes('timeout')) {
        setError("La récupération des réservations a pris trop de temps. Veuillez réessayer.");
      } else {
        console.error("❌ Erreur lors du chargement des réservations :", err);
        setError("Erreur lors du chargement des réservations.");
      }
    } finally {
      setLoadingReservations(false);
    }
  }, []);

  const loadDriver = useCallback(async () => {
    try {
      setLoadingDriver(true);
      const data = await fetchCompanyDriver();
      // Le service renvoie déjà un ARRAY normalisé
      setDriver(Array.isArray(data) ? data : (data?.driver ?? []));
      setError(null); // Réinitialiser l'erreur en cas de succès
    } catch (err) {
      // Gérer spécifiquement les erreurs de timeout
      if (err.code === 'ECONNABORTED' || err.message?.includes('timeout')) {
        setError("La récupération des chauffeurs a pris trop de temps. Veuillez réessayer.");
      } else {
        console.error("❌ Erreur lors du chargement des chauffeurs :", err);
        setError("Erreur lors du chargement des chauffeurs.");
      }
    } finally {
      setLoadingDriver(false);
    }
  }, []);

  // Chargement initial de toutes les données
  useEffect(() => {
    loadCompany();
    loadReservations();
    loadDriver();
  }, [loadCompany, loadReservations, loadDriver]);

  return {
    company,
    reservations,
    driver,
    loadingReservations,
    loadingDriver,
    error,
    reloadCompany: loadCompany,
    reloadReservations: loadReservations,
    reloadDriver: loadDriver,
  };
};

export default useCompanyData;