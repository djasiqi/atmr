
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
      console.log("\u2705 Company charg\u00e9e :", data);
    } catch (err) {
      console.error("\u274c Erreur lors du chargement de l'entreprise :", err);
      setError("Erreur lors du chargement de l'entreprise.");
    }
  }, []);

  const loadReservations = useCallback(async () => {
    try {
      setLoadingReservations(true);
      const data = await fetchCompanyReservations();
      // Le service renvoie d\u00e9j\u00e0 un ARRAY normalis\u00e9
      setReservations(Array.isArray(data) ? data : (data?.reservations ?? []));
      setError(null); // R\u00e9initialiser l'erreur en cas de succ\u00e8s
    } catch (err) {
      // G\u00e9rer sp\u00e9cifiquement les erreurs de timeout
      if (err.code === 'ECONNABORTED' || err.message?.includes('timeout')) {
        setError("La r\u00e9cup\u00e9ration des r\u00e9servations a pris trop de temps. Veuillez r\u00e9essayer.");
      } else {
        console.error("\u274c Erreur lors du chargement des r\u00e9servations :", err);
        setError("Erreur lors du chargement des r\u00e9servations.");
      }
    } finally {
      setLoadingReservations(false);
    }
  }, []);

  const loadDriver = useCallback(async () => {
    try {
      setLoadingDriver(true);
      const data = await fetchCompanyDriver();
      // Le service renvoie d\u00e9j\u00e0 un ARRAY normalis\u00e9
      setDriver(Array.isArray(data) ? data : (data?.driver ?? []));
      setError(null); // R\u00e9initialiser l'erreur en cas de succ\u00e8s
    } catch (err) {
      // G\u00e9rer sp\u00e9cifiquement les erreurs de timeout
      if (err.code === 'ECONNABORTED' || err.message?.includes('timeout')) {
        setError("La r\u00e9cup\u00e9ration des chauffeurs a pris trop de temps. Veuillez r\u00e9essayer.");
      } else {
        console.error("\u274c Erreur lors du chargement des chauffeurs :", err);
        setError("Erreur lors du chargement des chauffeurs.");
      }
    } finally {
      setLoadingDriver(false);
    }
  }, []);

  // Chargement initial de toutes les donn\u00e9es
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
