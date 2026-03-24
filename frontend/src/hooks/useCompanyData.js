import { useState, useCallback, useEffect } from 'react';
import {
  fetchCompanyReservations,
  fetchCompanyDriversCanonical,
  fetchCompanyInfo,
} from '../services/companyService';
import { getCompanySocket, joinCompanyRoom } from '../services/companySocket';
import { mergeOrUpdateDriverInList } from '../utils/mergeDriverLiveUpdate';
import { getAccessToken } from './useAuthToken';

const useCompanyData = ({ day } = {}) => {
  const [reservations, setReservations] = useState([]);
  const [driver, setDriver] = useState([]);
  const [loadingReservations, setLoadingReservations] = useState(true);
  const [loadingDriver, setLoadingDriver] = useState(true);
  const [loadingCompany, setLoadingCompany] = useState(true);
  const [error, setError] = useState(null);
  const [company, setCompany] = useState(null);

  const loadCompany = useCallback(async () => {
    try {
      setLoadingCompany(true);
      setError(null);
      
      // ✅ Vérifier l'authentification : soit token dans localStorage (mobile), soit infos utilisateur (web avec cookies httpOnly)
      // Si on utilise des cookies httpOnly, le token n'est pas dans localStorage, mais les infos utilisateur sont stockées
      // Dans ce cas, on peut quand même faire la requête car les cookies seront envoyés automatiquement avec withCredentials: true
      const token = getAccessToken();
      const hasToken = !!token;
      const hasUser = !!localStorage.getItem('user');
      
      if (!hasToken && !hasUser) {
        setError("Authentification manquante. Veuillez vous reconnecter.");
        setLoadingCompany(false);
        return;
      }
      
      // Si on a un token OU des infos utilisateur, on peut faire la requête
      // (les cookies httpOnly seront envoyés automatiquement si pas de token)
      const data = await fetchCompanyInfo();
      
      // Vérifier si fetchCompanyInfo a retourné un objet d'erreur
      if (data?.error === true) {
        setError("Erreur lors du chargement de l'entreprise.");
        setCompany(null);
      } else {
        setCompany(data);
      }
    } catch (err) {
      // Ne pas logger les erreurs 403/404/401 comme des erreurs critiques (permissions manquantes ou company non trouvée)
      const status = err?.response?.status;
      if (status !== 403 && status !== 404 && status !== 401) {
        console.error("❌ Erreur lors du chargement de l'entreprise :", err);
      }
      setError("Erreur lors du chargement de l'entreprise.");
      setCompany(null);
    } finally {
      setLoadingCompany(false);
    }
  }, []);

  const loadReservations = useCallback(async (opts = {}) => {
    const silent = Boolean(opts.silent);
    try {
      if (!silent) {
        setLoadingReservations(true);
      }
      const data = await fetchCompanyReservations(day);
      // Le service renvoie déjà un ARRAY normalisé
      setReservations(Array.isArray(data) ? data : (data?.reservations ?? []));
      setError(null); // Réinitialiser l'erreur en cas de succès
    } catch (err) {
      // Gérer spécifiquement les erreurs de timeout
      if (err.code === 'ECONNABORTED' || err.message?.includes('timeout')) {
        setError('La récupération des réservations a pris trop de temps. Veuillez réessayer.');
      } else {
        // Ne pas logger les erreurs 403/404/401 comme des erreurs critiques
        const status = err?.response?.status;
        if (status !== 403 && status !== 404 && status !== 401) {
          console.error('❌ Erreur lors du chargement des réservations :', err);
        }
        setError('Erreur lors du chargement des réservations.');
      }
    } finally {
      if (!silent) {
        setLoadingReservations(false);
      }
    }
  }, [day]);

  /** Affiche tout de suite une réservation créée (réponse POST) avant le GET /reservations. */
  const upsertReservation = useCallback((booking) => {
    if (!booking || booking.id == null) return;
    setReservations((prev) => {
      const list = Array.isArray(prev) ? prev : [];
      const id = Number(booking.id);
      const idx = list.findIndex((r) => Number(r.id) === id);
      if (idx >= 0) {
        const next = [...list];
        next[idx] = { ...next[idx], ...booking };
        return next;
      }
      return [booking, ...list];
    });
  }, []);

  const loadDriver = useCallback(async () => {
    try {
      setLoadingDriver(true);
      const data = await fetchCompanyDriversCanonical();
      // Le service renvoie déjà un ARRAY normalisé
      setDriver(Array.isArray(data) ? data : (data?.driver ?? []));
      setError(null); // Réinitialiser l'erreur en cas de succès
    } catch (err) {
      // Gérer spécifiquement les erreurs de timeout
      if (err.code === 'ECONNABORTED' || err.message?.includes('timeout')) {
        setError('La récupération des chauffeurs a pris trop de temps. Veuillez réessayer.');
      } else {
        // Ne pas logger les erreurs 403/404/401 comme des erreurs critiques
        const status = err?.response?.status;
        if (status !== 403 && status !== 404 && status !== 401) {
          console.error('❌ Erreur lors du chargement des chauffeurs :', err);
        }
        setError('Erreur lors du chargement des chauffeurs.');
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

  // Room entreprise + positions temps réel (même logique que useDriver) pour la carte dashboard / dispatch
  useEffect(() => {
    const cid = company?.id;
    if (cid && Number(cid) > 0) {
      joinCompanyRoom(Number(cid)).catch(() => {});
    }
  }, [company?.id]);

  useEffect(() => {
    const socket = getCompanySocket();
    if (!socket) return;

    const applyDelta = (payload, fromLiveState = false) => {
      setDriver((prev) =>
        mergeOrUpdateDriverInList(prev, payload, fromLiveState, company?.id ?? null)
      );
    };

    const onLiveState = (payload) => applyDelta(payload, true);
    const onLocationUpdate = (payload) => applyDelta(payload, false);
    const onReconnected = () => {
      loadDriver().catch(() => {});
    };

    socket.on('driver_live_state_update', onLiveState);
    socket.on('driver_location_update', onLocationUpdate);
    if (typeof window !== 'undefined') {
      window.addEventListener('company_socket_reconnected', onReconnected);
    }

    return () => {
      socket.off('driver_live_state_update', onLiveState);
      socket.off('driver_location_update', onLocationUpdate);
      if (typeof window !== 'undefined') {
        window.removeEventListener('company_socket_reconnected', onReconnected);
      }
    };
  }, [loadDriver, company?.id]);

  return {
    company,
    reservations,
    driver,
    loadingCompany,
    loadingReservations,
    loadingDriver,
    error,
    reloadCompany: loadCompany,
    reloadReservations: loadReservations,
    reloadDriver: loadDriver,
    upsertReservation,
  };
};

export default useCompanyData;
