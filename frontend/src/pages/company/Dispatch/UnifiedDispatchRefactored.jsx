// frontend/src/pages/company/Dispatch/UnifiedDispatchRefactored.jsx
/**
 * 📊 PAGE UNIFIÉE : DISPATCH & PLANIFICATION (Version refactorisée)
 *
 * S'adapte automatiquement selon le mode configuré :
 * - MANUAL : Interface simple pour assignation manuelle
 * - SEMI_AUTO : Interface avec suggestions à valider
 * - FULLY_AUTO : Interface de surveillance avec journal d'activité
 */

import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { Toaster } from 'sonner';
import CompanyHeader from '../../../components/layout/Header/CompanyHeader';
import CompanySidebar from '../../../components/layout/Sidebar/CompanySidebar/CompanySidebar';
import useCompanySocket from '../../../hooks/useCompanySocket';
import useDispatchStatus from '../../../hooks/useDispatchStatus';
import useCompanyData from '../../../hooks/useCompanyData';
import useCompanyAuthToken from '../../../hooks/useCompanyAuthToken';

// Hooks personnalisés
import { useDispatchData } from '../../../hooks/useDispatchData';
import { useLiveDelays } from '../../../hooks/useLiveDelays';
import { useDispatchMode } from '../../../hooks/useDispatchMode';
import { useAssignmentActions } from '../../../hooks/useAssignmentActions';

// Services
import {
  runDispatchForDay,
  dispatchNowForReservation,
  triggerReturnBooking,
  scheduleReservation,
  fetchDispatchStatus,
  updateReservation,
} from '../../../services/companyService';
import {
  getOptimizerStatus,
  startRealTimeOptimizer,
  stopRealTimeOptimizer,
  applySuggestion,
} from '../../../services/dispatchMonitoringService';
import { showSuccess, showError } from '../../../utils/toast';
import { toast } from 'sonner';

// Composants
import DispatchHeader from './components/DispatchHeader';
import ManualModePanel from './components/ManualModePanel';
import SemiAutoPanel from './components/SemiAutoPanel';
import FullyAutoPanel from './components/FullyAutoPanel';
import AdvancedSettings from './components/AdvancedSettings';
import ReservationModals from '../../../components/reservations/ReservationModals';
import DispatchProgress from './components/DispatchProgress';
import ReservationDetailPanel from '../../company/Reservations/components/ReservationDetailPanel';
import ChatWidget from '../../../components/widgets/ChatWidget';

// Import dynamique des styles par mode
import commonStyles from './modes/Common.module.css';
import manualStyles from './modes/Manual.module.css';
import semiAutoStyles from './modes/SemiAuto.module.css';
import fullyAutoStyles from './modes/FullyAuto.module.css';

// Fonction pour fusionner les styles selon le mode actif
const getModeStyles = (mode) => {
  const modeSpecificStyles = {
    manual: manualStyles,
    semi_auto: semiAutoStyles,
    fully_auto: fullyAutoStyles,
  };

  // Fusionner les styles communs avec les styles spécifiques au mode
  return { ...commonStyles, ...(modeSpecificStyles[mode] || semiAutoStyles) };
};

// Helpers
const makeToday = () => {
  const d = new Date();
  const pad = (n) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
};

const UnifiedDispatchRefactored = () => {
  // Récupérer les données de l'entreprise et les chauffeurs
  const { company, driver: driversList } = useCompanyData();

  // État principal
  const [date, setDate] = useState(makeToday());
  const [regularFirst, setRegularFirst] = useState(true);
  const [allowEmergency, setAllowEmergency] = useState(true);

  // 🆕 État pour overrides (chargé depuis DB au montage)
  const [overrides, setOverrides] = useState(null);
  const [showAdvancedSettings, setShowAdvancedSettings] = useState(false);
  // ⚡ Mode dispatch rapide (< 1 minute)
  const [fastMode, setFastMode] = useState(false);
  const [_loadingOverrides, setLoadingOverrides] = useState(true);

  // Panel lateral de details
  const [selectedDispatch, setSelectedDispatch] = useState(null);

  // ✅ Fonction pour charger les paramètres avancés (définie en dehors du useEffect pour être réutilisable)
  const loadAdvancedSettings = React.useCallback(async () => {
    console.log('🔍 [Dispatch] Début chargement paramètres avancés...');
    try {
      const apiClient = (await import('../../../utils/apiClient')).default;
      console.log('✅ [Dispatch] apiClient chargé, appel API en cours...');
      const { data } = await apiClient.get('/company_dispatch/advanced_settings');
      console.log('📦 [Dispatch] Réponse API reçue:', data);

      if (data.dispatch_overrides) {
        setOverrides(data.dispatch_overrides);
        console.log(
          '🔄 [Dispatch] Paramètres avancés chargés depuis la DB:',
          data.dispatch_overrides
        );
      } else {
        setOverrides(null); // ✅ Réinitialiser si pas de paramètres
        console.log('📌 [Dispatch] Aucun paramètre avancé configuré (utilise valeurs par défaut)');
      }
    } catch (err) {
      console.error('❌ [Dispatch] Erreur chargement paramètres avancés:', err);
      console.error('❌ [Dispatch] Détails erreur:', err.response?.status, err.response?.data);
    } finally {
      setLoadingOverrides(false);
      console.log('✅ [Dispatch] Chargement paramètres terminé');
    }
  }, []);

  // États pour les modals (conservé pour compatibilité, mais maintenant géré par assignModal)
  // const [selectedReservationForAssignment, setSelectedReservationForAssignment] = useState(null);
  const [scheduleModalOpen, setScheduleModalOpen] = useState(false);
  const [scheduleModalReservation, setScheduleModalReservation] = useState(null);
  const [assignModalOpen, setAssignModalOpen] = useState(false);
  const [assignModalReservation, setAssignModalReservation] = useState(null);
  const [deleteModalOpen, setDeleteModalOpen] = useState(false);
  const [deleteModalReservation, setDeleteModalReservation] = useState(null);

  // État pour le tri (Mode Manuel) - defaut: priorite metier
  const [sortBy, setSortBy] = useState('priority'); // 'priority', 'time', 'client', 'status'
  const [sortOrder, setSortOrder] = useState('asc'); // 'asc', 'desc'

  // États UI
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [dispatchSuccess, setDispatchSuccess] = useState(null);

  // États de progression du dispatch
  const [dispatchProgressState, setDispatchProgressState] = useState({
    isActive: false,
    status: 'idle', // 'idle' | 'queued' | 'processing' | 'completed' | 'failed'
    dispatchRunId: null,
    startTime: null,
    assignmentsCount: null,
  });

  // Rôle utilisateur : delays/live réservé COMPANY/ADMIN — lecture company_user uniquement (évite 403 si token DRIVER)
  const { user, isCompanyAuthReady } = useCompanyAuthToken();
  const isCompanyOrAdmin = user && (user.isCompany || String(user.role || '').toLowerCase() === 'admin');

  // Hooks personnalisés
  const { dispatchMode, loadDispatchMode } = useDispatchMode();
  const {
    dispatches: allDispatches,
    loading: dispatchesLoading,
    error: dispatchesError,
    loadDispatches,
    setDispatches, // CT2: destructure pour optimistic update V13
  } = useDispatchData(date, dispatchMode);
  const { delays, summary: _summary, loadDelays } = useLiveDelays(date, isCompanyAuthReady && !!isCompanyOrAdmin);

  // 🆕 Ref pour compter les assignations réelles (mis à jour après chargement)
  // Utiliser une ref plutôt qu'un état pour éviter les re-renders inutiles
  const realAssignedCountRef = useRef(0);

  // 🆕 Ref pour tracker les dispatch_run_id complétés via WebSocket
  // Permet d'éviter de lancer le polling si le WebSocket a déjà signalé la completion
  const completedDispatchRunsRef = useRef(new Set());

  // 🆕 Ref pour tracker les dates pour lesquelles un dispatch a été complété récemment
  // (utilisé si les IDs ne correspondent pas entre HTTP et WebSocket)
  const completedDispatchDatesRef = useRef(new Map()); // Map<date, timestamp>

  const {
    handleAssignDriver,
    handleDeleteReservation,
    loading: _actionsLoading,
    error: actionsError,
    success: actionsSuccess,
  } = useAssignmentActions();

  // 🆕 Filtrer les courses CANCELED (ne pas les afficher dans le tableau)
  const dispatches = useMemo(() => {
    const filtered = (allDispatches || []).filter((d) => d.status !== 'canceled');

    // 🆕 Mettre à jour le compteur réel d'assignations dans la ref
    const assigned = filtered.filter((d) => d.driver?.id || d.driver_id).length;
    realAssignedCountRef.current = assigned;

    return filtered;
  }, [allDispatches]);

  // V11: Cle unifiee pour eviter bug booking_id vs id
  const getDispatchKey = useCallback((d) => d.booking_id ?? d.id, []);

  // V12: Enrichir drivers avec charge O(N)
  const driversWithLoad = useMemo(() => {
    const loadByDriverId = {};
    (dispatches || []).forEach((d) => {
      const dId = d.driver_id || d.driver?.id;
      if (dId) loadByDriverId[dId] = (loadByDriverId[dId] || 0) + 1;
    });
    return (driversList || [])
      .filter((d) => d.is_active)
      .map((driver) => ({
        ...driver,
        courseCount: loadByDriverId[driver.id] || 0,
      }));
  }, [driversList, dispatches]);

  // V11: Pre-calculer delayMap avec cle normalisee
  const delayMap = useMemo(() => {
    const map = {};
    (delays || []).forEach((d) => {
      const key = d.booking_id ?? d.id;
      map[key] = {
        minutes: Math.round(d.delay_minutes || d.pickup_delay_minutes || 0),
        severity: d.delay_severity || 'reasonable',
      };
    });
    return map;
  }, [delays]);

  // V9 + V13: Handler assignation directe (optimistic UI, WebSocket fait le refresh)
  const handleAssignDirect = useCallback(
    async (reservationId, driverId) => {
      const targetDriver = driversWithLoad.find((dr) => dr.id === driverId);
      setDispatches((prev) =>
        prev.map((d) =>
          getDispatchKey(d) === reservationId
            ? {
                ...d,
                driver_id: driverId,
                driver: targetDriver,
                status: 'assigned',
              }
            : d
        )
      );
      const success = await handleAssignDriver(reservationId, driverId);
      if (!success) {
        // V9: rollback optimistic — toast generique (pas de code HTTP dispo CT3)
        setDispatches((prev) =>
          prev.map((d) =>
            getDispatchKey(d) === reservationId
              ? { ...d, driver_id: null, driver: null, status: 'accepted' }
              : d
          )
        );
      }
      return success;
    },
    [handleAssignDriver, driversWithLoad, setDispatches, getDispatchKey]
  );

  // États pour l'optimiseur
  const [optimizerStatus, setOptimizerStatus] = useState(null);

  // ✅ Styles dynamiques selon le mode actif (avec fallback si mode pas encore chargé)
  const styles = getModeStyles(dispatchMode || 'semi_auto');

  // WebSocket pour temps réel
  const socket = useCompanySocket();
  const {
    label: dispatchLabel,
    progress: dispatchProgress,
    isRunning: isDispatching,
  } = useDispatchStatus(socket);

  // Charger le statut de l'optimiseur
  const loadOptimizerStatus = useCallback(async () => {
    // ✅ En développement, ne pas appeler l'optimizer (évite les erreurs 500)
    const isDevelopment = process.env.NODE_ENV === 'development';
    if (isDevelopment) {
      setOptimizerStatus(null);
      return;
    }
    
    try {
      const status = await getOptimizerStatus();
      if (status) {
        setOptimizerStatus(status);
      }
      // Si status est null (401 avec refresh réussi), ne pas définir d'erreur
    } catch (err) {
      // ⚡ Ignorer les erreurs 401 si le refresh est en cours ou réussi
      if (err?.response?.status === 401 && err?.config?._retryAfterRefresh) {
        // Refresh réussi, ne pas logger l'erreur
        return;
      }

      // Ne logger que les vraies erreurs (pas les 401 en cours de refresh)
      if (err?.response?.status !== 401) {
        console.error('[UnifiedDispatch] Error loading optimizer:', err);
      } else {
        console.debug('[UnifiedDispatch] 401 error, refresh token will be attempted');
      }
    }
  }, []);

  // Gérer la suppression d'une réservation (fonction interne)
  const _handleDeleteReservation = async (reservationIdOrObject) => {
    const reservationId =
      typeof reservationIdOrObject === 'object' ? reservationIdOrObject.id : reservationIdOrObject;

    try {
      await handleDeleteReservation(reservationId);
      loadDispatches();
    } catch (err) {
      console.error('Erreur suppression réservation:', err);
    }
  };

  // Gérer la planification de l'heure de retour
  const onScheduleReservation = (reservation) => {
    // Passer l'objet complet (pas juste l'ID)
    const resObj =
      typeof reservation === 'object' ? reservation : dispatches.find((r) => r.id === reservation);
    if (!resObj) return;
    setScheduleModalReservation(resObj);
    setScheduleModalOpen(true);
  };

  // Confirme l'heure du retour OU marque en 'Urgent +15 min'
  const handleConfirmReturnTime = async (data) => {
    setScheduleModalOpen(false);
    if (!scheduleModalReservation) return;

    const reservationId = scheduleModalReservation?.id ?? scheduleModalReservation;
    try {
      let payload = {};
      if (data?.urgent) {
        payload = { urgent: true, minutes_offset: data.minutes_offset ?? 15 };
        await triggerReturnBooking(reservationId, payload);
      } else if (typeof data === 'string') {
        // Format "YYYY-MM-DD HH:mm" pour scheduleReservation
        await scheduleReservation(reservationId, data);
      } else if (data?.return_time) {
        payload = { return_time: data.return_time };
        await triggerReturnBooking(reservationId, payload);
      }

      setScheduleModalReservation(null);
      loadDispatches();
      showSuccess('Heure de retour planifiée avec succès');
    } catch (err) {
      console.error('Erreur lors de la planification:', err);
      setScheduleModalReservation(null);
      showError(err?.response?.data?.error || 'Erreur lors de la planification');
    }
  };

  // Dispatch urgent: +15 min
  const onDispatchNow = async (reservation) => {
    const id = reservation?.id ?? reservation;
    if (!id) return;
    try {
      await dispatchNowForReservation(id, 15);
      loadDispatches();
      toast.success('Dispatch urgent déclenché avec succès');
    } catch (e) {
      // ⚡ Gestion améliorée : extraire le message du backend (identique à Dashboard)
      const errorData = e?.response?.data;
      const errorMessage = errorData?.message || errorData?.error;
      const status = e?.response?.status;

      // 🔍 Debug pour comprendre pourquoi la détection ne fonctionne pas
      console.debug('[DispatchNow] Error status:', status);
      console.debug('[DispatchNow] Error data:', errorData);
      console.debug('[DispatchNow] Error message:', errorMessage);

      // ⚡ Si c'est un retour avec aller non complété (400), c'est un comportement attendu
      // Vérifier dans error ET message pour détecter les retours
      const errorLower = (errorMessage || '').toLowerCase();
      const errorErrorLower = (errorData?.error || '').toLowerCase();
      const isReturnNotReady =
        status === 400 &&
        (errorLower.includes('retour') ||
          errorLower.includes('aller') ||
          errorErrorLower.includes('retour') ||
          errorErrorLower.includes('aller'));

      console.debug('[DispatchNow] isReturnNotReady:', isReturnNotReady);

      if (isReturnNotReady) {
        // ⚡ Message informatif (warning) au lieu d'une erreur
        // Utiliser le message détaillé du backend s'il existe (message contient plus de détails)
        const detailMessage =
          errorData?.message ||
          errorData?.error ||
          "Impossible de déclencher un retour d'urgence. La course aller doit être complétée avant de déclencher le retour.";

        console.debug('[DispatchNow] Showing warning:', detailMessage);
        toast.warning(detailMessage, {
          duration: 5000, // ⚡ Plus long pour que l'utilisateur puisse lire
        });
        // ⚡ Ne pas logger comme erreur car c'est un comportement attendu (utiliser debug)
        console.debug('Dispatch urgent refusé (comportement attendu):', detailMessage);
      } else {
        // ⚡ Vraie erreur : afficher et logger
        console.error('Dispatch urgent:', e);
        toast.error(errorMessage || 'Erreur lors du dispatch urgent.');
      }
    }
  };

  // Gère la suppression (ouvre la modale de confirmation)
  const onDeleteReservationClick = (reservation) => {
    // Passer l'objet complet
    const resObj =
      typeof reservation === 'object' ? reservation : dispatches.find((r) => r.id === reservation);
    if (!resObj) return;
    setDeleteModalReservation(resObj);
    setDeleteModalOpen(true);
  };

  // Confirme la suppression / annulation
  const handleConfirmDelete = async (reservationId, reasonCode, reasonText) => {
    if (!reservationId && !deleteModalReservation) return;

    const id = reservationId || deleteModalReservation?.id || deleteModalReservation;
    await handleDeleteReservation(id, reasonCode, reasonText);
    setDeleteModalOpen(false);
    setDeleteModalReservation(null);
    loadDispatches();
  };

  // Gérer l'assignation (ouvre la modale)
  const onAssignReservation = (reservation) => {
    // Passer l'objet complet
    const resObj =
      typeof reservation === 'object' ? reservation : dispatches.find((r) => r.id === reservation);
    if (!resObj) return;
    setAssignModalReservation(resObj);
    setAssignModalOpen(true);
  };

  // Confirme l'assignation de chauffeur
  const handleConfirmAssign = async (reservationId, driverId) => {
    const success = await handleAssignDriver(reservationId, driverId);
    if (success) {
      setAssignModalOpen(false);
      setAssignModalReservation(null);
      loadDispatches();
    }
  };

  // Lancer le dispatch
  const onRunDispatch = async () => {
    try {
      setDispatchSuccess(null);

      // ✅ FORCER allow_emergency selon overrides
      const finalAllowEmergency =
        overrides?.allow_emergency !== undefined ? overrides.allow_emergency : allowEmergency;

      // 🆕 Calculer le nombre de courses une seule fois
      const bookingCount = dispatches?.length || 0;

      // ⚡ Mode rapide : optimiser pour garantir < 1 minute
      let finalMode = dispatchMode;
      let finalRunAsync = true;
      let finalOverrides = overrides || {};

      if (fastMode) {
        // ⚡ Mode rapide : forcer heuristic_only et mode sync pour rapidité maximale
        finalMode = 'heuristic_only';
        finalRunAsync = false; // Mode sync = immédiat, pas de file d'attente
        // ⚡ Overrides optimisés pour vitesse
        finalOverrides = {
          ...finalOverrides,
          reset_existing: true, // Supprimer toutes les assignations existantes pour redispatch complet
          fast_mode: true, // Flag pour le backend
          mode: 'heuristic_only', // Forcer heuristic_only
          solver: {
            ...finalOverrides.solver,
            time_limit_sec: 10, // Limiter solver à 10s max (au cas où)
          },
          features: {
            ...finalOverrides.features,
            enable_solver: false, // Désactiver solver
            enable_rl_optimizer: false, // Désactiver RL
            enable_parallel_heuristics: true, // Activer parallélisme
          },
        };
        console.log('⚡ [Dispatch] Mode RAPIDE activé - garantit < 1 minute');
      } else {
        // Mode normal : optimisation automatique selon taille
        const bookingCount = dispatches?.length || 0;
        const isSmallDispatch = bookingCount <= 3;
        const isMediumDispatch = bookingCount <= 20 && bookingCount > 3;
        const shouldUseSync = isSmallDispatch;
        finalMode = isSmallDispatch || isMediumDispatch ? 'heuristic_only' : dispatchMode;
        finalRunAsync = !shouldUseSync;
        // ⚡ Toujours supprimer les assignations existantes pour redispatch complet
        finalOverrides = {
          ...finalOverrides,
          reset_existing: true, // Supprimer toutes les assignations existantes pour redispatch complet
        };
      }

      console.log('🚀 [Dispatch] Lancement avec paramètres:', {
        date,
        regularFirst,
        allowEmergency: finalAllowEmergency,
        mode: dispatchMode,
        finalMode,
        fastMode,
        bookingCount: dispatches?.length || 0,
        runAsync: finalRunAsync,
        overrides: finalOverrides,
        hasOverrides: !!finalOverrides && Object.keys(finalOverrides).length > 0,
      });

      const result = await runDispatchForDay({
        forDate: date,
        regularFirst: regularFirst,
        allowEmergency: finalAllowEmergency,
        mode: finalMode,
        runAsync: finalRunAsync,
        overrides: finalOverrides,
      });

      console.log('✅ [Dispatch] Résultat reçu:', result);

      // 🆕 Activer l'indicateur de progression
      const dispatchRunId = result?.id || result?.dispatch_run_id || null;
      const isAsync = result?.status === 'queued' || dispatchRunId;

      if (isAsync) {
        // Mode async : activer la progression
        setDispatchProgressState({
          isActive: true,
          status: result?.status === 'queued' ? 'queued' : 'processing',
          dispatchRunId: dispatchRunId,
          startTime: Date.now(),
          assignmentsCount: null,
        });
      } else {
        // Mode sync : progression rapide
        setDispatchProgressState({
          isActive: true,
          status: 'processing',
          dispatchRunId: null,
          startTime: Date.now(),
          assignmentsCount: null,
        });
      }

      // ⚡ Si le dispatch est en file d'attente (async), attendre le WebSocket avec fallback polling
      // Sinon (sync), rafraîchir immédiatement
      // 🆕 OPTIMISATION : Le polling n'est nécessaire QUE pour les dispatchs complexes (plusieurs courses)
      // Pour les petits dispatchs (≤3 courses), on utilise déjà le mode sync, donc pas de polling
      const isComplexDispatch = bookingCount > 3; // Plus de 3 courses = dispatch complexe nécessitant optimisation

      if (isAsync && isComplexDispatch) {
        // 🆕 Vérifier si le dispatch_run_id a déjà été complété via WebSocket
        // (peut arriver si le WebSocket répond avant que la requête HTTP timeout)
        const alreadyCompletedById =
          dispatchRunId && completedDispatchRunsRef.current.has(dispatchRunId);

        // 🆕 Vérifier aussi si on a reçu un WebSocket completion pour cette date récemment (dans les 60 secondes)
        // (utile si les IDs ne correspondent pas entre HTTP et WebSocket)
        const lastCompletionTime = completedDispatchDatesRef.current.get(date);
        const recentlyCompletedByDate =
          lastCompletionTime && Date.now() - lastCompletionTime < 60000;

        if (alreadyCompletedById || recentlyCompletedByDate) {
          console.log(
            `✅ [Dispatch] Dispatch déjà complété via WebSocket (ID: ${
              alreadyCompletedById ? dispatchRunId : 'N/A'
            }, Date: ${date}), pas de polling nécessaire`
          );
          // Le WebSocket a déjà géré la completion, pas besoin de polling
          return;
        }

        // ✅ Si dispatch_run_id est null, essayer de l'obtenir via /status
        let finalDispatchRunId = dispatchRunId;
        if (!finalDispatchRunId) {
          try {
            console.log(
              `[Dispatch] dispatch_run_id manquant, récupération via /status pour date=${date}`
            );
            const statusData = await fetchDispatchStatus(date);
            finalDispatchRunId = statusData.dispatch_run_id || statusData.active_dispatch_run?.id;
            if (finalDispatchRunId) {
              console.log(`[Dispatch] dispatch_run_id obtenu via /status: ${finalDispatchRunId}`);
              // Mettre à jour l'état avec le dispatch_run_id obtenu
              setDispatchProgressState((prev) => ({
                ...prev,
                dispatchRunId: finalDispatchRunId,
              }));
            }
          } catch (statusError) {
            console.warn('[Dispatch] Erreur lors de la récupération du statut:', statusError);
          }
        }

        console.log(
          `⏳ [Dispatch] Dispatch complexe (${bookingCount} courses) en file d'attente, dispatch_run_id=${finalDispatchRunId}, attente du WebSocket dispatch_run_completed...`
        );

        // ⚡ Fallback : Polling toutes les 3 secondes pendant max 2-3 minutes si WebSocket ne répond pas
        // Un dispatch complexe ne devrait normalement pas prendre plus de 1-2 minutes
        let pollAttempts = 0;
        const maxAttempts = 60; // 60 * 3s = 3 minutes max (plus raisonnable pour un dispatch)
        const pollInterval = setInterval(async () => {
          // 🆕 Vérifier si le dispatch a été complété pendant le polling (par ID ou par date)
          const completedById =
            finalDispatchRunId && completedDispatchRunsRef.current.has(finalDispatchRunId);
          const lastCompletionTime = completedDispatchDatesRef.current.get(date);
          const completedByDate = lastCompletionTime && Date.now() - lastCompletionTime < 60000;

          if (completedById || completedByDate) {
            clearInterval(pollInterval);
            window._dispatchPollInterval = null;
            console.log(
              `✅ [Dispatch] Polling arrêté : dispatch complété via WebSocket (ID: ${
                completedById ? finalDispatchRunId : 'N/A'
              }, Date: ${date})`
            );
            return;
          }

          pollAttempts++;
          const elapsedSeconds = pollAttempts * 3;
          console.log(
            `🔄 [Dispatch] Polling #${pollAttempts}/${maxAttempts} (${elapsedSeconds}s écoulées)...`
          );

          // ✅ Améliorer le polling : utiliser /status pour obtenir le statut réel
          try {
            const statusData = await fetchDispatchStatus(date);

            // ✅ Vérifier le statut via active_dispatch_run
            if (statusData.active_dispatch_run) {
              const activeStatus = statusData.active_dispatch_run.status;
              const activeAssignmentsCount = statusData.active_dispatch_run.assignments_count || 0;

              console.log(
                `[Dispatch] Status check: dispatch_run_id=${statusData.active_dispatch_run.id}, status=${activeStatus}, assignments=${activeAssignmentsCount}`
              );

              // Si le dispatch est terminé (COMPLETED ou FAILED)
              if (activeStatus === 'COMPLETED' || activeStatus === 'FAILED') {
                clearInterval(pollInterval);
                window._dispatchPollInterval = null;

                if (activeStatus === 'COMPLETED') {
                  console.log(
                    `✅ [Dispatch] Polling arrêté : dispatch complété (status=${activeStatus}, assignments=${activeAssignmentsCount})`
                  );

                  // Mettre à jour l'état de progression
                  setDispatchProgressState((prev) => ({
                    ...prev,
                    status: 'completed',
                    assignmentsCount: activeAssignmentsCount,
                  }));

                  setDispatchSuccess(
                    `✅ Dispatch terminé ! ${activeAssignmentsCount} course(s) assignée(s)`
                  );
                  setTimeout(() => setDispatchSuccess(null), 3000);
                } else {
                  console.warn(
                    `⚠️ [Dispatch] Polling arrêté : dispatch échoué (status=${activeStatus})`
                  );
                  setDispatchProgressState((prev) => ({
                    ...prev,
                    status: 'failed',
                  }));
                  toast.error('Le dispatch a échoué. Veuillez vérifier les logs backend.');
                }

                // Recharger les données
                await loadDispatches();
                loadDelays();

                // Masquer après 5 secondes
                setTimeout(() => {
                  setDispatchProgressState((prev) => ({
                    ...prev,
                    isActive: false,
                    status: 'idle',
                  }));
                }, 5000);
                return;
              }
            }

            // ✅ Vérifier aussi via compteurs d'assignments
            const statusAssignmentsCount = statusData.counters?.assignments || 0;
            if (pollAttempts > 5 && statusAssignmentsCount > 0) {
              // Après 15 secondes de polling, si on a des assignations, on considère que c'est terminé
              console.log(
                `✅ [Dispatch] Polling arrêté : ${statusAssignmentsCount} assignation(s) détectée(s) via /status, dispatch probablement terminé`
              );
              clearInterval(pollInterval);
              window._dispatchPollInterval = null;

              // Mettre à jour l'état de progression
              setDispatchProgressState((prev) => ({
                ...prev,
                status: 'completed',
                assignmentsCount: statusAssignmentsCount,
              }));

              setDispatchSuccess(
                `✅ Dispatch terminé ! ${statusAssignmentsCount} course(s) assignée(s)`
              );
              setTimeout(() => setDispatchSuccess(null), 3000);

              // Recharger les données
              await loadDispatches();
              loadDelays();

              // Masquer après 5 secondes
              setTimeout(() => {
                setDispatchProgressState((prev) => ({
                  ...prev,
                  isActive: false,
                  status: 'idle',
                }));
              }, 5000);
              return;
            }
          } catch (statusError) {
            console.warn('[Dispatch] Erreur lors de la vérification du statut:', statusError);
            // Continuer avec le polling normal en fallback
          }

          // 🆕 Charger les données pour vérifier si le dispatch est terminé (fallback)
          await loadDispatches();
          loadDelays();

          // 🆕 Vérifier si le dispatch est terminé en comptant les assignations (fallback)
          const currentAssignedCount = realAssignedCountRef.current;
          if (pollAttempts > 5 && currentAssignedCount > 0) {
            // Après 15 secondes de polling, si on a des assignations, on considère que c'est terminé
            console.log(
              `✅ [Dispatch] Polling arrêté : ${currentAssignedCount} assignation(s) détectée(s), dispatch probablement terminé`
            );
            clearInterval(pollInterval);
            window._dispatchPollInterval = null;

            // Mettre à jour l'état de progression
            setDispatchProgressState((prev) => ({
              ...prev,
              status: 'completed',
              assignmentsCount: currentAssignedCount,
            }));

            setDispatchSuccess(
              `✅ Dispatch terminé ! ${currentAssignedCount} course(s) assignée(s)`
            );
            setTimeout(() => setDispatchSuccess(null), 3000);

            // Masquer après 5 secondes
            setTimeout(() => {
              setDispatchProgressState((prev) => ({
                ...prev,
                isActive: false,
                status: 'idle',
              }));
            }, 5000);
            return;
          }

          // Arrêter le polling si on atteint le timeout (3 minutes)
          if (pollAttempts >= maxAttempts) {
            clearInterval(pollInterval);
            window._dispatchPollInterval = null;
            console.warn('⚠️ [Dispatch] Polling timeout atteint (3 minutes), arrêt du polling');
            toast.error(
              'Le dispatch prend trop de temps (timeout 3 minutes). Veuillez vérifier les logs backend ou rafraîchir manuellement.'
            );

            // 🆕 Marquer comme échoué dans l'indicateur de progression
            setDispatchProgressState((prev) => ({
              ...prev,
              status: 'failed',
            }));

            // Masquer après 5 secondes
            setTimeout(() => {
              setDispatchProgressState((prev) => ({
                ...prev,
                isActive: false,
                status: 'idle',
              }));
            }, 5000);
          }
        }, 3000); // 3 secondes entre chaque poll

        // Nettoyer le polling si le WebSocket répond (géré dans handleDispatchComplete)
        // On stocke l'interval dans window pour pouvoir le nettoyer
        window._dispatchPollInterval = pollInterval;
      } else {
        // Mode sync : rafraîchir immédiatement et marquer comme complété
        console.log('🔄 [Dispatch] Rafraîchissement immédiat (mode sync)...');
        setTimeout(() => {
          loadDispatches();
          loadDelays();

          // 🆕 Marquer comme complété pour sync
          setDispatchProgressState({
            isActive: true,
            status: 'completed',
            dispatchRunId: null,
            startTime: Date.now(),
            assignmentsCount: result?.assignments_count || dispatches?.length || 0,
          });

          // Masquer après 5 secondes
          setTimeout(() => {
            setDispatchProgressState((prev) => ({
              ...prev,
              isActive: false,
              status: 'idle',
            }));
          }, 5000);
        }, 1000);
      }

      // ✅ Vérifier s'il y a des erreurs de validation
      if (result?.validation?.has_errors) {
        const errors = result.validation.errors || [];
        const warnings = result.validation.warnings || [];

        // Afficher message détaillé
        let message = '⚠️ Dispatch créé avec des conflits temporels !\n\n';

        if (errors.length > 0) {
          message += '🔴 ERREURS CRITIQUES :\n';
          errors.forEach((err, idx) => {
            message += `  ${idx + 1}. ${err}\n`;
          });
        }

        if (warnings.length > 0) {
          message += '\n⚠️ AVERTISSEMENTS :\n';
          warnings.forEach((warn, idx) => {
            message += `  ${idx + 1}. ${warn}\n`;
          });
        }

        message += '\n💡 Vérifiez les assignations et réassignez manuellement si nécessaire.';

        showError(message);
        setDispatchSuccess(null);
      } else if (result?.validation?.warnings) {
        // Warnings seulement (pas d'erreurs)
        showSuccess(
          '🚀 Dispatch lancé avec succès !\n⚠️ Quelques avertissements détectés (voir logs)'
        );
        setDispatchSuccess('Dispatch lancé avec avertissements');
        setTimeout(() => setDispatchSuccess(null), 3000);
      } else {
        // Succès complet sans problème
        showSuccess('🚀 Dispatch lancé avec succès !');
        setDispatchSuccess('Dispatch lancé avec succès');
        setTimeout(() => setDispatchSuccess(null), 3000);
      }
    } catch (err) {
      console.error('[UnifiedDispatch] Error running dispatch:', err);
      showError('Erreur lors du lancement du dispatch');

      // 🆕 Marquer comme échoué dans l'indicateur de progression
      setDispatchProgressState((prev) => ({
        ...prev,
        status: 'failed',
      }));

      // Masquer après 5 secondes
      setTimeout(() => {
        setDispatchProgressState((prev) => ({
          ...prev,
          isActive: false,
          status: 'idle',
        }));
      }, 5000);
    }
  };

  // Charger les paramètres avancés depuis la DB au montage
  useEffect(() => {
    loadAdvancedSettings();
  }, [loadAdvancedSettings]);

  // ✅ Recharger les paramètres quand la modal est ouverte pour avoir les dernières valeurs
  useEffect(() => {
    if (showAdvancedSettings) {
      loadAdvancedSettings();
    }
  }, [showAdvancedSettings, loadAdvancedSettings]);

  // 🆕 Handler pour appliquer overrides (sauvegarde permanente dans la DB)
  const handleApplyOverrides = async (newOverrides) => {
    console.log('🎯 [Overrides] Sauvegarde paramètres avancés:', newOverrides);
    try {
      const apiClient = (await import('../../../utils/apiClient')).default;
      const { data } = await apiClient.put('/company_dispatch/advanced_settings', {
        dispatch_overrides: newOverrides,
      });
      setOverrides(data.dispatch_overrides);
      setShowAdvancedSettings(false);
      showSuccess('✅ Paramètres avancés sauvegardés avec succès !');
      console.log('💾 [Dispatch] Paramètres avancés sauvegardés:', data.dispatch_overrides);
    } catch (err) {
      console.error('[Dispatch] Erreur sauvegarde paramètres avancés:', err);
      showError('❌ Erreur lors de la sauvegarde des paramètres');
    }
  };

  // Gérer l'optimiseur
  const onStartOptimizer = async () => {
    try {
      await startRealTimeOptimizer();
      showSuccess('✅ Optimiseur démarré avec succès');
      loadOptimizerStatus();
    } catch (err) {
      console.error('[UnifiedDispatch] Error starting optimizer:', err);
      showError("Erreur lors du démarrage de l'optimiseur");
    }
  };

  const onStopOptimizer = async () => {
    try {
      await stopRealTimeOptimizer();
      showSuccess('⏸️ Optimiseur arrêté');
      loadOptimizerStatus();
    } catch (err) {
      console.error('[UnifiedDispatch] Error stopping optimizer:', err);
      showError("Erreur lors de l'arrêt de l'optimiseur");
    }
  };

  // Appliquer une suggestion
  const onApplySuggestion = async (suggestion) => {
    try {
      await applySuggestion(suggestion);
      showSuccess('✅ Suggestion appliquée avec succès');
      loadDispatches();
      loadDelays();
    } catch (err) {
      console.error('[UnifiedDispatch] Error applying suggestion:', err);
      showError("Erreur lors de l'application de la suggestion");
    }
  };

  // Chargement initial
  useEffect(() => {
    loadDispatches();
    loadDelays();
    loadOptimizerStatus();
    loadDispatchMode();
  }, [loadDispatches, loadDelays, loadOptimizerStatus, loadDispatchMode]);

  // Auto-refresh (désactivé en mode fully_auto - l'agent gère tout automatiquement)
  useEffect(() => {
    // ✅ En mode fully_auto, ne pas rafraîchir automatiquement
    // L'agent gère les assignations automatiquement et les mises à jour arrivent via WebSocket
    if (dispatchMode === 'fully_auto') {
      return; // Pas de rafraîchissement automatique en fully_auto
    }

    if (!autoRefresh) return;
    const interval = setInterval(() => {
      loadDelays();
      loadOptimizerStatus();
    }, 30000);
    return () => clearInterval(interval);
  }, [autoRefresh, loadDelays, loadOptimizerStatus, dispatchMode]);

  // Écoute WebSocket
  useEffect(() => {
    if (!socket) return;

    const handleDispatchComplete = (data) => {
      console.log('📨 [Dispatch] Événement dispatch_run_completed reçu:', data);
      const assignmentsCount = data?.assignments_count || 0;
      const dispatchRunId = data?.dispatch_run_id;

      // 🆕 Marquer ce dispatch_run_id comme complété pour éviter le polling inutile
      if (dispatchRunId) {
        completedDispatchRunsRef.current.add(dispatchRunId);
        console.log(`✅ [Dispatch] Dispatch ${dispatchRunId} marqué comme complété`);
      }

      // 🆕 Marquer aussi la date comme complétée (utile si les IDs ne correspondent pas)
      const dispatchDate = data?.date || date;
      if (dispatchDate) {
        completedDispatchDatesRef.current.set(dispatchDate, Date.now());
        console.log(`✅ [Dispatch] Date ${dispatchDate} marquée comme complétée`);
      }

      // ⚡ Arrêter le polling de fallback si actif
      if (window._dispatchPollInterval) {
        clearInterval(window._dispatchPollInterval);
        window._dispatchPollInterval = null;
        console.log('✅ [Dispatch] Polling de fallback arrêté (WebSocket a répondu)');
      }

      // ⚡ Rafraîchir les données après un délai plus long pour s'assurer que le backend a bien commit
      // Le backend peut prendre 1-2 secondes pour persister toutes les assignations
      setTimeout(async () => {
        console.log('🔄 [Dispatch] Rafraîchissement après completion (délai 1.5s)...');
        await loadDispatches();
        await loadDelays();

        // 🆕 Recharger une deuxième fois après un délai supplémentaire pour s'assurer
        // que toutes les assignations sont bien visibles
        setTimeout(async () => {
          console.log('🔄 [Dispatch] Rafraîchissement supplémentaire (délai 3s)...');
          await loadDispatches();

          // Attendre encore un peu pour que l'état dispatches soit mis à jour
          setTimeout(() => {
            // 🆕 Le compteur réel est déjà calculé dans le useMemo de dispatches
            // Utiliser realAssignedCountRef qui est mis à jour automatiquement
            const realCount = realAssignedCountRef.current;
            const finalCount = realCount > 0 ? realCount : assignmentsCount;

            console.log(
              `📊 [Dispatch] Assignations WebSocket: ${assignmentsCount}, Réelles dans les données: ${realCount}, Utilisé: ${finalCount}`
            );

            setDispatchSuccess(`✅ Dispatch terminé ! ${finalCount} course(s) assignée(s)`);
            setTimeout(() => setDispatchSuccess(null), 3000);

            // 🆕 Mettre à jour l'état de progression avec le compteur réel
            setDispatchProgressState((prev) => ({
              ...prev,
              status: 'completed',
              assignmentsCount: finalCount,
            }));
          }, 500); // Délai pour que l'état se mette à jour
        }, 1500); // Délai supplémentaire de 1.5s
      }, 1500); // Premier délai de 1.5s

      // 🆕 Masquer l'indicateur après 5 secondes
      setTimeout(() => {
        setDispatchProgressState((prev) => ({
          ...prev,
          isActive: false,
          status: 'idle',
        }));
      }, 5000);
    };

    const handleDispatchStarted = (data) => {
      console.log('🚀 [Dispatch] Événement dispatch_run_started reçu:', data);

      // 🆕 Mettre à jour le statut de "queued" à "processing"
      setDispatchProgressState((prev) => {
        if (prev.isActive && prev.status === 'queued') {
          return {
            ...prev,
            status: 'processing',
          };
        }
        return prev;
      });
    };

    const handleBookingUpdated = () => {
      // ✅ En mode fully_auto, ne rafraîchir que si nécessaire
      // L'agent gère les assignations automatiquement, pas besoin de rafraîchir à chaque événement
      if (dispatchMode === 'fully_auto') {
        // En fully_auto, on se fie aux événements dispatch_run_completed pour les rafraîchissements
        // Les événements booking_updated individuels ne nécessitent pas de rafraîchissement immédiat
        return;
      }
      loadDispatches();
      loadDelays();
    };

    socket.on('dispatch_run_started', handleDispatchStarted);
    socket.on('dispatch_run_completed', handleDispatchComplete);
    // ✅ FIX: Standardisé sur 'booking_updated' (underscore) pour cohérence avec backend
    socket.on('booking_updated', handleBookingUpdated);
    socket.on('new_booking', handleBookingUpdated);

    return () => {
      socket.off('dispatch_run_started', handleDispatchStarted);
      socket.off('dispatch_run_completed', handleDispatchComplete);
      socket.off('booking_updated', handleBookingUpdated);
      socket.off('new_booking', handleBookingUpdated);
    };
  }, [socket, loadDispatches, loadDelays, date, dispatchMode]);

  // Rendu du panneau selon le mode
  const renderModePanel = () => {
    const commonProps = {
      dispatches: dispatches || [],
      delays: delays || [],
      loading: dispatchesLoading,
      error: dispatchesError,
      styles,
      currentCompanyId: company?.id,
      onRefresh: () => { loadDispatches(); loadDelays(); },
    };

    switch (dispatchMode) {
      case 'manual':
        return (
          <ManualModePanel
            {...commonProps}
            sortBy={sortBy}
            setSortBy={setSortBy}
            sortOrder={sortOrder}
            setSortOrder={setSortOrder}
            selectedReservationForAssignment={assignModalReservation?.id || null}
            setSelectedReservationForAssignment={onAssignReservation}
            onSchedule={onScheduleReservation}
            onDispatchNow={onDispatchNow}
            onDelete={onDeleteReservationClick}
            currentDate={date}
            drivers={driversWithLoad}
            delayMap={delayMap}
            onAssignDirect={handleAssignDirect}
            onRowClick={(r) => setSelectedDispatch(r)}
            onGoToSemiAuto={() => {
              const companyId = window.location.pathname.split('/')[3] || '';
              window.location.href = `/dashboard/company/${companyId}/settings#operations`;
            }}
          />
        );
      case 'semi_auto':
        return (
          <SemiAutoPanel
            {...commonProps}
            onApplySuggestion={onApplySuggestion}
            onDeleteReservation={onDeleteReservationClick}
            onDispatchNow={onDispatchNow}
            onAssign={onAssignReservation} // 🆕 Permettre l'assignation manuelle pour les courses transférées
            currentDate={date}
          />
        );
      case 'fully_auto':
        return (
          <FullyAutoPanel
            {...commonProps}
            optimizerStatus={optimizerStatus}
            onStartOptimizer={onStartOptimizer}
            onStopOptimizer={onStopOptimizer}
            autoRefresh={autoRefresh}
            setAutoRefresh={setAutoRefresh}
            onDispatchNow={onDispatchNow}
          />
        );
      default:
        return <div>Mode non reconnu: {dispatchMode}</div>;
    }
  };

  return (
    <div className={styles.container}>
      {/* Toast notifications provider */}
      <Toaster position="top-right" richColors />

      <CompanyHeader />
      <div className={`${styles.mainContent} ${selectedDispatch ? styles.mainContentWithPanel : ''}`}>
        <CompanySidebar />
        <div className={styles.content}>
          <DispatchHeader
            date={date}
            setDate={setDate}
            regularFirst={regularFirst}
            setRegularFirst={setRegularFirst}
            allowEmergency={allowEmergency}
            setAllowEmergency={setAllowEmergency}
            onRunDispatch={onRunDispatch}
            loading={isDispatching}
            dispatchSuccess={dispatchSuccess}
            dispatchProgress={dispatchProgress}
            dispatchLabel={dispatchLabel}
            dispatchMode={dispatchMode}
            styles={styles}
            onShowAdvancedSettings={() => setShowAdvancedSettings(true)} // 🆕
            hasOverrides={overrides !== null} // 🆕
            fastMode={fastMode} // ⚡ Mode rapide
            setFastMode={setFastMode} // ⚡ Setter mode rapide
          />

          {/* 🆕 Panneau paramètres avancés */}
          {showAdvancedSettings && (
            <div className="modal-overlay" onClick={() => setShowAdvancedSettings(false)}>
              <div className="modal-content" onClick={(e) => e.stopPropagation()}>
                <button className="modal-close" onClick={() => setShowAdvancedSettings(false)}>
                  ✕
                </button>
                <AdvancedSettings
                  key={JSON.stringify(overrides)} // 🆕 Force remount si overrides change
                  onApply={handleApplyOverrides}
                  initialSettings={overrides || {}}
                />
              </div>
            </div>
          )}

          {/* 🆕 Indicateur de progression du dispatch */}
          <DispatchProgress
            isActive={dispatchProgressState.isActive}
            status={dispatchProgressState.status}
            dispatchRunId={dispatchProgressState.dispatchRunId}
            startTime={dispatchProgressState.startTime}
            estimatedDuration={dispatches?.length <= 3 ? 15000 : 60000} // 15s pour petits dispatchs, 60s pour grands
            assignmentsCount={dispatchProgressState.assignmentsCount}
            onClose={() => {
              setDispatchProgressState({
                isActive: false,
                status: 'idle',
                dispatchRunId: null,
                startTime: null,
                assignmentsCount: null,
              });
            }}
          />

          {renderModePanel()}

          {/* Messages d'erreur/succès des actions */}
          {actionsError && <div className={styles.errorMessage}>{actionsError}</div>}
          {actionsSuccess && <div className={styles.successMessage}>{actionsSuccess}</div>}
        </div>

        {/* Panel lateral de details */}
        {selectedDispatch && (
          <aside className={styles.dispatchDetailPanel}>
            <ReservationDetailPanel
              reservation={selectedDispatch}
              onClose={() => setSelectedDispatch(null)}
              onSave={async (id, data) => {
                const payload = { ...data };
                // Recombiner scheduled_date + scheduled_time en ISO 8601
                if (payload.scheduled_date && payload.scheduled_time) {
                  payload.scheduled_time = `${payload.scheduled_date}T${payload.scheduled_time}:00`;
                  delete payload.scheduled_date;
                } else if (payload.scheduled_date && !payload.scheduled_time) {
                  payload.scheduled_time = `${payload.scheduled_date}T00:00:00`;
                  delete payload.scheduled_date;
                } else {
                  delete payload.scheduled_date;
                }
                await updateReservation(id, payload);
                loadDispatches();
              }}
              onDelete={onDeleteReservationClick}
            />
          </aside>
        )}
      </div>

      {/* Modales centralisées */}
      <ReservationModals
        scheduleModalOpen={scheduleModalOpen}
        scheduleModalReservation={scheduleModalReservation}
        onScheduleConfirm={handleConfirmReturnTime}
        onScheduleClose={() => {
          setScheduleModalOpen(false);
          setScheduleModalReservation(null);
        }}
        assignModalOpen={assignModalOpen}
        assignModalReservation={assignModalReservation}
        assignModalDrivers={(driversList || []).filter((d) => d.is_active)}
        onAssignConfirm={handleConfirmAssign}
        onAssignClose={() => {
          setAssignModalOpen(false);
          setAssignModalReservation(null);
        }}
        deleteModalOpen={deleteModalOpen}
        deleteModalReservation={deleteModalReservation}
        onDeleteConfirm={handleConfirmDelete}
        onDeleteClose={() => {
          setDeleteModalOpen(false);
          setDeleteModalReservation(null);
        }}
      />

      {/* Bulle de discussion */}
      {company?.id && <ChatWidget companyId={company.id} />}
    </div>
  );
};

export default UnifiedDispatchRefactored;
