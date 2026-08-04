// src/pages/company/Dashboard/CompanyDashboard.jsx
import React, {
  useCallback,
  useState,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useTransition,
  lazy,
  Suspense,
} from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { FiZap, FiPlus, FiBarChart2, FiMessageSquare } from 'react-icons/fi';
import useCompanySocket, { useSocketConnected } from '../../../hooks/useCompanySocket';
import useDispatchStatus from '../../../hooks/useDispatchStatus';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import OverviewCards from './components/OverviewCards';
import ReservationTable from './components/ReservationTable';
import DriverTable from '../../driver/components/Dashboard/DriverTable';
import OpportunitiesSection from './components/OpportunitiesSection';
import DispatchModeStatusBar from './components/DispatchModeStatusBar';
import InstitutionOffersTable from './components/InstitutionOffersTable';
import ReservationFilterBar, { TIME_RANGES } from './components/ReservationFilterBar';
import QuickAssignPanel from './components/QuickAssignPanel';
import {
  acceptReservation,
  rejectReservation,
  assignDriver,
  updateDriverStatus,
  deleteDriver,
  fetchAssignedReservations,
  toggleDriverType,
  deleteReservation,
  dispatchNowForReservation,
  triggerReturnBooking,
  fetchDispatchDelays,
  fetchRequestOffers,
  acceptRequestOffer,
  rejectRequestOffer,
  fetchCompanyReservationsPaginated,
  fetchCompanyReservations,
} from '../../../services/companyService';
import useCompanyData from '../../../hooks/useCompanyData';
import useCompanyDriversForMap from '../../../hooks/useCompanyDriversForMap';
import useRealtimeDashboard from '../../../hooks/useRealtimeDashboard';
import { useDispatchMode } from '../../../hooks/useDispatchMode';
import { hasCompanyScopedAccessToken } from '../../../utils/webAuthSession';
import {
  recordCompanyDashboardRender,
  recordReactCommitMs,
  startCompanyDashboardPerfBaseline,
  isCompanyDashboardPerfEnabled,
} from '../../../utils/companyDashboardPerfInstrumentation';
import {
  perfMark,
  startCompanyDashboardWebVitals,
} from '../../../utils/companyDashboardWebPerf';
import {
  publishAuditReports,
  scheduleAuditReportPublish,
} from '../../../utils/companyDashboardAuditReports';
import { resetCompanyDashboardApiTiming } from '../../../utils/companyDashboardApiTiming';
import {
  bootstrapCompanyDashboardPerf,
  teardownCompanyDashboardPerfBootstrap,
} from '../../../utils/companyDashboardPerfBootstrap';
import { resetDuplicationReport } from '../../../utils/companyDashboardDuplicationReport';
import styles from './CompanyDashboard.module.css';
import Modal from '../../../components/common/Modal';
import { CompanyDashboardFreshnessBadge } from './components/CompanyDashboardFreshnessBadge';
import InlineDatePicker from '../../../components/ui/InlineDatePicker';
import { toast } from 'sonner';
import { logoutUser } from '../../../utils/apiClient';
import { lirieKeys, LIRIE_QK_PREFIX, lirieInvalidateCompanyReservationLists, listScopeHash } from '../../../queryKeys/lirie';
import {
  canonicalRealtimeTimeMs,
  shouldAcceptRealtimeEvent,
} from '../../../utils/realtimeEventGuard';
import {
  inspectSequence,
  commitAppliedSequence,
  bufferRealtimeEvent,
  drainContiguousBuffer,
  setSubscribedCursor,
  isRealtimeDegraded,
  isResyncRequired,
} from '../../../utils/companyRealtimeSequenceGate';
import { getAuthEnv } from '../../../utils/webAuthSession';
import {
  countConstrainedAssignedImminentDrivers,
  buildConstrainedImminentToastMessage,
  CONSTRAINED_IMMINENT_TOAST_ID,
} from '../../../utils/companyDriverConstrainedBanner';
import { filterVisibleInstitutionOffers } from '../../../utils/institutionOfferResponse';

const DriverLiveMap = lazy(() => import('./components/DriverLiveMap'));
const ReservationChart = lazy(() => import('./components/ReservationChart'));
const ManualBookingForm = lazy(() => import('./components/ManualBookingForm'));
const EditDriverForm = lazy(() => import('../components/EditDriverForm'));
const ChatWidget = lazy(() => import('../../../components/widgets/ChatWidget'));
const DemoInteractiveGuide = lazy(() => import('../../../components/demo/DemoInteractiveGuide'));
const ReservationModals = lazy(() => import('../../../components/reservations/ReservationModals'));
const TransferBookingModal = lazy(() => import('../../../components/reservations/TransferBookingModal'));

function makeToday() {
  const d = new Date();
  const pad = (n) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
}

function offsetCalendarYmd(dayOffset) {
  const d = new Date();
  d.setDate(d.getDate() + dayOffset);
  const pad = (n) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
}

const RESERVATIONS_DASHBOARD_SCOPE_HASH = listScopeHash({ flat: true, include_stats: false });
const REALTIME_MOUNT_GRACE_MS = 3000;

const CompanyDashboard = () => {
  recordCompanyDashboardRender();

  const dashMountCountRef = React.useRef(0);
  const dashStartMarkedRef = React.useRef(false);
  const shellVisibleMarkedRef = React.useRef(false);
  const criticalReadyMarkedRef = React.useRef(false);
  const queriesLoadedMarkedRef = React.useRef(false);
  const liveMapPerfMarkedRef = React.useRef(false);

  useLayoutEffect(() => {
    if (!isCompanyDashboardPerfEnabled()) return undefined;
    dashMountCountRef.current += 1;
    const source = dashMountCountRef.current === 1 ? 'mount' : 'remount';
    if (typeof window !== 'undefined') {
      window.__companyDashboardDashStartSource = source;
    }
    if (!dashStartMarkedRef.current) {
      dashStartMarkedRef.current = true;
      perfMark('dashboard_start');
      bootstrapCompanyDashboardPerf();
      resetCompanyDashboardApiTiming();
      resetDuplicationReport();
      startCompanyDashboardWebVitals();
    }
    return undefined;
  }, []);

  useEffect(() => {
    if (!isCompanyDashboardPerfEnabled()) return undefined;
    const cancelPublish = scheduleAuditReportPublish(10_000);
    return () => {
      cancelPublish();
      publishAuditReports(
        typeof window !== 'undefined' ? window.__companyDashboardSqlPerf : {}
      );
      teardownCompanyDashboardPerfBootstrap();
    };
  }, []);

  const location = useLocation();
  const navigate = useNavigate();
  const isDemoEnv = getAuthEnv() === 'demo';
  const fallbackDemoMission = isDemoEnv
    ? (
        localStorage.getItem('demo_recommended_journey') ||
        localStorage.getItem('demo_demo_recommended_journey') ||
        ''
      )
        .toString()
        .trim()
        .toLowerCase()
    : '';
  const demoMission = useMemo(
    () => {
      const mission = new URLSearchParams(location.search).get('demo_mission');
      if (mission) return mission;
      if (fallbackDemoMission === 'transport') return 'transporteur';
      return null;
    },
    [location.search, fallbackDemoMission]
  );
  const [dispatchDay, setDispatchDay] = useState(makeToday());
  const [reservationTab, setReservationTab] = useState('pending');

  const {
    company,
    reservations,
    driver,
    loadingReservations,
    loadingDriver,
    loadingCompany,
    reloadReservations,
    reloadDriver,
    upsertReservation,
    bootstrap,
    isBootstrapError,
    driversError,
    reservationsError,
  } = useCompanyData({ day: dispatchDay });

  const { driversForMap } = useCompanyDriversForMap(company?.id);
  const socketConnected = useSocketConnected();

  /** Succès réel profil + bootstrap + chauffeurs — jamais si erreur. */
  const criticalDataReady =
    Boolean(company?.id) &&
    !loadingCompany &&
    !loadingReservations &&
    !loadingDriver &&
    !isBootstrapError &&
    !driversError &&
    !reservationsError &&
    Array.isArray(reservations);

  /** Secondaires uniquement après le marqueur (évite de les compter critiques). */
  const [deferredQueriesEnabled, setDeferredQueriesEnabled] = useState(false);
  useEffect(() => {
    if (criticalDataReady) {
      setDeferredQueriesEnabled(true);
    } else {
      setDeferredQueriesEnabled(false);
    }
  }, [criticalDataReady]);

  const realtimeDegraded =
    Boolean(company?.id) &&
    (isRealtimeDegraded(company.id) ||
      bootstrap?.health?.realtime_sequence === 'degraded' ||
      bootstrap?.snapshot_cursor == null);

  /** Demandes PENDING visibles pour l'entreprise sur une fenêtre de dates (repère les courses hors jour sélectionné). */
  const pendingWindowStart = useMemo(() => offsetCalendarYmd(-2), []);
  const pendingWindowEnd = useMemo(() => offsetCalendarYmd(14), []);
  const { data: pendingWindowPayload } = useQuery({
    queryKey: [
      LIRIE_QK_PREFIX,
      'company-pending-window',
      company?.id,
      pendingWindowStart,
      pendingWindowEnd,
    ],
    queryFn: () =>
      fetchCompanyReservationsPaginated({
        startDate: pendingWindowStart,
        endDate: pendingWindowEnd,
        tab: 'pending',
        perPage: 100,
        page: 1,
        excludeCanceled: true,
      }),
    staleTime: 20_000,
    refetchInterval: socketConnected ? false : 30_000,
    enabled: Boolean(company?.id) && reservationTab === 'pending' && deferredQueriesEnabled,
  });
  const pendingWindowReservations = useMemo(
    () => (Array.isArray(pendingWindowPayload?.reservations) ? pendingWindowPayload.reservations : []),
    [pendingWindowPayload]
  );

  const socket = useCompanySocket();
  useDispatchStatus(socket);

  const { dispatchMode } = useDispatchMode();

  const isManualMode = dispatchMode === 'manual';
  const isAutoMode = dispatchMode === 'fully_auto' || dispatchMode === 'autonomous';
  const isSemiAutoMode = !!dispatchMode && !isManualMode && !isAutoMode;

  useEffect(() => {
    const handler = (e) => {
      const code = (e.detail?.code || e.detail?.message || '').toString();
      // Session morte / JWT manquant ou invalide → déconnexion auto, pas de toast fugace.
      const logoutCodes = ['AUTH_REQUIRED', 'AUTH_INVALID', 'TOKEN_EXPIRED'];
      const accessCodes = ['AUTH_FORBIDDEN', 'COMPANY_NOT_FOUND', 'DRIVER_OR_COMPANY_NOT_FOUND'];
      if (logoutCodes.some((c) => code.includes(c))) {
        toast.dismiss('socket-auth-rejected');
        void logoutUser({ preserveNext: true });
      } else if (accessCodes.some((c) => code.includes(c))) {
        const reason =
          e.detail?.reasonLabel ||
          'Accès refusé à la connexion temps réel. Reconnectez-vous.';
        toast.error(reason, {
          id: 'socket-auth-rejected',
          duration: 5000,
        });
      } else if (code.includes('RATE_LIMIT')) {
        toast.warning(
          e.detail?.reasonLabel ||
            'Trop de tentatives de connexion. Réessayez dans quelques instants.',
          {
            id: 'socket-rate-limit',
            duration: 5000,
          }
        );
      }
      // Sinon (erreurs transport: websocket error, xhr poll error, timeout, transport close...):
      // pas de toast. Socket.IO reconnecte tout seul et le circuit breaker affiche déjà
      // l'état "reconnecting" via COMPANY_SOCKET_STATE_EVENT. Le message générique précédent
      // ("Connexion temps réel refusée. Vérifiez votre authentification.") était trompeur.
    };
    window.addEventListener('socket_connection_rejected', handler);
    return () => window.removeEventListener('socket_connection_rejected', handler);
  }, []);

  const {
    loading: loadingRealtimeDashboard,
    qualityMetrics,
    opportunities,
    error: realtimeDashboardError,
  } = useRealtimeDashboard(dispatchDay, socketConnected ? 0 : 120000, {
    enabled: deferredQueriesEnabled,
  });

  const queryClient = useQueryClient();
  const useUnifiedDispatchWs =
    process.env.REACT_APP_LIRIE_DISPATCH_WS_UNIFIED === '1' ||
    process.env.REACT_APP_LIRIE_DISPATCH_WS_UNIFIED === 'true';
  const useDispatchDashboardWs =
    process.env.REACT_APP_DISPATCH_DASHBOARD_WS === '1' ||
    process.env.REACT_APP_DISPATCH_DASHBOARD_WS === 'true';
  const [, startTransition] = useTransition();
  const [showEditModal, setShowEditModal] = useState(false);
  const [driverToEdit, setDriverToEdit] = useState(null);
  const [showBookingModal, setShowBookingModal] = useState(false);
  const [showPerformance, setShowPerformance] = useState(false);
  const [chatWidgetActive, setChatWidgetActive] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [urgenceMode, setUrgenceMode] = useState(false);
  const [delaysOnly, setDelaysOnly] = useState(false);
  const [selectedDriver, setSelectedDriver] = useState(null);
  const [selectedInstitution, setSelectedInstitution] = useState(null);
  const [timeRange, setTimeRange] = useState('all');
  const [quickAssignOpen, setQuickAssignOpen] = useState(false);
  const [quickAssignOpportunity, setQuickAssignOpportunity] = useState(null);
  const [quickAssigning, setQuickAssigning] = useState(false);
  const [lastDataSyncAt, setLastDataSyncAt] = useState(null);
  const [liveMapEnabled, setLiveMapEnabled] = useState(() => {
    if (process.env.NODE_ENV === 'test') return true;
    if (typeof window === 'undefined') return false;
    const params = new URLSearchParams(window.location.search);
    if (params.get('live_map') === '1') return true;
    return localStorage.getItem('company_dashboard_live_map') === '1';
  });

  useEffect(() => {
    if (isCompanyDashboardPerfEnabled()) {
      startCompanyDashboardPerfBaseline();
      perfMark('dashboard_first_render');
    }
  }, []);

  useLayoutEffect(() => {
    if (!isCompanyDashboardPerfEnabled()) return undefined;
    const started = performance.now();
    return () => {
      recordReactCommitMs(performance.now() - started);
    };
  });

  useEffect(() => {
    if (!isCompanyDashboardPerfEnabled() || shellVisibleMarkedRef.current) return undefined;
    const frameId = requestAnimationFrame(() => {
      if (shellVisibleMarkedRef.current) return;
      shellVisibleMarkedRef.current = true;
      perfMark('dashboard_shell_visible');
    });
    return () => cancelAnimationFrame(frameId);
  }, []);

  useEffect(() => {
    if (!isCompanyDashboardPerfEnabled() || !criticalDataReady || criticalReadyMarkedRef.current) {
      return undefined;
    }
    const frameId = requestAnimationFrame(() => {
      if (criticalReadyMarkedRef.current) return;
      criticalReadyMarkedRef.current = true;
      perfMark('dashboard_critical_ready');
    });
    return () => cancelAnimationFrame(frameId);
  }, [criticalDataReady]);

  useEffect(() => {
    if (!isCompanyDashboardPerfEnabled() || !company?.id) return;
    if (queriesLoadedMarkedRef.current) return;
    if (!loadingReservations && !loadingDriver && !loadingRealtimeDashboard) {
      queriesLoadedMarkedRef.current = true;
      perfMark('dashboard_queries_loaded');
    }
  }, [
    company?.id,
    loadingReservations,
    loadingDriver,
    loadingRealtimeDashboard,
  ]);

  useEffect(() => {
    if (!isCompanyDashboardPerfEnabled() || !liveMapEnabled || liveMapPerfMarkedRef.current) return;
    liveMapPerfMarkedRef.current = true;
    perfMark('dashboard_live_map_enabled');
  }, [liveMapEnabled]);

  useEffect(() => {
    if ((reservations || []).length > 0) {
      setLastDataSyncAt(Date.now());
    }
  }, [reservations, qualityMetrics]);

  useEffect(() => {
    if (!loadingDriver && (driver?.length ?? 0) > 0) {
      setLastDataSyncAt(Date.now());
    }
  }, [loadingDriver, driver?.length]);

  // Bannière sticky : chauffeurs ASSIGNED + batterie restreinte + mission < 30 min
  useEffect(() => {
    if (!company?.id) return;
    const count = countConstrainedAssignedImminentDrivers(
      driversForMap,
      reservations,
      Date.now()
    );
    if (count > 0) {
      toast.warning(buildConstrainedImminentToastMessage(count), {
        id: CONSTRAINED_IMMINENT_TOAST_ID,
        duration: Infinity,
      });
    } else {
      toast.dismiss(CONSTRAINED_IMMINENT_TOAST_ID);
    }
  }, [company?.id, driversForMap, reservations]);

  useEffect(() => {
    const params = new URLSearchParams(location.search);
    if (params.get('live_map') === '1') {
      setLiveMapEnabled(true);
      localStorage.setItem('company_dashboard_live_map', '1');
    }
  }, [location.search]);

  const handleEnableLiveMap = useCallback(() => {
    const params = new URLSearchParams(location.search);
    params.set('live_map', '1');
    localStorage.setItem('company_dashboard_live_map', '1');
    setLiveMapEnabled(true);
    navigate(`${location.pathname}?${params.toString()}`, { replace: true });
  }, [location.pathname, location.search, navigate]);

  /**
   * Active la carte live quand la section entre dans le viewport (IntersectionObserver).
   * Pas de prefetch Maps global au shell (Lot 2).
   */
  const mapSectionRef = useRef(null);

  useEffect(() => {
    if (process.env.NODE_ENV === 'test') return;
    if (liveMapEnabled) return;
    const el = mapSectionRef.current;
    if (!el || typeof IntersectionObserver === 'undefined') {
      handleEnableLiveMap();
      return undefined;
    }
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          startTransition(() => handleEnableLiveMap());
          observer.disconnect();
        }
      },
      { root: null, rootMargin: '120px', threshold: 0.05 }
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, [liveMapEnabled, handleEnableLiveMap]);

  const handleEditDriver = (d) => {
    setDriverToEdit(d);
    setShowEditModal(true);
  };
  const handleCloseModal = () => {
    setShowEditModal(false);
    setDriverToEdit(null);
  };

  const handleToggleType = async (driverId) => {
    try {
      await toggleDriverType(driverId);
      startTransition(() => {
        reloadDriver();
      });
    } catch (err) {
      console.error('Erreur lors du changement de type du chauffeur :', err);
    }
  };

  const handleToggleAvailability = async (driverId) => {
    try {
      const d = (driver || []).find((x) => x.id === driverId);
      if (!d) return;
      await updateDriverStatus(driverId, { is_available: !d.is_available });
      startTransition(() => {
        reloadDriver();
      });
    } catch (err) {
      console.error('Erreur mise à jour disponibilité chauffeur :', err);
    }
  };

  const [scheduleModalOpen, setScheduleModalOpen] = useState(false);
  const [scheduleModalReservation, setScheduleModalReservation] = useState(null);
  const [assignModalOpen, setAssignModalOpen] = useState(false);
  const [assignModalReservation, setAssignModalReservation] = useState(null);
  const [transferModalOpen, setTransferModalOpen] = useState(false);
  const [transferModalReservation, setTransferModalReservation] = useState(null);
  const [deleteModalOpen, setDeleteModalOpen] = useState(false);
  const [deleteModalReservation, setDeleteModalReservation] = useState(null);

  const reservationsMap = useMemo(() => {
    const map = new Map();
    (reservations || []).forEach((r) => {
      if (r?.id) map.set(r.id, r);
    });
    return map;
  }, [reservations]);

  const handleScheduleReservation = (reservation) => {
    const resObj =
      typeof reservation === 'object'
        ? reservation
        : reservationsMap.get(reservation);
    if (!resObj) return;
    setScheduleModalReservation(resObj);
    setScheduleModalOpen(true);
  };

  const handleDispatchNow = async (reservation) => {
    const id = reservation?.id ?? reservation;
    if (!id) return;
    try {
      await dispatchNowForReservation(id, 15);
      startTransition(() => {
        reloadReservations();
        void lirieInvalidateCompanyReservationLists(queryClient);
      });
      toast.success('Dispatch urgent déclenché avec succès');
    } catch (e) {
      const errorData = e?.response?.data;
      const errorMessage = errorData?.message || errorData?.error;
      const status = e?.response?.status;

      if (process.env.NODE_ENV === 'development') {
        console.debug('[DispatchNow] Error status:', status);
        console.debug('[DispatchNow] Error data:', errorData);
        console.debug('[DispatchNow] Error message:', errorMessage);
      }

      const errorLower = (errorMessage || '').toLowerCase();
      const errorErrorLower = (errorData?.error || '').toLowerCase();
      const isReturnNotReady =
        status === 400 &&
        (errorLower.includes('retour') ||
          errorLower.includes('aller') ||
          errorErrorLower.includes('retour') ||
          errorErrorLower.includes('aller'));

      if (process.env.NODE_ENV === 'development') {
        console.debug('[DispatchNow] isReturnNotReady:', isReturnNotReady);
      }

      if (isReturnNotReady) {
        const detailMessage =
          errorData?.message ||
          errorData?.error ||
          "Impossible de déclencher un retour d'urgence. La course aller doit être complétée avant de déclencher le retour.";

        if (process.env.NODE_ENV === 'development') {
          console.debug('[DispatchNow] Showing warning:', detailMessage);
          console.debug('Dispatch urgent refusé (comportement attendu):', detailMessage);
        }
        toast.warning(detailMessage, { duration: 5000 });
      } else {
        console.error('Dispatch urgent:', e);
        toast.error(errorMessage || 'Erreur lors du dispatch urgent.');
      }
    }
  };

  const reservationsCacheKey = useMemo(
    () => lirieKeys.companyReservations(dispatchDay, RESERVATIONS_DASHBOARD_SCOPE_HASH),
    [dispatchDay]
  );

  const { data: dispatchedReservations = [], refetch: refetchAssigned } = useQuery({
    queryKey: lirieKeys.assignedReservations(dispatchDay),
    queryFn: async () => {
      const reservations = await queryClient.ensureQueryData({
        queryKey: reservationsCacheKey,
        queryFn: async () => {
          const data = await fetchCompanyReservations(dispatchDay, { fields: 'dashboard' });
          return Array.isArray(data) ? data : (data?.reservations ?? []);
        },
        staleTime: 30_000,
      });
      return fetchAssignedReservations(dispatchDay, {
        reservations: Array.isArray(reservations) ? reservations : undefined,
      });
    },
    staleTime: 30_000,
    enabled:
      !!company?.id && deferredQueriesEnabled && hasCompanyScopedAccessToken(),
  });

  const {
    data: delays = [],
    refetch: refetchDelays,
    isFetching: fetchingDelays,
    isError: delaysError,
  } = useQuery({
    queryKey: lirieKeys.dispatchDelays(dispatchDay),
    queryFn: () => fetchDispatchDelays(dispatchDay),
    initialData: [],
    staleTime: 20_000,
    enabled:
      !!company?.id && deferredQueriesEnabled && hasCompanyScopedAccessToken(),
  });

  const {
    data: institutionOffersData,
    refetch: refetchInstitutionOffers,
    isLoading: loadingInstitutionOffers,
  } = useQuery({
    queryKey: lirieKeys.institutionOffers(),
    queryFn: () => fetchRequestOffers('PENDING'),
    staleTime: 15_000,
    refetchInterval: socketConnected ? false : 30_000,
    enabled: !!company?.id && deferredQueriesEnabled,
  });
  const institutionOffers = useMemo(
    () => institutionOffersData?.offers || [],
    [institutionOffersData?.offers],
  );
  const visibleInstitutionOffers = useMemo(
    () => filterVisibleInstitutionOffers(institutionOffers),
    [institutionOffers],
  );
  const hiddenInstitutionOffersCount = institutionOffers.length - visibleInstitutionOffers.length;

  const handleNewReservation = useCallback(() => reloadReservations(), [reloadReservations]);
  useEffect(() => {
    if (!socket) return;
    socket.on('new_reservation', handleNewReservation);
    return () => socket.off('new_reservation', handleNewReservation);
  }, [socket, handleNewReservation]);

  const handleOfferUnavailable = useCallback(
    (payload) => {
      const offerId = payload?.offer_id ?? payload?.metadata?.offer_id;
      queryClient.setQueryData(lirieKeys.institutionOffers(), (old) => {
        if (!old?.offers) return old;
        const filtered = offerId
          ? old.offers.filter((o) => o.id !== offerId)
          : old.offers;
        return {
          ...old,
          offers: filtered,
          total: filtered.length,
        };
      });
      refetchInstitutionOffers();
      const institutionName =
        payload?.institution_name || payload?.metadata?.institution_name;
      toast.info(
        institutionName
          ? `Demande ${institutionName} — plus disponible`
          : 'Une demande institution n\'est plus disponible'
      );
    },
    [queryClient, refetchInstitutionOffers]
  );

  useEffect(() => {
    if (!socket) return;
    socket.on('offer_unavailable', handleOfferUnavailable);
    return () => socket.off('offer_unavailable', handleOfferUnavailable);
  }, [socket, handleOfferUnavailable]);

  const handleInstitutionOfferUpdated = useCallback(
    (payload) => {
      if (!payload?.offer_id && !payload?.transport_request_id) return;
      refetchInstitutionOffers();
      if (payload?.is_relaunch) {
        toast.info('Demande institution relancée — nouvelle offre disponible');
      }
    },
    [refetchInstitutionOffers]
  );

  useEffect(() => {
    if (!socket) return;
    socket.on('institution_offer_updated', handleInstitutionOfferUpdated);
    return () => socket.off('institution_offer_updated', handleInstitutionOfferUpdated);
  }, [socket, handleInstitutionOfferUpdated]);

  const handleNewCompanyNotification = useCallback(
    (payload) => {
      if (payload?.event_type !== 'new_request') return;
      refetchInstitutionOffers();
    },
    [refetchInstitutionOffers]
  );

  useEffect(() => {
    if (!socket) return;
    socket.on('new_company_notification', handleNewCompanyNotification);
    return () => socket.off('new_company_notification', handleNewCompanyNotification);
  }, [socket, handleNewCompanyNotification]);

  const graceBufferActiveRef = React.useRef(true);

  useEffect(() => {
    if (!criticalDataReady) return undefined;
    const t = setTimeout(() => {
      graceBufferActiveRef.current = false;
      if (!company?.id) return;
      const drained = drainContiguousBuffer(company.id);
      if (!drained.complete || drained.resyncRequired || isResyncRequired(company.id)) {
        startTransition(() => {
          reloadReservations?.();
          refetchAssigned?.();
          refetchDelays?.();
        });
        return;
      }
      if (drained.events.length > 0) {
        drained.events.forEach(({ seq }) => commitAppliedSequence(company.id, seq));
        startTransition(() => {
          refetchAssigned?.();
          reloadReservations?.();
          refetchDelays?.();
        });
      }
    }, REALTIME_MOUNT_GRACE_MS);
    return () => clearTimeout(t);
  }, [criticalDataReady, company?.id, reloadReservations, refetchAssigned, refetchDelays]);

  useEffect(() => {
    if (!socket) return;
    const onSubscribed = (payload) => {
      if (payload?.subscribed_cursor != null) {
        setSubscribedCursor(company?.id, payload.subscribed_cursor);
      } else if (payload?.subscribed_cursor === null) {
        setSubscribedCursor(company?.id, null);
      }
    };
    socket.on('company_room_subscribed', onSubscribed);
    socket.on('subscribed', onSubscribed);
    return () => {
      socket.off('company_room_subscribed', onSubscribed);
      socket.off('subscribed', onSubscribed);
    };
  }, [socket, company?.id]);

  useEffect(() => {
    if (!socket) return;
    const applyEffect = () => {
      startTransition(() => {
        refetchAssigned?.();
        reloadReservations?.();
        refetchDelays?.();
        queryClient.invalidateQueries({
          queryKey: lirieKeys.companyReservationsSummary(dispatchDay),
        });
      });
    };
    const acceptRealtime = (payload, entityKey = null) => {
      const inspected = inspectSequence(
        company?.id,
        payload?.event_seq,
        payload?.company_id
      );
      if (!inspected.accept) return false;
      if (inspected.gapDetected) {
        startTransition(() => {
          reloadReservations?.();
          refetchAssigned?.();
          refetchDelays?.();
        });
        return false;
      }
      if (graceBufferActiveRef.current || !criticalDataReady) {
        bufferRealtimeEvent(company?.id, payload?.event_seq, payload);
        return false;
      }
      const dedupeOk = shouldAcceptRealtimeEvent({
        eventId: payload?.event_id,
        entityKey,
        canonicalTimeMs: canonicalRealtimeTimeMs(payload),
      });
      if (!dedupeOk) return false;
      commitAppliedSequence(company?.id, payload?.event_seq);
      return true;
    };
    const refetchAll = () => {
      applyEffect();
    };
    const onAssignCreated = (data) => {
      if (!acceptRealtime(data, data?.booking_id ? `booking:${data.booking_id}` : null)) return;
      refetchAll();
    };
    const onAssignUpdated = (data) => {
      if (!acceptRealtime(data, data?.assignment_id ? `assignment:${data.assignment_id}` : null)) return;
      refetchAll();
    };
    const onAssignDeleted = (data) => {
      if (!acceptRealtime(data, data?.booking_id ? `booking:${data.booking_id}` : null)) return;
      refetchAll();
    };
    const onDispatchStatePatch = (data) => {
      if (!acceptRealtime(data, data?.reservation_id ? `booking:${data.reservation_id}` : null)) return;
      refetchAll();
    };
    const onDispatchDashboardSnapshot = (data) => {
      if (!acceptRealtime(data, dispatchDay ? `dispatch-dashboard:${dispatchDay}` : null)) return;
      queryClient.invalidateQueries({
        queryKey: lirieKeys.dispatchRealtimeDashboard(dispatchDay),
      });
    };
    const onDispatchProgress = (_p) => {};
    const onDispatchError = (err) => {
      if (!acceptRealtime(err)) return;
      console.error('dispatch_error:', err);
      refetchAll();
    };
    const onDispatchRunCompleted = (data) => {
      if (!acceptRealtime(data, data?.dispatch_run_id ? `dispatch-run:${data.dispatch_run_id}` : null)) return;
      if (process.env.NODE_ENV === 'development') {
        console.log('Dispatch run completed:', data);
      }
      refetchAll();
    };
    const onTransferReceived = (data) => {
      if (!acceptRealtime(data, data?.booking_id ? `booking:${data.booking_id}` : null)) return;
      if (process.env.NODE_ENV === 'development') {
        console.log('Transfert reçu:', data);
      }
      refetchAll();
    };
    const onTransferProposed = (data) => {
      if (!acceptRealtime(data, data?.booking_id ? `booking:${data.booking_id}` : null)) return;
      if (process.env.NODE_ENV === 'development') {
        console.log('Transfert proposé:', data);
      }
      refetchAll();
    };
    const onBookingUpdated = (data) => {
      if (!acceptRealtime(data, data?.booking_id ? `booking:${data.booking_id}` : null)) return;
      if (process.env.NODE_ENV === 'development') {
        console.log('Course mise à jour:', data);
      }
      refetchAll();
    };
    const onBookingReassigned = (data) => {
      if (!acceptRealtime(data, data?.booking_id ? `booking:${data.booking_id}` : null)) return;
      if (process.env.NODE_ENV === 'development') {
        console.log('Course réassignée:', data);
      }
      refetchAll();
    };
    if (useUnifiedDispatchWs) {
      socket.on('dispatch_state_patch', onDispatchStatePatch);
    } else {
      socket.on('dispatch_assignment_created', onAssignCreated);
      socket.on('dispatch_assignment_updated', onAssignUpdated);
      socket.on('dispatch_assignment_cancelled', onAssignDeleted);
    }
    if (useDispatchDashboardWs) {
      socket.on('dispatch_dashboard_snapshot', onDispatchDashboardSnapshot);
    }
    socket.on('dispatch_progress', onDispatchProgress);
    socket.on('dispatch_error', onDispatchError);
    socket.on('dispatch_run_completed', onDispatchRunCompleted);
    socket.on('transfer_received', onTransferReceived);
    socket.on('transfer_proposed', onTransferProposed);
    socket.on('booking_updated', onBookingUpdated);
    socket.on('booking_reassigned', onBookingReassigned);
    socket.on('booking_assigned', onBookingReassigned);
    return () => {
      if (useUnifiedDispatchWs) {
        socket.off('dispatch_state_patch', onDispatchStatePatch);
      } else {
        socket.off('dispatch_assignment_created', onAssignCreated);
        socket.off('dispatch_assignment_updated', onAssignUpdated);
        socket.off('dispatch_assignment_cancelled', onAssignDeleted);
      }
      if (useDispatchDashboardWs) {
        socket.off('dispatch_dashboard_snapshot', onDispatchDashboardSnapshot);
      }
      socket.off('dispatch_progress', onDispatchProgress);
      socket.off('dispatch_error', onDispatchError);
      socket.off('dispatch_run_completed', onDispatchRunCompleted);
      socket.off('transfer_received', onTransferReceived);
      socket.off('transfer_proposed', onTransferProposed);
      socket.off('booking_updated', onBookingUpdated);
      socket.off('booking_reassigned', onBookingReassigned);
      socket.off('booking_assigned', onBookingReassigned);
    };
  }, [
    socket,
    refetchAssigned,
    reloadReservations,
    refetchDelays,
    queryClient,
    dispatchDay,
    useUnifiedDispatchWs,
    useDispatchDashboardWs,
    company?.id,
    startTransition,
    criticalDataReady,
  ]);

  const handleAccept = async (id) => {
    try {
      await acceptReservation(id);
      startTransition(() => {
        reloadReservations();
      });
    } catch (err) {
      console.error('Erreur acceptation :', err);
    }
  };
  const handleReject = async (id) => {
    try {
      await rejectReservation(id);
      startTransition(() => {
        reloadReservations();
      });
    } catch (err) {
      console.error('Erreur rejet :', err);
    }
  };

  const handleAcceptOffer = async (offerId, proposedPickupTime) => {
    try {
      const result = await acceptRequestOffer(offerId, proposedPickupTime);
      toast.success(
        proposedPickupTime
          ? 'Offre acceptée avec horaire proposé — réservation créée'
          : 'Offre acceptée — réservation créée'
      );
      startTransition(() => {
        refetchInstitutionOffers();
        reloadReservations();
        void lirieInvalidateCompanyReservationLists(queryClient);
      });
      return result;
    } catch (err) {
      console.error('[handleAcceptOffer] error:', err);
      toast.error(err?.response?.data?.error || 'Erreur lors de l\'acceptation');
    }
  };

  const handleRejectOffer = async (offerId) => {
    try {
      await rejectRequestOffer(offerId);
      toast.success('Offre refusée');
      startTransition(() => {
        refetchInstitutionOffers();
      });
    } catch (err) {
      console.error('[handleRejectOffer] error:', err);
      toast.error(err?.response?.data?.error || 'Erreur lors du refus');
    }
  };

  const openAssignModal = (res) => {
    const resObj = typeof res === 'object' ? res : reservationsMap.get(res);
    if (!resObj) return;
    setAssignModalReservation(resObj);
    setAssignModalOpen(true);
  };
  const handleAssignDriver = async (reservationId, driverId) => {
    try {
      await assignDriver(reservationId, driverId);
      startTransition(() => {
        reloadReservations();
      });
      setAssignModalOpen(false);
      setAssignModalReservation(null);
    } catch (err) {
      console.error('Erreur assignation chauffeur :', err);
    }
  };

  const handleTriggerReturn = (reservation) => {
    const resObj =
      typeof reservation === 'object'
        ? reservation
        : reservationsMap.get(reservation);
    if (!resObj) return;
    setScheduleModalReservation(resObj);
    setScheduleModalOpen(true);
  };

  const handleOpenTransferModal = (reservation) => {
    const resObj =
      typeof reservation === 'object'
        ? reservation
        : reservationsMap.get(reservation);
    if (!resObj) return;
    setTransferModalReservation(resObj);
    setTransferModalOpen(true);
  };

  const handleTransferSuccess = () => {
    startTransition(() => {
      reloadReservations();
    });
    toast.success('Course transférée avec succès');
  };

  const toLocalIsoString = (date) => {
    const pad = (n) => n.toString().padStart(2, '0');
    const Y = date.getFullYear();
    const M = pad(date.getMonth() + 1);
    const D = pad(date.getDate());
    const h = pad(date.getHours());
    const m = pad(date.getMinutes());
    return `${Y}-${M}-${D}T${h}:${m}`;
  };

  const handleConfirmReturnTime = async (data) => {
    setScheduleModalOpen(false);
    if (!scheduleModalReservation) return;

    const reservationId = scheduleModalReservation?.id ?? scheduleModalReservation;
    try {
      let payload = {};
      if (data?.urgent) {
        payload = { urgent: true, minutes_offset: data.minutes_offset ?? 15 };
      } else if (typeof data === 'string') {
        payload = { return_time: data };
      } else if (data instanceof Date) {
        payload = { return_time: toLocalIsoString(data) };
      } else if (data?.return_time) {
        payload = { return_time: data.return_time };
      }
      await triggerReturnBooking(reservationId, payload);
      setScheduleModalReservation(null);
      startTransition(() => {
        reloadReservations();
        void lirieInvalidateCompanyReservationLists(queryClient);
      });
    } catch (err) {
      console.error('Retour :', err);
      alert(err?.response?.data?.error || 'Erreur serveur.');
    }
  };

  const handleDeleteReservationClick = (reservation) => {
    const resObj =
      typeof reservation === 'object'
        ? reservation
        : reservationsMap.get(reservation);
    if (!resObj) return;
    setDeleteModalReservation(resObj);
    setDeleteModalOpen(true);
  };

  const handleConfirmDelete = async (reservationId, reasonCode, reasonText) => {
    if (!reservationId && !deleteModalReservation) return;

    const id = reservationId || deleteModalReservation?.id || deleteModalReservation;
    await handleDeleteReservation(id, reasonCode, reasonText);
    setDeleteModalOpen(false);
    setDeleteModalReservation(null);
  };

  const handleManualBookingSuccess = (resp) => {
    const created = resp?.reservation ?? resp?.reservations?.[0];
    if (created?.id != null) {
      upsertReservation(created);
    }
    const ymd = String(created?.scheduled_time || resp?.reservation?.scheduled_time || '').slice(0, 10);
    if (ymd) setDispatchDay(ymd);
      startTransition(() => {
      reloadReservations({ silent: true });
      void lirieInvalidateCompanyReservationLists(queryClient);
    });
  };

  const pendingReservations = (reservations || []).filter(
    (r) => r.status?.toLowerCase() === 'pending'
  );
  const assignedReservations = (reservations || []).filter(
    (r) => r.status?.toLowerCase() === 'accepted' && !r.driver_id
  );

  const handleToggleDriver = async (driverId, current) => {
    try {
      await updateDriverStatus(driverId, !current);
      startTransition(() => {
        reloadDriver();
      });
    } catch (err) {
      console.error('Erreur mise à jour chauffeur :', err);
    }
  };
  const handleDeleteDriver = async (driverId) => {
    try {
      await deleteDriver(driverId);
      startTransition(() => {
        reloadDriver();
      });
    } catch (err) {
      console.error('Erreur suppression chauffeur :', err);
    }
  };
  const handleDeleteReservation = useCallback(
    async (reservation, reasonCode = null, reasonText = null) => {
      const id = reservation?.id ?? reservation;
      if (!id) {
        console.error('ID réservation manquant:', reservation);
        return;
      }

      const result = await deleteReservation(id, reasonCode, reasonText);
      if (process.env.NODE_ENV === 'development') {
        console.log('Réservation supprimée:', result);
      }

      startTransition(() => {
        reloadReservations();
        void lirieInvalidateCompanyReservationLists(queryClient);
      });

      toast.success(`Réservation #${id} supprimée avec succès`);
    },
    [reloadReservations, queryClient]
  );

  const activeBookings = useMemo(() => {
    const isActive = (b) =>
      b && !['completed', 'cancelled', 'no_show'].includes((b.status || '').toLowerCase());
    return (reservations || []).filter(isActive).map((r) => ({
      id: r.id,
      client_name: r.client_name || r.client?.full_name || '',
      status: r.status,
      pickup_time: r.scheduled_time || r.pickup_time,
      dropoff_time: r.dropoff_time,
      pickup_location: r.pickup_location_coords || r.pickup_location || r.pickup || null,
      dropoff_location: r.dropoff_location_coords || r.dropoff_location || r.dropoff || null,
    }));
  }, [reservations]);

  const assignmentsForMap = useMemo(() => {
    const rows = Array.isArray(dispatchedReservations)
      ? dispatchedReservations
      : Object.values(dispatchedReservations || {});
    return rows.map((row) => {
      const a = row.assignment || {};
      return {
        id: a.id ?? row.id,
        driver_id: a.driver_id ?? row.driver?.id ?? row.driver_id,
        is_on_trip: [
          'assigned',
          'in_progress',
          'onboard',
          'en_route_pickup',
          'en_route_dropoff',
        ].includes(String(a.status ?? row.status ?? '').toLowerCase()),
        route: row.route || a.route || [],
        booking: {
          id: row.id,
          client_name: row.client_name || row.client?.full_name || '',
          status: row.status,
          pickup_time: row.scheduled_time || row.pickup_time,
          dropoff_time: row.dropoff_time || null,
          pickup_location: row.pickup_location,
          dropoff_location: row.dropoff_location,
        },
      };
    });
  }, [dispatchedReservations]);

  const delaysByBooking = useMemo(() => {
    if (!Array.isArray(delays)) return delays;
    const map = {};
    for (const d of delays) {
      if (d && d.booking_id) {
        map[d.booking_id] = {
          delay_minutes: d.delay_minutes,
          delay_severity: d.delay_severity ?? null,
          is_dropoff: !d.is_pickup,
          pickup_eta: d.pickup_eta ?? null,
          dropoff_eta: d.dropoff_eta ?? null,
        };
      }
    }
    return map;
  }, [delays]);

  const activeDelayCount = useMemo(() => {
    if (!delaysByBooking || typeof delaysByBooking !== 'object' || Array.isArray(delaysByBooking)) return 0;
    return Object.values(delaysByBooking).filter((d) => d?.delay_minutes > 0).length;
  }, [delaysByBooking]);

  /** Seuil critique backend-authoritative (15 min) — aligné table / KPI. */
  const CRITICAL_DELAY_MINUTES =
    Number(bootstrap?.kpi?.critical_delay_minutes) > 0
      ? Number(bootstrap.kpi.critical_delay_minutes)
      : 15;

  const hasCriticalDelays = useMemo(
    () =>
      Object.values(delaysByBooking || {}).some(
        (d) => Number(d?.delay_minutes || 0) >= CRITICAL_DELAY_MINUTES
      ),
    [delaysByBooking, CRITICAL_DELAY_MINUTES]
  );

  const filterBySearch = useCallback(
    (list) => {
      if (!searchQuery.trim()) return list;
      const q = searchQuery.toLowerCase().trim();
      return list.filter((r) => {
        const name = (r.client?.full_name || r.client_name || '').toLowerCase();
        const pickup = (r.pickup_location || '').toLowerCase();
        const dropoff = (r.dropoff_location || '').toLowerCase();
        return name.includes(q) || pickup.includes(q) || dropoff.includes(q);
      });
    },
    [searchQuery]
  );

  const filterByDelaysOnly = useCallback(
    (list) => {
      if (!delaysOnly || !delaysByBooking) return list;
      return list.filter((r) => delaysByBooking[r.id]?.delay_minutes > 0);
    },
    [delaysOnly, delaysByBooking]
  );

  const filterByDriver = useCallback(
    (list) => {
      if (!selectedDriver) return list;
      return list.filter((r) => {
        const dId = String(selectedDriver);
        return String(r.driver_id) === dId || String(r.driver?.id) === dId;
      });
    },
    [selectedDriver]
  );

  const filterByInstitution = useCallback(
    (list) => {
      if (!selectedInstitution) return list;
      return list.filter((r) => r.client?.institution_name === selectedInstitution);
    },
    [selectedInstitution]
  );

  const filterByTimeRange = useCallback(
    (list) => {
      if (!timeRange || timeRange === 'all') return list;
      const range = TIME_RANGES.find((tr) => tr.id === timeRange);
      if (!range?.start) return list;
      return list.filter((r) => {
        const t = r.scheduled_time || r.pickup_time;
        if (!t) return true;
        const hour = new Date(t).getHours();
        return hour >= range.start && hour < range.end;
      });
    },
    [timeRange]
  );

  const applyStructuredFilters = useCallback(
    (list) => filterByTimeRange(filterByInstitution(filterByDriver(list))),
    [filterByDriver, filterByInstitution, filterByTimeRange]
  );

  const urgentReservations = useMemo(() => {
    if (!urgenceMode) return [];
    const activeStatuses = [
      'accepted', 'assigned', 'en_route', 'in_progress',
      'onboard', 'en_route_pickup', 'en_route_dropoff',
    ];
    return (reservations || []).filter((r) => {
      const delayInfo = delaysByBooking?.[r.id];
      if (delayInfo?.delay_minutes > 0) return true;
      if (delayInfo?.delay_severity === 'critical') return true;

      const status = (r.status || '').toLowerCase();
      if (!activeStatuses.includes(status)) return false;

      const scheduled = r.scheduled_time || r.pickup_time;
      if (!scheduled) return false;
      const scheduledMs = new Date(scheduled).getTime();
      if (Number.isNaN(scheduledMs)) return false;
      const minutesLate = (Date.now() - scheduledMs) / 60000;
      return minutesLate >= CRITICAL_DELAY_MINUTES;
    });
  }, [reservations, urgenceMode, delaysByBooking, CRITICAL_DELAY_MINUTES]);

  const displayPending = useMemo(
    () => applyStructuredFilters(filterByDelaysOnly(filterBySearch(pendingReservations))),
    [pendingReservations, filterBySearch, filterByDelaysOnly, applyStructuredFilters]
  );

  const pendingOnOtherDays = useMemo(() => {
    return pendingWindowReservations.filter((r) => {
      if ((r.status || '').toLowerCase() !== 'pending') return false;
      const st = String(r.scheduled_time || r.pickup_time || '');
      return Boolean(st) && !st.startsWith(dispatchDay);
    });
  }, [pendingWindowReservations, dispatchDay]);

  const pendingHiddenByFilters = useMemo(
    () =>
      reservationTab === 'pending' &&
      !urgenceMode &&
      pendingReservations.length > 0 &&
      displayPending.length === 0,
    [reservationTab, urgenceMode, pendingReservations.length, displayPending.length]
  );

  const firstOtherDayYmd = useMemo(() => {
    const r = pendingOnOtherDays[0];
    if (!r) return null;
    const st = String(r.scheduled_time || r.pickup_time || '');
    return st.length >= 10 ? st.slice(0, 10) : null;
  }, [pendingOnOtherDays]);

  const displayAssigned = useMemo(
    () => applyStructuredFilters(filterByDelaysOnly(filterBySearch(assignedReservations))),
    [assignedReservations, filterBySearch, filterByDelaysOnly, applyStructuredFilters]
  );

  const displayUrgent = useMemo(
    () => applyStructuredFilters(filterBySearch(urgentReservations)),
    [urgentReservations, filterBySearch, applyStructuredFilters]
  );

  const perfStats = useMemo(() => {
    const all = Array.isArray(reservations) ? reservations : [];
    const isCompleted = (s) => ['completed', 'return_completed', 'done', 'finished'].includes((s || '').toLowerCase());
    const dayFiltered = dispatchDay
      ? all.filter((r) => String(r.scheduled_time || r.pickup_time || '').startsWith(dispatchDay))
      : all;
    const completed = dayFiltered.filter((r) => isCompleted(r.status));
    const rev = completed.reduce((acc, r) => acc + (Number(r.amount || r.total_amount || 0) || 0), 0);
    return {
      total: dayFiltered.length,
      completed: completed.length,
      revenue: rev >= 1000 ? `${(rev / 1000).toFixed(1)}k` : String(Math.round(rev)),
    };
  }, [reservations, dispatchDay]);

  const institutionOptions = useMemo(() => {
    const names = new Set();
    (reservations || []).forEach((r) => {
      const inst = r.client?.institution_name;
      if (inst) names.add(inst);
    });
    return Array.from(names).sort();
  }, [reservations]);

  const handleRefresh = useCallback(() => {
    startTransition(() => {
      reloadReservations();
      refetchAssigned?.();
      refetchDelays?.();
      refetchInstitutionOffers?.();
      if (company?.id) {
        queryClient.invalidateQueries({
          queryKey: [
            LIRIE_QK_PREFIX,
            'company-pending-window',
            company.id,
            pendingWindowStart,
            pendingWindowEnd,
          ],
          exact: true,
        });
      }
    });
  }, [
    reloadReservations,
    refetchAssigned,
    refetchDelays,
    refetchInstitutionOffers,
    queryClient,
    company?.id,
    pendingWindowStart,
    pendingWindowEnd,
  ]);

  const handleOpportunityAction = useCallback((opp) => {
    setQuickAssignOpportunity(opp);
    setQuickAssignOpen(true);
  }, []);

  const handleQuickAssign = async (bookingId, driverId) => {
    setQuickAssigning(true);
    try {
      await assignDriver(bookingId, driverId);
      startTransition(() => {
        reloadReservations();
        refetchAssigned?.();
        refetchDelays?.();
        void lirieInvalidateCompanyReservationLists(queryClient);
      });
      toast.success('Assignation confirmée');
      setQuickAssignOpen(false);
      setQuickAssignOpportunity(null);
    } catch (err) {
      console.error('Quick assign error:', err);
      toast.error(err?.response?.data?.error || "Erreur lors de l'assignation");
    } finally {
      setQuickAssigning(false);
    }
  };

  return (
    <>
        <main
          className={styles.content}
          data-testid="company-dashboard"
          data-critical-ready={criticalDataReady ? '1' : '0'}
        >
          {criticalDataReady ? (
            <span data-testid="dashboard-critical-ready" hidden aria-hidden="true" />
          ) : null}
          {demoMission === 'transporteur' && (
            <Suspense fallback={null}>
              <DemoInteractiveGuide role="transporteur" />
            </Suspense>
          )}

          {/* ============ 1. HEADER CONTEXTUALISÉ ============ */}
          <header className={styles.dashboardHeader}>
            <div className={styles.headerLeft}>
              <h1 className={styles.headerTitle} data-tour-id="dashboard-transports">
                Tableau de bord Exploitation
              </h1>
              <div className={styles.headerMeta}>
                <InlineDatePicker
                  value={dispatchDay}
                  onChange={(iso) => setDispatchDay(iso)}
                />
                {activeDelayCount > 0 && (
                  <span className={styles.delayHeaderBadge}>
                    {activeDelayCount} retard{activeDelayCount !== 1 ? 's' : ''} actif{activeDelayCount !== 1 ? 's' : ''}
                  </span>
                )}
              </div>
            </div>
            <div className={styles.headerActions}>
              {company?.public_id && (
                <Link
                  to={`${isDemoEnv ? '/demo/dashboard' : '/dashboard'}/company/${company.public_id}/dispatch`}
                  className={styles.headerBtnSecondary}
                >
                  <FiZap size={16} />
                  Optimiser les courses
                </Link>
              )}
              <button
                onClick={() => setShowBookingModal(true)}
                className={styles.headerBtnPrimary}
                data-tour-id="create-booking"
              >
                <FiPlus size={16} />
                Nouvelle réservation
              </button>
            </div>
          </header>
          <CompanyDashboardFreshnessBadge
            lastHttpSyncAt={lastDataSyncAt}
            isSyncing={loadingReservations || loadingDriver || loadingRealtimeDashboard}
            realtimeConnected={socketConnected}
            realtimeDegraded={realtimeDegraded}
            className={styles.dashboardFreshness}
          />

          {/* ============ 2. KPI + MODE DISPATCH ============ */}
          <OverviewCards
            reservations={reservations}
            pendingReservations={pendingReservations}
            assignedReservations={assignedReservations}
            driver={driver}
            day={dispatchDay}
            delayCount={delaysError ? 0 : activeDelayCount || 0}
            hasCriticalDelays={!delaysError && !!hasCriticalDelays}
            kpiStats={bootstrap?.kpi || null}
            delaysError={Boolean(delaysError)}
            driversError={Boolean(driversError)}
            bookingsTruncated={Boolean(bootstrap?.bookings_truncated)}
            bookingsLimit={bootstrap?.bookings_limit ?? null}
          />

          <DispatchModeStatusBar
            mode={dispatchMode}
            opportunities={opportunities}
          />

          {/* ============ 3. VUE OPÉRATIONNELLE ============ */}
          {/* Mode Auto : IA prioritaire au-dessus de la carte */}
          {isAutoMode && (
            <OpportunitiesSection
              opportunities={opportunities}
              companyPublicId={company?.public_id}
              loading={loadingRealtimeDashboard}
              error={Boolean(realtimeDashboardError)}
              onAction={handleOpportunityAction}
            />
          )}

          {/* Pleine largeur sauf semi-auto (colonne IA à droite).
              Important : tant que dispatchMode est null (chargement), ne pas
              utiliser twoColumnLayout — sinon la carte s'affiche à ~55 % puis
              « saute » en pleine largeur quand le mode arrive. */}
          <div className={isSemiAutoMode ? styles.twoColumnLayout : styles.singleColumnLayout}>
            <div className={isSemiAutoMode ? styles.leftColumn : styles.fullColumn}>
              <section
                ref={mapSectionRef}
                className={styles.mapSection}
                data-tour-id="dispatch-assign"
              >
                {liveMapEnabled ? (
                  <Suspense
                    fallback={
                      <div className={styles.mapDeferred}>
                        <p className={styles.mapDeferredTitle}>Chargement de la carte en cours</p>
                      </div>
                    }
                  >
                    <DriverLiveMap
                      date={dispatchDay}
                      drivers={driversForMap}
                      bookings={activeBookings}
                      assignments={assignmentsForMap}
                      delays={delaysByBooking}
                    />
                  </Suspense>
                ) : (
                  <div className={styles.mapDeferred}>
                    <p className={styles.mapDeferredTitle}>Chargement de la carte en cours</p>
                    <p className={styles.mapDeferredText}>
                      Le reste du tableau de bord s&apos;affiche d&apos;abord, puis la carte
                      s&apos;active toute seule (Google Maps) pour ne pas bloquer l&apos;interface.
                    </p>
                    <button
                      type="button"
                      className={styles.mapDeferredButton}
                      onClick={handleEnableLiveMap}
                    >
                      Activer maintenant
                    </button>
                  </div>
                )}
                {fetchingDelays && <small className={styles.hint}>Mise à jour des retards...</small>}
              </section>
            </div>

            {/* Mode Semi-Auto : IA en colonne droite */}
            {isSemiAutoMode && (
              <div className={styles.rightColumn}>
                <OpportunitiesSection
                  opportunities={opportunities}
                  companyPublicId={company?.public_id}
                  loading={loadingRealtimeDashboard}
                  error={Boolean(realtimeDashboardError)}
                  onAction={handleOpportunityAction}
                />
              </div>
            )}
          </div>

          {/* ============ 4. RÉSERVATIONS — PLEINE LARGEUR ============ */}
          <section className={`${styles.reservationsFullSection} ${urgenceMode ? styles.urgenceSection : ''}`} data-tour-id="dispatch-followup">
            {!urgenceMode &&
              reservationTab === 'pending' &&
              !loadingReservations &&
              pendingOnOtherDays.length > 0 &&
              displayPending.length === 0 && (
                <div className={styles.pendingScopeBanner} role="status">
                  <p>
                    <strong>{pendingOnOtherDays.length}</strong> demande
                    {pendingOnOtherDays.length > 1 ? 's' : ''} en attente sur d&apos;autres dates que le jour affiché (
                    <strong>{dispatchDay}</strong>). Les courses du portail client sont filtrées par{' '}
                    <strong>date de prise en charge</strong> : choisissez la bonne date en haut à gauche.
                  </p>
                  <div className={styles.pendingScopeActions}>
                    {firstOtherDayYmd ? (
                      <button
                        type="button"
                        className={styles.pendingScopeJumpBtn}
                        onClick={() => setDispatchDay(firstOtherDayYmd)}
                      >
                        Afficher le {firstOtherDayYmd.split('-').reverse().join('.')}
                      </button>
                    ) : null}
                  </div>
                </div>
              )}
            {!urgenceMode && reservationTab === 'pending' && pendingHiddenByFilters ? (
              <div className={styles.pendingScopeBanner} role="status">
                <p>
                  Des courses <strong>en attente</strong> existent pour ce jour, mais les filtres (chauffeur,
                  horaire, recherche…) les masquent. Réinitialisez les filtres ou élargissez la plage horaire.
                </p>
              </div>
            ) : null}
            <ReservationFilterBar
              searchQuery={searchQuery}
              onSearchChange={setSearchQuery}
              urgenceMode={urgenceMode}
              onToggleUrgence={() => {
                setUrgenceMode((m) => !m);
                if (!urgenceMode) setDelaysOnly(false);
              }}
              delaysOnly={delaysOnly}
              onToggleDelaysOnly={() => setDelaysOnly((d) => !d)}
              drivers={driver || []}
              selectedDriver={selectedDriver}
              onDriverChange={setSelectedDriver}
              institutions={institutionOptions}
              selectedInstitution={selectedInstitution}
              onInstitutionChange={setSelectedInstitution}
              timeRange={timeRange}
              onTimeRangeChange={setTimeRange}
              onRefresh={handleRefresh}
              visibleCount={
                urgenceMode
                  ? displayUrgent.length
                  : reservationTab === 'pending'
                    ? displayPending.length
                    : reservationTab === 'institution'
                      ? visibleInstitutionOffers.length
                      : displayAssigned.length
              }
              totalCount={
                urgenceMode
                  ? urgentReservations.length
                  : reservationTab === 'pending'
                    ? pendingReservations.length
                    : reservationTab === 'institution'
                      ? visibleInstitutionOffers.length
                      : assignedReservations.length
              }
              activeDelayCount={activeDelayCount}
            />

            {!urgenceMode && (
              <div className={styles.tabsHeader} data-active-tab={reservationTab}>
                <button
                  type="button"
                  className={`${styles.tab} ${reservationTab === 'pending' ? styles.tabActive : ''}`}
                  data-tour-id="tab-pending"
                  onClick={() => setReservationTab('pending')}
                  aria-label="À décider"
                >
                  À décider
                  <span className={styles.tabBadge}>
                    {bootstrap?.kpi?.pending_decision ?? pendingReservations.length}
                  </span>
                </button>
                <button
                  type="button"
                  className={`${styles.tab} ${reservationTab === 'institution' ? styles.tabActive : ''}`}
                  data-tour-id="tab-institutions"
                  onClick={() => setReservationTab('institution')}
                >
                  Institutions
                  <span className={styles.tabBadge}>{visibleInstitutionOffers.length}</span>
                </button>
                <button
                  type="button"
                  className={`${styles.tab} ${reservationTab === 'assigned' ? styles.tabActive : ''}`}
                  data-tour-id="tab-assigned"
                  onClick={() => setReservationTab('assigned')}
                  aria-label="Sans chauffeur"
                >
                  Sans chauffeur
                  <span className={styles.tabBadge}>
                    {bootstrap?.kpi?.unassigned ?? assignedReservations.length}
                  </span>
                </button>
              </div>
            )}

            {!urgenceMode && reservationTab === 'assigned' ? (
              <p className={styles.tabHint} role="note">
                Les demandes à accepter (marché ouvert, sans chauffeur assigné) sont listées sous
                « En attente » avec le statut « pending ». Cet onglet regroupe les courses déjà
                acceptées en attente d&apos;assignation chauffeur.
              </p>
            ) : null}
            {!urgenceMode && reservationTab === 'institution' && hiddenInstitutionOffersCount > 0 ? (
              <div className={styles.pendingScopeBanner} role="status">
                <p>
                  {hiddenInstitutionOffersCount === 1 ? (
                    <>Une demande institution <strong>expirée</strong> n&apos;est plus affichée ici. L&apos;institution peut la relancer pour vous la renvoyer.</>
                  ) : (
                    <>{hiddenInstitutionOffersCount} demandes institutions <strong>expirées</strong> ne sont plus affichées ici. L&apos;institution peut les relancer.</>
                  )}
                </p>
              </div>
            ) : null}

            {urgenceMode ? (
              <ReservationTable
                reservations={displayUrgent}
                loading={loadingReservations}
                delays={delaysByBooking}
                onAccept={handleAccept}
                onReject={handleReject}
                onAssign={openAssignModal}
                onTransfer={handleOpenTransferModal}
                onTriggerReturn={handleTriggerReturn}
                onDelete={handleDeleteReservationClick}
                onSchedule={handleScheduleReservation}
                onDispatchNow={handleDispatchNow}
                currentCompanyId={company?.id}
              />
            ) : (
              <>
                {reservationTab === 'pending' && (
                  <ReservationTable
                    reservations={displayPending}
                    loading={loadingReservations}
                    delays={delaysByBooking}
                    onAccept={handleAccept}
                    onReject={handleReject}
                    onAssign={openAssignModal}
                    onTransfer={handleOpenTransferModal}
                    onTriggerReturn={handleTriggerReturn}
                    onDelete={handleDeleteReservationClick}
                    onSchedule={handleScheduleReservation}
                    onDispatchNow={handleDispatchNow}
                    hideSchedule={true}
                    currentCompanyId={company?.id}
                  />
                )}

                {reservationTab === 'institution' && (
                  <InstitutionOffersTable
                    offers={visibleInstitutionOffers}
                    loading={loadingInstitutionOffers}
                    onAccept={handleAcceptOffer}
                    onReject={handleRejectOffer}
                  />
                )}

                {reservationTab === 'assigned' && (
                  <ReservationTable
                    reservations={displayAssigned}
                    loading={loadingReservations}
                    delays={delaysByBooking}
                    onAssign={openAssignModal}
                    onTransfer={handleOpenTransferModal}
                    onTriggerReturn={handleTriggerReturn}
                    onDelete={handleDeleteReservationClick}
                    onSchedule={handleScheduleReservation}
                    onDispatchNow={handleDispatchNow}
                    hideSchedule={true}
                    currentCompanyId={company?.id}
                  />
                )}
              </>
            )}
          </section>

          {/* ============ 5. PERFORMANCE & FLOTTE (collapsible) ============ */}
          <section className={styles.performanceSection} data-tour-id="generate-invoice">
            <div
              className={styles.performanceHeader}
              onClick={() => setShowPerformance(!showPerformance)}
            >
              <div className={styles.performanceTitleRow}>
                <FiBarChart2 size={18} />
                <h2 className={styles.performanceTitle}>Performance & Flotte</h2>
              </div>
              <div className={styles.performanceInline}>
                <span className={styles.perfStat}>
                  {perfStats.completed}/{perfStats.total} réalisées
                </span>
                <span className={styles.perfStatSep}>|</span>
                <span className={styles.perfStat}>
                  {perfStats.revenue} CHF
                </span>
                {qualityMetrics?.on_time_rate !== undefined && (
                  <>
                    <span className={styles.perfStatSep}>|</span>
                    <span className={styles.perfStat}>
                      {Math.round(qualityMetrics.on_time_rate)}% ponctualité
                    </span>
                  </>
                )}
                <span className={styles.perfStatSep}>|</span>
                <span className={styles.perfStat}>
                  {(driver || []).filter((d) => d.is_active).length} chauffeurs
                </span>
              </div>
              <span className={styles.collapseIcon}>{showPerformance ? '▼' : '▶'}</span>
            </div>
            {showPerformance && (
              <div className={styles.performanceContent}>
                <div className={styles.performanceGrid}>
                  <div className={styles.performanceChart}>
                    <Suspense fallback={<div className={styles.mapDeferred}>Chargement du graphique…</div>}>
                      <ReservationChart reservations={reservations} />
                    </Suspense>
                  </div>
                  <div className={styles.performanceDrivers}>
                    <DriverTable
                      driver={driver}
                      loading={loadingDriver}
                      onToggle={handleToggleDriver}
                      onDelete={handleDeleteDriver}
                      onEdit={handleEditDriver}
                      onToggleAvailability={handleToggleAvailability}
                      onToggleType={handleToggleType}
                    />
                  </div>
                </div>
              </div>
            )}
          </section>
        </main>

        {/* Modal réservation manuelle */}
        {showBookingModal && (
          <Modal onClose={() => setShowBookingModal(false)} size="xl" className="modal-booking">
            <Suspense fallback={<div>Chargement du formulaire…</div>}>
              <ManualBookingForm
                onSubmitStart={() => setShowBookingModal(false)}
                onSuccess={(booking) => {
                  handleManualBookingSuccess(booking);
                }}
                onClose={() => setShowBookingModal(false)}
              />
            </Suspense>
          </Modal>
        )}

        {/* Modal édition chauffeur */}
        {showEditModal && driverToEdit && (
          <Modal onClose={handleCloseModal}>
            <h3>Modifier le chauffeur {driverToEdit.username}</h3>
            <Suspense fallback={<div>Chargement…</div>}>
              <EditDriverForm driver={driverToEdit} onClose={handleCloseModal} />
            </Suspense>
          </Modal>
        )}

      {company?.id && !chatWidgetActive && (
        <button
          type="button"
          className="chat-widget-button"
          onClick={() => setChatWidgetActive(true)}
          aria-label="Ouvrir le chat équipe"
        >
          <FiMessageSquare size={24} />
        </button>
      )}
      {company?.id && chatWidgetActive && (
        <Suspense fallback={null}>
          <ChatWidget companyId={company.id} startOpen />
        </Suspense>
      )}

      {(scheduleModalOpen || assignModalOpen || deleteModalOpen) && (
        <Suspense fallback={null}>
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
            assignModalDrivers={(driver || []).filter((d) => d.is_active)}
            onAssignConfirm={handleAssignDriver}
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
        </Suspense>
      )}

      {transferModalOpen && (
        <Suspense fallback={null}>
          <TransferBookingModal
            isOpen={transferModalOpen}
            onClose={() => {
              setTransferModalOpen(false);
              setTransferModalReservation(null);
            }}
            reservation={transferModalReservation}
            onSuccess={handleTransferSuccess}
          />
        </Suspense>
      )}

      {/* Panel assignation rapide (Intelligence Dispatch) */}
      <QuickAssignPanel
        isOpen={quickAssignOpen}
        onClose={() => {
          setQuickAssignOpen(false);
          setQuickAssignOpportunity(null);
        }}
        opportunity={quickAssignOpportunity}
        booking={
          quickAssignOpportunity?.booking_id
            ? reservationsMap.get(quickAssignOpportunity.booking_id)
            : null
        }
        drivers={driver || []}
        onAssign={handleQuickAssign}
        assigning={quickAssigning}
      />
    </>
  );
};

export default CompanyDashboard;
