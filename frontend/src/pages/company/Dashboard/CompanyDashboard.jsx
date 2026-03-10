// src/pages/company/Dashboard/CompanyDashboard.jsx
import React, { useCallback, useState, useEffect, useMemo, useTransition } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { FiZap, FiPlus, FiBarChart2 } from 'react-icons/fi';
import useCompanySocket from '../../../hooks/useCompanySocket';
import useDispatchStatus from '../../../hooks/useDispatchStatus';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import CompanySidebar from '../../../components/layout/Sidebar/CompanySidebar/CompanySidebar';
import OverviewCards from './components/OverviewCards';
import ReservationChart from './components/ReservationChart';
import ReservationTable from './components/ReservationTable';
import DriverTable from '../../driver/components/Dashboard/DriverTable';
import ReservationModals from '../../../components/reservations/ReservationModals';
import TransferBookingModal from '../../../components/reservations/TransferBookingModal';
import DriverLiveMap from './components/DriverLiveMap';
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
} from '../../../services/companyService';
import useCompanyData from '../../../hooks/useCompanyData';
import useCompanyAuthToken from '../../../hooks/useCompanyAuthToken';
import useDispatchDelays from '../../../hooks/useDispatchDelays';
import useRealtimeDashboard from '../../../hooks/useRealtimeDashboard';
import { useDispatchMode } from '../../../hooks/useDispatchMode';
import styles from './CompanyDashboard.module.css';
import ManualBookingForm from './components/ManualBookingForm';
import ChatWidget from '../../../components/widgets/ChatWidget';
import EditDriverForm from '../components/EditDriverForm';
import Modal from '../../../components/common/Modal';
import CompanyHeader from '../../../components/layout/Header/CompanyHeader';
import InlineDatePicker from '../../../components/ui/InlineDatePicker';
import { Toaster, toast } from 'sonner';
import DemoInteractiveGuide from '../../../components/demo/DemoInteractiveGuide';

function makeToday() {
  const d = new Date();
  const pad = (n) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
}

const CompanyDashboard = () => {
  const location = useLocation();
  const isDemoEnv = (localStorage.getItem('lirie_auth_env') || '').toLowerCase() === 'demo';
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

  const {
    company,
    reservations,
    driver,
    loadingReservations,
    loadingDriver,
    reloadReservations,
    reloadDriver,
  } = useCompanyData({ day: dispatchDay });

  const socket = useCompanySocket();
  useDispatchStatus(socket);

  const { user, isCompanyAuthReady } = useCompanyAuthToken();
  const isCompanyOrAdmin = user && (user.isCompany || String(user.role || '').toLowerCase() === 'admin');

  const { delayCount, hasCriticalDelays, hasDelays } = useDispatchDelays(dispatchDay, 120000, isCompanyAuthReady && !!isCompanyOrAdmin);

  const { dispatchMode } = useDispatchMode();

  const isManualMode = dispatchMode === 'manual';
  const isAutoMode = dispatchMode === 'fully_auto' || dispatchMode === 'autonomous';
  const isSemiAutoMode = !!dispatchMode && !isManualMode && !isAutoMode;

  useEffect(() => {
    const handler = (e) => {
      const code = (e.detail?.code || e.detail?.message || '').toString();
      const authCodes = ['AUTH_REQUIRED', 'AUTH_INVALID', 'TOKEN_EXPIRED', 'AUTH_FORBIDDEN', 'COMPANY_NOT_FOUND', 'DRIVER_OR_COMPANY_NOT_FOUND'];
      if (authCodes.some((c) => code.includes(c))) {
        toast.error('Session expirée ou accès refusé. Reconnectez-vous.', { duration: 5000 });
      } else if (code.includes('RATE_LIMIT')) {
        toast.warning('Trop de tentatives de connexion. Réessayez dans quelques instants.', { duration: 5000 });
      } else if (code.includes('CONNECT_ERROR') || code) {
        toast.error('Connexion temps réel refusée. Vérifiez votre authentification.', { duration: 5000 });
      }
    };
    window.addEventListener('socket_connection_rejected', handler);
    return () => window.removeEventListener('socket_connection_rejected', handler);
  }, []);

  const {
    loading: loadingRealtimeDashboard,
    qualityMetrics,
    opportunities,
  } = useRealtimeDashboard(dispatchDay, 120000);

  const queryClient = useQueryClient();
  const [, startTransition] = useTransition();
  const [showEditModal, setShowEditModal] = useState(false);
  const [driverToEdit, setDriverToEdit] = useState(null);
  const [showBookingModal, setShowBookingModal] = useState(false);
  const [showPerformance, setShowPerformance] = useState(false);
  const [reservationTab, setReservationTab] = useState('pending');
  const [searchQuery, setSearchQuery] = useState('');
  const [urgenceMode, setUrgenceMode] = useState(false);
  const [delaysOnly, setDelaysOnly] = useState(false);
  const [selectedDriver, setSelectedDriver] = useState(null);
  const [selectedInstitution, setSelectedInstitution] = useState(null);
  const [timeRange, setTimeRange] = useState('all');
  const [quickAssignOpen, setQuickAssignOpen] = useState(false);
  const [quickAssignOpportunity, setQuickAssignOpportunity] = useState(null);
  const [quickAssigning, setQuickAssigning] = useState(false);

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
        queryClient.invalidateQueries(['reservations']);
      });
      toast.success('Dispatch urgent déclenché avec succès');
    } catch (e) {
      const errorData = e?.response?.data;
      const errorMessage = errorData?.message || errorData?.error;
      const status = e?.response?.status;

      console.debug('[DispatchNow] Error status:', status);
      console.debug('[DispatchNow] Error data:', errorData);
      console.debug('[DispatchNow] Error message:', errorMessage);

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
        const detailMessage =
          errorData?.message ||
          errorData?.error ||
          "Impossible de déclencher un retour d'urgence. La course aller doit être complétée avant de déclencher le retour.";

        console.debug('[DispatchNow] Showing warning:', detailMessage);
        toast.warning(detailMessage, { duration: 5000 });
        console.debug('Dispatch urgent refusé (comportement attendu):', detailMessage);
      } else {
        console.error('Dispatch urgent:', e);
        toast.error(errorMessage || 'Erreur lors du dispatch urgent.');
      }
    }
  };

  const { data: dispatchedReservations = [], refetch: refetchAssigned } = useQuery({
    queryKey: ['assigned-reservations', dispatchDay],
    queryFn: () => fetchAssignedReservations(dispatchDay),
    staleTime: 30_000,
    enabled: !!company?.id,
  });

  const {
    data: delays = [],
    refetch: refetchDelays,
    isFetching: fetchingDelays,
  } = useQuery({
    queryKey: ['dispatch-delays', dispatchDay],
    queryFn: () => fetchDispatchDelays(dispatchDay),
    initialData: [],
    staleTime: 20_000,
    enabled: !!company?.id,
  });

  const {
    data: institutionOffersData,
    refetch: refetchInstitutionOffers,
    isLoading: loadingInstitutionOffers,
  } = useQuery({
    queryKey: ['institution-offers'],
    queryFn: () => fetchRequestOffers('PENDING'),
    staleTime: 15_000,
    refetchInterval: 30_000,
    enabled: !!company?.id,
  });
  const institutionOffers = institutionOffersData?.offers || [];

  const handleNewReservation = useCallback(() => reloadReservations(), [reloadReservations]);
  useEffect(() => {
    if (!socket) return;
    socket.on('new_reservation', handleNewReservation);
    return () => socket.off('new_reservation', handleNewReservation);
  }, [socket, handleNewReservation]);

  useEffect(() => {
    if (!socket) return;
    const refetchAll = () => {
      startTransition(() => {
        refetchAssigned?.();
        reloadReservations?.();
        refetchDelays?.();
      });
    };
    const onAssignCreated = () => refetchAll();
    const onAssignUpdated = () => refetchAll();
    const onAssignDeleted = () => refetchAll();
    const onDispatchProgress = (_p) => {};
    const onDispatchError = (err) => {
      console.error('dispatch_error:', err);
      refetchAll();
    };
    const onDispatchRunCompleted = (data) => {
      console.log('Dispatch run completed:', data);
      refetchAll();
    };
    const onTransferReceived = (data) => {
      console.log('Transfert reçu:', data);
      refetchAll();
    };
    const onTransferProposed = (data) => {
      console.log('Transfert proposé:', data);
      refetchAll();
    };
    const onBookingUpdated = (data) => {
      console.log('Course mise à jour:', data);
      refetchAll();
    };
    socket.on('dispatch_assignment_created', onAssignCreated);
    socket.on('dispatch_assignment_updated', onAssignUpdated);
    socket.on('dispatch_assignment_cancelled', onAssignDeleted);
    socket.on('dispatch_progress', onDispatchProgress);
    socket.on('dispatch_error', onDispatchError);
    socket.on('dispatch_run_completed', onDispatchRunCompleted);
    socket.on('transfer_received', onTransferReceived);
    socket.on('transfer_proposed', onTransferProposed);
    socket.on('booking_updated', onBookingUpdated);
    return () => {
      socket.off('dispatch_assignment_created', onAssignCreated);
      socket.off('dispatch_assignment_updated', onAssignUpdated);
      socket.off('dispatch_assignment_cancelled', onAssignDeleted);
      socket.off('dispatch_progress', onDispatchProgress);
      socket.off('dispatch_error', onDispatchError);
      socket.off('dispatch_run_completed', onDispatchRunCompleted);
      socket.off('transfer_received', onTransferReceived);
      socket.off('transfer_proposed', onTransferProposed);
      socket.off('booking_updated', onBookingUpdated);
    };
  }, [socket, refetchAssigned, reloadReservations, refetchDelays]);

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
        queryClient.invalidateQueries(['reservations']);
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
        queryClient.invalidateQueries(['reservations']);
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
    const ymd = String(resp?.reservation?.scheduled_time || '').slice(0, 10);
    if (ymd) setDispatchDay(ymd);
    startTransition(() => {
      reloadReservations();
      queryClient.invalidateQueries(['reservations']);
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
      console.log('Réservation supprimée:', result);

      startTransition(() => {
        reloadReservations();
        queryClient.invalidateQueries(['reservations']);
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
          is_dropoff: !d.is_pickup,
        };
      }
    }
    return map;
  }, [delays]);

  const activeDelayCount = useMemo(() => {
    if (!delaysByBooking || typeof delaysByBooking !== 'object' || Array.isArray(delaysByBooking)) return 0;
    return Object.values(delaysByBooking).filter((d) => d?.delay_minutes > 0).length;
  }, [delaysByBooking]);

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
      'pending', 'accepted', 'assigned', 'en_route', 'in_progress',
      'onboard', 'en_route_pickup', 'en_route_dropoff',
    ];
    return (reservations || []).filter((r) => {
      const status = (r.status || '').toLowerCase();
      return activeStatuses.includes(status) || (delaysByBooking && delaysByBooking[r.id]?.delay_minutes > 0);
    });
  }, [reservations, urgenceMode, delaysByBooking]);

  const displayPending = useMemo(
    () => applyStructuredFilters(filterByDelaysOnly(filterBySearch(pendingReservations))),
    [pendingReservations, filterBySearch, filterByDelaysOnly, applyStructuredFilters]
  );

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
    });
  }, [reloadReservations, refetchAssigned, refetchDelays, refetchInstitutionOffers]);

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
        queryClient.invalidateQueries(['reservations']);
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
    <div className={styles.companyContainer}>
      <Toaster position="top-right" richColors />
      <CompanyHeader />

      <div className={styles.dashboard}>
        <CompanySidebar />
        <main className={styles.content}>
          {demoMission === 'transporteur' && <DemoInteractiveGuide role="transporteur" />}

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
                {hasDelays && (
                  <span className={styles.delayHeaderBadge}>
                    {delayCount} retard{delayCount !== 1 ? 's' : ''} actif{delayCount !== 1 ? 's' : ''}
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

          {/* ============ 2. KPI + MODE DISPATCH ============ */}
          <OverviewCards
            reservations={reservations}
            pendingReservations={pendingReservations}
            assignedReservations={assignedReservations}
            driver={driver}
            day={dispatchDay}
            delayCount={delayCount || 0}
            hasCriticalDelays={!!hasCriticalDelays}
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
              onAction={handleOpportunityAction}
            />
          )}

          <div className={isManualMode ? styles.singleColumnLayout : styles.twoColumnLayout}>
            <div className={isManualMode ? styles.fullColumn : styles.leftColumn}>
              <section className={styles.mapSection} data-tour-id="dispatch-assign">
                <DriverLiveMap
                  date={dispatchDay}
                  drivers={driver || []}
                  bookings={activeBookings}
                  assignments={assignmentsForMap}
                  delays={delaysByBooking}
                />
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
                  onAction={handleOpportunityAction}
                />
              </div>
            )}
          </div>

          {/* ============ 4. RÉSERVATIONS — PLEINE LARGEUR ============ */}
          <section className={`${styles.reservationsFullSection} ${urgenceMode ? styles.urgenceSection : ''}`} data-tour-id="dispatch-followup">
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
                      ? institutionOffers.length
                      : displayAssigned.length
              }
              totalCount={
                urgenceMode
                  ? (reservations || []).length
                  : reservationTab === 'pending'
                    ? pendingReservations.length
                    : reservationTab === 'institution'
                      ? institutionOffers.length
                      : assignedReservations.length
              }
              activeDelayCount={activeDelayCount}
            />

            {!urgenceMode && (
              <div className={styles.tabsHeader} data-active-tab={reservationTab}>
                <button
                  className={`${styles.tab} ${reservationTab === 'pending' ? styles.tabActive : ''}`}
                  data-tour-id="tab-pending"
                  onClick={() => setReservationTab('pending')}
                >
                  En attente
                  <span className={styles.tabBadge}>{pendingReservations.length}</span>
                </button>
                <button
                  className={`${styles.tab} ${reservationTab === 'institution' ? styles.tabActive : ''}`}
                  data-tour-id="tab-institutions"
                  onClick={() => setReservationTab('institution')}
                >
                  Institutions
                  <span className={styles.tabBadge}>{institutionOffers.length}</span>
                </button>
                <button
                  className={`${styles.tab} ${reservationTab === 'assigned' ? styles.tabActive : ''}`}
                  data-tour-id="tab-assigned"
                  onClick={() => setReservationTab('assigned')}
                >
                  Assignation chauffeur
                  <span className={styles.tabBadge}>{assignedReservations.length}</span>
                </button>
              </div>
            )}

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
                    offers={institutionOffers}
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
                    <ReservationChart reservations={reservations} />
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
            <ManualBookingForm
              onSubmitStart={() => setShowBookingModal(false)}
              onSuccess={(booking) => {
                handleManualBookingSuccess(booking);
              }}
              onClose={() => setShowBookingModal(false)}
            />
          </Modal>
        )}

        {/* Modal édition chauffeur */}
        {showEditModal && driverToEdit && (
          <Modal onClose={handleCloseModal}>
            <h3>Modifier le chauffeur {driverToEdit.username}</h3>
            <EditDriverForm driver={driverToEdit} onClose={handleCloseModal} />
          </Modal>
        )}
      </div>

      {company?.id && <ChatWidget companyId={company.id} />}

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

      {/* Modal de transfert */}
      <TransferBookingModal
        isOpen={transferModalOpen}
        onClose={() => {
          setTransferModalOpen(false);
          setTransferModalReservation(null);
        }}
        reservation={transferModalReservation}
        onSuccess={handleTransferSuccess}
      />

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
    </div>
  );
};

export default CompanyDashboard;
