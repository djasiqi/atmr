import React, { useEffect, useState, useCallback, useMemo, useRef, Suspense, lazy } from 'react';
import { useQuery, keepPreviousData, useQueryClient } from '@tanstack/react-query';
import { useSearchParams } from 'react-router-dom';
import { FiPlus, FiDownload, FiInbox, FiChevronLeft, FiChevronRight, FiChevronDown } from 'react-icons/fi';
import { useLirieCompany } from '../../../hooks/useLirieCompany';
import useUrlSearchSync from '../../../hooks/useUrlSearchSync';
import {
  fetchCompanyReservationsPaginated,
  deleteReservation,
  acceptReservation,
  rejectReservation,
  scheduleReservation,
  dispatchNowForReservation,
  updateReservation,
  fetchRequestOffers,
  acceptRequestOffer,
  rejectRequestOffer,
} from '../../../services/companyService';
import { buildOfferIdentity } from '../../../utils/bookingIdentity';
import { getConfirmedScheduleParts, formatSchedulePartLabel } from '../../../utils/formatLegTime';
import { canRespondToInstitutionOffer, isInstitutionOfferExpired } from '../../../utils/institutionOfferResponse';
import { computeAcceptNowPickupIso } from '../../../utils/institutionOfferActions';
import ReservationTable from '../Dashboard/components/ReservationTable';
import ReservationTableSkeleton from '../Dashboard/components/ReservationTableSkeleton';
import ProposeOfferTimeModal from '../Dashboard/components/ProposeOfferTimeModal';
import ReservationStats from './components/ReservationStats';
import ReservationFilters from './components/ReservationFilters';
import TopClients from './components/TopClients';
import Modal from '../../../components/common/Modal';
import { toast } from 'sonner';
import { isCompletedStatus } from '../../../utils/reservationStatusUtils';
import { lirieKeys, lirieInvalidateCompanyReservationLists } from '../../../queryKeys/lirie';
import styles from './CompanyReservations.module.css';

// Lot 4 perf — jamais montés au premier rendu : carte, formulaire, panneaux de détail
// et modales ne pèsent sur le bundle/le réseau qu'à l'ouverture effective.
const ReservationMapView = lazy(() => import('./components/ReservationMapView'));
const ManualBookingForm = lazy(() => import('../Dashboard/components/ManualBookingForm'));
const InstitutionOfferDetailPanel = lazy(() => import('./components/InstitutionOfferDetailPanel'));
const ReservationDetailPanel = lazy(() => import('./components/ReservationDetailPanel'));
const CancellationModal = lazy(() => import('../../../components/reservations/CancellationModal'));
const ReservationModals = lazy(() => import('../../../components/reservations/ReservationModals'));
const TransferBookingModal = lazy(() => import('../../../components/reservations/TransferBookingModal'));

const PER_PAGE_OPTIONS = [10, 25, 50, 100];

const ISO_DAY_RE = /^\d{4}-\d{2}-\d{2}$/;

/** Jour ISO (YYYY-MM-DD) → libellé court JJ.MM. */
const shortDateFromIso = (iso) => {
  if (!ISO_DAY_RE.test(iso || '')) return '';
  const [, m, d] = iso.split('-');
  return `${d}.${m}`;
};

/**
 * Horaire d'affichage d'une demande institution (départ/RDV/retour),
 * dérivé des legs car `scheduled_time` peut être nul pour un RDV.
 */
const buildOfferScheduling = (req) => {
  const parts = getConfirmedScheduleParts(req);
  const timeLabel = parts.length ? parts.map(formatSchedulePartLabel).join(' · ') : '';
  const dateShort = shortDateFromIso(req.mission_date || req.scheduling?.mission_date);
  if (!timeLabel) {
    return { display_time: 'À définir', display_datetime: dateShort || 'À définir', time_defined: false };
  }
  return {
    display_time: timeLabel,
    display_datetime: dateShort ? `${dateShort} · ${timeLabel}` : timeLabel,
    time_defined: true,
  };
};

/** Normalise une offre institution en attente en ligne « réservation ». */
const buildInstitutionOfferRow = (offer) => {
  const req = offer.transport_request || {};
  const identity = buildOfferIdentity(offer);
  const canRespond = canRespondToInstitutionOffer(offer);
  const isExpired = isInstitutionOfferExpired(offer);

  return {
    id: `offer-${offer.id}`,
    __institutionOffer: true,
    __offerId: offer.id,
    __offer: offer,
    __offerCanRespond: canRespond,
    __offerExpired: isExpired,
    status: 'pending',
    is_return: false,
    is_round_trip: Boolean(req.is_round_trip),
    multi_stop: Boolean(req.multi_stop),
    route_group_id: null,
    amount: null,
    expires_at: offer.expires_at,
    __priceEstimate: offer.price_estimate || null,
    pickup_location: req.pickup_location,
    dropoff_location: req.dropoff_location,
    scheduling: req.scheduling || buildOfferScheduling(req),
    scheduled_time: req.scheduled_time,
    time_confirmed: req.pickup_time_confirmed,
    identity: {
      primary_label: identity.passengerLabel,
      secondary_label: identity.source?.name,
    },
  };
};

const EMPTY_STATS = {
  total: 0,
  pending: 0,
  inProgress: 0,
  completed: 0,
  canceled: 0,
  revenue: 0,
};

function computeStatsFromReservations(reservationsData) {
  return {
    total: reservationsData.length,
    pending: reservationsData.filter((r) => r.status === 'pending').length,
    inProgress: reservationsData.filter((r) =>
      ['accepted', 'assigned', 'in_progress', 'en_route'].includes(
        (r.status || '').toLowerCase()
      )
    ).length,
    completed: reservationsData.filter((r) => isCompletedStatus(r.status)).length,
    canceled: reservationsData.filter((r) => r.status === 'canceled').length,
    revenue: reservationsData
      .filter((r) => isCompletedStatus(r.status))
      .reduce((sum, r) => sum + (Number(r.amount) || 0), 0),
  };
}

function normalizeApiStats(raw) {
  if (!raw || typeof raw !== 'object') return null;
  return {
    total: Number(raw.total) || 0,
    pending: Number(raw.pending) || 0,
    inProgress: Number(raw.inProgress) || 0,
    completed: Number(raw.completed) || 0,
    canceled: Number(raw.canceled) || 0,
    revenue: Number(raw.revenue) || 0,
  };
}

function PerPageChip({ value, onChange }) {
  const [open, setOpen] = React.useState(false);
  const ref = React.useRef(null);

  React.useEffect(() => {
    if (!open) return;
    const onClick = (e) => { if (ref.current && !ref.current.contains(e.target)) setOpen(false); };
    const onKey = (e) => { if (e.key === 'Escape') setOpen(false); };
    document.addEventListener('mousedown', onClick);
    document.addEventListener('keydown', onKey);
    return () => { document.removeEventListener('mousedown', onClick); document.removeEventListener('keydown', onKey); };
  }, [open]);

  return (
    <div className={styles.perPageChipWrap} ref={ref}>
      <button
        type="button"
        className={styles.perPageChipBtn}
        onClick={() => setOpen((p) => !p)}
      >
        {value} / page
        <FiChevronDown size={11} className={`${styles.perPageArrow} ${open ? styles.perPageArrowOpen : ''}`} />
      </button>
      {open && (
        <div className={styles.perPageMenu}>
          {PER_PAGE_OPTIONS.map((n) => (
            <button
              key={n}
              type="button"
              className={`${styles.perPageOption} ${n === value ? styles.perPageOptionActive : ''}`}
              onClick={() => { onChange(n); setOpen(false); }}
            >
              {n}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

const CompanyReservations = () => {
  const { company } = useLirieCompany();
  const queryClient = useQueryClient();
  const [searchParams, setSearchParams] = useSearchParams();

  // Data / filtres (réservations via TanStack Query — cache, stale-while-revalidate, pas de reflash total)
  // ?date=YYYY-MM-DD (clic notification « Demande modifiée ») → préselectionne le jour.
  const [selectedDay, setSelectedDay] = useState(() => {
    const dateParam = searchParams.get('date');
    return /^\d{4}-\d{2}-\d{2}$/.test(dateParam || '') ? dateParam : 'all';
  });
  const [searchTerm, setSearchTerm] = useState('');
  const [debouncedSearchTerm, setDebouncedSearchTerm] = useState('');
  const [statusFilter] = useState('all');
  const [sortOrder, setSortOrder] = useState('desc');
  const [currentPage, setCurrentPage] = useState(1);
  const [reservationsPerPage, setReservationsPerPage] = useState(25);

  // Modal states
  const [selectedReservation, setSelectedReservation] = useState(null);
  const [showConfirmModal, setShowConfirmModal] = useState(false);
  const [reservationToDelete, setReservationToDelete] = useState(null);
  const [scheduleModalOpen, setScheduleModalOpen] = useState(false);
  const [scheduleModalReservation, setScheduleModalReservation] = useState(null);
  const [transferModalOpen, setTransferModalOpen] = useState(false);
  const [transferModalReservation, setTransferModalReservation] = useState(null);
  const [editModalOpen, setEditModalOpen] = useState(false);
  const [editModalReservation, setEditModalReservation] = useState(null);
  const [newBookingOpen, setNewBookingOpen] = useState(false);
  const [proposeOffer, setProposeOffer] = useState(null);
  const [selectedOffer, setSelectedOffer] = useState(null);

  // UI states (parent-owned)
  const [activeTab, setActiveTab] = useState('all');
  const [viewMode, setViewMode] = useState('table');
  const [alertFilter, setAlertFilter] = useState(null);
  const [topClientsOpen, setTopClientsOpen] = useState(false);
  const [alerts, setAlerts] = useState([]);

  const [exporting, setExporting] = useState(false);

  const searchInputRef = useRef(null);
  const listAbortRef = useRef(null);
  const { initialSearch, shouldFocus, consumeFocus, initialized } = useUrlSearchSync();

  useEffect(() => {
    const t = setTimeout(() => {
      const next = (searchTerm || '').trim().slice(0, 100);
      // min 2 caractères pour recherche serveur ; vide = liste non filtrée
      if (next.length === 1) {
        setDebouncedSearchTerm('');
        return;
      }
      setDebouncedSearchTerm(next);
    }, 350);
    return () => clearTimeout(t);
  }, [searchTerm]);

  const openReservationPanel = useCallback((reservation) => {
    setSelectedOffer(null);
    setSelectedReservation(reservation);
  }, []);

  const openOfferPanel = useCallback((offer) => {
    setSelectedReservation(null);
    setSelectedOffer(offer);
  }, []);

  const listQueryFilterScope = useMemo(
    () => ({
      selectedDay,
      currentPage,
      reservationsPerPage: Math.min(Math.max(reservationsPerPage, 1), 100),
      statusFilter,
      activeTab,
      searchTerm: debouncedSearchTerm,
      sortOrder: sortOrder === 'asc' || sortOrder === 'desc' ? sortOrder : 'desc',
    }),
    [selectedDay, currentPage, reservationsPerPage, statusFilter, activeTab, debouncedSearchTerm, sortOrder]
  );

  // Portée des KPI — volontairement SANS `currentPage`/`reservationsPerPage` : les agrégats API
  // (base_query) ne dépendent pas de la pagination, donc changer de page ne doit ni invalider
  // ni refetcher les stats (Lot 4 perf).
  const statsQueryScope = useMemo(
    () => ({
      selectedDay,
      statusFilter,
      activeTab,
      searchTerm: debouncedSearchTerm,
      sortOrder: sortOrder === 'asc' || sortOrder === 'desc' ? sortOrder : 'desc',
    }),
    [selectedDay, statusFilter, activeTab, debouncedSearchTerm, sortOrder]
  );

  const canLoadReservations = Boolean(company?.id);

  const {
    data: listPayload,
    isLoading: listInitialLoading,
    isRefetching: listRefetching,
    refetch: refetchReservations,
  } = useQuery({
    queryKey: canLoadReservations
      ? lirieKeys.companyReservationsPaginated(company.id, listQueryFilterScope)
      : ['lirie', 'company-reservations-paginated', 'disabled'],
    enabled: canLoadReservations,
    queryFn: async ({ signal }) => {
      if (listAbortRef.current) {
        try {
          listAbortRef.current.abort();
        } catch {
          // ignore
        }
      }
      listAbortRef.current = typeof AbortController !== 'undefined' ? new AbortController() : null;
      const mergedSignal = signal || listAbortRef.current?.signal;
      const isDateRange = selectedDay && selectedDay.includes(':');
      const apiParam = selectedDay === 'all' || isDateRange ? null : selectedDay;
      const [startDate, endDate] = isDateRange ? selectedDay.split(':') : [null, null];
      const page = Math.max(Number(currentPage) || 1, 1);
      const perPage = Math.min(Math.max(Number(reservationsPerPage) || 25, 1), 100);
      return fetchCompanyReservationsPaginated({
        date: apiParam,
        startDate: startDate || undefined,
        endDate: endDate || undefined,
        page,
        perPage,
        status: statusFilter !== 'all' ? statusFilter : undefined,
        tab: activeTab !== 'all' ? activeTab : undefined,
        search: debouncedSearchTerm || undefined,
        sortOrder: sortOrder === 'asc' || sortOrder === 'desc' ? sortOrder : 'desc',
        excludeCanceled: activeTab === 'all' && statusFilter !== 'canceled',
        signal: mergedSignal,
      });
    },
    placeholderData: keepPreviousData,
    staleTime: 30_000,
  });

  // KPI indépendants de la pagination — clé de cache SANS `page`, `per_page` minimal
  // pour ne transférer aucune ligne de réservation (Lot 4 perf).
  const statsAbortRef = useRef(null);
  const { data: statsPayload } = useQuery({
    queryKey: canLoadReservations
      ? lirieKeys.companyReservationsStats(company.id, statsQueryScope)
      : ['lirie', 'company-reservations-stats', 'disabled'],
    enabled: canLoadReservations,
    queryFn: async ({ signal }) => {
      if (statsAbortRef.current) {
        try {
          statsAbortRef.current.abort();
        } catch {
          // ignore
        }
      }
      statsAbortRef.current = typeof AbortController !== 'undefined' ? new AbortController() : null;
      const mergedSignal = signal || statsAbortRef.current?.signal;
      const isDateRange = selectedDay && selectedDay.includes(':');
      const apiParam = selectedDay === 'all' || isDateRange ? null : selectedDay;
      const [startDate, endDate] = isDateRange ? selectedDay.split(':') : [null, null];
      return fetchCompanyReservationsPaginated({
        date: apiParam,
        startDate: startDate || undefined,
        endDate: endDate || undefined,
        page: 1,
        perPage: 1,
        status: statusFilter !== 'all' ? statusFilter : undefined,
        tab: activeTab !== 'all' ? activeTab : undefined,
        search: debouncedSearchTerm || undefined,
        sortOrder: sortOrder === 'asc' || sortOrder === 'desc' ? sortOrder : 'desc',
        excludeCanceled: activeTab === 'all' && statusFilter !== 'canceled',
        signal: mergedSignal,
      });
    },
    staleTime: 30_000,
  });

  const reservations = useMemo(
    () => (Array.isArray(listPayload?.reservations) ? listPayload.reservations : []),
    [listPayload]
  );
  const totalReservations = listPayload?.total ?? 0;
  const totalPages = listPayload?.total_pages ?? 0;

  // Garde le panneau ouvert aligné sur la liste (ex. après acceptation d'une modif)
  // sans remonter tout le panneau : fusion ciblée des champs opérationnels.
  useEffect(() => {
    const selectedId = selectedReservation?.id;
    if (!selectedId) return;
    const fresh = reservations.find((r) => r.id === selectedId);
    if (!fresh) return;
    const syncKeys = [
      'scheduled_time',
      'pickup_location',
      'dropoff_location',
      'wheelchair_client_has',
      'wheelchair_need',
      'requires_wheelchair',
      'medical_facility',
      'hospital_service',
      'doctor_name',
      'notes_medical',
      'amount',
      'status',
      'edit_version',
      'active_change_request',
      'active_change_request_id',
    ];
    setSelectedReservation((prev) => {
      if (!prev || prev.id !== fresh.id) return prev;
      let changed = false;
      const next = { ...prev };
      for (const key of syncKeys) {
        if (!Object.prototype.hasOwnProperty.call(fresh, key)) continue;
        if (JSON.stringify(prev[key]) !== JSON.stringify(fresh[key])) {
          next[key] = fresh[key];
          changed = true;
        }
      }
      return changed ? next : prev;
    });
  }, [reservations, selectedReservation?.id]);

  // Demandes institution en attente, fusionnées dans le tableau pour le jour sélectionné.
  const isSpecificDay = ISO_DAY_RE.test(selectedDay);
  const targetOfferIdParam = searchParams.get('offer');
  const targetRequestIdParam = searchParams.get('request');
  const needsOfferDateResolution = Boolean(
    !isSpecificDay && (targetOfferIdParam || targetRequestIdParam)
  );
  const { data: pendingOffersData } = useQuery({
    queryKey: lirieKeys.institutionOffers(),
    queryFn: () => fetchRequestOffers('PENDING'),
    enabled: canLoadReservations && (isSpecificDay || needsOfferDateResolution),
    staleTime: 15_000,
  });

  const institutionOfferRows = useMemo(() => {
    if (!isSpecificDay) return [];
    const offers = pendingOffersData?.offers || [];
    return offers
      .filter((o) => {
        const req = o.transport_request || {};
        const day = req.mission_date || req.scheduling?.mission_date;
        return day === selectedDay;
      })
      .map(buildInstitutionOfferRow);
  }, [pendingOffersData, selectedDay, isSpecificDay]);

  // Notifications anciennes : pas de mission_date en métadonnées.
  // On retrouve l'offre/la demande en attente puis on sélectionne son jour.
  useEffect(() => {
    if (!needsOfferDateResolution) return;
    const offers = pendingOffersData?.offers || [];
    if (!offers.length) return;

    const targetOfferId = targetOfferIdParam ? Number(targetOfferIdParam) : null;
    const targetRequestId = targetRequestIdParam ? Number(targetRequestIdParam) : null;
    const match = offers.find((offer) => {
      const req = offer.transport_request || {};
      return (
        (targetOfferId && Number(offer.id) === targetOfferId)
        || (targetRequestId && Number(req.id) === targetRequestId)
      );
    });
    const req = match?.transport_request || {};
    const day = req.mission_date || req.scheduling?.mission_date;
    if (!ISO_DAY_RE.test(day || '')) return;

    setSelectedDay(day);
    setCurrentPage(1);
    setSearchParams((prev) => {
      prev.delete('offer');
      prev.delete('request');
      return prev;
    }, { replace: true });
  }, [
    needsOfferDateResolution,
    pendingOffersData,
    targetOfferIdParam,
    targetRequestIdParam,
    setSearchParams,
  ]);

  const acceptOfferById = useCallback(
    async (offerId, proposedPickupTime, offerForGuard = null) => {
      if (offerForGuard && !canRespondToInstitutionOffer(offerForGuard)) {
        toast.error('Offre expirée, vous ne pouvez plus répondre.');
        return;
      }

      try {
        await acceptRequestOffer(offerId, proposedPickupTime);
        toast.success(
          proposedPickupTime
            ? 'Offre planifiée — réservation créée'
            : 'Offre validée — réservation créée'
        );
        queryClient.invalidateQueries({ queryKey: lirieKeys.institutionOffers() });
        void lirieInvalidateCompanyReservationLists(queryClient);
        refetchReservations();
      } catch (err) {
        toast.error(err?.response?.data?.error || "Erreur lors de l'acceptation");
      }
    },
    [queryClient, refetchReservations]
  );

  const handleValidateInstitutionOffer = useCallback(
    (row) => acceptOfferById(row.__offerId, undefined, row.__offer),
    [acceptOfferById]
  );

  const handlePlanInstitutionOffer = useCallback(
    (row) => {
      if (!canRespondToInstitutionOffer(row.__offer)) {
        toast.error('Offre expirée, vous ne pouvez plus répondre.');
        return;
      }

      setProposeOffer(row.__offer);
    },
    []
  );

  const handleAcceptNowInstitutionOffer = useCallback(
    (row) => acceptOfferById(row.__offerId, computeAcceptNowPickupIso(), row.__offer),
    [acceptOfferById]
  );

  /** @deprecated alias — Valider */
  const handleAcceptInstitutionOffer = handleValidateInstitutionOffer;

  /** @deprecated alias — Planifier */
  const handleProposeInstitutionOffer = handlePlanInstitutionOffer;

  const handleRejectInstitutionOffer = useCallback(
    async (offerId, offerForGuard = null) => {
      if (offerForGuard && !canRespondToInstitutionOffer(offerForGuard)) {
        toast.error('Offre expirée, vous ne pouvez plus répondre.');
        return;
      }

      try {
        await rejectRequestOffer(offerId);
        toast.success('Offre refusée');
        queryClient.invalidateQueries({ queryKey: lirieKeys.institutionOffers() });
      } catch (err) {
        toast.error(err?.response?.data?.error || 'Erreur lors du refus');
      }
    },
    [queryClient]
  );

  // KPI = agrégats API (même période / visibilité que le compteur total) — pas seulement la page courante.
  // Source primaire : query stats dédiée (clé sans page) ; repli sur les stats de la liste
  // (ex. premier rendu avant résolution de la query stats) puis calcul local.
  const stats = useMemo(() => {
    const fromStatsQuery = normalizeApiStats(statsPayload?.stats);
    if (fromStatsQuery) return fromStatsQuery;
    const fromApi = normalizeApiStats(listPayload?.stats);
    if (fromApi) return fromApi;
    if (!listPayload) return EMPTY_STATS;
    return computeStatsFromReservations(
      Array.isArray(listPayload.reservations) ? listPayload.reservations : []
    );
  }, [statsPayload, listPayload]);

  // Force table mode for date ranges
  useEffect(() => {
    const isDateRange = selectedDay && selectedDay.includes(':');
    if (isDateRange && viewMode === 'map') {
      setViewMode('table');
    }
  }, [selectedDay, viewMode]);

  // Generate alerts
  const generateAlerts = (reservationsData) => {
    const newAlerts = [];

    reservationsData
      .filter((r) => r.status === 'assigned' || r.status === 'in_progress')
      .forEach((r) => {
        const scheduledTime = new Date(r.scheduled_time);
        const now = new Date();
        const delayMinutes = Math.floor((now - scheduledTime) / (1000 * 60));

        if (delayMinutes > 15) {
          newAlerts.push({
            id: `delay-${r.id}`,
            type: 'delay',
            severity: delayMinutes > 30 ? 'high' : 'medium',
            message: `Course #${r.id} en retard`,
            reservation: r,
          });
        }
      });

    const unassignedCount = reservationsData.filter(
      (r) => r.status === 'accepted' && !r.driver_id
    ).length;
    if (unassignedCount > 0) {
      newAlerts.push({
        id: 'unassigned',
        type: 'unassigned',
        severity: 'medium',
        message: `${unassignedCount} sans chauffeur`,
        count: unassignedCount,
      });
    }

    setAlerts(newAlerts);
  };

  useEffect(() => {
    if (!listPayload) return;
    const reservationsData = Array.isArray(listPayload.reservations) ? listPayload.reservations : [];
    generateAlerts(reservationsData);
  }, [listPayload]);

  const afterListMutation = useCallback(() => {
    void lirieInvalidateCompanyReservationLists(queryClient);
  }, [queryClient]);

  const refetchListOnly = useCallback(() => {
    void refetchReservations();
  }, [refetchReservations]);

  const showListSkeleton = !canLoadReservations || listInitialLoading;

  // Applique ?date= puis nettoie l'URL (clic notification, y compris si la page est déjà montée).
  useEffect(() => {
    const dateParam = searchParams.get('date');
    if (!/^\d{4}-\d{2}-\d{2}$/.test(dateParam || '')) return;
    setSelectedDay(dateParam);
    setCurrentPage(1);
    setSearchParams((prev) => {
      prev.delete('date');
      return prev;
    }, { replace: true });
  }, [searchParams, setSearchParams]);

  // Auto-open reservation from ?booking= query param (e.g. from notification click)
  useEffect(() => {
    const bookingIdParam = searchParams.get('booking');
    if (!bookingIdParam || showListSkeleton) return;

    const bookingId = Number(bookingIdParam);
    if (!bookingId) return;

    // Try to find in current page first
    const found = reservations.find((r) => r.id === bookingId);
    if (found) {
      openReservationPanel(found);
      setSearchParams((prev) => {
        prev.delete('booking');
        return prev;
      }, { replace: true });
      return;
    }

    // If not in current page, fetch it via search
    const fetchAndOpen = async () => {
      try {
        const data = await fetchCompanyReservationsPaginated({
          search: String(bookingId),
          page: 1,
          perPage: 5,
          sortOrder: 'desc',
        });
        const match = (data?.reservations || []).find((r) => r.id === bookingId);
        if (match) {
          openReservationPanel(match);
        }
      } catch (err) {
        console.error('[CompanyReservations] Auto-open booking error:', err);
      } finally {
        setSearchParams((prev) => {
          prev.delete('booking');
          return prev;
        }, { replace: true });
      }
    };
    fetchAndOpen();
  }, [searchParams, reservations, showListSkeleton, setSearchParams, openReservationPanel]);

  useEffect(() => {
    if (!initialized) return;
    if (initialSearch && initialSearch !== searchTerm) {
      setSearchTerm(initialSearch);
    }
    if (shouldFocus) {
      window.scrollTo({ top: 0, behavior: 'smooth' });
      requestAnimationFrame(() => {
        searchInputRef.current?.focus();
      });
      consumeFocus();
    }
  }, [initialized, initialSearch, shouldFocus, consumeFocus, searchTerm]);

  useEffect(() => {
    setCurrentPage(1);
  }, [selectedDay, debouncedSearchTerm, statusFilter, sortOrder, activeTab, reservationsPerPage]);

  // --- CTA "Voir" handler ---
  const handleFilterByAlert = useCallback((type) => {
    if (type === 'delays') {
      setAlertFilter('delays');
    } else if (type === 'unassigned') {
      setAlertFilter('unassigned');
      setActiveTab('all');
    } else if (type === 'cancelled') {
      setAlertFilter(null);
      setActiveTab('canceled');
    }
  }, []);

  const handleClearAlertFilter = useCallback(() => {
    setAlertFilter(null);
  }, []);

  // Tab change always resets alertFilter
  const handleTabChange = useCallback((tabId) => {
    setActiveTab(tabId);
    setAlertFilter(null);
  }, []);

  // Apply alertFilter in display filtering
  const filteredReservations = useMemo(() => {
    if (!alertFilter) return reservations;

    if (alertFilter === 'delays') {
      return reservations.filter((r) => {
        const scheduledTime = new Date(r.scheduled_time);
        const now = new Date();
        const delayMinutes = Math.floor((now - scheduledTime) / (1000 * 60));
        return delayMinutes > 0 && ['assigned', 'in_progress'].includes(r.status);
      });
    }

    if (alertFilter === 'unassigned') {
      return reservations.filter((r) => r.status === 'accepted' && !r.driver_id);
    }

    return reservations;
  }, [reservations, alertFilter]);

  // Lignes du tableau = demandes institution en attente (jour sélectionné) + réservations.
  const tableReservations = useMemo(
    () => [...institutionOfferRows, ...filteredReservations],
    [institutionOfferRows, filteredReservations]
  );

  // Map reservations (single day only)
  const mapReservations = useMemo(() => {
    if (selectedDay === 'all') {
      const today = new Date();
      today.setHours(0, 0, 0, 0);
      const tomorrow = new Date(today);
      tomorrow.setDate(tomorrow.getDate() + 1);
      return reservations.filter((r) => {
        const d = new Date(r.scheduled_time || r.pickup_time);
        return d >= today && d < tomorrow;
      });
    }

    if (selectedDay && selectedDay.includes(':')) {
      const [s] = selectedDay.split(':');
      const start = new Date(s);
      start.setHours(0, 0, 0, 0);
      const end = new Date(start);
      end.setDate(end.getDate() + 1);
      return reservations.filter((r) => {
        const d = new Date(r.scheduled_time || r.pickup_time);
        return d >= start && d < end;
      });
    }

    const target = new Date(selectedDay);
    target.setHours(0, 0, 0, 0);
    const next = new Date(target);
    next.setDate(next.getDate() + 1);
    return reservations.filter((r) => {
      const d = new Date(r.scheduled_time || r.pickup_time);
      return d >= target && d < next;
    });
  }, [reservations, selectedDay]);

  // Visible page numbers
  const pageNumbers = useMemo(() => {
    if (totalPages <= 1) return [];
    const pages = [];
    const maxVisible = 5;
    let start = Math.max(1, currentPage - Math.floor(maxVisible / 2));
    let end = Math.min(totalPages, start + maxVisible - 1);
    if (end - start < maxVisible - 1) {
      start = Math.max(1, end - maxVisible + 1);
    }
    for (let i = start; i <= end; i++) {
      pages.push(i);
    }
    return pages;
  }, [currentPage, totalPages]);

  // --- Action handlers ---
  const handleDeleteRequest = (reservation) => {
    setReservationToDelete(reservation);
    setShowConfirmModal(true);
  };

  const handleCloseConfirmModal = () => {
    setShowConfirmModal(false);
    setReservationToDelete(null);
  };

  const handleConfirmDelete = async (reservationId, reasonCode = null, reasonText = null) => {
    const id = reservationId || reservationToDelete?.id;
    if (!id) return;
    await deleteReservation(id, reasonCode, reasonText);
    handleCloseConfirmModal();
    afterListMutation();
  };

  const handleAccept = async (reservationId) => {
    try {
      await acceptReservation(reservationId);
      afterListMutation();
    } catch (err) {
      console.error("Erreur lors de l'acceptation:", err);
    }
  };

  const handleReject = async (reservationId) => {
    try {
      await rejectReservation(reservationId);
      afterListMutation();
    } catch (err) {
      console.error('Erreur lors du rejet:', err);
    }
  };

  const handleEdit = (reservation) => {
    const resObj =
      typeof reservation === 'object'
        ? reservation
        : reservations.find((r) => r.id === reservation);
    if (!resObj) return;
    openReservationPanel(resObj);
  };

  const handleConfirmEdit = async (updatedData) => {
    if (!editModalReservation) return;
    try {
      await updateReservation(editModalReservation.id, updatedData);
      setEditModalOpen(false);
      setEditModalReservation(null);
      afterListMutation();
    } catch (err) {
      console.error("Erreur lors de l'edition:", err);
      throw err;
    }
  };

  const handleSchedule = (reservation) => {
    const resObj =
      typeof reservation === 'object'
        ? reservation
        : reservations.find((r) => r.id === reservation);
    if (!resObj) return;
    setScheduleModalReservation(resObj);
    setScheduleModalOpen(true);
  };

  const handleConfirmSchedule = async (data) => {
    setScheduleModalOpen(false);
    if (!scheduleModalReservation) return;

    try {
      let isoDatetime;
      if (typeof data === 'string') {
        isoDatetime = data;
      } else if (data?.return_time) {
        isoDatetime = data.return_time.replace('T', ' ');
      } else {
        throw new Error('Format de date invalide');
      }

      await scheduleReservation(scheduleModalReservation.id, isoDatetime);
      afterListMutation();
      setScheduleModalReservation(null);
    } catch (err) {
      console.error('Erreur lors de la planification:', err);
      setScheduleModalReservation(null);
      throw err;
    }
  };

  const handleOpenTransferModal = (reservation) => {
    const resObj =
      typeof reservation === 'object'
        ? reservation
        : reservations.find((r) => r.id === reservation);
    if (!resObj) return;
    setTransferModalReservation(resObj);
    setTransferModalOpen(true);
  };

  const handleTransferSuccess = () => {
    afterListMutation();
    toast.success('Course transferee avec succes');
  };

  const handleDispatchNow = async (reservation) => {
    try {
      await dispatchNowForReservation(reservation.id, 15);
      afterListMutation();
    } catch (err) {
      console.error('Erreur lors du dispatch urgent:', err);
    }
  };

  // Export Excel handler
  const handleExport = useCallback(async (profile = 'operational') => {
    if (exporting) return;
    setExporting(true);
    try {
      const dataToExport = filteredReservations.length > 0 ? filteredReservations : reservations;
      const periodLabel = selectedDay === 'all'
        ? 'Toutes les periodes'
        : selectedDay.includes(':')
          ? `Du ${selectedDay.split(':')[0]} au ${selectedDay.split(':')[1]}`
          : selectedDay;

      const { exportReservationsExcel } = await import('../../../utils/exportReservationsExcel');
      const fileName = await exportReservationsExcel(dataToExport, {
        companyName: company?.name || 'Entreprise',
        periodLabel,
        stats,
        profile,
      });
      toast.success(`Export termine : ${fileName}`);
    } catch (err) {
      console.error("Erreur lors de l'export:", err);
      toast.error("Erreur lors de l'export Excel");
    } finally {
      setExporting(false);
    }
  }, [exporting, filteredReservations, reservations, selectedDay, company, stats]);

  // Tabs
  const tabs = [
    { id: 'all', label: 'Toutes', count: stats.total },
    { id: 'pending', label: 'En attente', count: stats.pending },
    { id: 'in_progress', label: 'En cours', count: stats.inProgress },
    { id: 'completed', label: 'Terminees', count: stats.completed },
    { id: 'canceled', label: 'Annulees', count: stats.canceled },
  ];

  return (
    <div
      className={`${styles.contentArea} ${selectedReservation ? styles.contentAreaWithPanel : ''}`}
      data-tour-id="reservations-page"
    >
        <main className={styles.content} data-tour-id="reservations-board">

          {/* ===== ZONE A - Page Header ===== */}
          <div className={styles.pageHeader}>
            <div className={styles.pageHeaderLeft}>
              <h1 className={styles.pageTitle}>Reservations</h1>
              <p className={styles.pageSubtitle}>
                Gerez vos reservations et suivez leur statut en temps reel
              </p>
            </div>
            <div className={styles.pageHeaderActions}>
              <button
                type="button"
                className={styles.btnPrimary}
                onClick={() => setNewBookingOpen(true)}
              >
                <FiPlus size={16} />
                Nouvelle reservation
              </button>
              <button
                type="button"
                className={styles.btnSecondary}
                onClick={() => handleExport('operational')}
                disabled={exporting || (totalReservations === 0 && showListSkeleton)}
                title={totalReservations === 0 ? 'Aucune donnee a exporter' : 'Export operationnel (Passager + Origine)'}
              >
                <FiDownload size={16} className={exporting ? styles.exportSpin : ''} />
                {exporting ? 'Export...' : 'Exporter'}
              </button>
              <button
                type="button"
                className={styles.btnSecondary}
                onClick={() => handleExport('accounting')}
                disabled={exporting || (totalReservations === 0 && showListSkeleton)}
                title="Export comptable (amont, proprietaire, executant, payeur)"
              >
                Export compta
              </button>
            </div>
          </div>

          {/* ===== ZONE B - Command Bar (sticky, alerts integrated) ===== */}
          <ReservationFilters
            selectedDay={selectedDay}
            setSelectedDay={setSelectedDay}
            searchTerm={searchTerm}
            setSearchTerm={setSearchTerm}
            sortOrder={sortOrder}
            setSortOrder={setSortOrder}
            searchInputRef={searchInputRef}
            viewMode={viewMode}
            setViewMode={setViewMode}
            alertFilter={alertFilter}
            onClearAlertFilter={handleClearAlertFilter}
            onRefresh={refetchListOnly}
            totalResults={totalReservations}
            alerts={alerts}
            onFilterByAlert={handleFilterByAlert}
          />

          {/* ===== ZONE C - Resume ===== */}
          <ReservationStats stats={stats} />

          {/* ===== ZONE D - Liste principale ===== */}

          {/* Tabs pills */}
          <div className={styles.tabsPills}>
            {tabs.map((tab) => (
              <button
                key={tab.id}
                type="button"
                className={`${styles.pill} ${activeTab === tab.id ? styles.pillActive : ''}`}
                onClick={() => handleTabChange(tab.id)}
              >
                {tab.label}
                <span className={styles.pillCount}>{tab.count}</span>
              </button>
            ))}
          </div>

          {/* Main content : premier chargement = squelette tableau ; rechargements = contenu + barre d’activité */}
          {showListSkeleton ? (
            <ReservationTableSkeleton rowCount={Math.min(12, Math.max(6, reservationsPerPage))} />
          ) : totalReservations === 0 && institutionOfferRows.length === 0 && !alertFilter ? (
            <div className={styles.emptyState}>
              <FiInbox size={40} className={styles.emptyIcon} />
              <h3 className={styles.emptyTitle}>Aucune reservation trouvee</h3>
              <p className={styles.emptySubtitle}>
                Aucune reservation ne correspond a vos criteres de recherche.
              </p>
              <button
                type="button"
                className={styles.emptyCta}
                onClick={() => setNewBookingOpen(true)}
              >
                <FiPlus size={14} />
                Creer une reservation
              </button>
            </div>
          ) : (
            <div
              className={listRefetching ? styles.listBlockRefreshing : styles.listBlock}
              aria-busy={listRefetching}
            >
              {listRefetching && <div className={styles.listRefreshBar} role="status" aria-label="Mise à jour des réservations" />}
              {viewMode === 'table' ? (
                <>
                  <ReservationTable
                    reservations={tableReservations}
                    onRowClick={(row) => {
                      if (row.__institutionOffer) {
                        openOfferPanel(row.__offer);
                      } else {
                        openReservationPanel(row);
                      }
                    }}
                    onDelete={handleDeleteRequest}
                    onAccept={handleAccept}
                    onReject={handleReject}
                    onEdit={handleEdit}
                    onTransfer={handleOpenTransferModal}
                    onSchedule={handleSchedule}
                    onDispatchNow={handleDispatchNow}
                    onValidateInstitutionOffer={handleValidateInstitutionOffer}
                    onPlanInstitutionOffer={handlePlanInstitutionOffer}
                    onAcceptNowInstitutionOffer={handleAcceptNowInstitutionOffer}
                    onAcceptInstitutionOffer={handleAcceptInstitutionOffer}
                    onProposeInstitutionOffer={handleProposeInstitutionOffer}
                    onRejectInstitutionOffer={handleRejectInstitutionOffer}
                    hideAssign={true}
                    hideUrgent={true}
                    currentCompanyId={company?.id}
                  />

                  {/* Pagination enrichie */}
                  <div className={styles.paginationContainer}>
                    <div className={styles.paginationInfo}>
                      <span className={styles.resultCount}>
                        {totalReservations > 0
                          ? `${totalReservations} reservation${totalReservations > 1 ? 's' : ''}`
                          : 'Aucune reservation'}
                      </span>
                      <PerPageChip
                        value={reservationsPerPage}
                        onChange={(v) => { setReservationsPerPage(v); setCurrentPage(1); }}
                      />
                    </div>

                    {totalPages > 1 && (
                      <div className={styles.pagination}>
                        <button
                          disabled={currentPage === 1}
                          onClick={() => setCurrentPage(currentPage - 1)}
                          className={styles.paginationButton}
                          title="Page precedente"
                        >
                          <FiChevronLeft size={14} />
                        </button>

                        {pageNumbers.map((num) => (
                          <button
                            key={num}
                            onClick={() => setCurrentPage(num)}
                            className={`${styles.pageNumber} ${currentPage === num ? styles.pageNumberActive : ''}`}
                          >
                            {num}
                          </button>
                        ))}

                        <button
                          disabled={currentPage === totalPages}
                          onClick={() => setCurrentPage(currentPage + 1)}
                          className={styles.paginationButton}
                          title="Page suivante"
                        >
                          <FiChevronRight size={14} />
                        </button>
                      </div>
                    )}
                  </div>
                </>
              ) : (
                <Suspense fallback={<div className={styles.listBlock}>Chargement de la carte…</div>}>
                  <ReservationMapView reservations={mapReservations} />
                </Suspense>
              )}
            </div>
          )}

          {/* Top clients collapsible */}
          <TopClients
            reservations={reservations}
            isOpen={topClientsOpen}
            onToggle={() => setTopClientsOpen((prev) => !prev)}
          />

          {showConfirmModal && (
            <Suspense fallback={null}>
              <CancellationModal
                isOpen={showConfirmModal}
                reservation={reservationToDelete}
                onConfirm={handleConfirmDelete}
                onClose={handleCloseConfirmModal}
              />
            </Suspense>
          )}

          {(scheduleModalOpen || editModalOpen) && (
            <Suspense fallback={null}>
              <ReservationModals
                scheduleModalOpen={scheduleModalOpen}
                scheduleModalReservation={scheduleModalReservation}
                onScheduleConfirm={handleConfirmSchedule}
                onScheduleClose={() => {
                  setScheduleModalOpen(false);
                  setScheduleModalReservation(null);
                }}
                assignModalOpen={false}
                assignModalReservation={null}
                assignModalDrivers={[]}
                onAssignConfirm={() => {}}
                onAssignClose={() => {}}
                editModalOpen={editModalOpen}
                editModalReservation={editModalReservation}
                onEditConfirm={handleConfirmEdit}
                onEditClose={() => {
                  setEditModalOpen(false);
                  setEditModalReservation(null);
                }}
                deleteModalOpen={false}
                deleteModalReservation={null}
                onDeleteConfirm={() => {}}
                onDeleteClose={() => {}}
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

          {newBookingOpen && (
            <Modal onClose={() => setNewBookingOpen(false)} size="xl" className="modal-booking">
              <Suspense fallback={<div>Chargement du formulaire…</div>}>
                <ManualBookingForm
                  onSuccess={() => {
                    setNewBookingOpen(false);
                    afterListMutation();
                    toast.success('Réservation créée');
                  }}
                  onClose={() => setNewBookingOpen(false)}
                />
              </Suspense>
            </Modal>
          )}

          {proposeOffer && (
            <ProposeOfferTimeModal
              offer={proposeOffer}
              onConfirm={(offerId, isoTime) => {
                acceptOfferById(offerId, isoTime, proposeOffer);
                setProposeOffer(null);
              }}
              onClose={() => setProposeOffer(null)}
            />
          )}
        </main>

        {/* Side panel */}
        {selectedReservation && (
          <aside className={styles.detailPanel}>
            <Suspense fallback={null}>
              <ReservationDetailPanel
                reservation={selectedReservation}
                onClose={() => setSelectedReservation(null)}
                onSave={async (id, data) => {
                  await updateReservation(id, data);
                  afterListMutation();
                }}
                onDelete={handleDeleteRequest}
                onReservationUpdated={(updated) => {
                  if (updated?.id) setSelectedReservation(updated);
                  afterListMutation();
                }}
              />
            </Suspense>
          </aside>
        )}

        {/* Side panel — demande institution en attente (lecture seule + actions) */}
        {selectedOffer && (
          <aside className={styles.detailPanel}>
            <Suspense fallback={null}>
              <InstitutionOfferDetailPanel
                offer={selectedOffer}
                onClose={() => setSelectedOffer(null)}
                onValidate={(offer) => {
                  acceptOfferById(offer.id, undefined, offer);
                  setSelectedOffer(null);
                }}
                onPlan={(offer) => {
                  if (!canRespondToInstitutionOffer(offer)) {
                    toast.error('Offre expirée, vous ne pouvez plus répondre.');
                    return;
                  }

                  setProposeOffer(offer);
                  setSelectedOffer(null);
                }}
                onAcceptNow={(offer) => {
                  acceptOfferById(offer.id, computeAcceptNowPickupIso(), offer);
                  setSelectedOffer(null);
                }}
                onReject={(offerId) => {
                  handleRejectInstitutionOffer(offerId, selectedOffer);
                  setSelectedOffer(null);
                }}
              />
            </Suspense>
          </aside>
        )}
    </div>
  );
};

export default CompanyReservations;
