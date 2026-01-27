// frontend/src/pages/company/BillingReview/BillingReviewPage.jsx
import React, { useState, useEffect, useCallback, useMemo, useDeferredValue, useRef } from 'react';
import { useSearchParams } from 'react-router-dom';
import styles from './BillingReviewPage.module.css';
import CompanyHeader from '../../../components/layout/Header/CompanyHeader';
import CompanySidebar from '../../../components/layout/Sidebar/CompanySidebar/CompanySidebar';
import {
  batchSetBookingPayer,
  fetchMonthlyReview,
  lockBooking,
  setBookingPayer,
  unlockBooking,
} from '../../../services/billingReviewService';
import useAuthToken from '../../../hooks/useAuthToken';
import BillingReviewFilters from './components/BillingReviewFilters';
import BillingReviewTable from './components/BillingReviewTable';
import SetPayerModal from './components/SetPayerModal';
import LockUnlockModal from './components/LockUnlockModal';
import BillingReviewDrawer from './components/BillingReviewDrawer';
import { getRecipientSourceLabel } from '../../../utils/billingRecipient';
import useUrlSearchSync from '../../../hooks/useUrlSearchSync';

const ALLOWED_STATUSES = new Set(['draft', 'needs_review', 'ready', 'locked']);
const KNOWN_SOURCES = new Set([
  'transport_voucher',
  'client_stay',
  'default_client',
  'manual_override',
  'import',
  'system_rule',
]);

const parseBool = (value) => value === '1' || value === 'true';
const clampInt = (value, min, max) => Math.min(Math.max(value, min), max);
const parseIntParam = (value) => {
  if (!value) return null;
  const parsed = parseInt(value, 10);
  return Number.isNaN(parsed) ? null : parsed;
};
const parseId = (value) => {
  const parsed = parseIntParam(value);
  if (!parsed || parsed < 0) return null;
  return parsed;
};

const BillingReviewPage = () => {
  const user = useAuthToken();
  const companyId = user?.companyId || user?.company_id;
  const companyPublicId = user?.public_id;
  const searchInputRef = useRef(null);
  const { initialSearch, shouldFocus, consumeFocus, initialized } = useUrlSearchSync();
  const [searchParams, setSearchParams] = useSearchParams();
  const didSyncFromUrlRef = useRef(false);
  const debugBilling = searchParams.get('debugBilling') === '1';

  const [bookings, setBookings] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Filtres
  const [filters, setFilters] = useState({
    year: new Date().getFullYear(),
    month: new Date().getMonth() + 1,
    status: null,
    billing_party_id: null,
    clinic_id: null,
  });

  const [localFilters, setLocalFilters] = useState({
    source: null,
    needs_review_only: false,
    search: '',
  });
  const defaultFiltersRef = useRef(filters);
  const defaultLocalFiltersRef = useRef(localFilters);
  const [pendingSource, setPendingSource] = useState(null);

  // Modals
  const [showSetPayerModal, setShowSetPayerModal] = useState(false);
  const [showLockModal, setShowLockModal] = useState(false);
  const [showUnlockModal, setShowUnlockModal] = useState(false);
  const [selectedBooking, setSelectedBooking] = useState(null);
  const [selectedBookingIds, setSelectedBookingIds] = useState([]);
  const [drawerBooking, setDrawerBooking] = useState(null);

  // Charger les données
  const loadData = useCallback(async () => {
    if (!companyId) return;

    try {
      setLoading(true);
      setError(null);
      const response = await fetchMonthlyReview({
        company_id: companyId,
        year: filters.year,
        month: filters.month,
        status: filters.status,
        billing_party_id: filters.billing_party_id,
        clinic_id: filters.clinic_id,
      });
      setBookings(response.data || []);
    } catch (err) {
      console.error('Erreur lors du chargement:', err);
      setError('Erreur lors du chargement des données.');
      setBookings([]);
    } finally {
      setLoading(false);
    }
  }, [companyId, filters]);

  const sourceOptions = useMemo(() => {
    const uniqueSources = new Map();
    bookings.forEach((booking) => {
      if (booking?.billing_source) {
        const key = String(booking.billing_source);
        if (!uniqueSources.has(key)) {
          uniqueSources.set(key, {
            value: key,
            label: getRecipientSourceLabel({ billing_source: key }),
          });
        }
      }
    });
    return Array.from(uniqueSources.values());
  }, [bookings]);

  const observedSources = useMemo(() => {
    const sources = new Set();
    bookings.forEach((booking) => {
      if (booking?.billing_source) {
        sources.add(String(booking.billing_source));
      }
    });
    return sources;
  }, [bookings]);

  const effectiveSourceWhitelist = useMemo(() => {
    const merged = new Set(KNOWN_SOURCES);
    observedSources.forEach((value) => merged.add(value));
    return merged;
  }, [observedSources]);

  const deferredSearch = useDeferredValue(localFilters.search);

  const visibleBookings = useMemo(() => {
    const search = (deferredSearch || '').toLowerCase().trim();
    return bookings.filter((booking) => {
      if (localFilters.source && booking?.billing_source !== localFilters.source) {
        return false;
      }
      if (localFilters.needs_review_only && booking?.status !== 'needs_review') {
        return false;
      }
      if (search) {
        const name = String(booking?.patient_name || '').toLowerCase();
        if (!name.includes(search)) {
          return false;
        }
      }
      return true;
    });
  }, [bookings, localFilters.needs_review_only, localFilters.source, deferredSearch]);

  const emptyStateMessage = localFilters.needs_review_only
    ? 'Aucun à vérifier.'
    : 'Aucun résultat.';


  const buildSearchParams = useCallback(() => {
    const params = new URLSearchParams();

    if (filters.year !== defaultFiltersRef.current.year) {
      params.set('year', String(filters.year));
    }
    if (filters.month !== defaultFiltersRef.current.month) {
      params.set('month', String(filters.month));
    }
    if (filters.status !== defaultFiltersRef.current.status && filters.status) {
      params.set('status', String(filters.status));
    }
    if (
      filters.billing_party_id !== defaultFiltersRef.current.billing_party_id
      && filters.billing_party_id
    ) {
      params.set('billing_party_id', String(filters.billing_party_id));
    }
    if (filters.clinic_id !== defaultFiltersRef.current.clinic_id && filters.clinic_id) {
      params.set('clinic_id', String(filters.clinic_id));
    }

    if (localFilters.source !== defaultLocalFiltersRef.current.source && localFilters.source) {
      params.set('source', String(localFilters.source));
    }
    if (
      localFilters.needs_review_only !== defaultLocalFiltersRef.current.needs_review_only
      && localFilters.needs_review_only
    ) {
      params.set('needs_review_only', '1');
    }
    if (localFilters.search !== defaultLocalFiltersRef.current.search && localFilters.search) {
      params.set('search', String(localFilters.search));
    }

    return params;
  }, [filters, localFilters]);

  useEffect(() => {
    const hasAnyParam =
      searchParams.has('year')
      || searchParams.has('month')
      || searchParams.has('status')
      || searchParams.has('billing_party_id')
      || searchParams.has('clinic_id')
      || searchParams.has('source')
      || searchParams.has('needs_review_only')
      || searchParams.has('search')
      || searchParams.has('q');

    if (!hasAnyParam) {
      const defaultFilters = defaultFiltersRef.current;
      const defaultLocalFilters = defaultLocalFiltersRef.current;
      const filtersChanged = Object.keys(defaultFilters).some(
        (key) => defaultFilters[key] !== filters[key]
      );
      const localFiltersChanged = Object.keys(defaultLocalFilters).some(
        (key) => defaultLocalFilters[key] !== localFilters[key]
      );
      if (filtersChanged) {
        setFilters(defaultFilters);
      }
      if (localFiltersChanged) {
        setLocalFilters(defaultLocalFilters);
      }
      didSyncFromUrlRef.current = true;
      return;
    }

    const nextFilters = {
      year: searchParams.has('year')
        ? clampInt(
          parseIntParam(searchParams.get('year')) || defaultFiltersRef.current.year,
          2000,
          2100
        )
        : filters.year,
      month: searchParams.has('month')
        ? clampInt(
          parseIntParam(searchParams.get('month')) || defaultFiltersRef.current.month,
          1,
          12
        )
        : filters.month,
      status: searchParams.has('status')
        ? (ALLOWED_STATUSES.has(searchParams.get('status') || '')
            ? searchParams.get('status')
            : null)
        : filters.status,
      billing_party_id: searchParams.has('billing_party_id')
        ? parseId(searchParams.get('billing_party_id'))
        : filters.billing_party_id,
      clinic_id: searchParams.has('clinic_id')
        ? parseId(searchParams.get('clinic_id'))
        : filters.clinic_id,
    };

    const requestedSource = searchParams.has('source')
      ? String(searchParams.get('source') || '')
      : null;
    const nextLocal = {
      source: requestedSource
        ? (KNOWN_SOURCES.has(requestedSource) ? requestedSource : null)
        : localFilters.source,
      needs_review_only: searchParams.has('needs_review_only')
        ? parseBool(searchParams.get('needs_review_only'))
        : localFilters.needs_review_only,
      search: searchParams.has('search') || searchParams.has('q')
        ? (searchParams.get('search') || searchParams.get('q') || '').trim()
        : localFilters.search,
    };

    const filtersChanged = Object.keys(nextFilters).some(
      (key) => nextFilters[key] !== filters[key]
    );
    const localFiltersChanged = Object.keys(nextLocal).some(
      (key) => nextLocal[key] !== localFilters[key]
    );

    if (requestedSource && !KNOWN_SOURCES.has(requestedSource)) {
      if (debugBilling) {
        console.log('[BillingReview] Source pending:', requestedSource);
      }
      setPendingSource(requestedSource);
    } else if (!requestedSource) {
      setPendingSource(null);
    }

    if (filtersChanged) {
      setFilters((prev) => ({ ...prev, ...nextFilters }));
    }

    if (localFiltersChanged) {
      setLocalFilters((prev) => ({ ...prev, ...nextLocal }));
    }

    didSyncFromUrlRef.current = true;
  }, [searchParams, filters, localFilters, debugBilling]);

  useEffect(() => {
    if (!pendingSource) return;
    if (observedSources.has(pendingSource)) {
      if (debugBilling) {
        console.log('[BillingReview] Source acceptée (observée):', pendingSource);
      }
      if (localFilters.source !== pendingSource) {
        setLocalFilters((prev) => ({ ...prev, source: pendingSource }));
      }
      setPendingSource(null);
      return;
    }
    if (effectiveSourceWhitelist.has(pendingSource)) {
      if (debugBilling) {
        console.log('[BillingReview] Source acceptée (whitelist):', pendingSource);
      }
      return;
    }
    if (localFilters.source !== null) {
      if (debugBilling) {
        console.log('[BillingReview] Source rejetée:', pendingSource);
      }
      setLocalFilters((prev) => ({ ...prev, source: null }));
    }
    setPendingSource(null);
  }, [pendingSource, observedSources, effectiveSourceWhitelist, localFilters.source, debugBilling]);

  useEffect(() => {
    if (!didSyncFromUrlRef.current) return;
    const nextParams = buildSearchParams();
    const currentParams = new URLSearchParams(searchParams);
    currentParams.delete('focusSearch');

    if (nextParams.toString() !== currentParams.toString()) {
      setSearchParams(nextParams, { replace: true });
    }
  }, [buildSearchParams, searchParams, setSearchParams]);

  useEffect(() => {
    if (!initialized) return;

    if (initialSearch && initialSearch !== localFilters.search) {
      setLocalFilters((prev) => ({ ...prev, search: initialSearch }));
    }

    if (shouldFocus) {
      window.scrollTo({ top: 0, behavior: 'smooth' });
      requestAnimationFrame(() => {
        searchInputRef.current?.focus();
      });
      consumeFocus();
    }
  }, [initialized, initialSearch, shouldFocus, consumeFocus, localFilters.search]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  // Handlers pour les actions
  const handleSetPayer = async (bookingId, data) => {
    try {
      await setBookingPayer(bookingId, data);
      await loadData();
      setShowSetPayerModal(false);
      setSelectedBooking(null);
    } catch (err) {
      console.error('Erreur lors de la modification du payeur:', err);
      throw err;
    }
  };

  const handleLock = async (bookingId, reason) => {
    try {
      await lockBooking(bookingId, { reason });
      await loadData();
      setShowLockModal(false);
      setSelectedBooking(null);
    } catch (err) {
      console.error('Erreur lors du verrouillage:', err);
      throw err;
    }
  };

  const handleUnlock = async (bookingId, reason) => {
    try {
      await unlockBooking(bookingId, { reason });
      await loadData();
      setShowUnlockModal(false);
      setSelectedBooking(null);
    } catch (err) {
      console.error('Erreur lors du déverrouillage:', err);
      throw err;
    }
  };

  // Ouvrir modals
  const openSetPayerModal = (booking) => {
    setSelectedBooking(booking);
    setShowSetPayerModal(true);
  };

  const openLockModal = (booking) => {
    setSelectedBooking(booking);
    setShowLockModal(true);
  };

  const openUnlockModal = (booking) => {
    setSelectedBooking(booking);
    setShowUnlockModal(true);
  };

  const handleBatchAction = (bookingIds) => {
    // Pour l'action batch, on utilise le même modal mais avec plusieurs IDs
    setSelectedBooking({ booking_id: bookingIds[0], isBatch: true, booking_ids: bookingIds });
    setShowSetPayerModal(true);
  };

  const handleBatchSetPayer = async (bookingIds, data) => {
    try {
      await batchSetBookingPayer({
        booking_ids: bookingIds,
        ...data,
      });
      await loadData();
      setShowSetPayerModal(false);
      setSelectedBooking(null);
      setSelectedBookingIds([]);
    } catch (err) {
      console.error('Erreur lors de la modification batch du payeur:', err);
      throw err;
    }
  };

  return (
    <div className={styles.companyContainer}>
      <CompanyHeader />
      <div className={styles.dashboard}>
        <CompanySidebar />
        <main className={styles.content}>
          <div className={styles.header}>
            <h1>Contrôle facturation</h1>
            <p className={styles.subtitle}>
              Vérification et correction des décisions de facturation avant émission
            </p>
          </div>

          {error && <div className={styles.error}>{error}</div>}

          <BillingReviewFilters
            filters={filters}
            onFiltersChange={setFilters}
            companyId={companyId}
            sourceOptions={sourceOptions}
            localFilters={localFilters}
            onLocalFiltersChange={setLocalFilters}
            searchInputRef={searchInputRef}
          />

          <BillingReviewTable
            bookings={visibleBookings}
            loading={loading}
            onSetPayer={openSetPayerModal}
            onLock={openLockModal}
            onUnlock={openUnlockModal}
            isAdmin={user?.role === 'admin'}
            selectedIds={selectedBookingIds}
            onSelectionChange={setSelectedBookingIds}
            onBatchAction={handleBatchAction}
            onRowClick={(booking) => setDrawerBooking(booking)}
            emptyMessage={emptyStateMessage}
          />

          {showSetPayerModal && selectedBooking && (
            <SetPayerModal
              booking={selectedBooking}
              companyId={companyId}
              onClose={() => {
                setShowSetPayerModal(false);
                setSelectedBooking(null);
                setSelectedBookingIds([]);
              }}
              onSave={
                selectedBooking.isBatch
                  ? (_, data) => handleBatchSetPayer(selectedBooking.booking_ids, data)
                  : handleSetPayer
              }
            />
          )}

          {showLockModal && selectedBooking && (
            <LockUnlockModal
              booking={selectedBooking}
              mode="lock"
              onClose={() => {
                setShowLockModal(false);
                setSelectedBooking(null);
              }}
              onConfirm={handleLock}
            />
          )}

          {showUnlockModal && selectedBooking && (
            <LockUnlockModal
              booking={selectedBooking}
              mode="unlock"
              onClose={() => {
                setShowUnlockModal(false);
                setSelectedBooking(null);
              }}
              onConfirm={handleUnlock}
            />
          )}

          <BillingReviewDrawer
            booking={drawerBooking}
            isOpen={!!drawerBooking}
            onClose={() => setDrawerBooking(null)}
            companyPublicId={companyPublicId}
            onOpenSetPayer={(booking) => {
              setDrawerBooking(null);
              openSetPayerModal(booking);
            }}
          />
        </main>
      </div>
    </div>
  );
};

export default BillingReviewPage;
