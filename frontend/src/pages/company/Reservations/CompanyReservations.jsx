import React, { useEffect, useState, useCallback, useMemo, useRef } from 'react';
import { useSearchParams } from 'react-router-dom';
import { FiPlus, FiDownload, FiInbox, FiChevronLeft, FiChevronRight, FiChevronDown } from 'react-icons/fi';
import CompanyHeader from '../../../components/layout/Header/CompanyHeader';
import CompanySidebar from '../../../components/layout/Sidebar/CompanySidebar/CompanySidebar';
import useCompanyData from '../../../hooks/useCompanyData';
import useUrlSearchSync from '../../../hooks/useUrlSearchSync';
import {
  fetchCompanyReservationsPaginated,
  deleteReservation,
  acceptReservation,
  rejectReservation,
  scheduleReservation,
  dispatchNowForReservation,
  updateReservation,
} from '../../../services/companyService';
import ReservationTable from '../Dashboard/components/ReservationTable';
import ReservationDetailPanel from './components/ReservationDetailPanel';
import CancellationModal from '../../../components/reservations/CancellationModal';
import ReservationStats from './components/ReservationStats';
import ReservationFilters from './components/ReservationFilters';
import ReservationMapView from './components/ReservationMapView';
import TopClients from './components/TopClients';
import ReservationModals from '../../../components/reservations/ReservationModals';
import TransferBookingModal from '../../../components/reservations/TransferBookingModal';
import ManualBookingForm from '../Dashboard/components/ManualBookingForm';
import Modal from '../../../components/common/Modal';
import { toast } from 'sonner';
import { isCompletedStatus } from '../../../utils/reservationStatusUtils';
import { exportReservationsExcel } from '../../../utils/exportReservationsExcel';
import styles from './CompanyReservations.module.css';

const PER_PAGE_OPTIONS = [10, 25, 50, 100];

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
  const { company } = useCompanyData();
  const [searchParams, setSearchParams] = useSearchParams();

  // Data states
  const [reservations, setReservations] = useState([]);
  const [selectedDay, setSelectedDay] = useState('all');
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter] = useState('all');
  const [sortOrder, setSortOrder] = useState('desc');
  const [currentPage, setCurrentPage] = useState(1);
  const [reservationsPerPage, setReservationsPerPage] = useState(25);
  const [totalReservations, setTotalReservations] = useState(0);
  const [totalPages, setTotalPages] = useState(0);

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

  // UI states (parent-owned)
  const [activeTab, setActiveTab] = useState('all');
  const [viewMode, setViewMode] = useState('table');
  const [alertFilter, setAlertFilter] = useState(null);
  const [topClientsOpen, setTopClientsOpen] = useState(false);
  const [alerts, setAlerts] = useState([]);
  const [stats, setStats] = useState({
    total: 0,
    pending: 0,
    inProgress: 0,
    completed: 0,
    canceled: 0,
    revenue: 0,
  });

  const [exporting, setExporting] = useState(false);

  const searchInputRef = useRef(null);
  const { initialSearch, shouldFocus, consumeFocus, initialized } = useUrlSearchSync();

  // Force table mode for date ranges
  useEffect(() => {
    const isDateRange = selectedDay && selectedDay.includes(':');
    if (isDateRange && viewMode === 'map') {
      setViewMode('table');
    }
  }, [selectedDay, viewMode]);

  // Calculate stats
  const calculateStats = useCallback((reservationsData) => {
    const newStats = {
      total: reservationsData.length,
      pending: reservationsData.filter((r) => r.status === 'pending').length,
      inProgress: reservationsData.filter((r) =>
        ['accepted', 'assigned', 'in_progress'].includes(r.status)
      ).length,
      completed: reservationsData.filter((r) => isCompletedStatus(r.status)).length,
      canceled: reservationsData.filter((r) => r.status === 'canceled').length,
      revenue: reservationsData
        .filter((r) => isCompletedStatus(r.status))
        .reduce((sum, r) => sum + (Number(r.amount) || 0), 0),
    };
    setStats(newStats);
  }, []);

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

  // Load reservations
  const loadReservations = useCallback(async () => {
    try {
      setLoading(true);
      const isDateRange = selectedDay && selectedDay.includes(':');
      const apiParam = selectedDay === 'all' || isDateRange ? null : selectedDay;
      const [startDate, endDate] = isDateRange ? selectedDay.split(':') : [null, null];

      const params = {
        date: apiParam,
        startDate: startDate || undefined,
        endDate: endDate || undefined,
        page: currentPage,
        perPage: reservationsPerPage,
        status: statusFilter !== 'all' ? statusFilter : undefined,
        tab: activeTab !== 'all' ? activeTab : undefined,
        search: searchTerm ? searchTerm.trim() || undefined : undefined,
        sortOrder,
        excludeCanceled: activeTab === 'all' && statusFilter !== 'canceled',
      };

      const data = await fetchCompanyReservationsPaginated(params);
      const reservationsData = Array.isArray(data?.reservations) ? data.reservations : [];

      setReservations(reservationsData);
      setTotalReservations(data?.total ?? reservationsData.length);
      setTotalPages(data?.total_pages ?? 0);

      if (data?.stats) {
        setStats(data.stats);
      } else {
        calculateStats(reservationsData);
      }

      generateAlerts(reservationsData);
    } catch (err) {
      console.error('Erreur lors du chargement des reservations :', err);
    } finally {
      setLoading(false);
    }
  }, [
    selectedDay,
    calculateStats,
    currentPage,
    reservationsPerPage,
    statusFilter,
    searchTerm,
    sortOrder,
    activeTab,
  ]);

  useEffect(() => {
    loadReservations();
  }, [loadReservations]);

  // Auto-open reservation from ?booking= query param (e.g. from notification click)
  useEffect(() => {
    const bookingIdParam = searchParams.get('booking');
    if (!bookingIdParam || loading) return;

    const bookingId = Number(bookingIdParam);
    if (!bookingId) return;

    // Try to find in current page first
    const found = reservations.find((r) => r.id === bookingId);
    if (found) {
      setSelectedReservation(found);
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
          setSelectedReservation(match);
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
  }, [searchParams, reservations, loading, setSearchParams]);

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
  }, [selectedDay, searchTerm, statusFilter, sortOrder, activeTab, reservationsPerPage]);

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
    setReservations((prev) => prev.filter((r) => r.id !== id));
    handleCloseConfirmModal();
  };

  const handleAccept = async (reservationId) => {
    try {
      await acceptReservation(reservationId);
      setReservations((prev) =>
        prev.map((r) => (r.id === reservationId ? { ...r, status: 'accepted' } : r))
      );
      loadReservations();
    } catch (err) {
      console.error("Erreur lors de l'acceptation:", err);
    }
  };

  const handleReject = async (reservationId) => {
    try {
      await rejectReservation(reservationId);
      setReservations((prev) =>
        prev.map((r) => (r.id === reservationId ? { ...r, status: 'rejected' } : r))
      );
      loadReservations();
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
    setSelectedReservation(resObj);
  };

  const handleConfirmEdit = async (updatedData) => {
    if (!editModalReservation) return;
    try {
      await updateReservation(editModalReservation.id, updatedData);
      setEditModalOpen(false);
      setEditModalReservation(null);
      loadReservations();
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
      loadReservations();
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
    loadReservations();
    toast.success('Course transferee avec succes');
  };

  const handleDispatchNow = async (reservation) => {
    try {
      await dispatchNowForReservation(reservation.id, 15);
      loadReservations();
    } catch (err) {
      console.error('Erreur lors du dispatch urgent:', err);
    }
  };

  // Export Excel handler
  const handleExport = useCallback(async () => {
    if (exporting) return;
    setExporting(true);
    try {
      const dataToExport = filteredReservations.length > 0 ? filteredReservations : reservations;
      const periodLabel = selectedDay === 'all'
        ? 'Toutes les periodes'
        : selectedDay.includes(':')
          ? `Du ${selectedDay.split(':')[0]} au ${selectedDay.split(':')[1]}`
          : selectedDay;

      const fileName = await exportReservationsExcel(dataToExport, {
        companyName: company?.name || 'Entreprise',
        periodLabel,
        stats,
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
    <div className={styles.companyContainer}>
      <CompanyHeader />
      <div className={styles.dashboard}>
        <CompanySidebar />
        <div className={`${styles.contentArea} ${selectedReservation ? styles.contentAreaWithPanel : ''}`}>
        <main className={styles.content}>

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
                onClick={handleExport}
                disabled={exporting || (totalReservations === 0 && !loading)}
                title={totalReservations === 0 ? 'Aucune donnee a exporter' : 'Exporter en Excel'}
              >
                <FiDownload size={16} className={exporting ? styles.exportSpin : ''} />
                {exporting ? 'Export...' : 'Exporter'}
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
            onRefresh={loadReservations}
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

          {/* Main content */}
          {loading ? (
            <div className={styles.loadingState}>
              <div className={styles.spinner} />
              <p>Chargement des reservations...</p>
            </div>
          ) : totalReservations === 0 && !alertFilter ? (
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
            <>
              {viewMode === 'table' ? (
                <>
                  <ReservationTable
                    reservations={filteredReservations}
                    onRowClick={(reservation) => setSelectedReservation(reservation)}
                    onDelete={handleDeleteRequest}
                    onAccept={handleAccept}
                    onReject={handleReject}
                    onEdit={handleEdit}
                    onTransfer={handleOpenTransferModal}
                    onSchedule={handleSchedule}
                    onDispatchNow={handleDispatchNow}
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
                <ReservationMapView reservations={mapReservations} />
              )}
            </>
          )}

          {/* Top clients collapsible */}
          <TopClients
            reservations={reservations}
            isOpen={topClientsOpen}
            onToggle={() => setTopClientsOpen((prev) => !prev)}
          />

          <CancellationModal
            isOpen={showConfirmModal}
            reservation={reservationToDelete}
            onConfirm={handleConfirmDelete}
            onClose={handleCloseConfirmModal}
          />

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

          <TransferBookingModal
            isOpen={transferModalOpen}
            onClose={() => {
              setTransferModalOpen(false);
              setTransferModalReservation(null);
            }}
            reservation={transferModalReservation}
            onSuccess={handleTransferSuccess}
          />

          {newBookingOpen && (
            <Modal onClose={() => setNewBookingOpen(false)} size="xl" className="modal-booking">
              <ManualBookingForm
                onSuccess={() => {
                  setNewBookingOpen(false);
                  loadReservations();
                  toast.success('Réservation créée');
                }}
                onClose={() => setNewBookingOpen(false)}
              />
            </Modal>
          )}
        </main>

        {/* Side panel */}
        {selectedReservation && (
          <aside className={styles.detailPanel}>
            <ReservationDetailPanel
              reservation={selectedReservation}
              onClose={() => setSelectedReservation(null)}
              onSave={async (id, data) => {
                await updateReservation(id, data);
                loadReservations();
              }}
              onDelete={handleDeleteRequest}
            />
          </aside>
        )}
        </div>
      </div>
    </div>
  );
};

export default CompanyReservations;
