import React, { useEffect, useState, useCallback, useMemo, useRef } from 'react';
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
import ReservationDetailsModal from '../Dashboard/components/ReservationDetailsModal';
import ConfirmationModal from '../../../components/common/ConfirmationModal';
import ReservationStats from './components/ReservationStats';
import ReservationFilters from './components/ReservationFilters';
import ReservationMapView from './components/ReservationMapView';
import ReservationAlerts from './components/ReservationAlerts';
import TopClients from './components/TopClients';
import ReservationModals from '../../../components/reservations/ReservationModals';
import TransferBookingModal from '../../../components/reservations/TransferBookingModal';
import { toast } from 'sonner';
import { isCompletedStatus } from '../../../utils/reservationStatusUtils';
import styles from './CompanyReservations.module.css';

const CompanyReservations = () => {
  // ✅ Récupérer les infos de l'entreprise connectée pour gérer les transferts
  const { company } = useCompanyData();
  
  // États existants
  const [reservations, setReservations] = useState([]);
  const [selectedDay, setSelectedDay] = useState('all'); // Par défaut : toutes les dates
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter] = useState('all');
  const [sortOrder, setSortOrder] = useState('desc'); // Par défaut : ordre décroissant (plus récent d'abord)
  const [currentPage, setCurrentPage] = useState(1);
  const [reservationsPerPage, setReservationsPerPage] = useState(25); // Nombre de réservations par page
  const [totalReservations, setTotalReservations] = useState(0);
  const [totalPages, setTotalPages] = useState(0);
  const [selectedReservation, setSelectedReservation] = useState(null);
  const [showConfirmModal, setShowConfirmModal] = useState(false);
  const [reservationToDelete, setReservationToDelete] = useState(null);
  const [scheduleModalOpen, setScheduleModalOpen] = useState(false);
  const [scheduleModalReservation, setScheduleModalReservation] = useState(null);
  const [transferModalOpen, setTransferModalOpen] = useState(false);
  const [transferModalReservation, setTransferModalReservation] = useState(null);
  const searchInputRef = useRef(null);
  const { initialSearch, shouldFocus, consumeFocus, initialized } = useUrlSearchSync();

  // Nouveaux états pour les améliorations
  const [activeTab, setActiveTab] = useState('all');
  const [viewMode, setViewMode] = useState('table'); // "table" ou "map"
  const [alerts, setAlerts] = useState([]);
  const [stats, setStats] = useState({
    total: 0,
    pending: 0,
    inProgress: 0,
    completed: 0,
    canceled: 0,
    revenue: 0,
  });

  // Forcer le mode tableau quand une plage de dates est sélectionnée
  useEffect(() => {
    const isDateRange = selectedDay && selectedDay.includes(':');
    if (isDateRange && viewMode === 'map') {
      setViewMode('table');
    }
  }, [selectedDay, viewMode]);

  // Calculer les statistiques
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

  // Chargement des réservations avec calculs des statistiques et alertes
  const loadReservations = useCallback(async () => {
    try {
      setLoading(true);
      // Si "Toutes les dates" ou une plage de dates, charger toutes les réservations
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

      if (process.env.REACT_APP_DEBUG_FILTERS === 'true') {
        // eslint-disable-next-line no-console
        console.debug('[Reservations] loadReservations params:', params);
      }

      const data = await fetchCompanyReservationsPaginated(params);

      const reservationsData = Array.isArray(data?.reservations)
        ? data.reservations
        : [];

      if (process.env.REACT_APP_DEBUG_FILTERS === 'true') {
        // eslint-disable-next-line no-console
        console.debug('[Reservations] API response:', {
          total: data?.total,
          count: reservationsData.length,
          sample: reservationsData[0]?.client_name,
        });
      }

      setReservations(reservationsData);
      setTotalReservations(data?.total ?? reservationsData.length);
      setTotalPages(data?.total_pages ?? 0);

      // Calculer les statistiques
      if (data?.stats) {
        setStats(data.stats);
      } else {
        calculateStats(reservationsData);
      }

      // Générer les alertes
      generateAlerts(reservationsData);
    } catch (err) {
      console.error('Erreur lors du chargement des réservations :', err);
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

  // Générer les alertes
  const generateAlerts = (reservationsData) => {
    const newAlerts = [];

    // Alertes de retard
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
            message: `Course #${r.id} en retard de ${delayMinutes} minutes`,
            reservation: r,
          });
        }
      });

    // Alertes de chauffeurs non assignés
    const unassignedCount = reservationsData.filter(
      (r) => r.status === 'accepted' && !r.driver_id
    ).length;
    if (unassignedCount > 0) {
      newAlerts.push({
        id: 'unassigned',
        type: 'unassigned',
        severity: 'medium',
        message: `${unassignedCount} course(s) sans chauffeur assigné`,
        count: unassignedCount,
      });
    }

    setAlerts(newAlerts);
  };

  useEffect(() => {
    loadReservations();
  }, [loadReservations]);

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

  // Dans le composant CompanyReservations

  const handleDeleteRequest = (reservation) => {
    setReservationToDelete(reservation);
    setShowConfirmModal(true);
  };

  const handleCloseConfirmModal = () => {
    setShowConfirmModal(false);
    setReservationToDelete(null);
  };

  const handleConfirmDelete = async () => {
    if (!reservationToDelete) return;
    try {
      await deleteReservation(reservationToDelete.id);
      setReservations((prev) => prev.filter((r) => r.id !== reservationToDelete.id));
    } catch (err) {
      console.error('Erreur lors de la suppression:', err);
    } finally {
      handleCloseConfirmModal();
    }
  };

  // Gestion des actions sur les réservations
  const handleAccept = async (reservationId) => {
    try {
      await acceptReservation(reservationId);
      // Mettre à jour la réservation dans la liste locale
      setReservations((prev) =>
        prev.map((r) => (r.id === reservationId ? { ...r, status: 'accepted' } : r))
      );
      // Recharger les réservations pour avoir les données fraîches
      loadReservations();
    } catch (err) {
      console.error("Erreur lors de l'acceptation:", err);
    }
  };

  const handleReject = async (reservationId) => {
    try {
      await rejectReservation(reservationId);
      // Mettre à jour localement
      setReservations((prev) =>
        prev.map((r) => (r.id === reservationId ? { ...r, status: 'rejected' } : r))
      );
      loadReservations();
    } catch (err) {
      console.error('Erreur lors du rejet:', err);
    }
  };

  // États pour la modale d'édition
  const [editModalOpen, setEditModalOpen] = useState(false);
  const [editModalReservation, setEditModalReservation] = useState(null);

  const handleEdit = (reservation) => {
    // Passer l'objet complet
    const resObj =
      typeof reservation === 'object'
        ? reservation
        : reservations.find((r) => r.id === reservation);
    if (!resObj) return;
    setEditModalReservation(resObj);
    setEditModalOpen(true);
  };

  const handleConfirmEdit = async (updatedData) => {
    if (!editModalReservation) return;
    try {
      await updateReservation(editModalReservation.id, updatedData);
      setEditModalOpen(false);
      setEditModalReservation(null);
      loadReservations();
    } catch (err) {
      console.error("Erreur lors de l'édition:", err);
      throw err;
    }
  };

  const handleSchedule = (reservation) => {
    // Passer l'objet complet
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
        // Format "YYYY-MM-DD HH:mm"
        isoDatetime = data;
      } else if (data?.return_time) {
        // Format { return_time: "YYYY-MM-DDTHH:mm" }
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
      throw err; // Laisser le modal afficher l'erreur
    }
  };

  // Handler pour ouvrir le modal de transfert
  const handleOpenTransferModal = (reservation) => {
    const resObj =
      typeof reservation === 'object'
        ? reservation
        : reservations.find((r) => r.id === reservation);
    if (!resObj) return;
    setTransferModalReservation(resObj);
    setTransferModalOpen(true);
  };

  // Callback après transfert réussi
  const handleTransferSuccess = () => {
    loadReservations();
    toast.success('Course transférée avec succès');
  };

  const handleDispatchNow = async (reservation) => {
    try {
      // Dispatch urgent : +15 min depuis maintenant
      await dispatchNowForReservation(reservation.id, 15);
      loadReservations();
    } catch (err) {
      console.error('Erreur lors du dispatch urgent:', err);
      alert(err?.response?.data?.error || 'Erreur lors du dispatch urgent');
    }
  };

  // Gestion des onglets
  const tabs = [
    { id: 'all', label: 'Toutes', count: stats.total },
    { id: 'pending', label: 'En attente', count: stats.pending },
    { id: 'in_progress', label: 'En cours', count: stats.inProgress },
    { id: 'completed', label: 'Terminées', count: stats.completed },
    { id: 'canceled', label: 'Annulées', count: stats.canceled },
  ];

  // Fonction pour formater l'affichage de la période sélectionnée
  const _getDateDisplay = () => {
    if (selectedDay === 'all') {
      return 'Toutes les dates';
    }

    if (selectedDay && selectedDay.includes(':')) {
      // Plage de dates
      const [startDate, endDate] = selectedDay.split(':');
      const start = new Date(startDate).toLocaleDateString('fr-FR', {
        day: 'numeric',
        month: 'long',
        year: 'numeric',
      });
      const end = new Date(endDate).toLocaleDateString('fr-FR', {
        day: 'numeric',
        month: 'long',
        year: 'numeric',
      });
      return `Du ${start} au ${end}`;
    }

    // Date unique
    return new Date(selectedDay).toLocaleDateString('fr-FR', {
      weekday: 'long',
      year: 'numeric',
      month: 'long',
      day: 'numeric',
    });
  };

  // Réservations pour la carte (une seule journée uniquement)
  const mapReservations = useMemo(() => {
    // Si "toutes les dates" sélectionné, utiliser aujourd'hui
    if (selectedDay === 'all') {
      const today = new Date();
      today.setHours(0, 0, 0, 0);
      const tomorrow = new Date(today);
      tomorrow.setDate(tomorrow.getDate() + 1);

      const filtered = reservations.filter((r) => {
        const reservationDate = new Date(r.scheduled_time || r.pickup_time);
        return reservationDate >= today && reservationDate < tomorrow;
      });

      return filtered;
    }

    // Si plage de dates, utiliser la première date uniquement
    if (selectedDay && selectedDay.includes(':')) {
      const [startDate] = selectedDay.split(':');
      const start = new Date(startDate);
      start.setHours(0, 0, 0, 0);
      const end = new Date(start);
      end.setDate(end.getDate() + 1);

      const filtered = reservations.filter((r) => {
        const reservationDate = new Date(r.scheduled_time || r.pickup_time);
        return reservationDate >= start && reservationDate < end;
      });

      return filtered;
    }

    // Date unique : utiliser cette date
    const targetDate = new Date(selectedDay);
    targetDate.setHours(0, 0, 0, 0);
    const nextDay = new Date(targetDate);
    nextDay.setDate(nextDay.getDate() + 1);

    const filtered = reservations.filter((r) => {
      const reservationDate = new Date(r.scheduled_time || r.pickup_time);
      return reservationDate >= targetDate && reservationDate < nextDay;
    });

    return filtered;
  }, [reservations, selectedDay]);

  return (
    <div className={styles.companyContainer}>
      <CompanyHeader />
      <div className={styles.dashboard}>
        <CompanySidebar />
        <main className={styles.content}>
          {/* Section Header + Filtres */}
          <section className={styles.headerSection}>
            {/* En-tête avec titre et vue */}
            <div className={styles.pageHeader}>
              <div className={styles.headerLeft}>
                <h1>📋 Réservations</h1>
                <p className={styles.subtitle}>
                  Gérez toutes vos réservations et suivez leur statut en temps réel
                </p>
              </div>
              <div className={styles.headerRight}>
                <button
                  className={`${styles.viewToggle} ${viewMode === 'table' ? styles.active : ''}`}
                  onClick={() => setViewMode('table')}
                >
                  📋 Tableau
                </button>
                <button
                  className={`${styles.viewToggle} ${viewMode === 'map' ? styles.active : ''} ${
                    selectedDay && selectedDay.includes(':') ? styles.disabled : ''
                  }`}
                  onClick={() => {
                    // Désactiver la carte pour les plages de dates
                    if (!(selectedDay && selectedDay.includes(':'))) {
                      setViewMode('map');
                    }
                  }}
                  disabled={selectedDay && selectedDay.includes(':')}
                  title={
                    selectedDay && selectedDay.includes(':')
                      ? "La carte n'est disponible que pour une seule journée"
                      : 'Afficher la carte'
                  }
                >
                  🗺️ Carte
                </button>
              </div>
            </div>

            {/* Filtres dans le même conteneur */}
            <ReservationFilters
              selectedDay={selectedDay}
              setSelectedDay={setSelectedDay}
              searchTerm={searchTerm}
              setSearchTerm={setSearchTerm}
              sortOrder={sortOrder}
              setSortOrder={setSortOrder}
              searchInputRef={searchInputRef}
            />
          </section>

          {/* Widgets de statistiques KPI */}
          <ReservationStats stats={stats} />

          {/* Alertes */}
          {alerts.length > 0 && <ReservationAlerts alerts={alerts} />}

          {/* Onglets */}
          <div className={styles.tabsContainer}>
            <div className={styles.tabs}>
              {tabs.map((tab) => (
                <button
                  key={tab.id}
                  className={`${styles.tab} ${activeTab === tab.id ? styles.active : ''}`}
                  onClick={() => setActiveTab(tab.id)}
                >
                  <span>{tab.label}</span>
                  <span className={styles.tabBadge}>{tab.count}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Contenu principal */}
          {loading ? (
            <div className={styles.loading}>
              <div className={styles.spinner}></div>
              <p>Chargement des réservations...</p>
            </div>
          ) : totalReservations === 0 ? (
            <div className={styles.emptyState}>
              <div className={styles.emptyIcon}>📋</div>
              <h3>Aucune réservation trouvée</h3>
              <p>Aucune réservation ne correspond à vos critères de recherche.</p>
            </div>
          ) : (
            <>
              {viewMode === 'table' ? (
                <>
                  <ReservationTable
                    reservations={reservations}
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
                  {/* Pagination avec sélecteur d'éléments par page */}
                  <div className={styles.paginationContainer}>
                    <div className={styles.paginationInfo}>
                      <span className={styles.resultCount}>
                        {totalReservations} résultat
                        {totalReservations > 1 ? 's' : ''} trouvé
                        {totalReservations > 1 ? 's' : ''}
                      </span>
                      <div className={styles.perPageSelector}>
                        <label htmlFor="perPage">Afficher:</label>
                        <select
                          id="perPage"
                          value={reservationsPerPage}
                          onChange={(e) => {
                            setReservationsPerPage(Number(e.target.value));
                            setCurrentPage(1); // Réinitialiser à la page 1
                          }}
                          className={styles.perPageSelect}
                        >
                          <option value={10}>10</option>
                          <option value={25}>25</option>
                          <option value={50}>50</option>
                          <option value={100}>100</option>
                        </select>
                      </div>
                    </div>

                    {totalPages > 1 && (
                      <div className={styles.pagination}>
                        <button
                          disabled={currentPage === 1}
                          onClick={() => setCurrentPage(currentPage - 1)}
                          className={styles.paginationButton}
                        >
                          ← Précédent
                        </button>
                        <span className={styles.pageInfo}>
                          Page {currentPage} sur {totalPages || 1}
                        </span>
                        <button
                          disabled={currentPage === totalPages}
                          onClick={() => setCurrentPage(currentPage + 1)}
                          className={styles.paginationButton}
                        >
                          Suivant →
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

          {/* Widgets supplémentaires */}
          <div className={styles.widgetsGrid}>
            <TopClients reservations={reservations} />
          </div>

          {/* Modals */}
          {selectedReservation && (
            <ReservationDetailsModal
              reservation={selectedReservation}
              onClose={() => setSelectedReservation(null)}
            />
          )}

          <ConfirmationModal
            isOpen={showConfirmModal}
            onClose={handleCloseConfirmModal}
            onConfirm={handleConfirmDelete}
            title={(() => {
              if (!reservationToDelete) return "Confirmer l'action";

              const status = reservationToDelete.status?.toLowerCase();

              // ASSIGNED → Annulation
              if (status === 'assigned') {
                return `Annuler la Réservation n°${reservationToDelete.id}`;
              }
              // PENDING, ACCEPTED → Suppression
              return `Supprimer la Réservation n°${reservationToDelete.id}`;
            })()}
            confirmText={(() => {
              if (!reservationToDelete) return 'Confirmer';

              const status = reservationToDelete.status?.toLowerCase();
              return status === 'assigned' ? 'Oui, annuler' : 'Oui, supprimer';
            })()}
          >
            {reservationToDelete &&
              (() => {
                const status = reservationToDelete.status?.toLowerCase();
                const isCancel = status === 'assigned';

                return (
                  <>
                    <p>
                      {isCancel ? (
                        <>
                          Êtes-vous sûr de vouloir <strong>annuler</strong> la réservation pour{' '}
                          <strong>{reservationToDelete.client_name}</strong> ?
                        </>
                      ) : (
                        <>
                          Êtes-vous sûr de vouloir <strong>supprimer</strong> la réservation pour{' '}
                          <strong>{reservationToDelete.client_name}</strong> ?
                        </>
                      )}
                    </p>
                    <p
                      style={{
                        color: isCancel ? '#f59e0b' : '#ef4444',
                        fontStyle: 'italic',
                        marginTop: '16px',
                      }}
                    >
                      {isCancel ? (
                        <>
                          🚗 <strong>Course assignée à un chauffeur</strong> : La réservation sera
                          annulée et conservée dans l'historique. Le chauffeur sera automatiquement
                          libéré.
                        </>
                      ) : (
                        <>
                          ⚠️ Cette action est irréversible. La réservation sera définitivement
                          supprimée de la base de données.
                        </>
                      )}
                    </p>
                  </>
                );
              })()}
          </ConfirmationModal>

          {/* Modales centralisées */}
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
        </main>
      </div>
    </div>
  );
};

export default CompanyReservations;
