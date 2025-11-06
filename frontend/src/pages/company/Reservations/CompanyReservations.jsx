import React, { useEffect, useState, useCallback, useMemo } from 'react';
import CompanyHeader from '../../../components/layout/Header/CompanyHeader';
import CompanySidebar from '../../../components/layout/Sidebar/CompanySidebar/CompanySidebar';
import {
  fetchCompanyReservations,
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
import styles from './CompanyReservations.module.css';

const CompanyReservations = () => {
  // États existants
  const [reservations, setReservations] = useState([]);
  const [filteredReservations, setFilteredReservations] = useState([]);
  const [selectedDay, setSelectedDay] = useState('all'); // Par défaut : toutes les dates
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');
  const [sortOrder, setSortOrder] = useState('desc'); // Par défaut : ordre décroissant (plus récent d'abord)
  const [currentPage, setCurrentPage] = useState(1);
  const [reservationsPerPage, setReservationsPerPage] = useState(10); // Nombre de réservations par page
  const [selectedReservation, setSelectedReservation] = useState(null);
  const [showConfirmModal, setShowConfirmModal] = useState(false);
  const [reservationToDelete, setReservationToDelete] = useState(null);
  const [scheduleModalOpen, setScheduleModalOpen] = useState(false);
  const [scheduleModalReservation, setScheduleModalReservation] = useState(null);

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

  // Chargement des réservations avec calculs des statistiques et alertes
  const loadReservations = useCallback(async () => {
    try {
      setLoading(true);
      // Si "Toutes les dates" ou une plage de dates, charger toutes les réservations
      const isDateRange = selectedDay && selectedDay.includes(':');
      const apiParam = selectedDay === 'all' || isDateRange ? null : selectedDay;

      const data = await fetchCompanyReservations(apiParam);
      let reservationsData = Array.isArray(data) ? data : data.reservations || [];

      // Filtrer côté client si c'est une plage de dates
      if (isDateRange) {
        const [startDate, endDate] = selectedDay.split(':');
        const start = new Date(startDate);
        const end = new Date(endDate);
        end.setHours(23, 59, 59, 999); // Inclure toute la journée de fin

        reservationsData = reservationsData.filter((r) => {
          const reservationDate = new Date(r.scheduled_time || r.pickup_time);
          return reservationDate >= start && reservationDate <= end;
        });
      }

      setReservations(reservationsData);

      // Calculer les statistiques
      calculateStats(reservationsData);

      // Générer les alertes
      generateAlerts(reservationsData);
    } catch (err) {
      console.error('Erreur lors du chargement des réservations :', err);
    } finally {
      setLoading(false);
    }
  }, [selectedDay]);

  // Calculer les statistiques
  const calculateStats = (reservationsData) => {
    const newStats = {
      total: reservationsData.length,
      pending: reservationsData.filter((r) => r.status === 'pending').length,
      inProgress: reservationsData.filter((r) =>
        ['accepted', 'assigned', 'in_progress'].includes(r.status)
      ).length,
      completed: reservationsData.filter((r) => r.status === 'completed').length,
      canceled: reservationsData.filter((r) => r.status === 'canceled').length,
      revenue: reservationsData
        .filter((r) => r.status === 'completed')
        .reduce((sum, r) => sum + (Number(r.amount) || 0), 0),
    };
    setStats(newStats);
  };

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
    const resObj = typeof reservation === 'object' ? reservation : reservations.find((r) => r.id === reservation);
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
      console.error('Erreur lors de l\'édition:', err);
      throw err;
    }
  };

  const handleSchedule = (reservation) => {
    // Passer l'objet complet
    const resObj = typeof reservation === 'object' ? reservation : reservations.find((r) => r.id === reservation);
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

  // Filtrer et trier les réservations avec onglets
  useEffect(() => {
    let filtered = [...reservations];

    // Filtre par onglet
    if (activeTab !== 'all') {
      filtered = filtered.filter((r) => {
        switch (activeTab) {
          case 'pending':
            return r.status === 'pending';
          case 'in_progress':
            return ['accepted', 'assigned', 'in_progress'].includes(r.status);
          case 'completed':
            return r.status === 'completed';
          case 'canceled':
            return r.status === 'canceled';
          default:
            return true;
        }
      });
    } else {
      // ✅ Onglet "Toutes" : Masquer automatiquement les courses annulées
      filtered = filtered.filter((r) => r.status !== 'canceled' && r.status !== 'CANCELED');
    }

    // Filtres de recherche améliorés (ID, Client, Adresse, Email, Téléphone)
    if (searchTerm) {
      const q = searchTerm.toLowerCase().trim();
      filtered = filtered.filter((r) => {
        // Recherche par ID (exact ou partiel)
        const id = String(r.id || '');
        if (id.includes(q)) return true;

        // Recherche par nom du client
        const name = (
          r.customer_name ||
          r.client?.full_name ||
          r.client?.username ||
          ''
        ).toLowerCase();
        if (name.includes(q)) return true;

        // Recherche par email
        const email = (r.client?.email || r.customer_email || '').toLowerCase();
        if (email.includes(q)) return true;

        // Recherche par téléphone
        const phone = (r.client?.phone || r.customer_phone || '').replace(/\s/g, '');
        const qPhone = q.replace(/\s/g, '');
        if (phone.includes(qPhone)) return true;

        // Recherche par adresse de départ
        const pickup = (r.pickup_location || '').toLowerCase();
        if (pickup.includes(q)) return true;

        // Recherche par adresse d'arrivée
        const dropoff = (r.dropoff_location || '').toLowerCase();
        if (dropoff.includes(q)) return true;

        // Recherche par chauffeur assigné
        const driverName = (r.driver?.username || r.driver?.full_name || '').toLowerCase();
        if (driverName.includes(q)) return true;

        return false;
      });
    }

    if (statusFilter !== 'all') {
      filtered = filtered.filter((r) => (r.status || '').toLowerCase() === statusFilter);
    }

    filtered.sort((a, b) => {
      const dateA = new Date(a.scheduled_time);
      const dateB = new Date(b.scheduled_time);
      return sortOrder === 'asc' ? dateA - dateB : dateB - dateA;
    });

    setFilteredReservations(filtered);
    setCurrentPage(1);
  }, [reservations, searchTerm, statusFilter, sortOrder, activeTab]);

  // Pagination
  const currentReservations = useMemo(() => {
    const indexOfLast = currentPage * reservationsPerPage;
    const indexOfFirst = indexOfLast - reservationsPerPage;
    return filteredReservations.slice(indexOfFirst, indexOfLast);
  }, [filteredReservations, currentPage, reservationsPerPage]);

  const totalPages = Math.ceil(filteredReservations.length / reservationsPerPage);

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

      const filtered = filteredReservations.filter((r) => {
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

      const filtered = filteredReservations.filter((r) => {
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

    const filtered = filteredReservations.filter((r) => {
      const reservationDate = new Date(r.scheduled_time || r.pickup_time);
      return reservationDate >= targetDate && reservationDate < nextDay;
    });

    return filtered;
  }, [filteredReservations, selectedDay]);

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
              statusFilter={statusFilter}
              setStatusFilter={setStatusFilter}
              sortOrder={sortOrder}
              setSortOrder={setSortOrder}
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
          ) : filteredReservations.length === 0 ? (
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
                    reservations={currentReservations}
                    onRowClick={(reservation) => setSelectedReservation(reservation)}
                    onDelete={handleDeleteRequest}
                    onAccept={handleAccept}
                    onReject={handleReject}
                    onEdit={handleEdit}
                    onSchedule={handleSchedule}
                    onDispatchNow={handleDispatchNow}
                    hideAssign={true}
                    hideUrgent={true}
                  />
                  {/* Pagination avec sélecteur d'éléments par page */}
                  <div className={styles.paginationContainer}>
                    <div className={styles.paginationInfo}>
                      <span className={styles.resultCount}>
                        {filteredReservations.length} résultat
                        {filteredReservations.length > 1 ? 's' : ''} trouvé
                        {filteredReservations.length > 1 ? 's' : ''}
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
                          <option value={filteredReservations.length}>
                            Tous ({filteredReservations.length})
                          </option>
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
                          Page {currentPage} sur {totalPages}
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
                          <strong>{reservationToDelete.customer_name}</strong> ?
                        </>
                      ) : (
                        <>
                          Êtes-vous sûr de vouloir <strong>supprimer</strong> la réservation pour{' '}
                          <strong>{reservationToDelete.customer_name}</strong> ?
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
        </main>
      </div>
    </div>
  );
};

export default CompanyReservations;
