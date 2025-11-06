// src/pages/company/Dashboard/CompanyDashboard.jsx
import React, { useCallback, useState, useEffect, useMemo } from 'react';
import useCompanySocket from '../../../hooks/useCompanySocket';
import useDispatchStatus from '../../../hooks/useDispatchStatus';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import CompanySidebar from '../../../components/layout/Sidebar/CompanySidebar/CompanySidebar';
import OverviewCards from './components/OverviewCards';
import ReservationChart from './components/ReservationChart';
import ReservationTable from './components/ReservationTable';
import DriverTable from '../../driver/components/Dashboard/DriverTable';
import ReservationModals from '../../../components/reservations/ReservationModals';
import DriverLiveMap from './components/DriverLiveMap';
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
} from '../../../services/companyService';
import useCompanyData from '../../../hooks/useCompanyData';
import useDispatchDelays from '../../../hooks/useDispatchDelays';
import styles from './CompanyDashboard.module.css';
import ManualBookingForm from './components/ManualBookingForm';
import ChatWidget from '../../../components/widgets/ChatWidget';
import EditDriverForm from '../components/EditDriverForm';
import Modal from '../../../components/common/Modal';
import CompanyHeader from '../../../components/layout/Header/CompanyHeader';
import { Toaster, toast } from 'sonner';

function makeToday() {
  const d = new Date();
  const pad = (n) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
}

const CompanyDashboard = () => {
  // 1) State d’abord
  const [dispatchDay, setDispatchDay] = useState(makeToday());

  // 2) Hooks qui dépendent de dispatchDay
  const {
    company,
    reservations,
    driver,
    loadingReservations,
    loadingDriver,
    reloadReservations,
    reloadDriver,
  } = useCompanyData({ day: dispatchDay });
  // WebSocket entreprise
  const socket = useCompanySocket();
  useDispatchStatus(socket); // Monitor dispatch status via WebSocket

  // 🆕 Hook pour les retards dispatch (refresh toutes les 2 minutes)
  const { delayCount, hasCriticalDelays, hasDelays } = useDispatchDelays(dispatchDay, 120000);

  // queryClient pour invalidation
  const queryClient = useQueryClient();
  const [showEditModal, setShowEditModal] = useState(false);
  const [driverToEdit, setDriverToEdit] = useState(null);
  const [showBookingModal, setShowBookingModal] = useState(false);
  const [showDriversSection, setShowDriversSection] = useState(false);
  const [showChartSection, setShowChartSection] = useState(false);
  const [reservationTab, setReservationTab] = useState('pending'); // "pending" | "assigned"

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
      reloadDriver();
    } catch (err) {
      console.error('Erreur lors du changement de type du chauffeur :', err);
    }
  };

  // Dispo chauffeur (nettoyée & unifiée)
  const handleToggleAvailability = async (driverId) => {
    try {
      const d = (driver || []).find((x) => x.id === driverId);
      if (!d) return;
      await updateDriverStatus(driverId, { is_available: !d.is_available });
      await reloadDriver();
    } catch (err) {
      console.error('Erreur mise à jour disponibilité chauffeur :', err);
    }
  };

  // États pour les modales centralisées
  const [scheduleModalOpen, setScheduleModalOpen] = useState(false);
  const [scheduleModalReservation, setScheduleModalReservation] = useState(null);
  const [assignModalOpen, setAssignModalOpen] = useState(false);
  const [assignModalReservation, setAssignModalReservation] = useState(null);
  const [deleteModalOpen, setDeleteModalOpen] = useState(false);
  const [deleteModalReservation, setDeleteModalReservation] = useState(null);

  const handleScheduleReservation = (reservation) => {
    // Passer l'objet complet (pas juste l'ID)
    const resObj =
      typeof reservation === 'object'
        ? reservation
        : reservations.find((r) => r.id === reservation);
    if (!resObj) return;
    setScheduleModalReservation(resObj);
    setScheduleModalOpen(true);
  };

  // Dispatch urgent: +15 min
  const handleDispatchNow = async (reservation) => {
    const id = reservation?.id ?? reservation;
    if (!id) return;
    try {
      await dispatchNowForReservation(id, 15);
      await reloadReservations();
      await queryClient.invalidateQueries(['reservations']);
      toast.success('Dispatch urgent déclenché avec succès');
    } catch (e) {
      // ⚡ Gestion améliorée : extraire le message du backend
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

  // Modale d'assignation

  // Liste des dispatches (assignations/affectations) pour la carte et la table
  const { data: dispatchedReservations = [], refetch: refetchAssigned } = useQuery({
    queryKey: ['assigned-reservations', dispatchDay],
    queryFn: () => fetchAssignedReservations(dispatchDay),
    staleTime: 30_000,
    enabled: !!company?.id, // ✅ évite les 403 bruités
  });

  // Retards du jour (pour DriverLiveMap)
  const {
    data: delays = [],
    refetch: refetchDelays,
    isFetching: fetchingDelays,
  } = useQuery({
    queryKey: ['dispatch-delays', dispatchDay],
    queryFn: () => fetchDispatchDelays(dispatchDay),
    initialData: [],
    staleTime: 20_000,
    enabled: !!company?.id, // ✅ idem
  });

  // WS: nouvelles réservations -> recharger la liste
  const handleNewReservation = useCallback(() => reloadReservations(), [reloadReservations]);
  useEffect(() => {
    if (!socket) return;
    socket.on('new_reservation', handleNewReservation);
    return () => socket.off('new_reservation', handleNewReservation);
  }, [socket, handleNewReservation]);

  // WS: changements d'assignations & progression du dispatch
  useEffect(() => {
    if (!socket) return;
    const refetchAll = () => {
      refetchAssigned?.();
      reloadReservations?.();
      refetchDelays?.();
    };
    const onAssignCreated = () => refetchAll();
    const onAssignUpdated = () => refetchAll();
    const onAssignDeleted = () => refetchAll();
    const onDispatchProgress = (_p) => {
      // Optionnel: logger ou afficher un toast/loader granulaire
      // console.debug("dispatch_progress", _p);
    };
    const onDispatchError = (err) => {
      console.error('dispatch_error:', err);
      // Optionnel: notifier l’utilisateur
      refetchAll();
    };
    const onDispatchRunCompleted = (data) => {
      // eslint-disable-next-line no-console
      console.log('Dispatch run completed:', data);
      // window.alert(err?.message || "Une erreur est survenue pendant l'optimisation.");
      refetchAll();
    };
    socket.on('assignment_created', onAssignCreated);
    socket.on('assignment_updated', onAssignUpdated);
    socket.on('assignment_deleted', onAssignDeleted);
    socket.on('dispatch_progress', onDispatchProgress);
    socket.on('dispatch_error', onDispatchError);
    socket.on('dispatch_run_completed', onDispatchRunCompleted);
    return () => {
      socket.off('assignment_created', onAssignCreated);
      socket.off('assignment_updated', onAssignUpdated);
      socket.off('assignment_deleted', onAssignDeleted);
      socket.off('dispatch_progress', onDispatchProgress);
      socket.off('dispatch_error', onDispatchError);
      socket.off('dispatch_run_completed', onDispatchRunCompleted);
    };
  }, [socket, refetchAssigned, reloadReservations, refetchDelays]);

  // Callbacks réservations
  const handleAccept = async (id) => {
    try {
      await acceptReservation(id);
      reloadReservations();
    } catch (err) {
      console.error('Erreur acceptation :', err);
    }
  };
  const handleReject = async (id) => {
    try {
      await rejectReservation(id);
      reloadReservations();
    } catch (err) {
      console.error('Erreur rejet :', err);
    }
  };
  const openAssignModal = (res) => {
    // Passer l'objet complet
    const resObj = typeof res === 'object' ? res : reservations.find((r) => r.id === res);
    if (!resObj) return;
    setAssignModalReservation(resObj);
    setAssignModalOpen(true);
  };
  const handleAssignDriver = async (reservationId, driverId) => {
    try {
      await assignDriver(reservationId, driverId);
      reloadReservations();
      setAssignModalOpen(false);
      setAssignModalReservation(null);
    } catch (err) {
      console.error('Erreur assignation chauffeur :', err);
    }
  };

  // Ouvre la modale de retour
  const handleTriggerReturn = (reservation) => {
    // Passer l'objet complet (pas juste l'ID)
    const resObj =
      typeof reservation === 'object'
        ? reservation
        : reservations.find((r) => r.id === reservation);
    if (!resObj) return;
    setScheduleModalReservation(resObj);
    setScheduleModalOpen(true);
  };

  // Transforme en ISO local sans offset ni "Z"
  const toLocalIsoString = (date) => {
    const pad = (n) => n.toString().padStart(2, '0');
    const Y = date.getFullYear();
    const M = pad(date.getMonth() + 1);
    const D = pad(date.getDate());
    const h = pad(date.getHours());
    const m = pad(date.getMinutes());
    return `${Y}-${M}-${D}T${h}:${m}`;
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
      } else if (typeof data === 'string') {
        payload = { return_time: data };
      } else if (data instanceof Date) {
        payload = { return_time: toLocalIsoString(data) };
      } else if (data?.return_time) {
        payload = { return_time: data.return_time };
      }
      await triggerReturnBooking(reservationId, payload);
      setScheduleModalReservation(null);
      await reloadReservations();
      await queryClient.invalidateQueries(['reservations']);
    } catch (err) {
      console.error('Retour :', err);
      alert(err?.response?.data?.error || 'Erreur serveur.');
    }
  };

  const handleDeleteReservationClick = (reservation) => {
    // Passer l'objet complet
    const resObj =
      typeof reservation === 'object'
        ? reservation
        : reservations.find((r) => r.id === reservation);
    if (!resObj) return;
    setDeleteModalReservation(resObj);
    setDeleteModalOpen(true);
  };

  const handleConfirmDelete = async () => {
    if (!deleteModalReservation) return;

    const id = deleteModalReservation?.id ?? deleteModalReservation;
    await handleDeleteReservation(id);
    setDeleteModalOpen(false);
    setDeleteModalReservation(null);
  };

  // Après ajout manuel
  const handleManualBookingSuccess = (resp) => {
    const ymd = String(resp?.reservation?.scheduled_time || '').slice(0, 10);
    if (ymd) setDispatchDay(ymd);
    reloadReservations();
    queryClient.invalidateQueries(['reservations']);
  };

  // Filtrage listes
  const pendingReservations = (reservations || []).filter(
    (r) => r.status?.toLowerCase() === 'pending'
  );
  // Fix: Filter for reservations that are accepted but don't have a driver assigned
  const assignedReservations = (reservations || []).filter(
    (r) => r.status?.toLowerCase() === 'accepted' && !r.driver_id
  );

  // Callbacks chauffeurs (liste)
  const handleToggleDriver = async (driverId, current) => {
    try {
      await updateDriverStatus(driverId, !current);
      reloadDriver();
    } catch (err) {
      console.error('Erreur mise à jour chauffeur :', err);
    }
  };
  const handleDeleteDriver = async (driverId) => {
    try {
      await deleteDriver(driverId);
      reloadDriver();
    } catch (err) {
      console.error('Erreur suppression chauffeur :', err);
    }
  };
  const handleDeleteReservation = useCallback(
    async (reservation) => {
      const id = reservation?.id ?? reservation;
      if (!id) {
        console.error('❌ ID réservation manquant:', reservation);
        return;
      }

      try {
        console.log('🗑️ Suppression en cours de la réservation', id);
        const result = await deleteReservation(id);
        console.log('✅ Réservation supprimée:', result);

        // Rafraîchir les données
        await reloadReservations();
        await queryClient.invalidateQueries(['reservations']);

        alert(`✅ Réservation #${id} supprimée avec succès`);
      } catch (e) {
        console.error('❌ Erreur suppression réservation:', e);
        const errorMsg = e?.response?.data?.error || e?.message || 'Suppression impossible.';
        alert(`❌ Erreur: ${errorMsg}`);
      }
    },
    [reloadReservations, queryClient]
  );

  // ---------- Données pour DriverLiveMap ----------

  // Bookings "actifs" du jour (en dehors de completed/cancelled/no_show)
  const activeBookings = useMemo(() => {
    const isActive = (b) =>
      b && !['completed', 'cancelled', 'no_show'].includes((b.status || '').toLowerCase());
    return (reservations || []).filter(isActive).map((r) => ({
      id: r.id,
      client_name: r.customer_name || r.client?.full_name || '',
      status: r.status,
      pickup_time: r.scheduled_time || r.pickup_time, // Fallback
      dropoff_time: r.dropoff_time,
      pickup_location: r.pickup_location_coords || r.pickup_location || r.pickup || null,
      dropoff_location: r.dropoff_location_coords || r.dropoff_location || r.dropoff || null,
    }));
  }, [reservations]);

  // Assignments pour la carte :
  // on part de dispatchedReservations (booking + assignment)
  const assignmentsForMap = useMemo(() => {
    const rows = Array.isArray(dispatchedReservations)
      ? dispatchedReservations
      : Object.values(dispatchedReservations || {});
    return rows.map((row) => {
      const a = row.assignment || {};
      return {
        // identifiant d'assignation si dispo, sinon fallback booking
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
          client_name: row.customer_name || row.client?.full_name || '',
          status: row.status,
          pickup_time: row.scheduled_time || row.pickup_time,
          dropoff_time: row.dropoff_time || null,
          pickup_location: row.pickup_location,
          dropoff_location: row.dropoff_location,
        },
      };
    });
  }, [dispatchedReservations]);

  // Delays normalisés { [bookingId]: {delay_minutes, is_dropoff?} }
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

  return (
    <div className={styles.companyContainer}>
      {/* ⚡ Toaster pour les notifications toast */}
      <Toaster position="top-right" richColors />
      <CompanyHeader />

      {/* 🆕 Badge d'alerte pour les retards détectés */}
      {hasDelays && (
        <div className={styles.delayAlert}>
          <div className={styles.delayAlertContent}>
            <span className={styles.delayAlertIcon}>{hasCriticalDelays ? '🚨' : '⚠️'}</span>
            <span className={styles.delayAlertText}>
              {hasCriticalDelays ? (
                <strong>{delayCount} retard(s) critique(s) détecté(s) aujourd'hui !</strong>
              ) : (
                <>{delayCount} retard(s) détecté(s) aujourd'hui</>
              )}
            </span>
            <a
              href={company?.public_id ? `/dashboard/company/${company.public_id}/dispatch` : '#'}
              className={styles.delayAlertLink}
            >
              Voir les détails et suggestions →
            </a>
          </div>
        </div>
      )}

      <div className={styles.dashboard}>
        <CompanySidebar />
        <main className={styles.content}>
          {/* ======= KPIs compacts en haut ======= */}
          <OverviewCards
            reservations={reservations}
            pendingReservations={pendingReservations}
            assignedReservations={assignedReservations}
            driver={driver}
            day={dispatchDay}
          />

          {/* ======= Layout 2 colonnes ======= */}
          <div className={styles.twoColumnLayout}>
            {/* Colonne gauche : Carte + Actions */}
            <div className={styles.leftColumn}>
              {/* Carte */}
              <section className={styles.mapSection}>
                <DriverLiveMap
                  date={dispatchDay}
                  drivers={driver || []}
                  bookings={activeBookings}
                  assignments={assignmentsForMap}
                  delays={delaysByBooking}
                />
                {fetchingDelays && <small className={styles.hint}>Mise à jour des retards…</small>}
              </section>

              {/* Actions rapides */}
              <div className={styles.quickActions}>
                {/* Dispatch & Planification */}
                <a
                  href={`/dashboard/company/${company?.public_id}/dispatch`}
                  className={styles.actionButton}
                >
                  <span className={styles.actionIcon}>🚀</span>
                  <span className={styles.actionText}>
                    <strong>Dispatch & Planification</strong>
                    <small>Optimiser les courses</small>
                  </span>
                </a>

                {/* Créer une réservation */}
                <button onClick={() => setShowBookingModal(true)} className={styles.actionButton}>
                  <span className={styles.actionIcon}>➕</span>
                  <span className={styles.actionText}>
                    <strong>Nouvelle réservation</strong>
                    <small>Créer manuellement</small>
                  </span>
                </button>
              </div>

              {/* Graphique (collapsible) */}
              <section className={styles.compactSection}>
                <div
                  className={styles.collapsibleHeader}
                  onClick={() => setShowChartSection(!showChartSection)}
                >
                  <h2>📊 Statistiques</h2>
                  <span className={styles.collapseIcon}>{showChartSection ? '▼' : '▶'}</span>
                </div>
                {showChartSection && (
                  <div style={{ padding: '16px' }}>
                    <ReservationChart reservations={reservations} />
                  </div>
                )}
              </section>

              {/* Section chauffeurs (collapsible) */}
              <section className={styles.compactSection}>
                <div
                  className={styles.collapsibleHeader}
                  onClick={() => setShowDriversSection(!showDriversSection)}
                >
                  <h2>👥 Chauffeurs ({(driver || []).length})</h2>
                  <span className={styles.collapseIcon}>{showDriversSection ? '▼' : '▶'}</span>
                </div>
                {showDriversSection && (
                  <DriverTable
                    driver={driver}
                    loading={loadingDriver}
                    onToggle={handleToggleDriver}
                    onDelete={handleDeleteDriver}
                    onEdit={handleEditDriver}
                    onToggleAvailability={handleToggleAvailability}
                    onToggleType={handleToggleType}
                  />
                )}
              </section>
            </div>

            {/* Colonne droite : Réservations avec onglets */}
            <div className={styles.rightColumn}>
              {/* Onglets sans conteneur */}
              <div className={styles.tabsHeader} data-active-tab={reservationTab}>
                <button
                  className={`${styles.tab} ${
                    reservationTab === 'pending' ? styles.tabActive : ''
                  }`}
                  onClick={() => setReservationTab('pending')}
                >
                  📋 En attente{' '}
                  <span className={styles.tabBadge}>{pendingReservations.length}</span>
                </button>
                <button
                  className={`${styles.tab} ${
                    reservationTab === 'assigned' ? styles.tabActive : ''
                  }`}
                  onClick={() => setReservationTab('assigned')}
                >
                  ⏳ Assignation chauffeur{' '}
                  <span className={styles.tabBadge}>{assignedReservations.length}</span>
                </button>
              </div>

              {/* Tableaux directement sans conteneur */}
              {reservationTab === 'pending' && (
                <ReservationTable
                  reservations={pendingReservations}
                  loading={loadingReservations}
                  onAccept={handleAccept}
                  onReject={handleReject}
                  onAssign={openAssignModal}
                  onTriggerReturn={handleTriggerReturn}
                  onDelete={handleDeleteReservationClick}
                  onSchedule={handleScheduleReservation}
                  onDispatchNow={handleDispatchNow}
                  hideSchedule={true}
                  hideEdit={true}
                />
              )}

              {reservationTab === 'assigned' && (
                <ReservationTable
                  reservations={assignedReservations}
                  loading={loadingReservations}
                  onAssign={openAssignModal}
                  onTriggerReturn={handleTriggerReturn}
                  onDelete={handleDeleteReservationClick}
                  onSchedule={handleScheduleReservation}
                  onDispatchNow={handleDispatchNow}
                  hideSchedule={true}
                  hideEdit={true}
                />
              )}
            </div>
          </div>
        </main>

        {/* Modal réservation manuelle */}
        {showBookingModal && (
          <Modal onClose={() => setShowBookingModal(false)}>
            <h3>Créer une réservation manuelle</h3>
            <ManualBookingForm
              onSuccess={(booking) => {
                handleManualBookingSuccess(booking);
                setShowBookingModal(false);
              }}
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
    </div>
  );
};

export default CompanyDashboard;
