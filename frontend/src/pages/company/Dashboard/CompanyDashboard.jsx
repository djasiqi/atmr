// src/pages/company/Dashboard/CompanyDashboard.jsx
import React, { useCallback, useState, useEffect, useMemo } from "react";
import useCompanySocket from "../../../hooks/useCompanySocket";
import useDispatchStatus from "../../../hooks/useDispatchStatus";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import CompanySidebar from "../../../components/layout/Sidebar/CompanySidebar/CompanySidebar";
import OverviewCards from "./components/OverviewCards";
import ReservationChart from "./components/ReservationChart";
import ReservationTable from "./components/ReservationTable";
import DriverTable from "../../driver/components/Dashboard/DriverTable";
import AssignmentModal from "./AssignmentModal";
import DriverLiveMap from "./components/DriverLiveMap";
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
} from "../../../services/companyService";
import useCompanyData from "../../../hooks/useCompanyData";
import styles from "./CompanyDashboard.module.css";
import DispatchTable from "../components/DispatchTable";
import ManualBookingForm from "./components/ManualBookingForm";
import ChatWidget from "../../../components/widgets/ChatWidget";
import ReturnTimeModal from "./components/ReturnTimeModal";
import EditDriverForm from "../components/EditDriverForm";
import Modal from "../../../components/common/Modal";
import CompanyHeader from "../../../components/layout/Header/CompanyHeader";

function makeToday() {
  const d = new Date();
  const pad = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
}

const CompanyDashboard = () => {
  // Données globales
  const {
    company,
    reservations,
    driver,
    loadingReservations,
    loadingDriver,
    reloadReservations,
    reloadDriver,
  } = useCompanyData();

  // WebSocket entreprise
  const socket = useCompanySocket();
  const { status } = useDispatchStatus(socket);

  // Date de travail unifiée (table + carte)
  const [dispatchDay, setDispatchDay] = useState(makeToday());

  // queryClient pour invalidation
  const queryClient = useQueryClient();
  const [showEditModal, setShowEditModal] = useState(false);
  const [driverToEdit, setDriverToEdit] = useState(null);

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
      console.error("Erreur lors du changement de type du chauffeur :", err);
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
      console.error("Erreur mise à jour disponibilité chauffeur :", err);
    }
  };

  // Ouvre la modale pour planifier le retour (ou choisir Urgent)
  const [openReturnModal, setOpenReturnModal] = useState(false);
  const [selectedReturnReservation, setSelectedReturnReservation] = useState(null);
  const handleScheduleReservation = (reservation) => {
    const id = reservation?.id ?? reservation;
    if (!id) return;
    setSelectedReturnReservation(id);
    setOpenReturnModal(true);
  };

  // Dispatch urgent: +15 min
  const handleDispatchNow = async (reservation) => {
    const id = reservation?.id ?? reservation;
    if (!id) return;
    try {
      await dispatchNowForReservation(id, 15);
      await reloadReservations();
      await queryClient.invalidateQueries(["reservations"]);
    } catch (e) {
      console.error("Dispatch urgent:", e);
      alert(e?.response?.data?.error || "Erreur lors du dispatch urgent.");
    }
  };

  // Modale d'assignation
  const [selectedReservation, setSelectedReservation] = useState(null);

  // Liste des dispatches (assignations/affectations) pour la carte et la table
  const {
    data: dispatchedReservations = [],
    refetch: refetchAssigned,
    isLoading: loadingDispatches,
  } = useQuery({
    queryKey: ["assigned-reservations", dispatchDay],
    queryFn: () => fetchAssignedReservations(dispatchDay),
    staleTime: 30_000,
  });

  // Retards du jour (pour DriverLiveMap)
  const {
    data: delays = [],
    refetch: refetchDelays,
    isFetching: fetchingDelays,
  } = useQuery({
    queryKey: ["dispatch-delays", dispatchDay],
    queryFn: () => fetchDispatchDelays(dispatchDay),
    initialData: [],
    staleTime: 20_000,
  });

  // WS: nouvelles réservations -> recharger la liste
  const handleNewReservation = useCallback(() => reloadReservations(), [reloadReservations]);
  useEffect(() => {
    if (!socket) return;
    socket.on("new_reservation", handleNewReservation);
    return () => socket.off("new_reservation", handleNewReservation);
  }, [socket, handleNewReservation]);

  // WS: fin / statut d'optimisation -> rafraîchir dispatches + réservations + retards
  useEffect(() => {
    if (!socket) return;
    const handleDispatchCompleted = () => {
      refetchAssigned?.();
      reloadReservations?.();
      refetchDelays?.();
    };
    const handleDispatchStatus = (payload = {}) => {
      if (payload.is_running === false) {
        refetchAssigned?.();
        reloadReservations?.();
        refetchDelays?.();
      }
    };
    socket.on("dispatch_completed", handleDispatchCompleted);
    socket.on("dispatch_status", handleDispatchStatus);
    return () => {
      socket.off("dispatch_completed", handleDispatchCompleted);
      socket.off("dispatch_status", handleDispatchStatus);
    };
  }, [socket, refetchAssigned, reloadReservations, refetchDelays]);

  // Callbacks réservations
  const handleAccept = async (id) => {
    try {
      await acceptReservation(id);
      reloadReservations();
    } catch (err) {
      console.error("Erreur acceptation :", err);
    }
  };
  const handleReject = async (id) => {
    try {
      await rejectReservation(id);
      reloadReservations();
    } catch (err) {
      console.error("Erreur rejet :", err);
    }
  };
  const openAssignModal = (res) => setSelectedReservation(res);
  const handleAssignDriver = async (reservationId, driverId) => {
    try {
      await assignDriver(reservationId, driverId);
      reloadReservations();
      setSelectedReservation(null);
    } catch (err) {
      console.error("Erreur assignation chauffeur :", err);
    }
  };

  // Ouvre la modale de retour
  const handleTriggerReturn = (reservationId) => {
    setSelectedReturnReservation(reservationId);
    setOpenReturnModal(true);
  };

  // Transforme en ISO local sans offset ni "Z"
  const toLocalIsoString = (date) => {
    const pad = (n) => n.toString().padStart(2, "0");
    const Y = date.getFullYear();
    const M = pad(date.getMonth() + 1);
    const D = pad(date.getDate());
    const h = pad(date.getHours());
    const m = pad(date.getMinutes());
    return `${Y}-${M}-${D}T${h}:${m}`;
  };

  // Confirme l'heure du retour OU marque en 'Urgent +15 min'
  const handleConfirmReturnTime = async (data) => {
    setOpenReturnModal(false);
    if (!selectedReturnReservation) return;
    try {
      let payload = {};
      if (data?.urgent) {
        payload = { urgent: true, minutes_offset: data.minutes_offset ?? 15 };
      } else if (typeof data === "string") {
        payload = { return_time: data };
      } else if (data instanceof Date) {
        payload = { return_time: toLocalIsoString(data) };
      } else if (data?.return_time) {
        payload = { return_time: data.return_time };
      }
      await triggerReturnBooking(selectedReturnReservation, payload);
      setSelectedReturnReservation(null);
      await reloadReservations();
      await queryClient.invalidateQueries(["reservations"]);
    } catch (err) {
      console.error("Retour :", err);
      alert(err?.response?.data?.error || "Erreur serveur.");
    }
  };

  // Après ajout manuel
  const handleManualBookingSuccess = () => {
    reloadReservations();
    queryClient.invalidateQueries(["reservations"]);
  };

  // Filtrage listes
  const pendingReservations = (reservations || []).filter(
    (r) => r.status?.toLowerCase() === "pending"
  );
  // Fix: Filter for reservations that are accepted but don't have a driver assigned
  const assignedReservations = (reservations || []).filter((r) => 
    r.status?.toLowerCase() === "accepted" && !r.driver_id
  );

  // Callbacks chauffeurs (liste)
  const handleToggleDriver = async (driverId, current) => {
    try {
      await updateDriverStatus(driverId, !current);
      reloadDriver();
    } catch (err) {
      console.error("Erreur mise à jour chauffeur :", err);
    }
  };
  const handleDeleteDriver = async (driverId) => {
    try {
      await deleteDriver(driverId);
      reloadDriver();
    } catch (err) {
      console.error("Erreur suppression chauffeur :", err);
    }
  };
  const handleDeleteReservation = useCallback(
    async (reservation) => {
      const id = reservation?.id ?? reservation;
      if (!id) return;
      if (!window.confirm("Supprimer cette réservation ?")) return;
      try {
        await deleteReservation(id);
        await reloadReservations();
        await queryClient.invalidateQueries(["reservations"]);
      } catch (e) {
        console.error("Suppression réservations :", e);
        alert(e?.response?.data?.error || "Suppression impossible.");
      }
    },
    [reloadReservations, queryClient]
  );

  // ---------- Données pour DriverLiveMap ----------

  // Drivers
  const driversList = useMemo(() => driver || [], [driver]);

  // Bookings “actifs” du jour (en dehors de completed/cancelled/no_show)
  const activeBookings = useMemo(() => {
    const isActive = (b) =>
      b &&
      !["completed", "cancelled", "no_show"].includes(
        (b.status || "").toLowerCase()
      );
    return (reservations || [])
      .filter(isActive)
      .map((r) => ({
        id: r.id,
        client_name: r.customer_name || r.client?.full_name || "",
        status: r.status,
        pickup_time: r.scheduled_time || r.pickup_time, // Fallback
        dropoff_time: r.dropoff_time,
        pickup_location:
          r.pickup_location_coords || r.pickup_location || r.pickup || null,
        dropoff_location:
          r.dropoff_location_coords || r.dropoff_location || r.dropoff || null,
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
        is_on_trip: ["assigned", "in_progress", "onboard", "en_route_pickup", "en_route_dropoff"].includes(
          String(a.status ?? row.status ?? "").toLowerCase()
        ),
        route: row.route || a.route || [],
        booking: {
          id: row.id,
          client_name: row.customer_name || row.client?.full_name || "",
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
      <CompanyHeader />
      <div className={styles.dashboard}>
        <CompanySidebar />
        <main className={styles.content}>
          {/* ======= Carte ======= */}
          <section className={styles.mapSection}>
            <div className={styles.sectionHeader}>
              <h2>Localisation des chauffeurs</h2>
              <div className={styles.inlineControls}>
                <label>
                  Jour:&nbsp;
                  <input
                    type="date"
                    value={dispatchDay}
                    onChange={(e) => setDispatchDay(e.target.value)}
                  />
                </label>
              </div>
            </div>

            <DriverLiveMap
              date={dispatchDay}
              drivers={driversList}
              bookings={activeBookings}
              assignments={assignmentsForMap}
              delays={delaysByBooking}
            />
            {fetchingDelays && (
              <small className={styles.hint}>Mise à jour des retards…</small>
            )}
          </section>

          {/* ======= Planification & suivi ======= */}
          <section className={styles.dispatchSection} data-dispatch-status={status}>
            <h2>Planifier & suivre une journée</h2>
            {loadingDispatches && <p>Chargement des dispatches…</p>}
            <DispatchTable
              showPlanner
              initialDispatchDay={dispatchDay}
              initialRegularFirst={true}
              initialAllowEmergency={true}
              dispatches={
                Array.isArray(dispatchedReservations)
                  ? dispatchedReservations
                  : Object.values(dispatchedReservations || {})
              }
              reload={() => {
                refetchAssigned();
                refetchDelays();
              }}
            />
          </section>

          <OverviewCards
            reservations={reservations}
            pendingReservations={pendingReservations}
            assignedReservations={assignedReservations}
            driver={driver}
            day={dispatchDay}
          />
          <ReservationChart reservations={reservations} />

          <section className={styles.reservationsSection}>
            <h2>Réservations en attente</h2>
            <ReservationTable
              reservations={pendingReservations}
              loading={loadingReservations}
              onAccept={handleAccept}
              onReject={handleReject}
              onAssign={openAssignModal}
              onTriggerReturn={handleTriggerReturn}
              onDelete={handleDeleteReservation}
              onSchedule={handleScheduleReservation}
              onDispatchNow={handleDispatchNow}
            />
          </section>

          <section className={styles.reservationsSection}>
            <h2>Réservations en attente d'assignation chauffeur</h2>
            <ReservationTable
              reservations={assignedReservations}
              loading={loadingReservations}
              onAssign={openAssignModal}
              onTriggerReturn={handleTriggerReturn}
              onDelete={handleDeleteReservation}
              onSchedule={handleScheduleReservation}
              onDispatchNow={handleDispatchNow}
            />
          </section>

          <section className={styles.driverSection}>
            <h2>Chauffeurs de l'entreprise</h2>
            <DriverTable
              driver={driver}
              loading={loadingDriver}
              onToggle={handleToggleDriver}
              onDelete={handleDeleteDriver}
              onEdit={handleEditDriver}
              onToggleAvailability={handleToggleAvailability}
              onToggleType={handleToggleType}
            />
          </section>

          {showEditModal && driverToEdit && (
            <Modal onClose={handleCloseModal}>
              <h3>Modifier le chauffeur {driverToEdit.username}</h3>
              <EditDriverForm driver={driverToEdit} onClose={handleCloseModal} />
            </Modal>
          )}

          {selectedReservation && (
            <AssignmentModal
              reservation={selectedReservation}
              driver={(driver || []).filter((d) => d.is_active)}
              onAssign={handleAssignDriver}
              onClose={() => setSelectedReservation(null)}
            />
          )}

          <section className={styles.manualBookingSection}>
            <h2>Créer une réservation manuelle</h2>
            <ManualBookingForm onSuccess={handleManualBookingSuccess} />
          </section>
        </main>
      </div>

      {company?.id && <ChatWidget companyId={company.id} />}

      {/* Modale Retour */}
      {openReturnModal && (
        <ReturnTimeModal
          open={openReturnModal}
          onClose={() => setOpenReturnModal(false)}
          onConfirm={handleConfirmReturnTime}
        />
      )}
    </div>
  );
};

export default CompanyDashboard;
