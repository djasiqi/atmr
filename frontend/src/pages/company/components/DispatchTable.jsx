import React, { useEffect, useMemo, useRef, useState } from "react";
import PropTypes from "prop-types";
import {
  LinearProgress,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Chip,
  Tooltip,
  Alert,
} from "@mui/material";
import { FiRefreshCw } from "react-icons/fi";
import { MdSwapHoriz } from "react-icons/md";
import styles from "./DispatchTable.module.css";
import { renderBookingDateTime } from "../../../utils/formatDate";

import useCompanySocket from "../../../hooks/useCompanySocket";
import useDispatchStatus from "../../../hooks/useDispatchStatus";
import { runDispatchForDay, fetchDispatchRunById } from "../../../services/companyService";
// Utilitaires locaux simples
const pad2 = (n) => String(n).padStart(2, "0");
const toYMD = (d) =>
  `${d.getFullYear()}-${pad2(d.getMonth() + 1)}-${pad2(d.getDate())}`;

const isActiveStatus = (s) =>
  ["scheduled", "en_route_pickup", "arrived_pickup", "onboard", "en_route_dropoff", "arrived_dropoff"].includes(
    (s || "").toLowerCase()
  );

/**
 * Tableau des courses dispatchées avec suivi du moteur.
 * - Reçoit des mises à jour temps réel (Socket)
 * - Affiche des alertes de retard
 * - Permet la réassignation manuelle d'un booking
 */
const DispatchTable = ({
  dispatches,
  reload,
  showPlanner = true,
  initialDispatchDay,
  initialRegularFirst = true,
  initialAllowEmergency = true,
  drivers = [],
  onReassign, // (assignmentId, newDriverId) => Promise<void>
}) => {
  // --- État moteur via WebSocket et polling ---
  const socket = useCompanySocket();
  const { 
    label, 
    progress, 
    isRunning, 
    setUpdatedAt: setStatusUpdatedAt,
    handleDispatchJobResponse 
  } = useDispatchStatus(socket);
  
  const isDispatching = isRunning;
  const statusLabel = isDispatching ? label : "Planification à jour";

  // --- Panneau planification ---
  const [dispatchDay, setDispatchDay] = useState(
    initialDispatchDay || toYMD(new Date())
  );
  const [regularFirst, setRegularFirst] = useState(initialRegularFirst);
  const [allowEmergency, setAllowEmergency] = useState(initialAllowEmergency);

  const handleOptimizeDay = async () => {
  if (!dispatchDay) return;
  
  try {
    console.log(`Triggering dispatch for date: ${dispatchDay}, regularFirst: ${regularFirst}, allowEmergency: ${allowEmergency}`);
    
    // Call the dispatch service and handle the response
    const response = await runDispatchForDay({
      forDate: dispatchDay,
      regularFirst,
      allowEmergency,
      // Force runAsync to true to ensure the job is queued
      runAsync: true,
    });
    
    console.log('Dispatch response:', response);
    
    // Handle the response with our enhanced hook
    handleDispatchJobResponse(response);

    // --- Fallback polling si on a un run_id mais pas d'event socket
    if (response?.dispatch_run_id) {
      let attempts = 0;
      const maxAttempts = 90; // ~3 minutes
      const poll = async () => {
        try {
          const run = await fetchDispatchRunById(response.dispatch_run_id);
          // status attendu: queued|running|completed|failed (selon ton modèle)
          if (run?.status === "completed" || run?.status === "failed") {
            // Use the date from the response if available
            const reloadDate = response.for_date || dispatchDay;
            reload?.(reloadDate);
            setUpdatedAt(Date.now());
            setStatusUpdatedAt(Date.now());
            return; // stop
          }
        } catch (e) {
          // on ignore l'erreur ponctuelle et on réessaie
        }
        attempts += 1;
        if (attempts < maxAttempts) {
          setTimeout(poll, 2000);
        }
      };
      setTimeout(poll, 2000);
    }
    
    // Reload data after a short delay to ensure backend has processed the request
    setTimeout(() => {
      reload?.();
      setUpdatedAt(Date.now());
    }, 2000);
    
  } catch (err) {
    console.error("Dispatch failed:", err);
    
    // Provide more detailed error information
    const errorMessage = err?.response?.data?.message || 
                        err?.response?.data?.error || 
                        err?.message ||
                        "Erreur lors de la planification.";
                        
    alert(`Erreur de dispatch: ${errorMessage}`);
  }
};

  // Écouter l'événement de fin de dispatch
  useEffect(() => {
    if (!socket) return;
    
    const handleDispatchCompleted = (data) => {
      console.log("Dispatch run completed event received:", data);
      
      // Ensure we have the necessary data
      if (!data) {
        console.error("Invalid dispatch_run_completed event data");
        return;
      }
      
      // Log the data for debugging
      console.log("Dispatch run completed with data:", {
        dispatch_run_id: data.dispatch_run_id,
        assignments_count: data.assignments_count,
        date: data.date
      });
      
      // If we have a date, use it for reloading
      const reloadDate = data.date || dispatchDay;
      
      // Reload assignments for the specific date
      if (reloadDate) {
        console.log(`Reloading assignments for date: ${reloadDate}`);
        reload?.(reloadDate);
      } else {
        // Fallback to general reload
        console.log("Reloading assignments (no specific date)");
        reload?.();
      }
      
      // Update timestamps
      setUpdatedAt(Date.now());
    };

    // Declare a handler named for proper removal
    const onDispatchRunCompleted = (data) => {
      console.log("Dispatch run completed:", data);
      // Verify that the structure is as expected
      if (data && (data.dispatch_run_id || data.date)) {
        handleDispatchCompleted(data);
      } else {
        console.error("Structure d'événement dispatch_run_completed invalide:", data);
      }
    };

    socket.on("dispatch_run_completed", onDispatchRunCompleted);

    return () => {
      socket.off("dispatch_run_completed", onDispatchRunCompleted);
    };
  }, [socket, reload, dispatchDay]);

  // --- "dernière mise à jour" ---
  const [updatedAt, setUpdatedAt] = useState(Date.now());
  const [relativeNow, setRelativeNow] = useState(Date.now());
  useEffect(() => {
    const id = setInterval(() => setRelativeNow(Date.now()), 60_000);
    return () => clearInterval(id);
  }, []);
  const updatedLabel = (() => {
    const delta = Math.max(0, relativeNow - updatedAt);
    const mins = Math.floor(delta / 60000);
    if (mins === 0) return "il y a quelques secondes";
    if (mins === 1) return "il y a 1 minute";
    return `il y a ${mins} minutes`;
  })();

  // --- Auto-refresh quand le moteur s'arrête ---
  const prevIsRunning = useRef(isRunning);
  useEffect(() => {
    if (prevIsRunning.current && !isRunning) {
      reload?.();
      setUpdatedAt(Date.now());
      setStatusUpdatedAt(Date.now());
    }
    prevIsRunning.current = isRunning;
  }, [isRunning, reload, setStatusUpdatedAt]);

  // --- Données locales + retard ---
  const [rows, setRows] = useState(() => normalizeAndSort(dispatches));
  const [delays, setDelays] = useState({}); // { [bookingId]: { delay_minutes, ... } }

  useEffect(() => {
    setRows(normalizeAndSort(dispatches));
  }, [dispatches]);

  // --- Abonnement Socket pour la date sélectionnée + évènements temps réel ---
  useEffect(() => {
    if (!socket) return;

    // Essaye de souscrire par date si le backend expose ces rooms
    try {
      socket.emit("subscribe:date", dispatchDay);
    } catch (_) {}

    const onAssignmentCreated = (data) => {
      setRows((prev) =>
        prev.map((b) =>
          b.id === data.booking_id
            ? {
                ...b,
                assignment: {
                  id: data.assignment_id,
                  driver_id: data.driver_id,
                  status: "assigned",
                  estimated_pickup_arrival: data.estimated_pickup_arrival,
                  estimated_dropoff_arrival: data.estimated_dropoff_arrival,
                },
              }
            : b
        )
      );
    };

    const onAssignmentUpdated = (data) => {
      setRows((prev) =>
        prev.map((b) =>
          b.assignment && b.assignment.id === data.assignment_id
            ? { ...b, assignment: { ...b.assignment, ...data.updates } }
            : b
        )
      );
    };

    const onAssignmentCancelled = (data) => {
      setRows((prev) =>
        prev.map((b) =>
          b.id === data.booking_id ? { ...b, assignment: null } : b
        )
      );
    };

    const onDelayDetected = (data) => {
      setDelays((prev) => ({
        ...prev,
        [data.booking_id]: {
          assignment_id: data.assignment_id,
          delay_minutes: data.delay_minutes,
          estimated_arrival: data.estimated_arrival,
          scheduled_time: data.scheduled_time,
          is_dropoff: data.is_dropoff || false,
          // Optionnel si ton backend envoie une alternative
          has_alternative: data.has_alternative,
          alternative_driver_id: data.alternative_driver_id,
          alternative_estimated_arrival: data.alternative_estimated_arrival,
          alternative_delay_minutes: data.alternative_delay_minutes,
        },
      }));
    };

    const onBookingStatusChanged = (data) => {
      setRows((prev) =>
        prev.map((b) =>
          b.id === data.booking_id ? { ...b, status: data.status } : b
        )
      );
      if (["completed", "cancelled"].includes(data.status)) {
        setDelays((prev) => {
          const cp = { ...prev };
          delete cp[data.booking_id];
          return cp;
        });
      }
    };

    // NB : on écoute aussi si tu souhaites ajuster visuellement les ETAs
    const onDriverLocationUpdated = (_data) => {
      // placeholder (les ETAs doivent en pratique venir du serveur)
    };

    socket.on("dispatch:assignment:created", onAssignmentCreated);
    socket.on("dispatch:assignment:updated", onAssignmentUpdated);
    socket.on("dispatch:assignment:cancelled", onAssignmentCancelled);
    socket.on("dispatch:delay:detected", onDelayDetected);
    socket.on("booking:status:changed", onBookingStatusChanged);
    socket.on("driver:location:updated", onDriverLocationUpdated);

    return () => {
      socket.off("dispatch:assignment:created", onAssignmentCreated);
      socket.off("dispatch:assignment:updated", onAssignmentUpdated);
      socket.off("dispatch:assignment:cancelled", onAssignmentCancelled);
      socket.off("dispatch:delay:detected", onDelayDetected);
      socket.off("booking:status:changed", onBookingStatusChanged);
      socket.off("driver:location:updated", onDriverLocationUpdated);
      try {
        socket.emit("unsubscribe:date", dispatchDay);
      } catch (_) {}
    };
  }, [socket, dispatchDay]);

  // --- Réassignation ---
  const [reModalOpen, setReModalOpen] = useState(false);
  const [selectedBooking, setSelectedBooking] = useState(null);
  const [selectedDriver, setSelectedDriver] = useState("");

  const openReassign = (booking) => {
    setSelectedBooking(booking);
    setSelectedDriver("");
    setReModalOpen(true);
  };

  const confirmReassign = async () => {
    if (selectedBooking?.assignment?.id && selectedDriver) {
      try {
        await onReassign?.(selectedBooking.assignment.id, selectedDriver);
        setReModalOpen(false);
      } catch (e) {
        alert("Échec de la réassignation.");
      }
    }
  };

  const availableDrivers = useMemo(() => {
    if (!selectedBooking) return [];
    const currentId = selectedBooking.assignment?.driver_id;
    return (drivers || [])
      .filter(
        (d) =>
          d.id !== currentId &&
          (
            (typeof d.status === "string" && d.status.toLowerCase() === "available") ||
            d.is_available === true
          )
      )
      .sort((a, b) => {
        const an = a.name || a.username || "";
        const bn = b.name || b.username || "";
        return an.localeCompare(bn);
      });
  }, [selectedBooking, drivers]);

  const handleRefresh = async () => {
    await reload?.();
    setUpdatedAt(Date.now());
  };

  // --- Stats pied de tableau ---
  const total = rows.length;
  const completed = rows.filter((b) => (b.status || "").toLowerCase() === "completed").length;
  const cancelled = rows.filter((b) => (b.status || "").toLowerCase() === "cancelled").length;
  const inProgress = rows.filter((b) => isActiveStatus(b.status)).length;
  const delayedCount = Object.keys(delays).length;

  return (
    <div className={styles.dispatchTableContainer}>
      {/* --- Panneau Planification --- */}
      {showPlanner && (
        <div className={styles.plannerBar}>
          <div className={styles.plannerRow}>
            <label>
              Jour:&nbsp;
              <input
                type="date"
                value={dispatchDay}
                onChange={(e) => setDispatchDay(e.target.value)}
                disabled={isDispatching}
              />
            </label>
            <label className={styles.inlineCheckbox}>
              <input
                type="checkbox"
                checked={regularFirst}
                onChange={(e) => setRegularFirst(e.target.checked)}
                disabled={isDispatching}
              />
              Réguliers d'abord
            </label>
            <label className={styles.inlineCheckbox}>
              <input
                type="checkbox"
                checked={allowEmergency}
                onChange={(e) => setAllowEmergency(e.target.checked)}
                disabled={isDispatching}
              />
              Autoriser urgences si nécessaire
            </label>
            <button
              className={styles.optimizeBtn}
              type="button"
              onClick={handleOptimizeDay}
              disabled={isDispatching || !dispatchDay}
              title={
                isDispatching
                  ? `Moteur en cours (${progress || 0}%)`
                  : !dispatchDay
                  ? "Sélectionne une date"
                  : "Lancer l'optimisation"
              }
            >
              {isDispatching ? "Optimisation..." : "Optimiser cette journée"}
            </button>
          </div>
          <small className={styles.hint}>
            Astuce : sélectionne le jour, puis "Optimiser cette journée". Le
            moteur traitera d'abord les réguliers, puis n'utilisera les
            chauffeurs d'urgence que si nécessaire (selon les options).
          </small>
        </div>
      )}

      {/* --- En-tête : état + progression + refresh --- */}
      <div className={styles.headerBar}>
        <div className={styles.left}>
          <span
            className={`${styles.statusPill} ${
              isDispatching ? styles.optimizing : styles.completed
            }`}
          >
            {statusLabel}
          </span>
          {isDispatching && (
            <div className={styles.progressWrap}>
              <LinearProgress variant="determinate" value={progress || 10} />
            </div>
          )}
        </div>
        <div className={styles.right}>
          <span className={styles.updatedAt}>Mis à jour {updatedLabel}</span>
          <button
            className={styles.refreshBtn}
            onClick={handleRefresh}
            disabled={isDispatching}
            title="Rafraîchir"
          >
            <FiRefreshCw />
          </button>
        </div>
      </div>
      {isDispatching && <LinearProgress />}

      {/* --- Tableau --- */}
      <table className={styles.dispatchTable}>
        <thead>
          <tr>
            <th>ID</th>
            <th>Client</th>
            <th>Date / Heure</th>
            <th>Pickup</th>
            <th>Dropoff</th>
            <th>Chauffeur assigné</th>
            <th>Statut</th>
            <th>Retard / Actions</th>
          </tr>
        </thead>
        <tbody>
          {rows.length > 0 ? (
            rows.map((b) => {
              const delay = delays[b.id];
              const hasAssignment = !!b.assignment;
              const assignedDriver = hasAssignment
                ? (drivers.find((d) => d.id === b.assignment.driver_id) || {})
                : {};
              const driverName =
                b?.driver_username ||
                b?.driver?.username ||
                assignedDriver.username ||
                assignedDriver.name ||
                "Non assigné";

              return (
                <tr key={b.id}>
                  <td>{b.id}</td>
                  <td>{b.customer_name || b.client?.full_name || "—"}</td>
                  <td>{renderBookingDateTime(b)}</td>
                  <td>{b.pickup_location || "—"}</td>
                  <td>{b.dropoff_location || "—"}</td>
                  <td>{driverName}</td>
                  <td>
                    <Chip
                      size="small"
                      label={b.status || "—"}
                      color={
                        (b.status || "").toLowerCase() === "completed"
                          ? "success"
                          : (b.status || "").toLowerCase() === "cancelled"
                          ? "error"
                          : "default"
                      }
                      variant="outlined"
                    />
                  </td>
                  <td>
                    <div className={styles.actionsCell}>
                      {delay && (
                        <Tooltip
                          title={`${
                            delay.is_dropoff ? "Dropoff" : "Pickup"
                          } en retard de ${delay.delay_minutes} min`}
                        >
                          <Chip
                            size="small"
                            color={delay.delay_minutes < 10 ? "warning" : "error"}
                            label={`${delay.delay_minutes} min de retard`}
                          />
                        </Tooltip>
                      )}
                      {onReassign && hasAssignment && isActiveStatus(b.status) && (
                        <Tooltip title="Réassigner">
                          <button
                            className={styles.iconBtn}
                            onClick={() => openReassign(b)}
                            aria-label="Réassigner"
                          >
                            <MdSwapHoriz />
                          </button>
                        </Tooltip>
                      )}
                    </div>
                  </td>
                </tr>
              );
            })
          ) : (
            <tr>
              <td colSpan="8" style={{ textAlign: "center" }}>
                Aucun dispatch à afficher.
              </td>
            </tr>
          )}
        </tbody>
        <tfoot>
          <tr>
            <td colSpan="8">
              <div className={styles.footerStats}>
                <span>Total : {total}</span>
                <span>En cours : {inProgress}</span>
                <span>Terminées : {completed}</span>
                <span>Annulées : {cancelled}</span>
                {delayedCount > 0 && (
                  <span className={styles.warning}>
                    Retards : {delayedCount}
                  </span>
                )}
              </div>
            </td>
          </tr>
        </tfoot>
      </table>

      {/* --- Modal réassignation --- */}
      <Dialog open={!!onReassign && reModalOpen} onClose={() => setReModalOpen(false)} fullWidth>
        <DialogTitle>Réassigner la course</DialogTitle>
        <DialogContent dividers>
          {selectedBooking && (
            <>
              <div className={styles.modalBlock}>
                <strong>Course #{selectedBooking.id}</strong>
                <div>Client : {selectedBooking.customer_name || selectedBooking.client?.full_name || "—"}</div>
                <div>Pickup : {selectedBooking.pickup_location || "—"}</div>
                <div>Dropoff : {selectedBooking.dropoff_location || "—"}</div>
                <div>Date/Heure : {renderBookingDateTime(selectedBooking)}</div>
                {delays[selectedBooking.id] && (
                  <div style={{ marginTop: 8 }}>
                    <Alert
                      severity="warning"
                      variant="outlined"
                      sx={{ borderStyle: "dashed" }}
                    >
                      Retard estimé : {delays[selectedBooking.id].delay_minutes} min
                    </Alert>
                  </div>
                )}
              </div>
              <FormControl fullWidth sx={{ mt: 2 }}>
                <InputLabel id="driver-select-label">Nouveau chauffeur</InputLabel>
                <Select
                  labelId="driver-select-label"
                  label="Nouveau chauffeur"
                  value={selectedDriver}
                  onChange={(e) => setSelectedDriver(e.target.value)}
                >
                  {availableDrivers.map((d) => (
                    <MenuItem key={d.id} value={d.id}>
                      {d.name} {d.is_emergency_driver ? "(Urgence)" : ""}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
              {availableDrivers.length === 0 && (
                <Alert sx={{ mt: 2 }} severity="info">
                  Aucun chauffeur disponible sur ce créneau.
                </Alert>
              )}
              {delays[selectedBooking.id]?.has_alternative && (
                <Alert sx={{ mt: 2 }} severity="success">
                  Suggestion :{" "}
                  {
                    drivers.find(
                      (d) =>
                        d.id ===
                        delays[selectedBooking.id].alternative_driver_id
                    )?.name
                  }{" "}
                  (arrivée ~ {delays[selectedBooking.id].alternative_delay_minutes} min de
                  retard)
                </Alert>
              )}
            </>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setReModalOpen(false)}>Annuler</Button>
          <Button
            variant="contained"
            onClick={confirmReassign}
            disabled={!selectedDriver}
          >
            Confirmer
          </Button>
        </DialogActions>
      </Dialog>
    </div>
  );
};

function normalizeAndSort(dispatches) {
  const list = Array.isArray(dispatches)
    ? dispatches
    : dispatches
    ? Object.values(dispatches)
    : [];
  return [...list].sort((a, b) => {
    const aMs = a?.scheduled_time
      ? Date.parse(a.scheduled_time)
      : Number.POSITIVE_INFINITY;
    const bMs = b?.scheduled_time
      ? Date.parse(b.scheduled_time)
      : Number.POSITIVE_INFINITY;
    return aMs - bMs;
  });
}

DispatchTable.propTypes = {
  dispatches: PropTypes.oneOfType([
    PropTypes.arrayOf(
      PropTypes.shape({
        id: PropTypes.number.isRequired,
        customer_name: PropTypes.string,
        client: PropTypes.shape({ full_name: PropTypes.string }),
        scheduled_time: PropTypes.string, // ISO
        pickup_location: PropTypes.string,
        dropoff_location: PropTypes.string,
        driver_username: PropTypes.string,
        driver: PropTypes.oneOfType([
          PropTypes.string,
          PropTypes.shape({ username: PropTypes.string }),
        ]),
        assignment: PropTypes.shape({
          id: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
          driver_id: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
          status: PropTypes.string,
          estimated_pickup_arrival: PropTypes.string,
          estimated_dropoff_arrival: PropTypes.string,
        }),
        is_return: PropTypes.bool,
        status: PropTypes.string,
      })
    ),
    PropTypes.object,
  ]).isRequired,
  reload: PropTypes.func.isRequired,
  showPlanner: PropTypes.bool,
  initialDispatchDay: PropTypes.string, // "YYYY-MM-DD"
  initialRegularFirst: PropTypes.bool,
  initialAllowEmergency: PropTypes.bool,
  drivers: PropTypes.arrayOf(
    PropTypes.shape({
      id: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
      name: PropTypes.string,
      username: PropTypes.string,
      status: PropTypes.string,      // "available" / ...
      is_available: PropTypes.bool,  // bool backend
      is_emergency_driver: PropTypes.bool,
    })
  ),
  onReassign: PropTypes.func, // (assignmentId, newDriverId)
};

export default DispatchTable;