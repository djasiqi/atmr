/**
 * VirtualizedDispatchTable.jsx
 *
 * Composant virtualisé pour DispatchTable utilisant react-window.
 *
 * Ce composant virtualise le rendu des lignes du tableau DispatchTable
 * pour améliorer les performances avec de grandes listes (100+ éléments).
 *
 * @module components/virtualized/VirtualizedDispatchTable
 */

import React, { useEffect, useMemo, useRef, useState, useCallback } from 'react';
import PropTypes from 'prop-types';
import { List } from 'react-window';
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
  Alert,
} from '@mui/material';
import { FiRefreshCw } from 'react-icons/fi';
import styles from '../../pages/company/components/DispatchTable.module.css';
import { renderBookingDateTime } from '../../utils/formatDate';
import { Chip, Tooltip } from '@mui/material';

import useCompanySocket from '../../hooks/useCompanySocket';
import useDispatchStatus from '../../hooks/useDispatchStatus';
import { useHybridDataSync } from '../../hooks/useHybridDataSync';
import {
  runDispatchForDay,
  fetchDispatchRunById,
  fetchDispatchDelays,
} from '../../services/companyService';
import {
  canonicalRealtimeTimeMs,
  shouldAcceptRealtimeEvent,
} from '../../utils/realtimeEventGuard';

// Utilitaires locaux simples
const pad2 = (n) => String(n).padStart(2, '0');
const toYMD = (d) => `${d.getFullYear()}-${pad2(d.getMonth() + 1)}-${pad2(d.getDate())}`;

const isActiveStatus = (s) =>
  [
    'scheduled',
    'en_route_pickup',
    'arrived_pickup',
    'onboard',
    'en_route_dropoff',
    'arrived_dropoff',
  ].includes((s || '').toLowerCase());

function normalizeAndSort(dispatches) {
  const list = Array.isArray(dispatches) ? dispatches : dispatches ? Object.values(dispatches) : [];
  return [...list].sort((a, b) => {
    const aMs = a?.scheduled_time ? Date.parse(a.scheduled_time) : Number.POSITIVE_INFINITY;
    const bMs = b?.scheduled_time ? Date.parse(b.scheduled_time) : Number.POSITIVE_INFINITY;
    return aMs - bMs;
  });
}

/**
 * Composant VirtualizedDispatchTable
 *
 * Version virtualisée de DispatchTable qui n'affiche que les lignes visibles
 * dans le viewport, améliorant significativement les performances.
 */
const VirtualizedDispatchTable = ({
  dispatches,
  reload,
  showPlanner = true,
  initialDispatchDay,
  initialRegularFirst = true,
  initialAllowEmergency = true,
  drivers = [],
  onReassign,
}) => {
  // --- État moteur via WebSocket et polling ---
  const socket = useCompanySocket();
  const {
    label,
    progress,
    isRunning,
    setUpdatedAt: setStatusUpdatedAt,
    handleDispatchJobResponse,
  } = useDispatchStatus(socket);

  const isDispatching = isRunning;
  const statusLabel = isDispatching ? label : 'Planification à jour';

  // --- Panneau planification ---
  const [dispatchDay, setDispatchDay] = useState(initialDispatchDay || toYMD(new Date()));
  const [regularFirst, setRegularFirst] = useState(initialRegularFirst);
  const [allowEmergency, setAllowEmergency] = useState(initialAllowEmergency);

  const useUnifiedDispatchWs =
    process.env.REACT_APP_LIRIE_DISPATCH_WS_UNIFIED === '1' ||
    process.env.REACT_APP_LIRIE_DISPATCH_WS_UNIFIED === 'true';
  const acceptRealtime = useCallback(
    (payload, entityKey = null) =>
      shouldAcceptRealtimeEvent({
        eventId: payload?.event_id,
        entityKey,
        canonicalTimeMs: canonicalRealtimeTimeMs(payload),
      }),
    []
  );

  const handleOptimizeDay = async () => {
    if (!dispatchDay) return;

    try {
      console.log(
        `Triggering dispatch for date: ${dispatchDay}, regularFirst: ${regularFirst}, allowEmergency: ${allowEmergency}`
      );

      const response = await runDispatchForDay({
        forDate: dispatchDay,
        regularFirst,
        allowEmergency,
        runAsync: true,
      });

      console.log('Dispatch response:', response);
      handleDispatchJobResponse(response);

      if (response?.dispatch_run_id) {
        let attempts = 0;
        const maxAttempts = 10;
        let delay = 2000;
        const timeoutGlobal = 10 * 60 * 1000;
        const startTime = Date.now();

        const poll = async () => {
          const elapsed = Date.now() - startTime;
          if (elapsed >= timeoutGlobal) {
            console.warn(
              `[Dispatch] Timeout global atteint (10min) pour dispatch_run_id=${response.dispatch_run_id}`
            );
            return;
          }

          try {
            const run = await fetchDispatchRunById(response.dispatch_run_id);
            if (run?.status === 'completed' || run?.status === 'failed') {
              const reloadDate = response.for_date || dispatchDay;
              reload?.(reloadDate);
              setUpdatedAt(Date.now());
              setStatusUpdatedAt(Date.now());
              return;
            }
          } catch (e) {
            console.warn('[Dispatch] Erreur polling (tentative', attempts + 1, '):', e);
          }

          attempts += 1;
          if (attempts < maxAttempts) {
            delay = Math.min(delay * 1.5, 10000);
            setTimeout(poll, delay);
          } else {
            console.warn(
              `[Dispatch] Maximum attempts (${maxAttempts}) atteint pour dispatch_run_id=${response.dispatch_run_id}`
            );
          }
        };
        setTimeout(poll, delay);
      }

      setTimeout(() => {
        reload?.();
        setUpdatedAt(Date.now());
      }, 2000);
    } catch (err) {
      console.error('Dispatch failed:', err);
      const errorMessage =
        err?.response?.data?.message ||
        err?.response?.data?.error ||
        err?.message ||
        'Erreur lors de la planification.';
      alert(`Erreur de dispatch: ${errorMessage}`);
    }
  };

  // Écouter l'événement de fin de dispatch
  useEffect(() => {
    if (!socket) return;

    const handleDispatchCompleted = (data) => {
      console.log('Dispatch run completed event received:', data);
      if (!data) {
        console.error('Invalid dispatch_run_completed event data');
        return;
      }

      const reloadDate = data.date || dispatchDay;
      if (reloadDate) {
        console.log(`Reloading assignments for date: ${reloadDate}`);
        reload?.(reloadDate);
      } else {
        console.log('Reloading assignments (no specific date)');
        reload?.();
      }
      setUpdatedAt(Date.now());
    };

    const onDispatchRunCompleted = (data) => {
      if (!acceptRealtime(data, data?.dispatch_run_id ? `dispatch-run:${data.dispatch_run_id}` : null)) {
        return;
      }
      console.log('Dispatch run completed:', data);
      if (data && (data.dispatch_run_id || data.date)) {
        handleDispatchCompleted(data);
      } else {
        console.error("Structure d'événement dispatch_run_completed invalide:", data);
      }
    };

    socket.on('dispatch_run_completed', onDispatchRunCompleted);

    return () => {
      socket.off('dispatch_run_completed', onDispatchRunCompleted);
    };
  }, [socket, reload, dispatchDay, acceptRealtime]);

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
    if (mins === 0) return 'il y a quelques secondes';
    if (mins === 1) return 'il y a 1 minute';
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
  const [delays, setDelays] = useState({});

  useEffect(() => {
    setRows(normalizeAndSort(dispatches));
  }, [dispatches]);

  const processDelayData = useCallback((data) => {
    const map = {};
    for (const d of data || []) {
      const bid = d.booking_id;
      if (!bid) continue;
      const prev = map[bid]?.delay_minutes ?? 0;
      const cur = Number(d.delay_minutes ?? d.pickup_delay_minutes ?? d.dropoff_delay_minutes ?? 0);
      if (!map[bid] || cur > prev) {
        map[bid] = {
          booking_id: bid,
          delay_minutes: cur,
          is_dropoff: d.is_dropoff || false,
          estimated_arrival: d.estimated_arrival || d.pickup_eta || d.dropoff_eta || null,
          scheduled_time: d.scheduled_time || null,
        };
      }
    }
    setDelays(map);
  }, []);

  // Chargement initial des retards pour la journée sélectionnée
  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const data = await fetchDispatchDelays(dispatchDay);
        if (cancelled) return;
        processDelayData(data);
      } catch {}
    };
    load();
    return () => {
      cancelled = true;
    };
  }, [dispatchDay, processDelayData]);

  // Polling hybride socket-first : poll seulement si socket indisponible ou flux stale
  useHybridDataSync({
    fetchFn: async () => {
      const data = await fetchDispatchDelays(dispatchDay);
      processDelayData(data);
      return data;
    },
    socket,
    staleThreshold: 120000,
    pollIntervalConnected: 180000,
    pollIntervalDisconnected: 45000,
    onUpdate: (timestamp) => {
      console.log(
        JSON.stringify({
          event: 'hybrid_poll_update',
          timestamp: new Date(timestamp).toISOString(),
          socket_connected: socket?.connected || false,
        })
      );
    },
    dependencies: [dispatchDay],
  });

  // --- Abonnement Socket pour la date sélectionnée + évènements temps réel ---
  useEffect(() => {
    if (!socket) return;

    try {
      socket.emit('subscribe:date', dispatchDay);
    } catch (_) {}

    const onAssignmentCreated = (data) => {
      if (!acceptRealtime(data, data?.booking_id ? `booking:${data.booking_id}` : null)) return;
      setRows((prev) =>
        prev.map((b) =>
          b.id === data.booking_id
            ? {
                ...b,
                assignment: {
                  id: data.assignment_id,
                  driver_id: data.driver_id,
                  status: 'assigned',
                  estimated_pickup_arrival: data.estimated_pickup_arrival,
                  estimated_dropoff_arrival: data.estimated_dropoff_arrival,
                },
              }
            : b
        )
      );
    };

    const onAssignmentUpdated = (data) => {
      if (!acceptRealtime(data, data?.assignment_id ? `assignment:${data.assignment_id}` : null)) return;
      const patch = data.updates || data.fields || {};
      setRows((prev) =>
        prev.map((b) =>
          b.assignment && b.assignment.id === data.assignment_id
            ? { ...b, assignment: { ...b.assignment, ...patch } }
            : b
        )
      );
    };

    const onAssignmentCancelled = (data) => {
      if (!acceptRealtime(data, data?.booking_id ? `booking:${data.booking_id}` : null)) return;
      setRows((prev) =>
        prev.map((b) => (b.id === data.booking_id ? { ...b, assignment: null } : b))
      );
    };

    const onDispatchStatePatch = (data) => {
      if (!acceptRealtime(data, data?.reservation_id ? `booking:${data.reservation_id}` : null)) return;
      const op = data?.op;
      if (op === 'assignment_created') {
        onAssignmentCreated({
          booking_id: data.reservation_id,
          assignment_id: data.assignment_id,
          driver_id: data.driver_id,
        });
      } else if (op === 'assignment_updated') {
        onAssignmentUpdated({
          assignment_id: data.assignment_id,
          updates: data.fields || {},
        });
      } else if (op === 'assignment_cancelled') {
        onAssignmentCancelled({ booking_id: data.reservation_id });
      }
    };

    const onDelayDetected = (data) => {
      if (!acceptRealtime(data, data?.booking_id ? `booking:${data.booking_id}` : null)) return;
      setDelays((prev) => ({
        ...prev,
        [data.booking_id]: {
          assignment_id: data.assignment_id,
          delay_minutes: data.delay_minutes,
          estimated_arrival: data.estimated_arrival,
          scheduled_time: data.scheduled_time,
          is_dropoff: data.is_dropoff || false,
          has_alternative: data.has_alternative,
          alternative_driver_id: data.alternative_driver_id,
          alternative_estimated_arrival: data.alternative_estimated_arrival,
          alternative_delay_minutes: data.alternative_delay_minutes,
          driver_name: data.driver_name,
          driver_phone: data.driver_phone,
          driver_vehicle: data.driver_vehicle,
          driver_id: data.driver_id,
        },
      }));
    };

    const _onBookingStatusChanged = (data) => {
      if (!acceptRealtime(data, data?.booking_id ? `booking:${data.booking_id}` : null)) return;
      setRows((prev) =>
        prev.map((b) => (b.id === data.booking_id ? { ...b, status: data.status } : b))
      );
      if (['completed', 'cancelled'].includes(data.status)) {
        setDelays((prev) => {
          const cp = { ...prev };
          delete cp[data.booking_id];
          return cp;
        });
      }
    };

    let locTimer;
    const onDriverLocationUpdated = (data) => {
      if (!acceptRealtime(data, data?.driver_id ? `driver:${data.driver_id}` : null)) return;
      if (locTimer) clearTimeout(locTimer);
      locTimer = setTimeout(async () => {
        try {
          const data = await fetchDispatchDelays(dispatchDay);
          const map = {};
          for (const d of data || []) {
            const bid = d.booking_id;
            if (!bid) continue;
            const prev = map[bid]?.delay_minutes ?? 0;
            const cur = Number(
              d.delay_minutes ?? d.pickup_delay_minutes ?? d.dropoff_delay_minutes ?? 0
            );
            if (!map[bid] || cur > prev) {
              map[bid] = {
                booking_id: bid,
                delay_minutes: cur,
                is_dropoff: d.is_dropoff || false,
                estimated_arrival: d.estimated_arrival || d.pickup_eta || d.dropoff_eta || null,
                scheduled_time: d.scheduled_time || null,
              };
            }
          }
          setDelays(map);
        } catch {}
      }, 800);
    };

    if (useUnifiedDispatchWs) {
      socket.on('dispatch_state_patch', onDispatchStatePatch);
    } else {
      socket.on('dispatch_assignment_created', onAssignmentCreated);
      socket.on('dispatch_assignment_updated', onAssignmentUpdated);
      socket.on('dispatch_assignment_cancelled', onAssignmentCancelled);
    }
    socket.on('dispatch_delay_detected', onDelayDetected);
    // booking_status_changed supprimé (jamais émis par backend, remplacé par booking_updated)
    socket.on('driver_location_update', onDriverLocationUpdated);

    return () => {
      if (useUnifiedDispatchWs) {
        socket.off('dispatch_state_patch', onDispatchStatePatch);
      } else {
        socket.off('dispatch_assignment_created', onAssignmentCreated);
        socket.off('dispatch_assignment_updated', onAssignmentUpdated);
        socket.off('dispatch_assignment_cancelled', onAssignmentCancelled);
      }
      socket.off('dispatch_delay_detected', onDelayDetected);
      socket.off('driver_location_update', onDriverLocationUpdated);
      if (locTimer) clearTimeout(locTimer);
      try {
        socket.emit('unsubscribe:date', dispatchDay);
      } catch (_) {}
    };
  }, [socket, dispatchDay, useUnifiedDispatchWs, acceptRealtime]);

  // --- Réassignation ---
  const [reModalOpen, setReModalOpen] = useState(false);
  const [selectedBooking, _setSelectedBooking] = useState(null);
  const [selectedDriver, setSelectedDriver] = useState('');

  const confirmReassign = async () => {
    if (selectedBooking?.assignment?.id && selectedDriver) {
      try {
        await onReassign?.(selectedBooking.assignment.id, selectedDriver);
        setReModalOpen(false);
      } catch (e) {
        alert('Échec de la réassignation.');
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
          ((typeof d.status === 'string' && d.status.toLowerCase() === 'available') ||
            d.is_available === true)
      )
      .sort((a, b) => {
        const an = a.name || a.username || '';
        const bn = b.name || b.username || '';
        return an.localeCompare(bn);
      });
  }, [selectedBooking, drivers]);

  const handleRefresh = useCallback(async () => {
    await reload?.();
    setUpdatedAt(Date.now());
  }, [reload]);

  // --- Stats pied de tableau ---
  const total = rows.length;
  const completed = rows.filter((b) => (b.status || '').toLowerCase() === 'completed').length;
  const cancelled = rows.filter((b) => (b.status || '').toLowerCase() === 'cancelled').length;
  const inProgress = rows.filter((b) => isActiveStatus(b.status)).length;
  const delayedCount = Object.keys(delays).length;

  // --- Helpers retard/ETA ---
  const toDate = useCallback((v) => {
    try {
      return v ? new Date(v) : null;
    } catch (_) {
      return null;
    }
  }, []);

  const minutesBetween = useCallback((a, b) => {
    if (!a || !b) return null;
    return Math.round((a.getTime() - b.getTime()) / 60000);
  }, []);

  const timingStatus = useCallback(
    (b) => {
      // 1) signalements temps réel (prend le pas)
      const d = delays[b.id];
      if (d && typeof d.delay_minutes === 'number') {
        const mins = d.delay_minutes;
        return {
          kind: mins <= 0 ? 'on_time' : 'delayed',
          minutes: Math.max(0, mins),
          label: mins <= 0 ? "À l'heure" : `${mins} min de retard`,
        };
      }
      // 2) estimation de l'assignation (ETA prévue vs horaire)
      const sch = toDate(b.scheduled_time);
      const eta = toDate(b.assignment?.estimated_pickup_arrival);
      if (sch && eta) {
        const diff = minutesBetween(eta, sch);
        if (diff !== null) {
          if (diff <= 0) return { kind: 'on_time', minutes: 0, label: "À l'heure" };
          if (diff > 0 && diff < 10)
            return {
              kind: 'slightly_delayed',
              minutes: diff,
              label: `${diff} min de retard`,
            };
          return {
            kind: 'delayed',
            minutes: diff,
            label: `${diff} min de retard`,
          };
        }
      }
      // 3) impossibilité: pas d'assignation et statut actif/à venir
      const st = (b.status || '').toLowerCase();
      const isDone = st === 'completed' || st === 'cancelled';
      if (!b.assignment && !isDone) {
        return {
          kind: 'impossible',
          minutes: null,
          label: 'Impossible (aucun chauffeur)',
        };
      }
      return { kind: 'unknown', minutes: null, label: '—' };
    },
    [delays, toDate, minutesBetween]
  );

  // --- Virtualisation ---
  const listRef = useRef(null);
  const rowHeightsRef = useRef({});

  // Hauteur estimée par défaut (peut être ajustée)
  const DEFAULT_ROW_HEIGHT = 60;

  const getItemSize = useCallback(
    (index) => {
      // Retourner la hauteur stockée ou la hauteur par défaut
      return rowHeightsRef.current[index] || DEFAULT_ROW_HEIGHT;
    },
    []
  );

  // Mesurer la hauteur réelle après rendu (optionnel, pour optimisation)
  // Note: Cette fonction peut être utilisée pour mesurer dynamiquement les hauteurs
  // const setRowHeight = useCallback((index, size) => {
  //   if (rowHeightsRef.current[index] !== size) {
  //     rowHeightsRef.current[index] = size;
  //     if (listRef.current) {
  //       listRef.current.resetAfterIndex(index);
  //     }
  //   }
  // }, []);

  // Composant de ligne pour react-window
  // Note: react-window rend des divs, donc on doit adapter DispatchTableRow
  // pour qu'il rende des divs avec display: table-row au lieu de tr
  const Row = useCallback(
    ({ index, style }) => {
      const booking = rows[index];
      if (!booking) return null;

      const hasAssignment = !!booking.assignment;
      const assignedDriver = hasAssignment
        ? drivers.find((d) => d.id === booking.assignment.driver_id) || {}
        : {};

      // Résolution robuste du nom chauffeur (même logique que DispatchTableRow)
      let driverName = 'Non assigné';
      if (typeof booking?.driver === 'string' && booking.driver.trim()) {
        driverName = booking.driver.trim();
      } else if (booking?.driver?.full_name) {
        driverName = booking.driver.full_name;
      } else if (booking?.driver?.first_name || booking?.driver?.last_name) {
        driverName = `${booking.driver.first_name || ''} ${booking.driver.last_name || ''}`.trim();
      } else if (booking?.driver_username) {
        driverName = booking.driver_username;
      } else if (booking?.driver?.username) {
        driverName = booking.driver.username;
      } else if (booking?.driver_name) {
        driverName = booking.driver_name;
      } else if (booking?.driver_id) {
        const byId = drivers.find((d) => d.id === booking.driver_id);
        if (byId) {
          driverName =
            byId.full_name ||
            (byId.first_name || byId.last_name
              ? `${byId.first_name || ''} ${byId.last_name || ''}`.trim()
              : byId.username || byId.name || `#${byId.id}`);
        }
      } else if (assignedDriver) {
        driverName =
          assignedDriver.full_name ||
          (assignedDriver.first_name || assignedDriver.last_name
            ? `${assignedDriver.first_name || ''} ${assignedDriver.last_name || ''}`.trim()
            : assignedDriver.username || assignedDriver.name || 'Non assigné');
      }

      if ((booking.status || '').toLowerCase() === 'completed' && driverName === 'Non assigné') {
        driverName = 'Inconnu';
      }

      const t = timingStatus(booking);

      // Utiliser une structure de table pour aligner les colonnes avec le header
      // Les largeurs sont gérées par tableLayout: fixed sur le conteneur parent
      return (
        <div style={{ ...style, display: 'table-row', width: '100%' }} className={styles.virtualizedRow}>
          <div style={{ display: 'table-cell', padding: 'var(--spacing-sm) var(--spacing-md)', verticalAlign: 'middle' }}>
            {booking.id}
          </div>
          <div style={{ display: 'table-cell', padding: 'var(--spacing-sm) var(--spacing-md)', verticalAlign: 'middle' }}>
            {booking.client_name || booking.client?.full_name || '—'}
          </div>
          <div style={{ display: 'table-cell', padding: 'var(--spacing-sm) var(--spacing-md)', verticalAlign: 'middle' }}>
            {renderBookingDateTime(booking)}
          </div>
          <div style={{ display: 'table-cell', padding: 'var(--spacing-sm) var(--spacing-md)', verticalAlign: 'middle' }}>
            {booking.pickup_location || '—'}
          </div>
          <div style={{ display: 'table-cell', padding: 'var(--spacing-sm) var(--spacing-md)', verticalAlign: 'middle' }}>
            {booking.dropoff_location || '—'}
          </div>
          <div style={{ display: 'table-cell', padding: 'var(--spacing-sm) var(--spacing-md)', verticalAlign: 'middle' }}>
            {driverName}
          </div>
          <div style={{ display: 'table-cell', padding: 'var(--spacing-sm) var(--spacing-md)', verticalAlign: 'middle' }}>
            <Chip
              size="small"
              label={booking.status || '—'}
              color={
                (booking.status || '').toLowerCase() === 'completed'
                  ? 'success'
                  : (booking.status || '').toLowerCase() === 'cancelled'
                    ? 'error'
                    : 'default'
              }
              variant="outlined"
            />
          </div>
          <div style={{ display: 'table-cell', padding: 'var(--spacing-sm) var(--spacing-md)', verticalAlign: 'middle' }}>
            {t.kind === 'on_time' && (
              <Chip size="small" label={t.label} className={styles.statusChipOnTime} />
            )}
            {t.kind === 'slightly_delayed' && (
              <Tooltip title="Retard faible, OK si < 10 min">
                <Chip size="small" label={t.label} className={styles.statusChipSlightDelay} />
              </Tooltip>
            )}
            {t.kind === 'delayed' && (() => {
              const delayInfo = delays[booking.id];
              const tooltipContent = delayInfo?.driver_name ? (
                <div>
                  <div><strong>{delayInfo.driver_name}</strong></div>
                  {delayInfo.driver_phone && (
                    <div>
                      <a href={`tel:${delayInfo.driver_phone}`} className="text-white underline">
                        {delayInfo.driver_phone}
                      </a>
                    </div>
                  )}
                  {delayInfo.driver_vehicle && <div>Véhicule: {delayInfo.driver_vehicle}</div>}
                </div>
              ) : (
                'Retard important'
              );
              return (
                <Tooltip title={tooltipContent}>
                  <Chip size="small" label={t.label} className={styles.statusChipDelay} />
                </Tooltip>
              );
            })()}
            {t.kind === 'impossible' && (
              <div className={styles.actionsCell}>
                <Chip size="small" label={t.label} className={styles.statusChipImpossible} />
                <button
                  className={styles.iconBtn}
                  onClick={() => alert('Action: appeler le client')}
                  aria-label="Appeler le client"
                  title="Appeler le client"
                >
                  📞
                </button>
              </div>
            )}
            {t.kind === 'unknown' && <span>—</span>}
          </div>
        </div>
      );
    },
    [rows, delays, drivers, timingStatus]
  );

  // Hauteur du conteneur virtualisé
  // Ajustée pour afficher environ 10 lignes à la fois (60px par ligne)
  // Peut être ajustée dynamiquement selon la taille de l'écran si nécessaire
  const VIRTUALIZED_HEIGHT = useMemo(() => {
    if (typeof window !== 'undefined') {
      return Math.min(600, window.innerHeight * 0.6);
    }
    return 600; // Fallback pour SSR
  }, []);

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
                    ? 'Sélectionne une date'
                    : "Lancer l'optimisation"
              }
            >
              {isDispatching ? 'Optimisation...' : 'Optimiser cette journée'}
            </button>
          </div>
          <small className={styles.hint}>
            Astuce : sélectionne le jour, puis "Optimiser cette journée". Le moteur traitera d'abord
            les réguliers, puis n'utilisera les chauffeurs d'urgence que si nécessaire (selon les
            options).
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

      {/* --- Tableau virtualisé --- */}
      <div className={styles.virtualizedTableWrapper}>
        {/* Header fixe */}
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
        </table>
        {/* Body virtualisé - Structure hybride pour aligner les colonnes */}
        {rows.length > 0 ? (
          <div style={{ display: 'table', width: '100%', tableLayout: 'fixed' }} className={styles.dispatchTable}>
            <List
              ref={listRef}
              height={VIRTUALIZED_HEIGHT}
              itemCount={rows.length}
              itemSize={getItemSize}
              width="100%"
            >
              {Row}
            </List>
          </div>
        ) : (
          <table className={styles.dispatchTable}>
            <tbody>
              <tr>
                <td colSpan="8" className="text-center">
                  Aucun dispatch à afficher.
                </td>
              </tr>
            </tbody>
          </table>
        )}
        {/* Footer fixe */}
        <table className={styles.dispatchTable}>
          <tfoot>
            <tr>
              <td colSpan="8">
                <div className={styles.footerStats}>
                  <span>Total : {total}</span>
                  <span>En cours : {inProgress}</span>
                  <span>Terminées : {completed}</span>
                  <span>Annulées : {cancelled}</span>
                  {delayedCount > 0 && (
                    <span className={styles.warning}>Retards : {delayedCount}</span>
                  )}
                </div>
              </td>
            </tr>
          </tfoot>
        </table>
      </div>

      {/* --- Modal réassignation --- */}
      <Dialog open={!!onReassign && reModalOpen} onClose={() => setReModalOpen(false)} fullWidth>
        <DialogTitle>Réassigner la course</DialogTitle>
        <DialogContent dividers>
          {selectedBooking && (
            <>
              <div className={styles.modalBlock}>
                <strong>Course #{selectedBooking.id}</strong>
                <div>
                  Client :{' '}
                  {selectedBooking.client_name || selectedBooking.client?.full_name || '—'}
                </div>
                <div>Pickup : {selectedBooking.pickup_location || '—'}</div>
                <div>Dropoff : {selectedBooking.dropoff_location || '—'}</div>
                <div>Date/Heure : {renderBookingDateTime(selectedBooking)}</div>
                {delays[selectedBooking.id] && (
                  <div className="mt-sm">
                    <Alert severity="warning" variant="outlined" sx={{ borderStyle: 'dashed' }}>
                      <div>Retard estimé : {delays[selectedBooking.id].delay_minutes} min</div>
                      {delays[selectedBooking.id].driver_name && (
                        <div className="mt-xs">
                          <strong>Chauffeur :</strong> {delays[selectedBooking.id].driver_name}
                          {delays[selectedBooking.id].driver_phone && (
                            <span>
                              {' — '}
                              <a href={`tel:${delays[selectedBooking.id].driver_phone}`}>
                                {delays[selectedBooking.id].driver_phone}
                              </a>
                            </span>
                          )}
                          {delays[selectedBooking.id].driver_vehicle && (
                            <div className="mt-xxs">
                              <strong>Véhicule :</strong> {delays[selectedBooking.id].driver_vehicle}
                            </div>
                          )}
                        </div>
                      )}
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
                      {d.name} {d.is_emergency_driver ? '(Urgence)' : ''}
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
                  Suggestion :{' '}
                  {
                    drivers.find((d) => d.id === delays[selectedBooking.id].alternative_driver_id)
                      ?.name
                  }{' '}
                  (arrivée ~ {delays[selectedBooking.id].alternative_delay_minutes} min de retard)
                </Alert>
              )}
            </>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setReModalOpen(false)}>Annuler</Button>
          <Button variant="contained" onClick={confirmReassign} disabled={!selectedDriver}>
            Confirmer
          </Button>
        </DialogActions>
      </Dialog>
    </div>
  );
};

VirtualizedDispatchTable.propTypes = {
  dispatches: PropTypes.oneOfType([
    PropTypes.arrayOf(
      PropTypes.shape({
        id: PropTypes.number.isRequired,
        client_name: PropTypes.string,
        client: PropTypes.shape({ full_name: PropTypes.string }),
        scheduled_time: PropTypes.string,
        pickup_location: PropTypes.string,
        dropoff_location: PropTypes.string,
        driver_username: PropTypes.string,
        driver: PropTypes.oneOfType([PropTypes.string, PropTypes.shape({ username: PropTypes.string })]),
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
  initialDispatchDay: PropTypes.string,
  initialRegularFirst: PropTypes.bool,
  initialAllowEmergency: PropTypes.bool,
  drivers: PropTypes.arrayOf(PropTypes.object),
  onReassign: PropTypes.func,
};

export default VirtualizedDispatchTable;
