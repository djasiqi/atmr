/**
 * DispatchTableRow.jsx
 *
 * Composant pour rendre une ligne du tableau DispatchTable.
 * Extrait la logique de rendu d'une ligne pour faciliter la virtualisation.
 *
 * @module components/virtualized/DispatchTableRow
 */

import React from 'react';
import PropTypes from 'prop-types';
import { Chip, Tooltip } from '@mui/material';
import { renderBookingDateTime } from '../../../utils/formatDate';
import styles from '../../pages/company/components/DispatchTable.module.css';

/**
 * Composant DispatchTableRow
 *
 * Rend une ligne du tableau DispatchTable avec toutes ses colonnes.
 *
 * @param {Object} props - Props du composant
 * @param {Object} props.booking - Données du booking/dispatch
 * @param {Object} props.delays - Map des retards par booking_id
 * @param {Array} props.drivers - Liste des drivers disponibles
 * @param {Function} props.timingStatus - Fonction pour calculer le statut de timing
 * @param {Object} props.style - Style inline pour la virtualisation (react-window)
 *
 * @returns {JSX.Element} Ligne du tableau
 */
const DispatchTableRow = ({ booking: b, delays, drivers, timingStatus, style }) => {
  const hasAssignment = !!b.assignment;
  const assignedDriver = hasAssignment
    ? drivers.find((d) => d.id === b.assignment.driver_id) || {}
    : {};

  // Résolution robuste du nom chauffeur
  let driverName = 'Non assigné';
  if (typeof b?.driver === 'string' && b.driver.trim()) {
    driverName = b.driver.trim();
  } else if (b?.driver?.full_name) {
    driverName = b.driver.full_name;
  } else if (b?.driver?.first_name || b?.driver?.last_name) {
    driverName = `${b.driver.first_name || ''} ${b.driver.last_name || ''}`.trim();
  } else if (b?.driver_username) {
    driverName = b.driver_username;
  } else if (b?.driver?.username) {
    driverName = b.driver.username;
  } else if (b?.driver_name) {
    driverName = b.driver_name;
  } else if (b?.driver_id) {
    const byId = drivers.find((d) => d.id === b.driver_id);
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

  // Si la course est terminée mais aucun nom détecté, afficher "Inconnu"
  if ((b.status || '').toLowerCase() === 'completed' && driverName === 'Non assigné') {
    driverName = 'Inconnu';
  }

  const t = timingStatus(b);

  return (
    <tr style={style} key={b.id}>
      <td>{b.id}</td>
      <td>{b.customer_name || b.client?.full_name || '—'}</td>
      <td>{renderBookingDateTime(b)}</td>
      <td>{b.pickup_location || '—'}</td>
      <td>{b.dropoff_location || '—'}</td>
      <td>{driverName}</td>
      <td>
        <Chip
          size="small"
          label={b.status || '—'}
          color={
            (b.status || '').toLowerCase() === 'completed'
              ? 'success'
              : (b.status || '').toLowerCase() === 'cancelled'
                ? 'error'
                : 'default'
          }
          variant="outlined"
        />
      </td>
      <td>
        {t.kind === 'on_time' && (
          <Chip size="small" label={t.label} className={styles.statusChipOnTime} />
        )}
        {t.kind === 'slightly_delayed' && (
          <Tooltip title="Retard faible, OK si < 10 min">
            <Chip size="small" label={t.label} className={styles.statusChipSlightDelay} />
          </Tooltip>
        )}
        {t.kind === 'delayed' && (() => {
          const delayInfo = delays[b.id];
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
      </td>
    </tr>
  );
};

DispatchTableRow.propTypes = {
  booking: PropTypes.object.isRequired,
  delays: PropTypes.object.isRequired,
  drivers: PropTypes.arrayOf(PropTypes.object).isRequired,
  timingStatus: PropTypes.func.isRequired,
  style: PropTypes.object,
};

export default DispatchTableRow;

