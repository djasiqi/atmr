import React from 'react';
import styles from '../../Dashboard/DriverDashboard.module.css';
import { FiRepeat, FiUserX, FiUserCheck, FiTruck, FiAlertTriangle, FiPhone } from 'react-icons/fi';
import {
  isDriverConstrained,
  getDriverConstraintReason,
} from '../../../../utils/companyDriverProjections';

const DriverTable = ({ driver, loading, onToggle, onToggleType }) => {
  if (loading) return <p className={styles.emptyText}>Chargement des chauffeurs...</p>;
  if (!driver || driver.length === 0) return <p className={styles.emptyText}>Aucun chauffeur pour le moment.</p>;

  const getDriverName = (drv) => {
    const first = drv.first_name || '';
    const last = drv.last_name || '';
    const full = `${first} ${last}`.trim();
    return full || drv.full_name || drv.username || '\u2014';
  };

  return (
    <table className={styles.table}>
      <thead>
        <tr>
          <th>Chauffeur</th>
          <th>Type</th>
          <th>Statut</th>
          <th>Actions</th>
        </tr>
      </thead>
      <tbody>
        {driver.map((drv) => {
          const name = getDriverName(drv);
          const isEmergency = drv.driver_type === 'EMERGENCY';
          const vehicleInfo = drv.vehicle_assigned || drv.vehicle?.model;

          return (
            <tr key={drv.id} className={!drv.is_active ? styles.rowInactive : undefined}>
              <td className={styles.driverNameCell}>
                <span className={styles.driverName}>{name}</span>
                {vehicleInfo && (
                  <span className={styles.driverVehicle}>
                    <FiTruck size={10} /> {vehicleInfo}
                  </span>
                )}
              </td>
              <td>
                <span className={isEmergency ? styles.badgeEmergency : styles.badgeRegular}>
                  {isEmergency ? 'Urgence' : 'Regulier'}
                </span>
              </td>
              <td>
                <div className={styles.statusCell}>
                  <span className={drv.is_available ? styles.badgeAvailable : styles.badgeUnavailable}>
                    {drv.is_available ? 'Disponible' : 'Indisponible'}
                  </span>
                  {isDriverConstrained(drv) && (
                    <span
                      className={styles.badgeConstrained}
                      title={`Batterie restreinte — l'app du chauffeur signale un problème d'optimisation batterie (raison : ${
                        getDriverConstraintReason(drv) || 'inconnue'
                      }). Position figée.`}
                      role="status"
                    >
                      <FiAlertTriangle size={11} aria-hidden />
                      <span>Batterie restreinte</span>
                    </span>
                  )}
                  {isDriverConstrained(drv) && drv.phone ? (
                    <a
                      href={`tel:${drv.phone}`}
                      className={styles.contactLink}
                      title={`Contacter le chauffeur (${drv.phone})`}
                    >
                      <FiPhone size={11} aria-hidden />
                      <span>Contacter le chauffeur</span>
                    </a>
                  ) : null}
                </div>
              </td>
              <td>
                <button
                  onClick={() => onToggleType(drv.id)}
                  title="Changer le type (Regulier/Urgence)"
                  className={styles.actionButton}
                >
                  <FiRepeat size={13} />
                </button>
                <button
                  onClick={() => onToggle(drv.id, drv.is_active)}
                  title={drv.is_active ? 'Desactiver le compte' : 'Activer le compte'}
                  className={styles.actionButton}
                >
                  {drv.is_active ? <FiUserX size={13} /> : <FiUserCheck size={13} />}
                </button>
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
};

export default DriverTable;
