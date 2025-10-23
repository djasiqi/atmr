// src/pages/company/components/CompanyDriverTable.jsx
import React from 'react';
import { FiEdit, FiTrash2 } from 'react-icons/fi';
import styles from './CompanyDriverTable.module.css';

const CompanyDriverTable = ({ drivers, onEdit, onDeleteRequest }) => {
  return (
    <div className={styles.tableContainer}>
      <table className={styles.table}>
        <thead>
          <tr>
            <th>Nom Complet</th>
            <th>Véhicule</th>
            <th>Statut du Compte</th>
            <th style={{ textAlign: 'right' }}>Actions</th>
          </tr>
        </thead>
        <tbody>
          {(drivers || []).map((driver) => (
            <tr key={driver.id}>
              <td>
                <div className={styles.userCell}>
                  <img
                    src={driver.photo || '/default-avatar.png'}
                    alt={driver.username}
                    className={styles.avatar}
                  />
                  {/* On ne garde que la div pour le nom complet */}
                  <div className={styles.fullName}>
                    {driver.first_name} {driver.last_name}
                  </div>
                </div>
              </td>
              <td>
                <div className={styles.vehicleCell}>
                  <div className={styles.vehicleModel}>{driver.vehicle_assigned || 'N/A'}</div>
                  <div className={styles.vehicleBrand}>{driver.brand || 'N/A'}</div>
                </div>
              </td>
              <td>
                <span
                  className={`${styles.statusBadge} ${
                    driver.is_active ? styles.active : styles.inactive
                  }`}
                >
                  {driver.is_active ? 'Actif' : 'Inactif'}
                </span>
              </td>
              <td className={styles.actionsCell}>
                <button
                  onClick={() => onEdit(driver)}
                  title="Modifier les détails"
                  className={styles.actionButton}
                >
                  <FiEdit />
                </button>
                <button
                  onClick={() => onDeleteRequest(driver)}
                  title="Supprimer le chauffeur"
                  className={`${styles.actionButton} ${styles.deleteButton}`}
                >
                  <FiTrash2 />
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default CompanyDriverTable;
