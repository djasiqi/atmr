// C:\Users\jasiq\atmr\frontend\src\pages\driver\components\Dashboard\DriverTable.jsx
import React from "react";
import styles from '../../Dashboard/DriverDashboard.module.css';
import { FiRepeat, FiUserX, FiUserCheck } from "react-icons/fi";

// La prop "onToggleAvailability" a été retirée
const DriverTable = ({ driver, loading, onToggle, onToggleType }) => {
  if (loading) return <p>Chargement des chauffeurs...</p>;
  if (!driver || driver.length === 0) return <p>Aucun chauffeur pour le moment.</p>;

  return (
    <table className={styles.table}>
      <thead>
        <tr>
          <th>Nom d'utilisateur</th>
          <th>Type</th>
          <th>Disponibilité</th>
          <th>Statut Compte</th>
          <th>Actions</th>
        </tr>
      </thead>
      <tbody>
        {driver.map((drv) => (
          <tr key={drv.id}>
            <td>{drv.username}</td>
            <td>{drv.driver_type === 'EMERGENCY' ? 'Urgence' : 'Régulier'}</td>
            <td>{drv.is_available ? "Disponible" : "Indisponible"}</td>
            <td>{drv.is_active ? "Actif" : "Inactif"}</td>
            <td>
              {/* Bouton pour changer le TYPE */}
              <button 
                onClick={() => onToggleType(drv.id)} 
                title="Changer le type (Régulier/Urgence)" 
                className={styles.actionButton}
              >
                <FiRepeat />
              </button>

              {/* Le bouton pour changer la disponibilité a été supprimé */}

              {/* Bouton pour changer le STATUT DU COMPTE */}
              <button 
                onClick={() => onToggle(drv.id, drv.is_active)} 
                title={drv.is_active ? "Désactiver le compte" : "Activer le compte"} 
                className={styles.actionButton}
              >
                {drv.is_active ? <FiUserX /> : <FiUserCheck />}
              </button>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
};

export default DriverTable;