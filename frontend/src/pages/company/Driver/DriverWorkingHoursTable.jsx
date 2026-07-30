import React from 'react';
// L'import est déjà présent, c'est parfait
import styles from './DriverWorkingHoursTable.module.css';

function formatTime(minutes) {
  const h = Math.floor(minutes / 60);
  const m = minutes % 60;
  return `${h}h${m.toString().padStart(2, '0')}`;
}

export default function DriverWorkingHoursTable({ driverHoursData = [], onViewDetails }) {
  return (
    // 1. On ajoute le conteneur principal pour l'ombre et les bordures
    <div className={styles.tableContainer}>
      {/* 2. On applique la classe de base à la table */}
      <table className={styles.table}>
        <thead>
          <tr>
            <th>Chauffeur</th>
            <th className={styles.numericHeader}>Nombre de courses</th>
            <th className={styles.numericHeader}>Heures travaillees</th>
            {onViewDetails && <th className={styles.numericHeader}>Détail</th>}
          </tr>
        </thead>
        <tbody>
          {driverHoursData.length === 0 ? (
            <tr>
              {/* 3. On remplace le style en ligne par la classe CSS */}
              <td colSpan={onViewDetails ? 4 : 3} className={styles.noDataCell}>
                Aucune donnée d'heure disponible
              </td>
            </tr>
          ) : (
            driverHoursData.map((driver) => {
              const count = driver.count || 0;
              const totalMinutes = driver.totalMinutes || 0;
              return (
                <tr key={driver.driverId}>
                  {/* 4. On applique les classes de style aux cellules */}
                  <td className={styles.driverName}>{driver.driverName}</td>
                  <td className={styles.numericCell}>{count}</td>
                  <td className={styles.numericCell}>{formatTime(totalMinutes)}</td>
                  {onViewDetails && (
                    <td className={styles.numericCell}>
                      <button
                        type="button"
                        className={styles.detailsBtn}
                        onClick={() => onViewDetails(driver.driverId, driver.driverName)}
                        disabled={count === 0}
                      >
                        Voir
                      </button>
                    </td>
                  )}
                </tr>
              );
            })
          )}
        </tbody>
      </table>
    </div>
  );
}
