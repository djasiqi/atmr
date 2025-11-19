import React from 'react';
// L'import est déjà présent, c'est parfait
import styles from './DriverWorkingHoursTable.module.css';

function formatTime(minutes) {
  const h = Math.floor(minutes / 60);
  const m = minutes % 60;
  return `${h}h${m.toString().padStart(2, '0')}`;
}

export default function DriverWorkingHoursTable({ driverHoursData = [] }) {
  return (
    // 1. On ajoute le conteneur principal pour l'ombre et les bordures
    <div className={styles.tableContainer}>
      {/* 2. On applique la classe de base à la table */}
      <table className={styles.table}>
        <thead>
          <tr>
            <th>Chauffeur</th>
            <th style={{ textAlign: 'center' }}>Nombre de courses</th>
            <th style={{ textAlign: 'center' }}>Heures travaillées</th>
          </tr>
        </thead>
        <tbody>
          {driverHoursData.length === 0 ? (
            <tr>
              {/* 3. On remplace le style en ligne par la classe CSS */}
              <td colSpan={3} className={styles.noDataCell}>
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
                </tr>
              );
            })
          )}
        </tbody>
      </table>
    </div>
  );
}
