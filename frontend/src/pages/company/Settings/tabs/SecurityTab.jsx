// frontend/src/pages/company/Settings/tabs/SecurityTab.jsx
import React from 'react';
import styles from '../CompanySettings.module.css';

const SecurityTab = () => {
  // Données factices pour l'instant
  const recentActivity = [
    {
      id: 1,
      timestamp: '14/10/2025 12:30',
      user: 'Admin (vous)',
      action: 'Modification des paramètres de facturation',
      ip: '192.168.1.1',
    },
    {
      id: 2,
      timestamp: '14/10/2025 10:15',
      user: 'Admin (vous)',
      action: 'Lancement du dispatch',
      ip: '192.168.1.1',
    },
    {
      id: 3,
      timestamp: '13/10/2025 16:45',
      user: 'Admin (vous)',
      action: 'Ajout d\'un nouveau client',
      ip: '192.168.1.1',
    },
  ];

  const handleExportLogs = () => {
    // TODO: Implémenter l'export des logs
    alert('Export des logs en cours de développement...');
  };

  return (
    <>
      {/* Informations de compte */}
      <section className={styles.section}>
        <h2>🔐 Sécurité du compte</h2>

        <div className={styles.infoBox}>
          <div className={styles.infoRow}>
            <span className={styles.infoLabel}>Dernière connexion</span>
            <span className={styles.infoValue}>14/10/2025 à 12:30</span>
          </div>
          <div className={styles.infoRow}>
            <span className={styles.infoLabel}>Adresse IP</span>
            <span className={styles.infoValue}>192.168.1.1</span>
          </div>
          <div className={styles.infoRow}>
            <span className={styles.infoLabel}>Sessions actives</span>
            <span className={styles.infoValue}>1</span>
          </div>
        </div>
      </section>

      {/* Logs d'activité */}
      <section className={styles.section}>
        <h2>📝 Activité récente</h2>
      </section>

      {/* Tableau séparé - comme les autres pages */}
      <div className={styles.tableContainer}>
        <table className={styles.activityTable}>
          <thead>
            <tr>
              <th>Date & Heure</th>
              <th>Utilisateur</th>
              <th>Action</th>
              <th>IP</th>
            </tr>
          </thead>
          <tbody>
            {recentActivity.map((log) => (
              <tr key={log.id}>
                <td>{log.timestamp}</td>
                <td>{log.user}</td>
                <td>{log.action}</td>
                <td className={styles.ipCell}>{log.ip}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Actions après le tableau */}
      <div className={styles.actionsRow} style={{ marginTop: 'var(--spacing-lg)' }}>
        <button
          type="button"
          className={`${styles.button} ${styles.secondary}`}
          onClick={handleExportLogs}
        >
          📥 Exporter tous les logs (CSV)
        </button>
      </div>

      {/* Informations système */}
      <section className={styles.section}>
        <h2>ℹ️ Informations système</h2>

        <div className={styles.infoBox}>
          <div className={styles.infoRow}>
            <span className={styles.infoLabel}>Version API</span>
            <span className={styles.infoValue}>1.0.0</span>
          </div>
          <div className={styles.infoRow}>
            <span className={styles.infoLabel}>Environnement</span>
            <span className={styles.infoValue}>
              {process.env.NODE_ENV === 'production' ? 'Production' : 'Développement'}
            </span>
          </div>
          <div className={styles.infoRow}>
            <span className={styles.infoLabel}>Base de données</span>
            <span className={styles.infoValue}>PostgreSQL 16</span>
          </div>
        </div>
      </section>
    </>
  );
};

export default SecurityTab;

