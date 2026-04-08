import React from 'react';
import { NavLink, Outlet, useParams } from 'react-router-dom';
import shell from '../adminShell.module.css';
import styles from './AdminBillingHub.module.css';

/**
 * Hub unique « Facturation LIRIE » : pilotage analytique, relevés plateforme, paramètres transporteurs.
 * Les moteurs backend restent séparés ; seule l’entrée UX est unifiée.
 */
const AdminBillingHub = () => {
  const { public_id: adminId } = useParams();
  const base = `/dashboard/admin/${adminId}/billing`;

  return (
    <main className={shell.content}>
      <header className={styles.hubHeader}>
        <h1 className={styles.hubTitle}>Facturation LIRIE</h1>
        <p className={styles.hubSubtitle}>
          Pilotage du périmètre facturable, relevés contractuels plateforme (abonnement, commission
          institution, support) et paramètres par transporteur. Les exports CSV restent distincts par
          onglet.
        </p>
        <nav className={styles.tabs} aria-label="Sections facturation">
          <NavLink
            to={`${base}/pilotage`}
            className={({ isActive }) => `${styles.tab} ${isActive ? styles.tabActive : ''}`}
          >
            Pilotage facturable
          </NavLink>
          <NavLink
            to={`${base}/releves`}
            className={({ isActive }) => `${styles.tab} ${isActive ? styles.tabActive : ''}`}
          >
            Relevés plateforme
          </NavLink>
          <NavLink
            to={`${base}/config`}
            className={({ isActive }) => `${styles.tab} ${isActive ? styles.tabActive : ''}`}
          >
            Paramètres transporteurs
          </NavLink>
        </nav>
      </header>
      <div className={styles.outlet}>
        <Outlet />
      </div>
    </main>
  );
};

export default AdminBillingHub;
