import React from 'react';
import { NavLink, Outlet, useParams } from 'react-router-dom';
import shell from '../adminShell.module.css';
import styles from './AdminBillingHub.module.css';

/**
 * Hub « Facturation LIRIE » : synthèse, relevés, configuration transporteurs.
 */
const AdminBillingHub = () => {
  const { public_id: adminId } = useParams();
  const base = `/dashboard/admin/${adminId}/billing`;

  return (
    <main className={shell.content}>
      <header className={styles.hubHeader}>
        <h1 className={styles.hubTitle}>Facturation LIRIE</h1>
        <p className={styles.hubSubtitle}>
          Relevés mensuels aux transporteurs — abonnement, commission et support.
        </p>
        <nav className={styles.tabs} aria-label="Sections facturation">
          <NavLink
            to={base}
            end
            className={({ isActive }) => `${styles.tab} ${isActive ? styles.tabActive : ''}`}
          >
            Vue d&apos;ensemble
          </NavLink>
          <NavLink
            to={`${base}/releves`}
            className={({ isActive }) => `${styles.tab} ${isActive ? styles.tabActive : ''}`}
          >
            Relevés
          </NavLink>
          <NavLink
            to={`${base}/config`}
            className={({ isActive }) => `${styles.tab} ${isActive ? styles.tabActive : ''}`}
          >
            Entreprises
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
