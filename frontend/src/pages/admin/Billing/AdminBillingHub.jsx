import React from 'react';
import { Outlet, useLocation } from 'react-router-dom';
import shell from '../adminShell.module.css';
import styles from './AdminBillingHub.module.css';

/**
 * Hub Finance — titre + contenu. La sous-nav est fournie par AdminWorkspaceNav.
 */
const AdminBillingHub = () => {
  const location = useLocation();
  const subtitle = location.pathname.includes('/finance/factures')
    ? 'Factures légales émises aux transporteurs — échéances, paiements et avoirs.'
    : 'Relevés mensuels aux transporteurs — abonnement, commission et support.';

  return (
    <main className={shell.content}>
      <header className={styles.hubHeader}>
        <h1 className={styles.hubTitle}>Facturation LIRIE</h1>
        <p className={styles.hubSubtitle}>{subtitle}</p>
      </header>
      <div className={styles.outlet}>
        <Outlet />
      </div>
    </main>
  );
};

export default AdminBillingHub;
