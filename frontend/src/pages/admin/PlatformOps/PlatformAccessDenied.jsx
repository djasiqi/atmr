import React from 'react';
import { Link, useParams } from 'react-router-dom';
import styles from './AdminPlatformOps.module.css';

export default function PlatformAccessDenied() {
  const { public_id: adminId } = useParams();
  const base = `/dashboard/admin/${adminId}/platform-ops`;

  return (
    <div className={styles.tabPanel}>
      <div className={styles.errors} role="alert">
        <strong>Accès refusé</strong> — Vous n’avez pas les droits plateforme nécessaires pour ce
        segment. Contactez un administrateur ou utilisez un compte avec les bundles requis.
      </div>
      <p className={styles.tabPanelHint}>
        <Link to={`${base}/overview`}>Retour à la vue globale plateforme</Link>
        {' · '}
        <Link to={`/dashboard/admin/${adminId}`}>Tableau de bord admin</Link>
      </p>
    </div>
  );
}
