import React from 'react';
import HeaderDashboard from '../../../components/layout/Header/HeaderDashboard';
import AdminSidebar from '../../../components/layout/Sidebar/AdminSidebar/AdminSidebar';
import styles from './AdminInvoices.module.css';

const AdminInvoices = () => {
  return (
    <div className={styles.container}>
      <HeaderDashboard />
      <div className={styles.body}>
        <AdminSidebar />
        <main className={styles.content}>
          <section className={styles.hero}>
            <h1>Facturation</h1>
            <p>
              Cette section affichera prochainement les factures globales et les indicateurs clés
              pour les administrateurs.
            </p>
          </section>
          <section className={styles.placeholder}>
            <h2>🎯 Fonctionnalité en préparation</h2>
            <p>
              Les APIs `/api/v1/invoices/*` sont dédiées aux entreprises. Pour le rôle
              administrateur, la vue consolidated sera ajoutée lors de la tâche « Tests E2E
              versioning avancés ».
            </p>
            <p>
              En attendant, tu peux suivre les dernières factures via le tableau de bord principal
              ou les routes entreprises.
            </p>
          </section>
        </main>
      </div>
    </div>
  );
};

export default AdminInvoices;
