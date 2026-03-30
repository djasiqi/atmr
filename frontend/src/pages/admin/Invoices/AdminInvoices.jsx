import React from 'react';
import styles from './AdminInvoices.module.css';

const AdminInvoices = () => {
  return (
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
  );
};

export default AdminInvoices;
