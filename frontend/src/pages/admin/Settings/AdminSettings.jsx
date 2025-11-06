import React from 'react';
import HeaderDashboard from '../../../components/layout/Header/HeaderDashboard';
import Sidebar from '../../../components/layout/Sidebar/AdminSidebar/AdminSidebar';
import styles from './AdminSettings.module.css';

const AdminSettings = () => {
  return (
    <div className={styles.container}>
      <HeaderDashboard />
      <div className={styles.body}>
        <Sidebar />
        <main className={styles.content}>
          <section className={styles.hero}>
            <h1>Paramètres administrateur</h1>
            <p>
              Gestion centralisée des préférences plateformes, des notifications et des options
              d’administration.
            </p>
          </section>
          <section className={styles.placeholder}>
            <h2>🚧 Module en cours de conception</h2>
            <p>
              Cette vue sera remplie lors de la tâche « Tests E2E versioning avancés » afin de
              couvrir la configuration du rôle administrateur.
            </p>
            <p>
              Les paramètres critiques restent disponibles via les fichiers de configuration et les
              routes backend existantes.
            </p>
          </section>
        </main>
      </div>
    </div>
  );
};

export default AdminSettings;
