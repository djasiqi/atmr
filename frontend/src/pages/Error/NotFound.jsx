import React from 'react';
import styles from './NotFound.module.css'; // Assurez-vous d'avoir un fichier CSS pour le style

const NotFound = () => {
  return (
    <div className={styles.notFoundContainer}>
      <h1>404 - Page introuvable</h1>
      <p>Désolé, la page que vous recherchez n'existe pas.</p>
    </div>
  );
};

export default NotFound;
