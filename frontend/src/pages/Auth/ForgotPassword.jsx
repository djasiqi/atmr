import React from 'react';
import { Link } from 'react-router-dom';
import styles from './ForgotPassword.module.css';

const InfoIcon = () => (
  <svg className={styles.infoIcon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="10" />
    <line x1="12" y1="16" x2="12" y2="12" />
    <line x1="12" y1="8" x2="12.01" y2="8" />
  </svg>
);

const ArrowLeftIcon = () => (
  <svg className={styles.backIcon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <line x1="19" y1="12" x2="5" y2="12" />
    <polyline points="12 19 5 12 12 5" />
  </svg>
);

const ForgotPassword = () => {
  return (
    <div className={styles.pageWrapper}>
      <div className={styles.card}>
        <img src="/logo-lirie.png" alt="Lirie" className={styles.logo} />

        <h1 className={styles.title}>Mot de passe oublié ?</h1>
        <p className={styles.description}>
          La réinitialisation de mot de passe est gérée par votre administrateur pour des raisons de sécurité.
        </p>

        <div className={styles.infoBox}>
          <InfoIcon />
          <p className={styles.infoText}>
            <strong>Contactez l'administrateur de votre entreprise</strong> pour demander la réinitialisation de votre mot de passe. Il pourra générer un nouveau mot de passe temporaire depuis le panneau d'administration.
          </p>
        </div>

        <Link to="/login" className={styles.backButton}>
          <ArrowLeftIcon />
          Retour à la connexion
        </Link>

        <div className={styles.footer}>
          <p className={styles.footerText}>
            Lirie — Plateforme de transport sanitaire
          </p>
        </div>
      </div>
    </div>
  );
};

export default ForgotPassword;
