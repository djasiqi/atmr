import React from 'react';
import { Link } from 'react-router-dom';
import { showComingSoonToast } from '../../../utils/showComingSoonToast';
import styles from './Footer.module.css';

const Footer = () => {
  const cs = (e) => {
    e.preventDefault();
    showComingSoonToast();
  };

  return (
    <footer className={styles.footer}>
      <div className={styles.inner}>
        {/* ── Top : logo + colonnes ── */}
        <div className={styles.top}>
          <div className={styles.brand}>
            <img src="/logo-lirie.png" alt="Lirie" className={styles.logo} />
            <p className={styles.tagline}>
              Mobilité bienveillante en Suisse.
              <br />
              Transport médical & adapté.
            </p>
            <a href="mailto:info@lirie.ch" className={styles.email}>
              info@lirie.ch
            </a>
          </div>

          <nav className={styles.columns}>
            <div className={styles.column}>
              <h4 className={styles.columnTitle}>Plateforme</h4>
              <ul className={styles.columnList}>
                <li><button type="button" onClick={cs}>Réserver une course</button></li>
                <li><button type="button" onClick={cs}>Transport entreprise</button></li>
                <li><button type="button" onClick={cs}>Devenir chauffeur</button></li>
                <li><button type="button" onClick={cs}>Espace institution</button></li>
              </ul>
            </div>

            <div className={styles.column}>
              <h4 className={styles.columnTitle}>Entreprise</h4>
              <ul className={styles.columnList}>
                <li><button type="button" onClick={cs}>À propos</button></li>
                <li><button type="button" onClick={cs}>Nos valeurs</button></li>
                <li><button type="button" onClick={cs}>Offres d'emploi</button></li>
                <li><button type="button" onClick={cs}>Espace presse</button></li>
              </ul>
            </div>

            <div className={styles.column}>
              <h4 className={styles.columnTitle}>Ressources</h4>
              <ul className={styles.columnList}>
                <li><button type="button" onClick={cs}>Centre d'aide</button></li>
                <li><button type="button" onClick={cs}>Sécurité</button></li>
                <li><button type="button" onClick={cs}>Accessibilité</button></li>
                <li><button type="button" onClick={cs}>Développement durable</button></li>
              </ul>
            </div>
          </nav>
        </div>

        {/* ── Bottom : copyright + legal ── */}
        <div className={styles.bottom}>
          <span className={styles.copyright}>© {new Date().getFullYear()} Lirie — Suisse</span>

          <div className={styles.legal}>
            <Link to="/privacy">Confidentialité</Link>
            <span className={styles.legalDot} />
            <button type="button" onClick={cs}>Conditions</button>
            <span className={styles.legalDot} />
            <button type="button" onClick={cs}>Mentions légales</button>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
