import React from 'react';
import { Link } from 'react-router-dom';
import styles from './Footer.module.css';

const Footer = () => {
  return (
    <footer className={styles.footer}>
      <div className={styles.inner}>
        {/* ── Bottom : copyright + legal ── */}
        <div className={styles.bottom}>
          <span className={styles.copyright}>© {new Date().getFullYear()} Lirie — Suisse</span>

          <div className={styles.legal}>
            <Link to="/privacy">Confidentialité</Link>
            <span className={styles.legalDot} />
            <Link to="/contact">Contact</Link>
            <span className={styles.legalDot} />
            <Link to="/conditions">Conditions</Link>
            <span className={styles.legalDot} />
            <Link to="/mentions-legales">Mentions légales</Link>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
