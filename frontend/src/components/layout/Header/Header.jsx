import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import styles from './Header.module.css';

const SIGNUP_DISABLED =
  typeof process.env.REACT_APP_SIGNUP_DISABLED === 'string'
    ? process.env.REACT_APP_SIGNUP_DISABLED === 'true' ||
      process.env.REACT_APP_SIGNUP_DISABLED === '1'
    : true;

const Header = ({ hideAuthEntry = false }) => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  const toggleMenu = () => {
    setIsMenuOpen(!isMenuOpen);
  };

  return (
    <header className={styles.header}>
      {/* Logo */}
      <Link to="/" className={styles.logo}>
        <img src="/logo-lirie.png" alt="Lirie" className={styles.logoImg} width="120" height="30" />
      </Link>

      {/* Navigation menu */}
      <nav className={styles.nav}>
        <ul
          className={`${styles.navList} ${isMenuOpen ? styles.navListOpen : styles.navListClosed}`}
        >
          <li>
            <Link to="/deplacez-vous" className={styles.navLink}>
              Déplacez-vous
            </Link>
          </li>
          <li>
            <Link to="/conduire" className={styles.navLink}>
              Conduire
            </Link>
          </li>
          <li>
            <Link to="/professionnel" className={styles.navLink}>
              Professionnel
            </Link>
          </li>
          <li>
            <Link to="/a-propos" className={styles.navLink}>
              Pourquoi Lirie
            </Link>
          </li>
          <li>
            <Link to="/aide" className={styles.navLink}>
              Aide
            </Link>
          </li>
          <li>
            <Link to="/contact" className={styles.navLink}>
              Contact
            </Link>
          </li>
        </ul>
      </nav>

      {/* Auth : une seule entrée — connexion et inscription sur /login */}
      <div className={styles.authAndMenu}>
        {!hideAuthEntry && (
          <Link
            to="/login"
            className={styles.authEntry}
            title={
              SIGNUP_DISABLED
                ? 'Inscriptions suspendues – contactez info@lirie.ch (connexion toujours possible)'
                : 'Connexion et création de compte sur la même page'
            }
          >
            {SIGNUP_DISABLED ? 'Connexion' : 'Connexion ou inscription'}
          </Link>
        )}
        <button
          className={styles.hamburgerButton}
          onClick={toggleMenu}
          aria-label="Toggle Navigation"
        >
          ☰
        </button>
      </div>
    </header>
  );
};

export default Header;
