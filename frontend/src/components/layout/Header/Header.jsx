import React, { useState } from "react";
import { Link } from "react-router-dom";
import styles from "./Header.module.css";


const Header = () => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  const toggleMenu = () => {
    setIsMenuOpen(!isMenuOpen);
  };

  return (
    <header className={styles.header}>
      {/* Logo */}
      <Link to="/" className={styles.logo}>
        MonTransport
      </Link>

      {/* Navigation menu */}
      <nav className={styles.nav}>
        <ul
          className={`${styles.navList} ${
            isMenuOpen ? styles.navListOpen : styles.navListClosed
          }`}
        >
          <li>
            <Link to="/services" className={styles.navLink}>
              Déplacez-vous
            </Link>
          </li>
          <li>
            <Link to="/drive" className={styles.navLink}>
              Conduire
            </Link>
          </li>
          <li>
            <Link to="/pro" className={styles.navLink}>
              Professionnel
            </Link>
          </li>
          <li>
            <Link to="/about" className={styles.navLink}>
              À propos
            </Link>
          </li>
          <li>
            <Link to="/help" className={styles.navLink}>
              Aide
            </Link>
          </li>
        </ul>
      </nav>

      {/* Auth actions + Hamburger */}
      <div className={styles.authAndMenu}>
        <Link to="/login" className={styles.login}>
          Connexion
        </Link>
        <Link to="/signup" className={styles.signUp}>
          S'inscrire
        </Link>
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
