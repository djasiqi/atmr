import React, { useState } from "react";
import toast from "react-hot-toast";
import styles from "./Footer.module.css";

const COMING_SOON_TOAST_ID = "footer-coming-soon";

const showComingSoonToast = () => {
  toast.dismiss(COMING_SOON_TOAST_ID);
  toast.custom(
    (t) => (
      <div
        className={styles.toastCard}
        role="status"
        aria-live="polite"
        data-visible={t.visible}
      >
        <div className={styles.toastIcon} aria-hidden="true">
          🚧
        </div>
        <div className={styles.toastBody}>
          <strong className={styles.toastTitle}>Bientôt disponible</strong>
          <p className={styles.toastMessage}>
            Notre équipe finalise cette section. Écrivez-nous à{" "}
            <a href="mailto:info@lirie.ch" className={styles.toastLink}>
              info@lirie.ch
            </a>{" "}
            pour être informé du lancement.
          </p>
        </div>
        <button
          type="button"
          className={styles.toastClose}
          onClick={() => toast.dismiss(COMING_SOON_TOAST_ID)}
          aria-label="Fermer"
        >
          ×
        </button>
      </div>
    ),
    {
      id: COMING_SOON_TOAST_ID,
      duration: 5000,
      position: "top-right",
      style: {
        padding: 0,
        background: "transparent",
        boxShadow: "none",
      },
    }
  );
};

const Footer = () => {
  const [language, setLanguage] = useState("Français (Suisse)");
  const [location] = useState("En cours de développement");

  // Fonction pour changer la langue
  const handleLanguageChange = (event) => {
    setLanguage(event.target.value);
  };

  const handleComingSoon = (event) => {
    event.preventDefault();
    showComingSoonToast();
  };

  return (
    <footer className={styles.footer}>
      <div className={styles.topSection}>
        <div className={styles.logo}>
          <h2>MonTransport</h2>
          <p>
            Plateforme en cours de développement. Besoin d&apos;aide ?
            <a
              href="mailto:info@lirie.ch"
              className={styles.contactLink}
              onClick={(event) => event.stopPropagation()}
            >
              {" "}
              info@lirie.ch
            </a>
          </p>
        </div>
        <div className={styles.links}>
          {/* Première colonne */}
          <div className={styles.column}>
            <h3>Entreprise</h3>
            <ul>
              <li>
                <button
                  type="button"
                  className={styles.linkButton}
                  onClick={handleComingSoon}
                >
                  À propos
                </button>
              </li>
              <li>
                <button
                  type="button"
                  className={styles.linkButton}
                  onClick={handleComingSoon}
                >
                  Nos services
                </button>
              </li>
              <li>
                <button
                  type="button"
                  className={styles.linkButton}
                  onClick={handleComingSoon}
                >
                  Espace presse
                </button>
              </li>
              <li>
                <button
                  type="button"
                  className={styles.linkButton}
                  onClick={handleComingSoon}
                >
                  Investisseurs
                </button>
              </li>
              <li>
                <button
                  type="button"
                  className={styles.linkButton}
                  onClick={handleComingSoon}
                >
                  Blog
                </button>
              </li>
              <li>
                <button
                  type="button"
                  className={styles.linkButton}
                  onClick={handleComingSoon}
                >
                  Offres d'emploi
                </button>
              </li>
            </ul>
          </div>
          {/* Deuxième colonne */}
          <div className={styles.column}>
            <h3>Produits</h3>
            <ul>
              <li>
                <button
                  type="button"
                  className={styles.linkButton}
                  onClick={handleComingSoon}
                >
                  Déplacez-vous avec nous
                </button>
              </li>
              <li>
                <button
                  type="button"
                  className={styles.linkButton}
                  onClick={handleComingSoon}
                >
                  Conduire
                </button>
              </li>
              <li>
                <button
                  type="button"
                  className={styles.linkButton}
                  onClick={handleComingSoon}
                >
                  Livrez
                </button>
              </li>
              <li>
                <button
                  type="button"
                  className={styles.linkButton}
                  onClick={handleComingSoon}
                >
                  Commandez un repas
                </button>
              </li>
              <li>
                <button
                  type="button"
                  className={styles.linkButton}
                  onClick={handleComingSoon}
                >
                  Transport pour les entreprises
                </button>
              </li>
              <li>
                <button
                  type="button"
                  className={styles.linkButton}
                  onClick={handleComingSoon}
                >
                  Cartes-cadeaux
                </button>
              </li>
            </ul>
          </div>
          {/* Troisième colonne */}
          <div className={styles.column}>
            <h3>Citoyens du monde</h3>
            <ul>
              <li>
                <button
                  type="button"
                  className={styles.linkButton}
                  onClick={handleComingSoon}
                >
                  Sécurité
                </button>
              </li>
              <li>
                <button
                  type="button"
                  className={styles.linkButton}
                  onClick={handleComingSoon}
                >
                  Diversité et intégration
                </button>
              </li>
              <li>
                <button
                  type="button"
                  className={styles.linkButton}
                  onClick={handleComingSoon}
                >
                  Développement durable
                </button>
              </li>
            </ul>
          </div>
          {/* Quatrième colonne */}
          <div className={styles.column}>
            <h3>Déplacements</h3>
            <ul>
              <li>
                <button
                  type="button"
                  className={styles.linkButton}
                  onClick={handleComingSoon}
                >
                  Réservez
                </button>
              </li>
              <li>
                <button
                  type="button"
                  className={styles.linkButton}
                  onClick={handleComingSoon}
                >
                  Aéroports
                </button>
              </li>
              <li>
                <button
                  type="button"
                  className={styles.linkButton}
                  onClick={handleComingSoon}
                >
                  Villes
                </button>
              </li>
            </ul>
          </div>
        </div>
      </div>

      {/* Section inférieure */}
      <div className={styles.bottomSection}>
        <div className={styles.social}>
          <span>© 2025 MonTransport Inc.</span>
          <ul>
            <li>
              <button
                type="button"
                className={styles.linkButton}
                onClick={handleComingSoon}
              >
                Facebook
              </button>
            </li>
            <li>
              <button
                type="button"
                className={styles.linkButton}
                onClick={handleComingSoon}
              >
                Twitter
              </button>
            </li>
            <li>
              <button
                type="button"
                className={styles.linkButton}
                onClick={handleComingSoon}
              >
                YouTube
              </button>
            </li>
            <li>
              <button
                type="button"
                className={styles.linkButton}
                onClick={handleComingSoon}
              >
                LinkedIn
              </button>
            </li>
            <li>
              <button
                type="button"
                className={styles.linkButton}
                onClick={handleComingSoon}
              >
                Instagram
              </button>
            </li>
          </ul>
        </div>
        <div className={styles.language}>
          <select value={language} onChange={handleLanguageChange} className={styles.languageSelector}>
            <option value="Français (Suisse)">Français (Suisse)</option>
            <option value="English (US)">English (US)</option>
          </select>
          | <span>{location}</span>
        </div>
        <div className={styles.legal}>
          <button
            type="button"
            className={styles.linkButton}
            onClick={handleComingSoon}
          >
            Confidentialité
          </button>{" "}
          |{" "}
          <button
            type="button"
            className={styles.linkButton}
            onClick={handleComingSoon}
          >
            Accessibilité
          </button>{" "}
          |{" "}
          <button
            type="button"
            className={styles.linkButton}
            onClick={handleComingSoon}
          >
            Conditions
          </button>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
