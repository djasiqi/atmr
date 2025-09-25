import React, { useState, useEffect } from "react";
import styles from "./Footer.module.css";

const Footer = () => {
  const [language, setLanguage] = useState("Français (Suisse)");
  const [location, setLocation] = useState("Chargement...");

  // Fonction pour changer la langue
  const handleLanguageChange = (event) => {
    setLanguage(event.target.value);
  };

  // Fonction pour obtenir la localisation de l'utilisateur
  const fetchLocation = () => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        async (position) => {
          const { latitude, longitude } = position.coords;
          try {
            // Appel API pour convertir latitude/longitude en ville
            const response = await fetch(
              `https://geocode.xyz/${latitude},${longitude}?geoit=json`
            );
            const data = await response.json();
            setLocation(data.city || "Localisation indisponible");
          } catch (error) {
            setLocation("Erreur de localisation");
          }
        },
        () => {
          setLocation("Localisation refusée");
        }
      );
    } else {
      setLocation("Géolocalisation non supportée");
    }
  };

  // Obtenir la localisation au chargement
  useEffect(() => {
    fetchLocation();
  }, []);

  return (
    <footer className={styles.footer}>
      <div className={styles.topSection}>
        <div className={styles.logo}>
          <h2>MonTransport</h2>
          <p>Accédez au centre d'aide</p>
        </div>
        <div className={styles.links}>
          {/* Première colonne */}
          <div className={styles.column}>
            <h3>Entreprise</h3>
            <ul>
              <li><a href="/about">À propos</a></li>
              <li><a href="/services">Nos services</a></li>
              <li><a href="/press">Espace presse</a></li>
              <li><a href="/investors">Investisseurs</a></li>
              <li><a href="/blog">Blog</a></li>
              <li><a href="/careers">Offres d'emploi</a></li>
            </ul>
          </div>
          {/* Deuxième colonne */}
          <div className={styles.column}>
            <h3>Produits</h3>
            <ul>
              <li><a href="/ride">Déplacez-vous avec nous</a></li>
              <li><a href="/drive">Conduire</a></li>
              <li><a href="/deliver">Livrez</a></li>
              <li><a href="/meals">Commandez un repas</a></li>
              <li><a href="/business">Transport pour les entreprises</a></li>
              <li><a href="/gifts">Cartes-cadeaux</a></li>
            </ul>
          </div>
          {/* Troisième colonne */}
          <div className={styles.column}>
            <h3>Citoyens du monde</h3>
            <ul>
              <li><a href="/safety">Sécurité</a></li>
              <li><a href="/diversity">Diversité et intégration</a></li>
              <li><a href="/sustainability">Développement durable</a></li>
            </ul>
          </div>
          {/* Quatrième colonne */}
          <div className={styles.column}>
            <h3>Déplacements</h3>
            <ul>
              <li><a href="/book">Réservez</a></li>
              <li><a href="/airports">Aéroports</a></li>
              <li><a href="/cities">Villes</a></li>
            </ul>
          </div>
        </div>
      </div>

      {/* Section inférieure */}
      <div className={styles.bottomSection}>
        <div className={styles.social}>
          <span>© 2025 MonTransport Inc.</span>
          <ul>
            <li><a href="https://facebook.com">Facebook</a></li>
            <li><a href="https://twitter.com">Twitter</a></li>
            <li><a href="https://youtube.com">YouTube</a></li>
            <li><a href="https://linkedin.com">LinkedIn</a></li>
            <li><a href="https://instagram.com">Instagram</a></li>
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
          <a href="/privacy">Confidentialité</a> | <a href="/accessibility">Accessibilité</a> | <a href="/terms">Conditions</a>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
