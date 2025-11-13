import React from "react";
import styles from "./PrivacyPolicy.module.css";

const PrivacyPolicy = () => {
  const updatedAt = "13 novembre 2025";

  return (
    <article className={styles.container}>
      <header className={styles.section}>
        <h1>Politique de confidentialité</h1>
        <p className={styles.subtitle}>
          Dernière mise à jour : {updatedAt}
        </p>
        <div className={styles.card}>
          <p>
            La présente politique explique comment Lirie (ci-après « Lirie »,
            « nous ») collecte, utilise et protège les données personnelles
            dans le cadre de l’application mobile et web « Lirie Opérations ».
          </p>
        </div>
      </header>

      <section className={styles.section}>
        <h2>1. Responsable du traitement</h2>
        <p>
          Lirie – Avenue Ernest-Pictet 9, 1203 Genève, Suisse. <br />
          Contact :{" "}
          <a href="mailto:info@lirie.ch" className={styles.contactLink}>
            info@lirie.ch
          </a>
        </p>
      </section>

      <section className={styles.section}>
        <h2>2. Données collectées</h2>
        <p>
          Nous collectons uniquement les données nécessaires à la fourniture de
          nos services :
        </p>
        <ul>
          <li>
            <strong>Données de compte</strong> : nom, prénom, adresse e-mail,
            numéro de téléphone, rôle (chauffeur, entreprise, administrateur).
          </li>
          <li>
            <strong>Données professionnelles</strong> : informations sur les
            courses, disponibilité, documents d’exploitation.
          </li>
          <li>
            <strong>Données de localisation</strong> : coordonnées GPS en temps
            réel (chauffeurs) afin d’optimiser l’assignation et le suivi des
            missions.
          </li>
          <li>
            <strong>Données techniques</strong> : identifiants de session,
            journaux d’erreurs (Sentry), informations de performance.
          </li>
        </ul>
      </section>

      <section className={styles.section}>
        <h2>3. Finalités</h2>
        <p>Les données sont utilisées pour :</p>
        <ul>
          <li>Fournir l’accès sécurisé à l’application et aux tableaux de bord.</li>
          <li>Planifier, optimiser et suivre les missions de transport.</li>
          <li>
            Communication entre chauffeurs, dispatchers, entreprises et support.
          </li>
          <li>Suivre la qualité de service, mesurer les performances.</li>
          <li>Assurer la maintenance, la sécurité et la détection des fraudes.</li>
        </ul>
      </section>

      <section className={styles.section}>
        <h2>4. Base légale</h2>
        <p>
          Le traitement repose sur l’exécution du contrat (gestion des missions)
          et sur l’intérêt légitime de Lirie à assurer la sécurité et la qualité
          du service. Dans certains cas, votre consentement explicite peut être
          requis (notifications push, collecte de localisation en arrière-plan).
        </p>
      </section>

      <section className={styles.section}>
        <h2>5. Destinataires</h2>
        <p>
          Les données sont accessibles uniquement aux personnes autorisées
          (personnel Lirie, entreprises clientes, chauffeurs assignés). Nous
          pouvons recourir à des prestataires certifiés (hébergement, analytics,
          notifications, support). Aucun transfert commercial de données ne
          s’effectue sans votre accord.
        </p>
      </section>

      <section className={styles.section}>
        <h2>6. Conservation</h2>
        <p>
          Les données sont conservées pendant la durée du contrat et archivées
          au maximum 3 ans après sa fin, sauf obligation légale contraire.
          Les journaux techniques sont conservés 12 mois.
        </p>
      </section>

      <section className={styles.section}>
        <h2>7. Vos droits</h2>
        <p>
          Conformément à la LPD (Suisse) et au RGPD (UE), vous disposez des droits
          d’accès, de rectification, d’effacement, d’opposition, de limitation et
          de portabilité. Pour exercer vos droits, écrivez à{" "}
          <a href="mailto:info@lirie.ch" className={styles.contactLink}>
            info@lirie.ch
          </a>{" "}
          ou{" "}
          <a href="mailto:info@lirie.ch" className={styles.contactLink}>
            info@lirie.ch
          </a>
          .
        </p>
      </section>

      <section className={styles.section}>
        <h2>8. Sécurité</h2>
        <p>
          Lirie met en œuvre des mesures techniques et organisationnelles
          appropriées : chiffrement TLS, accès restreint, sauvegardes, audit des
          journaux. Nous demandons à nos prestataires de respecter les mêmes
          standards.
        </p>
      </section>

      <section className={styles.section}>
        <h2>9. Localisation et droits mobiles</h2>
        <p>
          L’application requiert l’accès à la localisation GPS pour les chauffeurs
          afin de suivre les missions. Vous pouvez retirer cet accès via les
          paramètres de votre appareil ; certaines fonctionnalités pourraient
          être limitées.
        </p>
      </section>

      <section className={styles.section}>
        <h2>10. Notifications</h2>
        <p>
          Les notifications push (missions, chat, alertes) sont envoyées via des
          services tiers (Firebase). Vous pouvez désactiver ces notifications
          depuis l’application ou dans les réglages système.
        </p>
      </section>

      <section className={styles.section}>
        <h2>11. Modifications</h2>
        <p>
          Nous pouvons mettre à jour la présente politique pour refléter les
          évolutions réglementaires ou fonctionnelles. Toute modification
          substantielle vous sera notifiée via l’application ou par e-mail.
        </p>
      </section>

      <section id="data-deletion" className={styles.section}>
        <h2>12. Suppression des données</h2>
        <p>
          Vous pouvez demander la suppression de vos données à tout moment en écrivant à
          <a href="mailto:info@lirie.ch" className={styles.contactLink}>
            info@lirie.ch
          </a>
          . Nous supprimons ou anonymisons les données dans un délai maximum de 30 jours, sauf
          obligation légale contraire. Certaines données opérationnelles (facturation, logs de sécurité)
          peuvent être conservées pendant la durée requise par la loi.
        </p>
      </section>

      <p className={styles.legalNotice}>
        Pour toute question relative à cette politique, contactez{" "}
        <a href="mailto:info@lirie.ch" className={styles.contactLink}>
          info@lirie.ch
        </a>
        .
      </p>
    </article>
  );
};

export default PrivacyPolicy;

