import React from 'react';
import { Link } from 'react-router-dom';
import { listCategories } from './contactCategories';
import styles from './Contact.module.css';

const Contact = () => {
  const categories = listCategories();

  return (
    <article className={styles.page}>
      <section className={styles.hero}>
        <div className={styles.heroInner}>
          <span className={styles.sectionTag}>Point d&apos;acces</span>
          <h1 className={styles.heroTitle}>Contact</h1>
          <p className={styles.heroSubtitle}>
            Point d&apos;acces pour toute demande relative a la plateforme LIRIE.
          </p>
          <p className={styles.heroSubnote}>Reponse generalement sous 24h ouvrees.</p>
        </div>
      </section>

      <section className={styles.listSection}>
        <div className={styles.listInner}>
          <span className={styles.sectionTag}>Orientation</span>
          <h2 className={styles.sectionTitle}>Selectionnez la nature de votre demande</h2>
          <nav aria-label="Categories de contact" className={styles.listCard}>
            <div className={styles.list}>
            {categories.map((item) => {
              return (
                <div key={item.key} className={styles.item}>
                  <Link className={styles.itemButton} to={item.route} aria-label={`${item.index} ${item.label}`}>
                    <span className={styles.itemMain}>
                      <span className={styles.itemIndex}>{item.index}</span>
                      <span className={styles.itemLabel}>{item.label}</span>
                    </span>
                    <span className={styles.itemChevron} aria-hidden="true">
                      →
                    </span>
                  </Link>
                  <div className={styles.itemContent}>
                    <p className={styles.itemDescription}>{item.description}</p>
                  </div>
                </div>
              );
            })}
            </div>
          </nav>
        </div>
      </section>

      <section className={styles.footerSection}>
        <div className={styles.footerInner}>
          <div className={styles.infoCard}>
            <span className={styles.miniTag}>Principes</span>
            <h3>Engagement</h3>
            <p>
              LIRIE traite des flux operationnels sensibles. Les demandes sont orientees selon des principes de
              neutralite, de tracabilite et de continuite de service.
            </p>
          </div>
          <div className={`${styles.infoCard} ${styles.infoCardSubtle}`}>
            <span className={styles.miniTag}>Processus</span>
            <h3>Traitement</h3>
            <ol>
              <li>Analyse de votre demande</li>
              <li>Orientation vers l&apos;equipe competente</li>
              <li>Reponse sous 24h ouvrees</li>
            </ol>
          </div>
          <div className={`${styles.infoCard} ${styles.infoCardSubtle}`}>
            <span className={styles.miniTag}>Contact direct</span>
            <h3>Coordonnees generales</h3>
            <p>
              <a href="mailto:info@lirie.ch">info@lirie.ch</a>
            </p>
            <p>Lun-Ven 08:00-18:00 (CET)</p>
            <p>Geneve, Suisse</p>
          </div>
        </div>
      </section>
    </article>
  );
};

export default Contact;
