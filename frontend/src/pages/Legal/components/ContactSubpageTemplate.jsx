import React from 'react';
import { Link } from 'react-router-dom';
import ContactFormBase from './ContactFormBase';
import styles from '../ContactSubpages.module.css';

const ContactSubpageTemplate = ({ category, config }) => (
  <article className={styles.page}>
    <div className={styles.inner}>
      <header className={styles.header}>
        <span className={styles.sectionTag}>Contact</span>
        <h1>{config.introTitle}</h1>
        <p>{config.introText}</p>
      </header>

      <ContactFormBase category={category} config={config} />

      <Link className={styles.backLink} to="/contact">
        &larr; Revenir aux categories
      </Link>
    </div>
  </article>
);

export default ContactSubpageTemplate;
