import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import styles from './PrivacyPolicy.module.css';

const UPDATED_AT = '13 avril 2026';

/* Petites icônes stroke — cohérentes avec le reste du site */
function IcoShield({ s = 20 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
    </svg>
  );
}
function IcoUsers({ s = 16 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
      <circle cx="9" cy="7" r="4" />
      <path d="M23 21v-2a4 4 0 0 0-3-3.87M16 3.13a4 4 0 0 1 0 7.75" />
    </svg>
  );
}
function IcoDatabase({ s = 16 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <ellipse cx="12" cy="5" rx="9" ry="3" />
      <path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3" />
      <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5" />
    </svg>
  );
}
function IcoHeart({ s = 16 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z" />
    </svg>
  );
}
function IcoFileText({ s = 16 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
      <polyline points="14 2 14 8 20 8" />
      <line x1="16" y1="13" x2="8" y2="13" />
      <line x1="16" y1="17" x2="8" y2="17" />
    </svg>
  );
}
function IcoLock({ s = 16 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <rect x="3" y="11" width="18" height="11" rx="2" />
      <path d="M7 11V7a5 5 0 0 1 10 0v4" />
    </svg>
  );
}
function IcoGlobe({ s = 16 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <circle cx="12" cy="12" r="10" />
      <line x1="2" y1="12" x2="22" y2="12" />
      <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
    </svg>
  );
}
function IcoMapPin({ s = 16 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z" />
      <circle cx="12" cy="10" r="3" />
    </svg>
  );
}
function IcoBell({ s = 16 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9" />
      <path d="M13.73 21a2 2 0 0 1-3.46 0" />
    </svg>
  );
}
function IcoCreditCard({ s = 16 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <rect x="1" y="4" width="22" height="16" rx="2" />
      <line x1="1" y1="10" x2="23" y2="10" />
    </svg>
  );
}
function IcoCookie({ s = 16 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M12 2a10 10 0 1 0 10 10 4 4 0 0 1-5-5 4 4 0 0 1-5-5" />
      <path d="M8.5 8.5v.01M16 15.5v.01M12 12v.01" />
    </svg>
  );
}
function IcoEye({ s = 16 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
      <circle cx="12" cy="12" r="3" />
    </svg>
  );
}
function IcoCheckCircle({ s = 14 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
      <polyline points="22 4 12 14.01 9 11.01" />
    </svg>
  );
}
function IcoTrash({ s = 16 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <polyline points="3 6 5 6 21 6" />
      <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
    </svg>
  );
}
function IcoRefresh({ s = 16 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <polyline points="23 4 23 10 17 10" />
      <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10" />
    </svg>
  );
}
function IcoMail({ s = 16 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z" />
      <polyline points="22,6 12,13 2,6" />
    </svg>
  );
}
function IcoChevR({ s = 12 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <polyline points="9 18 15 12 9 6" />
    </svg>
  );
}
function IcoExternal({ s = 12 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
      <polyline points="15 3 21 3 21 9" />
      <line x1="10" y1="14" x2="21" y2="3" />
    </svg>
  );
}

const SECTIONS = [
  { id: 'responsable', num: 1, label: 'Responsable du traitement', Icon: IcoShield },
  { id: 'champ-application', num: 2, label: "Champ d'application", Icon: IcoUsers },
  { id: 'donnees-collectees', num: 3, label: 'Données collectées', Icon: IcoDatabase },
  { id: 'donnees-sensibles', num: 4, label: 'Données sensibles', Icon: IcoHeart },
  { id: 'finalites', num: 5, label: 'Finalités du traitement', Icon: IcoFileText },
  { id: 'bases-legales', num: 6, label: 'Bases légales', Icon: IcoLock },
  { id: 'destinataires', num: 7, label: 'Destinataires', Icon: IcoUsers },
  { id: 'sous-traitants', num: 8, label: 'Sous-traitants techniques', Icon: IcoGlobe },
  { id: 'transferts-internationaux', num: 9, label: 'Transferts internationaux', Icon: IcoGlobe },
  { id: 'conservation', num: 10, label: 'Durée de conservation', Icon: IcoDatabase },
  { id: 'securite', num: 11, label: 'Sécurité', Icon: IcoLock },
  { id: 'localisation-gps', num: 12, label: 'Localisation GPS', Icon: IcoMapPin },
  { id: 'notifications', num: 13, label: 'Notifications push', Icon: IcoBell },
  { id: 'paiements', num: 14, label: 'Paiements en ligne', Icon: IcoCreditCard },
  { id: 'cookies', num: 15, label: 'Cookies (web)', Icon: IcoCookie },
  { id: 'droits', num: 16, label: 'Droits des personnes', Icon: IcoEye },
  { id: 'autorite-controle', num: 17, label: 'Autorité de contrôle', Icon: IcoCheckCircle },
  { id: 'data-deletion', num: 18, label: 'Suppression des données', Icon: IcoTrash },
  { id: 'modifications', num: 19, label: 'Modifications', Icon: IcoRefresh },
  { id: 'contact', num: 20, label: 'Contact', Icon: IcoMail },
];

const RIGHTS = [
  { title: 'Accès', desc: 'Obtenir une copie des données personnelles que nous détenons dans le cadre concerné.' },
  { title: 'Rectification', desc: 'Corriger des données inexactes ou incomplètes.' },
  { title: 'Effacement', desc: 'Demander la suppression dans les conditions légales (voir section 18).' },
  { title: 'Opposition', desc: "Vous opposer à certains traitements, notamment fondés sur l'intérêt légitime." },
  { title: 'Limitation', desc: 'Demander la limitation du traitement dans certaines circonstances.' },
  { title: 'Portabilité', desc: 'Recevoir vos données dans un format structuré et lisible par machine (RGPD, le cas échéant).' },
];

function SectionHead({ num, title, Icon }) {
  return (
    <div className={styles.sectionHead}>
      <div className={styles.sectionIcon}>
        <Icon s={16} />
      </div>
      <h2>
        <span className={styles.sectionNum}>{num}.</span>
        {title}
      </h2>
    </div>
  );
}

const PrivacyPolicy = () => {
  const [activeSection, setActiveSection] = useState(SECTIONS[0].id);

  useEffect(() => {
    const onScroll = () => {
      const y = window.scrollY + 110;
      let current = SECTIONS[0].id;
      for (const s of SECTIONS) {
        const el = document.getElementById(s.id);
        if (el && el.offsetTop <= y) current = s.id;
      }
      setActiveSection(current);
    };
    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  const scrollTo = (id) => {
    const el = document.getElementById(id);
    if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };

  return (
    <div className={styles.page}>
      <header className={styles.hero}>
        <div className={styles.heroInner}>
          <div className={styles.heroIcon} aria-hidden>
            <IcoShield s={26} />
          </div>
          <div className={styles.heroBody}>
            <h1>Politique de confidentialité</h1>
            <div className={styles.heroMeta}>
              <span className={styles.badge}>LPD · RGPD</span>
              <p className={styles.subtitle}>Dernière mise à jour : {UPDATED_AT}</p>
            </div>
            <p className={styles.lead}>
              Cette page explique comment <strong>Drin Jasiqi</strong>, exploitant du projet <strong>Lirie</strong> (« nous
              »), traite les données personnelles dans le cadre de la plateforme <strong>Lirie Opérations</strong>{' '}
              (coordination de transports, dont transports médicaux ou à vocation sanitaire). Elle complète les{' '}
              <Link to="/mentions-legales" className={styles.contactLink}>
                mentions légales
              </Link>{' '}
              et les{' '}
              <Link to="/conditions" className={styles.contactLink}>
                conditions générales d&apos;utilisation
              </Link>
              .
            </p>
          </div>
        </div>
      </header>

      <div className={styles.shell}>
        <div className={styles.layout}>
          <aside className={styles.toc} aria-label="Table des matières">
            <div className={styles.tocInner}>
              <p className={styles.tocTitle}>Sur cette page</p>
              <nav className={styles.tocNav} aria-label="Sections">
                {SECTIONS.map((s) => (
                  <button
                    key={s.id}
                    type="button"
                    className={`${styles.tocBtn}${activeSection === s.id ? ` ${styles.tocBtnActive}` : ''}`}
                    onClick={() => scrollTo(s.id)}
                    aria-current={activeSection === s.id ? 'location' : undefined}
                  >
                    <span className={styles.tocNum}>{s.num}</span>
                    <span className={styles.tocLabel}>{s.label}</span>
                    {activeSection === s.id ? (
                      <span className={styles.tocChev}>
                        <IcoChevR />
                      </span>
                    ) : null}
                  </button>
                ))}
              </nav>
              <div className={styles.tocContact}>
                <IcoMail s={14} />
                <a href="mailto:privacy@lirie.ch">privacy@lirie.ch</a>
              </div>
            </div>
          </aside>

          <article className={styles.article}>
            <section id="responsable" className={styles.section}>
              <SectionHead num={1} title="Responsable du traitement" Icon={IcoShield} />
              <div className={styles.infoCard}>
                <div className={styles.infoRow}>
                  <strong>Drin Jasiqi</strong>
                </div>
                <div className={styles.infoRow}>Exploitant du projet Lirie — plateforme Lirie Opérations</div>
                <div className={styles.infoRow}>
                  Avenue Ernest-Pictet 9, 1203 Genève — Suisse
                </div>
                <div className={styles.infoRow}>
                  <IcoMail s={14} />
                  <a href="mailto:info@lirie.ch" className={styles.contactLink}>
                    info@lirie.ch
                  </a>
                  <span className={styles.divider}>·</span>
                  <a href="mailto:privacy@lirie.ch" className={styles.contactLink}>
                    privacy@lirie.ch
                  </a>
                  <span className={styles.tag}>Protection des données</span>
                </div>
              </div>
              <p>
                Pour la forme juridique de l&apos;exploitation et l&apos;identification de l&apos;éditeur du site, voir
                nos{' '}
                <Link to="/mentions-legales" className={styles.contactLink}>
                  mentions légales
                </Link>
                .
              </p>
              <p>
                Pour toute question relative à la protection de vos données :{' '}
                <a href="mailto:privacy@lirie.ch" className={styles.contactLink}>
                  privacy@lirie.ch
                </a>
                . Les demandes d&apos;exercice de droits sont traitées dans les délais prévus par la loi (sections 16 et
                18).
              </p>
            </section>

            <section id="champ-application" className={styles.section}>
              <SectionHead num={2} title="Champ d'application" Icon={IcoUsers} />
              <p>
                La présente politique s&apos;applique aux traitements effectués via Lirie Opérations pour les catégories
                d&apos;utilisateurs suivantes, selon les modules et rôles activés :
              </p>
              <ul className={styles.listPlain}>
                <li>entreprises de transport partenaires et leurs équipes (dispatch, gestion) ;</li>
                <li>chauffeurs et conducteurs habilités ;</li>
                <li>
                  institutions ou organismes clientes (hôpitaux, EMS, services de coordination, etc.) et leurs
                  utilisateurs ;
                </li>
                <li>clients ou bénéficiaires utilisant un portail ou un parcours dédié (réservation, suivi) ;</li>
                <li>utilisateurs administratifs ou de support Lirie ;</li>
                <li>
                  tout visiteur du site web ou des applications mobiles dans la mesure où des données sont collectées
                  (cookies, journaux techniques).
                </li>
              </ul>
              <div className={`${styles.callout} ${styles.calloutInfo}`}>
                <strong>Rôles de traitement :</strong> selon les cas, Lirie agit comme{' '}
                <em>responsable du traitement</em>, <em>responsable conjoint</em> avec un partenaire, ou{' '}
                <em>sous-traitant</em> pour le compte d&apos;une institution ou d&apos;une entreprise cliente ; les rôles
                précis sont précisés dans les contrats applicables (section 7).
              </div>
            </section>

            <section id="donnees-collectees" className={styles.section}>
              <SectionHead num={3} title="Données collectées" Icon={IcoDatabase} />
              <p>
                Nous traitons uniquement les données <strong>proportionnées et nécessaires</strong> aux finalités décrites
                à la section 5, notamment :
              </p>
              <div className={styles.grid2}>
                <div className={styles.card}>
                  <div className={styles.cardTitle}>Identification et compte</div>
                  <p>
                    Nom, prénom, adresse e-mail, numéro de téléphone, fonction, rôle (chauffeur, entreprise, institution,
                    client, administrateur, etc.).
                  </p>
                </div>
                <div className={styles.card}>
                  <div className={styles.cardTitle}>Missions de transport</div>
                  <p>
                    Adresses et lieux de prise en charge et de destination, créneaux horaires, statuts de mission,
                    informations opérationnelles (véhicule, équipements, consignes d&apos;accès lorsqu&apos;elles sont
                    communiquées), échanges liés à la mission.
                  </p>
                </div>
                <div className={styles.card}>
                  <div className={styles.cardTitle}>Données professionnelles</div>
                  <p>Disponibilités, plannings, documents ou justificatifs transmis dans le cadre du service.</p>
                </div>
                <div className={styles.card}>
                  <div className={styles.cardTitle}>Localisation</div>
                  <p>
                    Coordonnées GPS ou équivalent pour les chauffeurs pendant les missions, lorsque la fonctionnalité est
                    activée (section 12).
                  </p>
                </div>
                <div className={styles.card} style={{ gridColumn: '1 / -1' }}>
                  <div className={styles.cardTitle}>Données techniques</div>
                  <p>
                    Identifiants de session, adresses IP, journaux d&apos;erreurs et de performance (par ex. via
                    Sentry), paramètres d&apos;appareil utiles au diagnostic.
                  </p>
                </div>
              </div>
            </section>

            <section id="donnees-sensibles" className={styles.section}>
              <SectionHead num={4} title="Données sensibles et contexte médical" Icon={IcoHeart} />
              <div className={`${styles.callout} ${styles.calloutBrand}`}>
                <strong>Contexte médical :</strong> certaines missions impliquent des informations liées à la{' '}
                <strong>mobilité</strong>, à l&apos;<strong>accompagnement</strong> ou, dans certains cas, à l&apos;
                <strong>état de santé</strong> ou aux besoins d&apos;accès (transport sanitaire coordonné, mobilité
                réduite, accompagnement spécifique).
              </div>
              <p>
                Ces données ne sont traitées que dans la mesure où elles sont <strong>nécessaires</strong> à
                l&apos;organisation et à la réalisation du transport, avec accès limité aux intervenants autorisés
                (personnel habilité côté Lirie, partenaire de transport, institution ou client selon le périmètre de la
                mission).
              </p>
              <p>
                Lirie ne sollicite pas davantage d&apos;informations sensibles que nécessaire et invite les utilisateurs à
                limiter les données communiquées aux seuls éléments utiles à la mission, conformément aux instructions des
                responsables du traitement côté institution ou entreprise lorsque celles-ci s&apos;appliquent.
              </p>
            </section>

            <section id="finalites" className={styles.section}>
              <SectionHead num={5} title="Finalités du traitement" Icon={IcoFileText} />
              <p>Les données sont utilisées notamment pour :</p>
              <ul className={`${styles.listPlain} ${styles.listCheck}`}>
                <li>fournir l&apos;accès sécurisé à la plateforme web et mobile et aux tableaux de bord ;</li>
                <li>coordonner les missions entre institutions, entreprises de transport et chauffeurs ;</li>
                <li>planifier, assigner, optimiser et suivre les missions en temps réel ;</li>
                <li>
                  assurer la traçabilité des interventions et la qualité de service (y compris preuves opérationnelles
                  lorsque la loi ou le contrat l&apos;exige) ;
                </li>
                <li>permettre la communication opérationnelle entre acteurs (missions, alertes, support) ;</li>
                <li>gérer la facturation ou les flux de paiement lorsque ces modules sont utilisés ;</li>
                <li>assurer la maintenance, la sécurité, la supervision technique et la prévention des abus ou fraudes.</li>
              </ul>
            </section>

            <section id="bases-legales" className={styles.section}>
              <SectionHead num={6} title="Bases légales" Icon={IcoLock} />
              <p>
                Selon le type de traitement et le rôle des personnes concernées, nous nous appuyons sur une ou plusieurs des
                bases suivantes, au sens de la <strong>LPD</strong> (Suisse) et, le cas échéant, du <strong>RGPD</strong>{' '}
                (UE) :
              </p>
              <div className={styles.grid2}>
                <div className={`${styles.card} ${styles.cardLegal}`}>
                  <div className={styles.cardTitle}>Exécution du contrat</div>
                  <p>
                    Fourniture de la plateforme, création et gestion des comptes, organisation et suivi des missions,
                    coordination opérationnelle, facturation lorsqu&apos;elle découle du contrat.
                  </p>
                </div>
                <div className={`${styles.card} ${styles.cardLegal}`}>
                  <div className={styles.cardTitle}>Intérêt légitime</div>
                  <p>
                    Sécurité des systèmes et des utilisateurs, amélioration mesurée du service, supervision technique
                    (logs, monitoring), prévention de la fraude ou des abus, lorsque cet intérêt n&apos;est pas prévalué
                    par les droits et libertés des personnes concernées.
                  </p>
                </div>
                <div className={`${styles.card} ${styles.cardLegal}`}>
                  <div className={styles.cardTitle}>Obligations légales</div>
                  <p>
                    Conservation ou communication de certaines données lorsque la loi l&apos;exige (obligations
                    comptables ou fiscales, pièces justificatives, réponses à une autorité compétente).
                  </p>
                </div>
                <div className={`${styles.card} ${styles.cardLegal}`}>
                  <div className={styles.cardTitle}>Consentement</div>
                  <p>
                    Lorsque requis pour des fonctionnalités facultatives (notifications marketing si elles existent,
                    réglages spécifiques) ou lorsque la loi ou les magasins d&apos;applications imposent un recueil
                    distinct. Le retrait du consentement n&apos;affecte pas la licéité des traitements antérieurs fondés
                    sur d&apos;autres bases.
                  </p>
                </div>
              </div>
              <div className={`${styles.callout} ${styles.calloutInfo}`}>
                <strong>Localisation des chauffeurs :</strong> la collecte en temps réel pendant une mission est en
                principe nécessaire à l&apos;<em>exécution des missions de transport</em> et à la{' '}
                <em>coordination opérationnelle</em>. Un consentement distinct peut être sollicité pour des traitements
                optionnels ou si la loi ou la plateforme de distribution l&apos;exige (section 12).
              </div>
            </section>

            <section id="destinataires" className={styles.section}>
              <SectionHead num={7} title="Destinataires des données" Icon={IcoUsers} />
              <p>
                Les données peuvent être communiquées aux catégories de destinataires suivantes, dans la limite de ce qui
                est <strong>nécessaire</strong> à chaque mission ou fonction :
              </p>
              <ul className={styles.listPlain}>
                <li>personnel autorisé de Lirie (support, exploitation, administration) ;</li>
                <li>entreprises de transport partenaires et leurs utilisateurs habilités (dispatch, gestion de flotte) ;</li>
                <li>institutions ou organismes clientes et leurs utilisateurs désignés ;</li>
                <li>chauffeurs ou conducteurs assignés à une mission ;</li>
                <li>
                  clients ou bénéficiaires, lorsque le parcours applicatif prévoit un accès limité à certaines informations
                  de mission ;
                </li>
                <li>prestataires techniques agissant en tant que sous-traitants (section 8).</li>
              </ul>
              <div className={`${styles.callout} ${styles.calloutInfo}`}>
                <strong>Relation triangulaire :</strong> la plateforme met en relation des responsables du traitement
                côté institution ou entreprise et des exécutants (chauffeurs), tout en traitant elle-même des données en
                tant que responsable ou, selon les cas, en qualité de <strong>sous-traitant</strong> ; les rôles et
                instructions sont définis contractuellement.
              </div>
            </section>

            <section id="sous-traitants" className={styles.section}>
              <SectionHead num={8} title="Sous-traitants techniques" Icon={IcoGlobe} />
              <p>
                Nous faisons appel à des prestataires qui traitent des données pour notre compte et selon nos instructions,
                dans le strict cadre d&apos;un contrat de sous-traitance lorsque la loi l&apos;exige :
              </p>
              <div className={styles.grid2}>
                <div className={`${styles.card} ${styles.providerCard}`}>
                  <div className={styles.cardTitle}>Hébergement &amp; infrastructure</div>
                  <p>Services cloud sécurisés pour l&apos;ensemble de la plateforme.</p>
                </div>
                <div className={`${styles.card} ${styles.providerCard}`}>
                  <div className={styles.cardTitle}>Supervision applicative</div>
                  <p>Outils de journalisation, monitoring et qualité (par ex. Sentry).</p>
                </div>
                <div className={`${styles.card} ${styles.providerCard}`}>
                  <div className={styles.cardTitle}>Notifications &amp; messagerie</div>
                  <p>Services push et messagerie (par ex. Firebase / Google Cloud).</p>
                </div>
                <div className={`${styles.card} ${styles.providerCard}`}>
                  <div className={styles.cardTitle}>Cartographie &amp; géolocalisation</div>
                  <p>Services d&apos;itinéraires, cartes et géolocalisation.</p>
                </div>
                <div className={`${styles.card} ${styles.providerCard}`} style={{ gridColumn: '1 / -1' }}>
                  <div className={styles.cardTitle}>Paiement en ligne</div>
                  <p>Prestataires certifiés PCI-DSS pour les transactions (section 14).</p>
                </div>
              </div>
              <p className={styles.note}>
                La liste peut évoluer ; des précisions peuvent figurer dans la documentation contractuelle. Pour plus de
                détails :{' '}
                <a href="mailto:privacy@lirie.ch" className={styles.contactLink}>
                  privacy@lirie.ch
                </a>
                .
              </p>
            </section>

            <section id="transferts-internationaux" className={styles.section}>
              <SectionHead num={9} title="Transferts internationaux" Icon={IcoGlobe} />
              <p>
                Certains prestataires techniques peuvent être situés en dehors de la Suisse ou de l&apos;Espace économique
                européen (EEE), ou traiter des données depuis ces pays. Lorsqu&apos;un tel transfert a lieu, nous veillons à
                ce que des <strong>garanties appropriées</strong> soient mises en place conformément à la LPD et, le cas
                échéant, au RGPD :
              </p>
              <ul className={styles.listPlain}>
                <li>clauses contractuelles types de la Commission européenne (CCT) ;</li>
                <li>mesures organisationnelles et techniques complémentaires ;</li>
                <li>décisions d&apos;adéquation lorsqu&apos;elles existent.</li>
              </ul>
            </section>

            <section id="conservation" className={styles.section}>
              <SectionHead num={10} title="Durée de conservation" Icon={IcoDatabase} />
              <p>
                Les données sont conservées pendant la durée nécessaire aux finalités poursuivies, puis supprimées ou
                anonymisées. Durées indicatives :
              </p>
              <div className={styles.table}>
                <div className={`${styles.tableRow} ${styles.tableHeader}`}>
                  <div className={styles.tableCell}>Type de données</div>
                  <div className={styles.tableCell}>Durée indicative</div>
                </div>
                <div className={styles.tableRow}>
                  <div className={styles.tableCell}>Données contractuelles et opérationnelles</div>
                  <div className={styles.tableCell}>Durée du contrat + 3 ans (sauf délai légal ou contractuel différent)</div>
                </div>
                <div className={styles.tableRow}>
                  <div className={styles.tableCell}>Journaux techniques (logs, erreurs)</div>
                  <div className={styles.tableCell}>12 mois en principe, sous réserve d&apos;exceptions liées à la sécurité</div>
                </div>
                <div className={styles.tableRow}>
                  <div className={styles.tableCell}>Facturation et comptabilité</div>
                  <div className={styles.tableCell}>Jusqu&apos;à 10 ans lorsque le droit suisse l&apos;impose</div>
                </div>
                <div className={styles.tableRow}>
                  <div className={styles.tableCell}>Preuves opérationnelles en cas de litige</div>
                  <div className={styles.tableCell}>Durée de la procédure + délais légaux applicables</div>
                </div>
              </div>
              <p className={styles.note}>
                Les données soumises à des obligations légales peuvent être conservées plus longtemps dans les limites
                imposées par le droit applicable.
              </p>
            </section>

            <section id="securite" className={styles.section}>
              <SectionHead num={11} title="Sécurité" Icon={IcoLock} />
              <p>Lirie met en œuvre des mesures techniques et organisationnelles appropriées au regard des risques :</p>
              <ul className={`${styles.listPlain} ${styles.listCheck}`}>
                <li>chiffrement des communications (TLS) pour les accès web et API lorsque applicable ;</li>
                <li>contrôle d&apos;accès par rôles et principe du moindre privilège ;</li>
                <li>authentification sécurisée et gestion des sessions ;</li>
                <li>journalisation d&apos;événements techniques et supervision des incidents ;</li>
                <li>sauvegardes et politiques de continuité adaptées ;</li>
                <li>
                  séparation des environnements (production / test / développement) lorsque cela est pertinent pour limiter
                  les risques.
                </li>
              </ul>
              <p>Nous exigeons contractuellement de nos sous-traitants un niveau de protection conforme aux usages du marché.</p>
            </section>

            <section id="localisation-gps" className={styles.section}>
              <SectionHead num={12} title="Localisation GPS (applications mobiles)" Icon={IcoMapPin} />
              <p>
                L&apos;application peut requérir l&apos;accès à la localisation des chauffeurs pendant les missions afin
                d&apos;assurer l&apos;assignation, le suivi en temps réel et la coordination (sections 5 et 6).
              </p>
              <div className={`${styles.callout} ${styles.calloutInfo}`}>
                <strong>Votre contrôle :</strong> vous pouvez retirer ou restreindre certains accès depuis les réglages du
                système ; sans localisation pendant une mission, certaines fonctions peuvent être limitées ou indisponibles.
              </div>
              <p>
                Pour les traitements optionnels ou en arrière-plan lorsque la loi ou les magasins d&apos;applications les
                exigent, des consentements ou bannières spécifiques peuvent être présentés dans l&apos;application.
              </p>
            </section>

            <section id="notifications" className={styles.section}>
              <SectionHead num={13} title="Notifications push" Icon={IcoBell} />
              <p>
                Les notifications relatives aux missions, au chat ou aux alertes opérationnelles peuvent être délivrées via
                des services tiers (par exemple <strong>Firebase Cloud Messaging</strong> / infrastructure Google Cloud).
              </p>
              <p>
                Vous pouvez désactiver les notifications depuis l&apos;application ou depuis les réglages du système
                d&apos;exploitation.
              </p>
            </section>

            <section id="paiements" className={styles.section}>
              <SectionHead num={14} title="Paiements en ligne" Icon={IcoCreditCard} />
              <p>
                Lorsqu&apos;un paiement en ligne est proposé, les données strictement nécessaires au règlement sont transmises
                au <strong>prestataire de paiement certifié</strong> chargé d&apos;héberger la transaction (par exemple
                flux de type Saferpay dans l&apos;écosystème des solutions Worldline).
              </p>
              <div className={`${styles.callout} ${styles.calloutBrand}`}>
                <div className={styles.calloutRow}>
                  <IcoLock s={15} />
                  <span>
                    Lirie n&apos;a <strong>pas accès</strong> au numéro complet de carte bancaire ni au cryptogramme visuel.
                    Le traitement des données de carte est régi par les conditions du prestataire, certifié PCI-DSS.
                  </span>
                </div>
              </div>
            </section>

            <section id="cookies" className={styles.section}>
              <SectionHead num={15} title="Cookies et technologies similaires (web)" Icon={IcoCookie} />
              <div className={styles.grid2}>
                <div className={`${styles.card} ${styles.cookieRequired}`}>
                  <div className={styles.cookieTitle}>
                    Cookies strictement nécessaires
                    <span className={`${styles.badgePill} ${styles.badgeReq}`}>Toujours actifs</span>
                  </div>
                  <p>
                    Session, préférences de base, protection anti-abus. Ne peuvent pas être désactivés sans impacter le
                    fonctionnement essentiel.
                  </p>
                </div>
                <div className={styles.card}>
                  <div className={styles.cookieTitle}>
                    Cookies analytiques
                    <span className={`${styles.badgePill} ${styles.badgeOpt}`}>Optionnel</span>
                  </div>
                  <p>
                    Mesure d&apos;audience et amélioration du service, sous réserve des exigences légales (information,
                    consentement ou refus). Paramétrables via une bannière ou une page dédiée lorsque ces traceurs sont
                    activés.
                  </p>
                </div>
              </div>
            </section>

            <section id="droits" className={styles.section}>
              <SectionHead num={16} title="Droits des personnes concernées" Icon={IcoEye} />
              <p>
                Conformément à la <strong>LPD</strong> (Suisse) et, lorsque le <strong>RGPD</strong> (UE) s&apos;applique,
                vous disposez en principe des droits suivants, dans les limites prévues par la loi :
              </p>
              <div className={styles.rightsGrid}>
                {RIGHTS.map((r) => (
                  <div key={r.title} className={styles.rightItem}>
                    <div className={styles.rightTitle}>
                      <IcoCheckCircle s={13} />
                      {r.title}
                    </div>
                    <p>{r.desc}</p>
                  </div>
                ))}
              </div>
              <div className={`${styles.callout} ${styles.calloutInfo}`}>
                <strong>Comment exercer vos droits :</strong> écrivez à{' '}
                <a href="mailto:privacy@lirie.ch" className={styles.contactLink}>
                  privacy@lirie.ch
                </a>{' '}
                ou à{' '}
                <a href="mailto:info@lirie.ch" className={styles.contactLink}>
                  info@lirie.ch
                </a>
                , ou utilisez le{' '}
                <Link to="/contact/support" className={styles.contactLink}>
                  formulaire d&apos;aide réservé aux clients
                </Link>
                , en précisant votre identité et la nature de la demande. Si Lirie agit comme sous-traitant, certaines
                demandes pourront être transmises au responsable du traitement client.
              </div>
            </section>

            <section id="autorite-controle" className={styles.section}>
              <SectionHead num={17} title="Autorité de contrôle et droit de réclamation" Icon={IcoCheckCircle} />
              <p>
                Vous avez le droit d&apos;introduire une <strong>réclamation</strong> auprès d&apos;une autorité de
                protection des données compétente.
              </p>
              <div className={styles.authority}>
                <span className={styles.flag} aria-hidden>
                  🇨🇭
                </span>
                <div>
                  <strong>Suisse — PFPDT</strong>
                  <br />
                  Préposé fédéral à la protection des données et à la transparence
                  <br />
                  <a
                    href="https://www.edoeb.admin.ch"
                    className={styles.contactLink}
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    www.edoeb.admin.ch <IcoExternal s={11} />
                  </a>
                </div>
              </div>
              <p className={styles.note}>
                Si vous résidez dans un autre État ou si le RGPD s&apos;applique, vous pouvez également vous adresser à
                l&apos;autorité de protection des données de votre pays ou canton, lorsque la loi le prévoit.
              </p>
            </section>

            <section id="data-deletion" className={styles.section}>
              <SectionHead num={18} title="Suppression des données" Icon={IcoTrash} />
              <p>
                Vous pouvez demander la suppression de vos données lorsque les conditions légales sont réunies, en
                écrivant à{' '}
                <a href="mailto:privacy@lirie.ch" className={styles.contactLink}>
                  privacy@lirie.ch
                </a>
                . Sauf obligations légales ou droits légitimes de conservation, Lirie supprime ou <strong>anonymise</strong>{' '}
                les données concernées dans un délai raisonnable (visée : jusqu&apos;à <strong>30 jours</strong> pour les
                demandes courantes, sous réserve de complexité ou d&apos;obligations de preuve).
              </p>
              <div className={`${styles.callout} ${styles.calloutInfo}`}>
                Certaines données (facturation, journaux de sécurité, preuves opérationnelles) peuvent être conservées pendant
                la durée requise par la loi ou un titre exécutoire, le cas échéant sous forme minimisée ou agrégée.
              </div>
            </section>

            <section id="modifications" className={styles.section}>
              <SectionHead num={19} title="Modifications de cette politique" Icon={IcoRefresh} />
              <p>
                Nous pouvons mettre à jour la présente politique pour refléter les évolutions légales, techniques ou
                fonctionnelles. Les modifications substantielles seront portées à votre connaissance par des moyens
                appropriés (notification dans l&apos;application, courriel aux coordonnées enregistrées, ou avis sur le
                site), selon les exigences applicables.
              </p>
              <p>La date en tête de page indique la version en vigueur.</p>
            </section>

            <section id="contact" className={`${styles.section} ${styles.sectionLast}`}>
              <SectionHead num={20} title="Contact" Icon={IcoMail} />
              <p>
                Pour toute question relative à cette politique ou au traitement de vos données : page{' '}
                <Link to="/contact" className={styles.contactLink}>
                  Contact
                </Link>
                , ou les adresses ci-dessous.
              </p>
              <div className={styles.contactGrid}>
                <a href="mailto:privacy@lirie.ch" className={styles.contactCard}>
                  <div className={styles.contactCardIcon}>
                    <IcoShield s={18} />
                  </div>
                  <div>
                    <div className={styles.contactCardTitle}>Protection des données</div>
                    <div className={styles.contactCardAddr}>privacy@lirie.ch</div>
                  </div>
                </a>
                <a href="mailto:info@lirie.ch" className={styles.contactCard}>
                  <div className={styles.contactCardIcon}>
                    <IcoMail s={18} />
                  </div>
                  <div>
                    <div className={styles.contactCardTitle}>Contact général</div>
                    <div className={styles.contactCardAddr}>info@lirie.ch</div>
                  </div>
                </a>
              </div>
            </section>

            <div className={styles.footerStamp}>
              <IcoShield s={14} />
              <span>
                Cadre LPD (Suisse) et RGPD (UE) le cas échéant — Dernière révision : {UPDATED_AT}
              </span>
            </div>
          </article>
        </div>
      </div>
    </div>
  );
};

export default PrivacyPolicy;
