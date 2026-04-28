import React, { useId, useState } from 'react';
import { Link } from 'react-router-dom';
import styles from './ConduirePage.module.css';

function IcoChevR({ s = 14 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <polyline points="9 18 15 12 9 6" />
    </svg>
  );
}
function IcoTruck({ s = 18 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <rect x="1" y="3" width="15" height="13" />
      <polygon points="16 8 20 8 23 11 23 16 16 16 16 8" />
      <circle cx="5.5" cy="18.5" r="2.5" />
      <circle cx="18.5" cy="18.5" r="2.5" />
    </svg>
  );
}
function IcoUsers({ s = 18 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
      <circle cx="9" cy="7" r="4" />
      <path d="M23 21v-2a4 4 0 0 0-3-3.87" />
      <path d="M16 3.13a4 4 0 0 1 0 7.75" />
    </svg>
  );
}
function IcoShield({ s = 18 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
    </svg>
  );
}
function IcoBriefcase({ s = 18 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <rect x="2" y="7" width="20" height="14" rx="2" />
      <path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16" />
    </svg>
  );
}
function IcoSmartphone({ s = 16 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <rect x="5" y="2" width="14" height="20" rx="2" />
      <line x1="12" y1="18" x2="12.01" y2="18" />
    </svg>
  );
}
function IcoMap({ s = 16 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <polygon points="1 6 1 22 8 18 16 22 23 18 23 2 16 6 8 2 1 6" />
      <line x1="8" y1="2" x2="8" y2="18" />
      <line x1="16" y1="6" x2="16" y2="22" />
    </svg>
  );
}
function IcoRadio({ s = 16 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M4.9 19.1C1 15.2 1 8.8 4.9 4.9" />
      <path d="M7.8 16.2c-2.3-2.3-2.3-6.1 0-8.5" />
      <circle cx="12" cy="12" r="2" />
      <path d="M16.2 7.8c2.3 2.3 2.3 6.1 0 8.5" />
      <path d="M19.1 4.9C23 8.8 23 15.1 19.1 19.1" />
    </svg>
  );
}
function IcoMessage({ s = 16 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
    </svg>
  );
}
function IcoHistory({ s = 16 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <polyline points="12 8 12 12 14 14" />
      <path d="M3.05 11a9 9 0 1 1 .5 4" />
      <path d="M3 2v4h4" />
    </svg>
  );
}
function IcoList({ s = 16 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <line x1="8" y1="6" x2="21" y2="6" />
      <line x1="8" y1="12" x2="21" y2="12" />
      <line x1="8" y1="18" x2="21" y2="18" />
      <line x1="3" y1="6" x2="3.01" y2="6" />
      <line x1="3" y1="12" x2="3.01" y2="12" />
      <line x1="3" y1="18" x2="3.01" y2="18" />
    </svg>
  );
}

function IcoBuilding({ s = 18 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <rect x="4" y="2" width="16" height="20" />
      <path d="M9 22V12h6v10" />
      <line x1="9" y1="7" x2="9.01" y2="7" />
      <line x1="15" y1="7" x2="15.01" y2="7" />
    </svg>
  );
}

function IcoCheck({ s = 14 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <polyline points="20 6 9 17 4 12" />
    </svg>
  );
}

function IcoInfo({ s = 18 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <circle cx="12" cy="12" r="10" />
      <line x1="12" y1="16" x2="12" y2="12" />
      <line x1="12" y1="8" x2="12.01" y2="8" />
    </svg>
  );
}

function IcoX({ s = 14 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <line x1="18" y1="6" x2="6" y2="18" />
      <line x1="6" y1="6" x2="18" y2="18" />
    </svg>
  );
}

const MISSION_TYPES = [
  { title: 'Transports médicaux planifiés', desc: 'Cliniques, hôpitaux, établissements de soins — créneaux coordonnés.' },
  { title: 'Transports PMR', desc: 'Véhicules adaptés, accessibilité et besoins d’accompagnement documentés.' },
  { title: 'Retours à domicile', desc: 'Après soins, interventions ou hospitalisations de jour.' },
  { title: 'Accompagnements institutionnels', desc: 'Mandats et consignes définis par l’établissement donneur d’ordre.' },
  { title: 'Trajets réguliers', desc: 'Dialyse, rééducation, soins ambulatoires récurrents.' },
  { title: 'Inter-établissements', desc: 'Transferts entre unités, sites ou structures de soins.' },
];

const AUDIENCE_CARDS = [
  {
    title: 'Chauffeurs salariés',
    desc: 'Rattachés à une entreprise de transport partenaire ou en discussion avec Lirie.',
    icon: 'truck',
  },
  {
    title: 'Entreprises de transport',
    desc: 'Opérateurs médicaux, sanitaires ou PMR souhaitant intégrer le réseau de coordination.',
    icon: 'building',
  },
  {
    title: 'Indépendants structurés',
    desc: 'Exercice dans une raison enregistrée : IDE, autorisations cantonales et assurances à jour.',
    icon: 'users',
  },
  {
    title: 'Responsables opérationnels',
    desc: 'Comprendre le modèle dispatch / coordination avant engagement contractuel.',
    icon: 'shield',
  },
];

const MODE_SALARIE_CHECKS = [
  'Missions coordonnées via la plateforme',
  'Contrat de travail avec votre employeur',
  'Accès et droits configurés par l’entreprise',
];

const MODE_INDEP_CHECKS = [
  'Structure enregistrée (p. ex. raison individuelle avec IDE)',
  'Autorisations cantonales requises pour l’activité',
  'Assurances et véhicule conformes au type de transport',
];

const PROCESS_STEPS = [
  { n: '1', title: 'Contact initial', text: 'Équipe partenaires transport ou employeur déjà raccordé à Lirie.' },
  { n: '2', title: 'Vérification du profil', text: 'Habilitations, véhicule, assurances et pièces attendues selon le profil.' },
  { n: '3', title: 'Validation & configuration', text: 'Validation par l’entreprise et paramétrage des accès sur la plateforme.' },
  { n: '4', title: 'Activation & formation', text: 'Ouverture des accès et prise en main des processus opérationnels.' },
];

const IN_PAGE_NAV = [
  { href: '#pour-qui', label: 'Public visé' },
  { href: '#modeles', label: 'Modèles' },
  { href: '#transparence-role', label: 'Rôle Lirie' },
  { href: '#recrutement', label: 'Recrutement' },
  { href: '#types-missions', label: 'Missions' },
  { href: '#parcours-mission', label: 'Déroulement' },
  { href: '#outils-lirie', label: 'Outils' },
  { href: '#faq-chauffeurs', label: 'FAQ' },
];

const TOOLS = [
  {
    title: 'Missions structurées',
    desc: 'Fil d’états opérationnels : chaque mission suit un cycle défini, lisible et traçable pour les équipes.',
    Icon: IcoList,
  },
  {
    title: 'Navigation intégrée',
    desc: 'Itinéraires et guidance lorsque le module est activé par votre entreprise sur la plateforme.',
    Icon: IcoMap,
  },
  {
    title: 'Suivi en temps réel',
    desc: 'Coordination et statuts pour les acteurs autorisés, selon les droits configurés par l’employeur.',
    Icon: IcoRadio,
  },
  {
    title: 'Canal dispatch',
    desc: 'Échanges avec la coordination ou le dispatch de votre structure partenaire.',
    Icon: IcoMessage,
  },
  {
    title: 'Historique & traçabilité',
    desc: 'Étapes horodatées et journaux accessibles aux parties habilitées sur la mission.',
    Icon: IcoHistory,
  },
  {
    title: 'Interface terrain',
    desc: 'Parcours pensé pour la mobilité : saisies courtes, lisibilité en conditions réelles.',
    Icon: IcoSmartphone,
  },
];

const EXIGENCE_ITEMS = [
  'Permis de conduire valide et catégories adaptées au type de transport.',
  'Autorisation de transport professionnel lorsque la loi l’exige.',
  'Véhicule conforme, dont adaptations PMR si applicable.',
  'Assurances et couvertures alignées sur l’activité exercée.',
  'Affiliation à une entreprise partenaire ou structure indépendante enregistrée (IDE, etc.).',
];

const LIRIE_ROLE_YES = [
  "Fournit l'outil de coordination et d'organisation des missions",
  "Assure la plateforme, l'hébergement et la disponibilité du service",
  'Vérifie les accréditations des transporteurs partenaires',
  'Met à disposition le suivi de mission aux acteurs habilités',
  'Assure la traçabilité utile à la coordination des transports',
];

const LIRIE_ROLE_NO = [
  "N'exécute pas les prestations de transport sur la voie publique",
  "N'intervient pas en qualité de transporteur",
  'Ne remplace pas le jugement clinique ni les protocoles médicaux',
  'Ne participe pas à la prise en charge médicale des personnes transportées',
  "N'est pas partie aux contrats de transport entre institution et partenaire",
];

function AudienceIcon({ name }) {
  if (name === 'truck') return <IcoTruck s={20} />;
  if (name === 'building') return <IcoBuilding s={20} />;
  if (name === 'users') return <IcoUsers s={20} />;
  return <IcoShield s={20} />;
}

const FAQ = [
  {
    q: 'Puis-je travailler directement pour Lirie comme employé ?',
    a: 'Non. Lirie n’emploie pas directement les chauffeurs. Vous intervenez via une entreprise de transport partenaire ou, si vous êtes indépendant, via votre propre structure enregistrée et autorisée.',
  },
  {
    q: 'Puis-je travailler comme indépendant avec l’application ?',
    a: 'Oui, uniquement si vous exercez dans le cadre d’une entreprise de transport légalement constituée (p. ex. raison individuelle avec IDE, autorisations et assurances requises selon le canton). L’application ne remplace pas ces obligations.',
  },
  {
    q: 'Puis-je utiliser mon véhicule personnel ?',
    a: 'Uniquement si votre véhicule et son usage respectent la réglementation applicable au transport professionnel concerné (y compris pour le PMR ou le transport sanitaire). L’entreprise qui engage la mission reste responsable de la conformité.',
  },
  {
    q: 'Qui me rémunère ?',
    a: 'Votre employeur (entreprise partenaire) ou, en indépendant, la facturation passe par votre structure : Lirie coordonne les missions sur la plateforme ; elle n’est pas votre employeur et ne fixe pas votre salaire.',
  },
  {
    q: 'Comment mon entreprise peut-elle rejoindre le réseau ?',
    a: 'Contactez l’équipe partenaires via le formulaire transport. Un interlocuteur Lirie vous accompagne dans la vérification de la conformité et la configuration des accès pour vos équipes.',
  },
];

const ConduirePage = () => {
  const [openFaq, setOpenFaq] = useState(null);
  const faqBaseId = useId();

  return (
    <div className={styles.page}>
      <header className={styles.hero}>
        <div className={styles.heroShell}>
          <div className={styles.heroGrid}>
            <div className={styles.heroMain}>
              <div className={styles.heroBadge}>
                <IcoTruck s={12} />
                <span className={styles.heroBadgeLabel}>Chauffeurs &amp; entreprises de transport</span>
              </div>
              <h1 className={styles.heroTitle}>
                Conduire avec <span className={styles.heroTitleAccent}>Lirie</span>
              </h1>
              <p className={styles.heroLead}>
                Rejoignez un réseau professionnel de <strong>coordination</strong> de transports médicaux planifiés.
                Lirie est une <strong>plateforme de dispatch</strong> : elle organise les missions et la traçabilité,
                sans se substituer à votre employeur, à votre structure ni à vos obligations réglementaires.
              </p>
              <div className={styles.heroCtas}>
                <Link to="/contact/transport" className={styles.btnPrimary}>
                  Rejoindre le réseau
                  <IcoChevR s={14} />
                </Link>
                <Link to="/contact" className={styles.btnSecondary}>
                  Contacter l’équipe
                </Link>
              </div>
              <div className={styles.heroQuickNav} role="navigation" aria-label="Pages liées">
                <Link to="/deplacez-vous" className={styles.heroQuickLink}>
                  Côté patients
                </Link>
                <span className={styles.heroQuickDot} aria-hidden />
                <Link to="/" className={styles.heroQuickLink}>
                  Accueil Lirie
                </Link>
              </div>
              <div className={styles.heroProof}>
                <div className={styles.heroProofDots} aria-hidden>
                  <span className={styles.heroProofDot} />
                  <span className={styles.heroProofDot} />
                  <span className={styles.heroProofDot} />
                </div>
                <p>
                  Les courses sont exécutées par des <strong>entreprises partenaires</strong> habilitées. Lirie
                  orchestre la mise en relation opérationnelle et le suivi pour les acteurs autorisés sur la mission.
                </p>
              </div>
              <nav className={styles.inPageNav} aria-label="Sur cette page">
                {IN_PAGE_NAV.map(({ href, label }) => (
                  <a key={href} href={href} className={styles.inPageNavLink}>
                    {label}
                  </a>
                ))}
              </nav>
            </div>
            <aside className={styles.heroAside} aria-label="En bref">
              <div className={styles.statCard}>
                <div className={styles.statIcon}>
                  <IcoShield s={20} />
                </div>
                <div>
                  <div className={styles.statVal}>Cadre structuré</div>
                  <div className={styles.statLabel}>Missions documentées, rôles et statuts clairs</div>
                </div>
              </div>
              <div className={styles.statCard}>
                <div className={styles.statIcon}>
                  <IcoUsers s={20} />
                </div>
                <div>
                  <div className={styles.statVal}>Deux parcours</div>
                  <div className={styles.statLabel}>Salarié partenaire ou indépendant via structure enregistrée</div>
                </div>
              </div>
              <div className={styles.statCard}>
                <div className={styles.statIcon}>
                  <IcoTruck s={20} />
                </div>
                <div>
                  <div className={styles.statVal}>Outil métier</div>
                  <div className={styles.statLabel}>Conçu pour équipes de transport, pas pour du grand public</div>
                </div>
              </div>
            </aside>
          </div>
        </div>
      </header>

      <div className={styles.reassurance}>
        <div className={styles.shell}>
          <ul className={styles.reassuranceList}>
            <li className={styles.reassuranceItem}>
              <div className={styles.reassuranceIcon}>
                <IcoShield s={20} />
              </div>
              <div>
                <div className={styles.reassuranceTitle}>Cadre professionnel</div>
                <p className={styles.reassuranceDesc}>
                  Missions structurées, traçabilité et rôles clairs entre institution, entreprise de transport et
                  exécutants.
                </p>
              </div>
            </li>
            <li className={styles.reassuranceItem}>
              <div className={styles.reassuranceIcon}>
                <IcoUsers s={20} />
              </div>
              <div>
                <div className={styles.reassuranceTitle}>Deux parcours possibles</div>
                <p className={styles.reassuranceDesc}>
                  Salarié d’une entreprise partenaire ou indépendant via une structure enregistrée — chaque situation a ses
                  règles.
                </p>
              </div>
            </li>
            <li className={styles.reassuranceItem}>
              <div className={styles.reassuranceIcon}>
                <IcoBriefcase s={20} />
              </div>
              <div>
                <div className={styles.reassuranceTitle}>Réseau partenaires</div>
                <p className={styles.reassuranceDesc}>
                  Lirie met en relation opérationnelle les acteurs ; les contrats de travail et de transport restent entre
                  les parties habilitées.
                </p>
              </div>
            </li>
          </ul>
        </div>
      </div>

      <main className={styles.main}>
        <section id="pour-qui" className={styles.section} aria-labelledby="pour-qui-heading">
          <div className={styles.sectionEyebrow}>
            <span className={styles.eyebrowLine} aria-hidden />
            Cible
          </div>
          <h2 id="pour-qui-heading" className={styles.sectionTitle}>
            Pour qui est cette page ?
          </h2>
          <p className={styles.sectionLead}>
            Professionnels du transport médical ou accompagné qui souhaitent intégrer ou comprendre le réseau Lirie.
          </p>
          <div className={styles.profiles}>
            {AUDIENCE_CARDS.map((card) => (
              <div key={card.title} className={styles.profileCardStatic}>
                <div className={styles.profileIconWrap}>
                  <AudienceIcon name={card.icon} />
                </div>
                <h3 className={styles.profileTitle}>{card.title}</h3>
                <p className={styles.profileQuote}>{card.desc}</p>
              </div>
            ))}
          </div>
          <div className={styles.profilesOutro}>
            <p className={styles.profilesOutroNote}>
              Quel que soit le profil ci-dessus, la démarche est la même&nbsp;: vous utilisez le formulaire transport.
              L’équipe Lirie en prend connaissance, affine le besoin si nécessaire et vous indique la suite adaptée
              (réseau partenaires, démarches, contacts utiles). Ce n’est pas un message adressé en direct à une
              entreprise du réseau.
            </p>
            <div className={styles.actionRow}>
              <Link to="/contact/transport" className={styles.btnPrimary}>
                Contacter l’équipe transport
                <IcoChevR s={13} />
              </Link>
            </div>
          </div>
        </section>

        <section id="modeles" className={styles.section} aria-labelledby="modes-heading">
          <div className={styles.sectionEyebrow}>
            <span className={styles.eyebrowLine} aria-hidden />
            Modèle
          </div>
          <h2 id="modes-heading" className={styles.sectionTitle}>
            Deux modes de collaboration possibles
          </h2>
          <p className={styles.sectionLead}>
            En Suisse, la distinction est essentielle pour éviter toute confusion entre <strong>coordination via une
            plateforme</strong> et <strong>relation d’emploi ou d’exploitation directe</strong>.
          </p>
          <div className={styles.modeGrid}>
            <div className={styles.modeCard}>
              <div className={styles.modeCardHeader}>
                <span className={styles.modeBadge}>
                  <IcoTruck s={14} />
                  Salarié
                </span>
                <h3 className={styles.modeCardTitle}>Chauffeur salarié d’une entreprise partenaire</h3>
              </div>
              <p className={styles.modeCardText}>
                Vous recevez vos missions dans le cadre de votre <strong>contrat de travail habituel</strong>. Lirie
                facilite la coordination, le dispatch et le suivi opérationnel. Votre{' '}
                <strong>employeur reste votre interlocuteur légal et social</strong> — Lirie n’intervient pas dans cette
                relation.
              </p>
              <ul className={styles.modeCheckList}>
                {MODE_SALARIE_CHECKS.map((line) => (
                  <li key={line}>
                    <span className={styles.modeCheckIcon} aria-hidden>
                      <IcoCheck s={13} />
                    </span>
                    {line}
                  </li>
                ))}
              </ul>
            </div>
            <div className={styles.modeCard}>
              <div className={styles.modeCardHeader}>
                <span className={styles.modeBadge}>
                  <IcoBuilding s={14} />
                  Indépendant
                </span>
                <h3 className={styles.modeCardTitle}>Chauffeur indépendant via une structure enregistrée</h3>
              </div>
              <p className={styles.modeCardText}>
                Vous exercez dans le cadre d’une <strong>entreprise de transport légalement constituée</strong> (raison
                individuelle avec IDE, autorisations cantonales, assurances et véhicule conformes). L’application ne
                dispense pas de ces <strong>obligations réglementaires préalables</strong>.
              </p>
              <ul className={styles.modeCheckList}>
                {MODE_INDEP_CHECKS.map((line) => (
                  <li key={line}>
                    <span className={styles.modeCheckIcon} aria-hidden>
                      <IcoCheck s={13} />
                    </span>
                    {line}
                  </li>
                ))}
              </ul>
            </div>
          </div>
          <div className={styles.calloutLegal}>
            <div className={styles.calloutLegalIcon} aria-hidden>
              <IcoInfo s={18} />
            </div>
            <div className={styles.calloutLegalBody}>
              <p>
                <strong>Lirie n’emploie pas directement les chauffeurs.</strong> Les missions sont réalisées sous la
                responsabilité des <strong>entreprises partenaires juridiquement indépendantes</strong>.
              </p>
              <p>
                Les chauffeurs indépendants doivent exercer dans le cadre d’une entreprise enregistrée disposant d’un{' '}
                <strong>numéro IDE</strong> et des <strong>autorisations nécessaires</strong> au transport professionnel
                selon la réglementation cantonale applicable.
              </p>
            </div>
          </div>
        </section>

        <section id="transparence-role" className={styles.section} aria-labelledby="role-lirie-heading">
          <div className={styles.sectionEyebrow}>
            <span className={styles.eyebrowLine} aria-hidden />
            Transparence
          </div>
          <h2 id="role-lirie-heading" className={styles.sectionTitle}>
            Le rôle exact de Lirie
          </h2>
          <p className={styles.sectionLead}>
            Même logique que pour les institutions : la plateforme coordonne ; le transport sur la voie publique relève
            des entreprises partenaires.
          </p>
          <div className={styles.roleCard}>
            <div className={styles.roleCol}>
              <div className={styles.roleColHead}>
                <span className={styles.roleColIconYes} aria-hidden>
                  <IcoCheck s={13} />
                </span>
                <span className={styles.roleColTitleYes}>Ce que Lirie assure</span>
              </div>
              <ul className={styles.roleList}>
                {LIRIE_ROLE_YES.map((item) => (
                  <li key={item} className={styles.roleItem}>
                    <span className={`${styles.roleItemGlyph} ${styles.roleItemGlyphYes}`} aria-hidden>
                      <IcoCheck s={11} />
                    </span>
                    <span>{item}</span>
                  </li>
                ))}
              </ul>
            </div>
            <div className={styles.roleDivider} aria-hidden />
            <div className={styles.roleCol}>
              <div className={styles.roleColHead}>
                <span className={styles.roleColIconNo} aria-hidden>
                  <IcoX s={13} />
                </span>
                <span className={styles.roleColTitleNo}>Ce que Lirie ne fait pas</span>
              </div>
              <ul className={styles.roleList}>
                {LIRIE_ROLE_NO.map((item) => (
                  <li key={item} className={styles.roleItem}>
                    <span className={`${styles.roleItemGlyph} ${styles.roleItemGlyphNo}`} aria-hidden>
                      <IcoX s={11} />
                    </span>
                    <span>{item}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
          <p className={styles.roleLegal}>
            Cadre juridique :{' '}
            <Link to="/conditions" className={styles.inlineLink}>
              conditions générales d’utilisation
            </Link>
            {' · '}
            <Link to="/mentions-legales" className={styles.inlineLink}>
              mentions légales
            </Link>
          </p>
        </section>

        <section id="recrutement" className={styles.section} aria-labelledby="jobs-heading">
          <div className={styles.sectionEyebrow}>
            <span className={styles.eyebrowLine} aria-hidden />
            Recrutement réseau
          </div>
          <h2 id="jobs-heading" className={styles.sectionTitle}>
            Opportunités professionnelles auprès des entreprises partenaires
          </h2>
          <div className={styles.jobsPanel}>
            <p className={styles.jobsIntro}>
              Les entreprises de transport partenaires du réseau Lirie peuvent recruter des chauffeurs pour des missions
              de transport médical planifié, accompagné ou PMR. Les relations de travail se concluent{' '}
              <strong>directement avec l’entreprise employeuse</strong>. Lirie agit uniquement comme plateforme de
              coordination des missions.
            </p>
            <div className={styles.jobsEmpty}>
              <div className={styles.jobsEmptyIcon} aria-hidden>
                <IcoBriefcase s={26} />
              </div>
              <h3 className={styles.jobsEmptyTitle}>Aucune offre publiée en ligne pour le moment</h3>
              <p className={styles.jobsEmptyText}>
                <strong>Deux possibilités</strong>&nbsp;: postuler ou solliciter le réseau (bouton de gauche), ou
                signaler une offre côté entreprise (bouton de droite). Un seul formulaire transport — indiquez l’objet
                dans votre message&nbsp;; l’équipe Lirie (transport) reçoit la demande et vous oriente selon le contexte
                local.
              </p>
            </div>
            <div className={styles.actionRow}>
              <Link to="/contact/transport" className={styles.btnPrimary}>
                Postuler ou contacter le réseau
                <IcoChevR s={13} />
              </Link>
              <Link to="/contact/transport" className={styles.btnOutline}>
                Faire connaître une offre (entreprise)
              </Link>
            </div>
            <p className={styles.jobsDisclaimer}>
              Les offres publiées sur cette page le sont par des entreprises partenaires juridiquement indépendantes. Les
              conditions d’engagement, de rémunération et d’activité relèvent exclusivement de ces entreprises.
            </p>
          </div>
        </section>

        <section id="types-missions" className={styles.section} aria-labelledby="missions-types-heading">
          <div className={styles.sectionEyebrow}>
            <span className={styles.eyebrowLine} aria-hidden />
            Missions
          </div>
          <h2 id="missions-types-heading" className={styles.sectionTitle}>
            Types de missions coordonnées
          </h2>
          <p className={styles.sectionLead}>
            La disponibilité effective dépend des conventions entre partenaires et institutions. Exemples de besoins
            fréquents côtés réseau :
          </p>
          <div className={styles.typesGrid}>
            {MISSION_TYPES.map((m) => (
              <div key={m.title} className={styles.typeCard}>
                <div className={styles.typeIcon} aria-hidden>
                  <IcoTruck s={18} />
                </div>
                <h3 className={styles.typeTitle}>{m.title}</h3>
                <p className={styles.typeDesc}>{m.desc}</p>
              </div>
            ))}
          </div>
        </section>

        <section id="parcours-mission" className={styles.section} aria-labelledby="workflow-heading">
          <div className={styles.sectionEyebrow}>
            <span className={styles.eyebrowLine} aria-hidden />
            Opérations
          </div>
          <h2 id="workflow-heading" className={styles.sectionTitle}>
            Fonctionnement d’une mission
          </h2>
          <p className={styles.sectionLead}>
            Parcours type côté chauffeur (selon configuration et droits) : réception de la mission, validation par
            l’entreprise, prise en charge, suivi dans l’application, clôture avec horodatages.
          </p>
          <ol className={styles.workflowSteps}>
            {[
              {
                n: '1',
                title: 'Réception',
                text: 'Notification de la mission — liste des courses ou attribution directe selon les règles de votre entreprise.',
              },
              {
                n: '2',
                title: 'Validation',
                text: 'Acceptation ou validation selon les processus définis par votre entreprise partenaire.',
              },
              {
                n: '3',
                title: 'Prise en charge',
                text: 'Prise en charge du patient ou du bénéficiaire conformément aux consignes de la mission.',
              },
              {
                n: '4',
                title: 'Suivi en temps réel',
                text: 'Mise à jour des statuts et coordination avec le dispatch si nécessaire, via la plateforme.',
              },
              {
                n: '5',
                title: 'Clôture horodatée',
                text: 'Étapes tracées et horodatées — journaux accessibles aux parties habilitées sur la mission.',
              },
            ].map((step) => (
              <li key={step.n} className={styles.workflowStep}>
                <div className={styles.workflowNum}>{step.n}</div>
                <div className={styles.workflowBody}>
                  <div className={styles.workflowTitle}>{step.title}</div>
                  <p className={styles.workflowText}>{step.text}</p>
                </div>
              </li>
            ))}
          </ol>
        </section>

        <section id="outils-lirie" className={styles.section} aria-labelledby="tools-heading">
          <div className={styles.sectionEyebrow}>
            <span className={styles.eyebrowLine} aria-hidden />
            Produit
          </div>
          <h2 id="tools-heading" className={styles.sectionTitle}>
            Outils fournis par Lirie
          </h2>
          <p className={styles.sectionLead}>
            Lirie se positionne comme <strong>outil professionnel</strong> pour les équipes de transport — pas comme une
            application grand public type VTC.
          </p>
          <div className={styles.toolGrid}>
            {TOOLS.map(({ title, desc, Icon }) => (
              <div key={title} className={styles.toolCard}>
                <div className={styles.toolCardIcon} aria-hidden>
                  <Icon s={18} />
                </div>
                <h3 className={styles.toolCardTitle}>{title}</h3>
                <p className={styles.toolCardDesc}>{desc}</p>
              </div>
            ))}
          </div>
        </section>

        <section id="cadre-responsabilites" className={styles.section} aria-labelledby="cadre-heading">
          <div className={styles.sectionEyebrow}>
            <span className={styles.eyebrowLine} aria-hidden />
            Responsabilités
          </div>
          <h2 id="cadre-heading" className={styles.sectionTitle}>
            Cadre d’intervention
          </h2>
          <div className={styles.calloutLegal}>
            <div className={styles.calloutLegalIcon} aria-hidden>
              <IcoInfo s={18} />
            </div>
            <div className={styles.calloutLegalBody}>
              <p>
                Lirie est une <strong>plateforme de coordination</strong>. Les missions sont exécutées sous la
                responsabilité des <strong>entreprises partenaires</strong>, conformément aux réglementations applicables
                (transport, assurance, protection des données, etc.).
              </p>
              <p>
                Pour le traitement des données et le rôle exact des acteurs, voir les{' '}
                <Link to="/mentions-legales" className={styles.inlineLink}>
                  mentions légales
                </Link>
                , les{' '}
                <Link to="/conditions" className={styles.inlineLink}>
                  conditions générales d’utilisation
                </Link>{' '}
                et la{' '}
                <Link to="/privacy" className={styles.inlineLink}>
                  politique de confidentialité
                </Link>
                .
              </p>
            </div>
          </div>
        </section>

        <section id="exigences" className={styles.section} aria-labelledby="exigences-heading">
          <div className={styles.sectionEyebrow}>
            <span className={styles.eyebrowLine} aria-hidden />
            Éligibilité
          </div>
          <h2 id="exigences-heading" className={styles.sectionTitle}>
            Exigences réglementaires et pratiques
          </h2>
          <p className={styles.sectionLead}>Selon le type de mission et le canton, les exigences peuvent notamment inclure :</p>
          <div className={styles.eligibilityGrid}>
            {EXIGENCE_ITEMS.map((line) => (
              <div key={line} className={styles.eligibilityCard}>
                <span className={styles.eligibilityCheck} aria-hidden>
                  <IcoCheck s={12} />
                </span>
                <p className={styles.eligibilityText}>{line}</p>
              </div>
            ))}
          </div>
          <p className={`${styles.sectionLead} ${styles.sectionLeadSpaced}`}>
            Lirie peut accompagner la <strong>mise en conformité opérationnelle</strong> sur la plateforme ; la
            conformité juridique et métier reste de la responsabilité de l’entreprise et des autorités compétentes.
          </p>
        </section>

        <section id="integration" className={styles.section} aria-labelledby="process-heading">
          <div className={styles.sectionEyebrow}>
            <span className={styles.eyebrowLine} aria-hidden />
            Parcours
          </div>
          <h2 id="process-heading" className={styles.sectionTitle}>
            Processus d’intégration au réseau
          </h2>
          <p className={styles.sectionLead}>
            Les étapes exactes dépendent de votre statut (salarié / entreprise / indépendant structuré). Schéma habituel :
          </p>
          <ul className={styles.timeline}>
            {PROCESS_STEPS.map((step) => (
              <li key={step.n} className={styles.step}>
                <div className={styles.stepNum}>{step.n}</div>
                <div className={styles.stepTitle}>{step.title}</div>
                <p className={styles.stepDesc}>{step.text}</p>
              </li>
            ))}
          </ul>
        </section>

        <section id="faq-chauffeurs" className={styles.section} aria-labelledby="faq-heading">
          <div className={styles.sectionEyebrow}>
            <span className={styles.eyebrowLine} aria-hidden />
            FAQ
          </div>
          <h2 id="faq-heading" className={styles.sectionTitle}>
            Questions fréquentes (chauffeurs)
          </h2>
          <p className={styles.sectionLead}>
            Réponses générales — votre situation peut varier selon le canton et le contrat. Pour le cadre exact de Lirie,
            voir aussi la section{' '}
            <a href="#transparence-role" className={styles.inlineLink}>
              Rôle Lirie
            </a>
            .
          </p>
          <div className={styles.faq}>
            {FAQ.map((item, i) => {
              const open = openFaq === i;
              const headingId = `${faqBaseId}-h-${i}`;
              const panelId = `${faqBaseId}-p-${i}`;
              return (
                <div key={item.q} className={styles.faqItem}>
                  <h3 className={styles.faqHeading}>
                    <button
                      type="button"
                      id={headingId}
                      className={`${styles.faqTrigger}${open ? ` ${styles.faqTriggerOpen}` : ''}`}
                      aria-expanded={open}
                      aria-controls={panelId}
                      onClick={() => setOpenFaq(open ? null : i)}
                    >
                      <span>{item.q}</span>
                      <span className={`${styles.faqIcon}${open ? ` ${styles.faqIconOpen}` : ''}`} aria-hidden>
                        +
                      </span>
                    </button>
                  </h3>
                  <div id={panelId} role="region" className={styles.faqPanel} aria-labelledby={headingId} hidden={!open}>
                    <p>{item.a}</p>
                  </div>
                </div>
              );
            })}
          </div>
        </section>
      </main>

      <div className={styles.finalBandBleed}>
        <div className={styles.finalBandShell}>
          <section className={styles.finalBandSection} aria-labelledby="final-cta-heading">
            <h2 id="final-cta-heading" className={styles.finalTitle}>
              Prochaine étape
            </h2>
            <p className={styles.finalSub}>
              Deux canaux : transport / réseau ou demande générale.
            </p>
            <div className={styles.finalCards}>
              <Link to="/contact/transport" className={styles.finalCard}>
                <span className={styles.finalCardLabel}>Transport &amp; réseau</span>
                <span className={styles.finalCardTitle}>Formulaire transport</span>
                <span className={styles.finalCardDesc}>
                  Chauffeur, structure ou entreprise : un formulaire. Décrivez votre besoin ; l’équipe Lirie (transport)
                  oriente (candidature, partenariat, offre…).
                </span>
                <span className={styles.finalCardAction}>
                  Ouvrir le formulaire <IcoChevR s={13} />
                </span>
              </Link>
              <Link to="/contact" className={styles.finalCard}>
                <span className={styles.finalCardLabel}>Autres sujets</span>
                <span className={styles.finalCardTitle}>Contact général Lirie</span>
                <span className={styles.finalCardDesc}>
                  Produit, institution, facturation ou demande hors volet partenaires transport.
                </span>
                <span className={styles.finalCardAction}>
                  Page contact <IcoChevR s={13} />
                </span>
              </Link>
            </div>
          </section>
        </div>
      </div>

      <div className={styles.bottomSpacer} aria-hidden />
    </div>
  );
};

export default ConduirePage;
