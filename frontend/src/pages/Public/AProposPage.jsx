import React, { useEffect, useRef } from 'react';
import { Link } from 'react-router-dom';
import styles from './AProposPage.module.css';

function useScrollReveal(containerRef) {
  useEffect(() => {
    const root = containerRef.current;
    if (!root) return undefined;

    const nodes = root.querySelectorAll('[data-reveal]');
    if (typeof window === 'undefined' || !nodes.length) return undefined;

    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      nodes.forEach((el) => {
        el.classList.add(styles.revealVisible);
      });
      return undefined;
    }

    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;
          entry.target.classList.add(styles.revealVisible);
          io.unobserve(entry.target);
        });
      },
      { threshold: 0.12, rootMargin: '0px 0px -8% 0px' }
    );

    nodes.forEach((n) => io.observe(n));
    return () => io.disconnect();
    // ref object stable ; styles.revealVisible vient du CSS module
  }, [containerRef]);
}

function IcoEye({ s = 20 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
      <circle cx="12" cy="12" r="3" />
    </svg>
  );
}

function IcoUsers({ s = 20 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
      <circle cx="9" cy="7" r="4" />
      <path d="M23 21v-2a4 4 0 0 0-3-3.87" />
      <path d="M16 3.13a4 4 0 0 1 0 7.75" />
    </svg>
  );
}

function IcoList({ s = 20 }) {
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

function IcoLayers({ s = 20 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <polygon points="12 2 2 7 12 12 22 7 12 2" />
      <polyline points="2 17 12 22 22 17" />
      <polyline points="2 12 12 17 22 12" />
    </svg>
  );
}

function IcoShare({ s = 20 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <circle cx="18" cy="5" r="3" />
      <circle cx="6" cy="12" r="3" />
      <circle cx="18" cy="19" r="3" />
      <line x1="8.59" y1="13.51" x2="15.42" y2="17.49" />
      <line x1="15.41" y1="6.51" x2="8.59" y2="10.49" />
    </svg>
  );
}

function IcoActivity({ s = 20 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
    </svg>
  );
}

function IcoHistory({ s = 20 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <polyline points="12 8 12 12 14 14" />
      <path d="M3.05 11a9 9 0 1 1 .5 4" />
      <path d="M3 2v4h4" />
    </svg>
  );
}

function IcoBuilding({ s = 20 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <rect x="4" y="2" width="16" height="20" />
      <path d="M9 22V12h6v10" />
      <line x1="9" y1="7" x2="9.01" y2="7" />
      <line x1="15" y1="7" x2="15.01" y2="7" />
    </svg>
  );
}

function IcoHome({ s = 20 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
      <polyline points="9 22 9 12 15 12 15 22" />
    </svg>
  );
}

function IcoHeart({ s = 20 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z" />
    </svg>
  );
}

function IcoLandmark({ s = 20 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <line x1="3" y1="22" x2="21" y2="22" />
      <line x1="6" y1="18" x2="6" y2="11" />
      <line x1="10" y1="18" x2="10" y2="11" />
      <line x1="14" y1="18" x2="14" y2="11" />
      <line x1="18" y1="18" x2="18" y2="11" />
      <polygon points="12 2 20 7 4 7 12 2" />
    </svg>
  );
}

function IcoTruck({ s = 20 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <rect x="1" y="3" width="15" height="13" />
      <polygon points="16 8 20 8 23 11 23 16 16 16 16 8" />
      <circle cx="5.5" cy="18.5" r="2.5" />
      <circle cx="18.5" cy="18.5" r="2.5" />
    </svg>
  );
}

function IcoMapPin({ s = 20 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z" />
      <circle cx="12" cy="10" r="3" />
    </svg>
  );
}

function IcoShield({ s = 20 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
    </svg>
  );
}

function IcoChevR({ s = 14 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <polyline points="9 18 15 12 9 6" />
    </svg>
  );
}

function IcoCheckSm({ s = 9 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <polyline points="20 6 9 17 4 12" />
    </svg>
  );
}

function IcoCollabArrow({ s = 28 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <polyline points="9 18 15 12 9 6" />
    </svg>
  );
}

const MISSION_CARDS = [
  {
    Icon: IcoEye,
    title: 'Visibilité accrue',
    desc: 'Vue unifiée sur les transports planifiés et l’état d’avancement de chaque mission.',
    delay: styles.revealDelay1,
  },
  {
    Icon: IcoUsers,
    title: 'Coordination fluide',
    desc: 'Interface partagée entre institutions, dispatch et entreprises de transport.',
    delay: styles.revealDelay2,
  },
  {
    Icon: IcoList,
    title: 'Suivi structuré',
    desc: 'Étapes clés documentées, historique complet, traçabilité exploitable.',
    delay: styles.revealDelay3,
  },
  {
    Icon: IcoShare,
    title: 'Continuité d’information',
    desc: 'Données partagées en temps réel entre tous les acteurs autorisés.',
    delay: styles.revealDelay4,
  },
];

const CONSTAT_ITEMS = [
  'Demandes transmises par téléphone ou courriel sans traçabilité',
  'Informations dispersées entre plusieurs interlocuteurs',
  'Visibilité limitée sur l’avancement des missions',
  'Coordination complexe avec plusieurs entreprises de transport',
  'Difficulté à reconstituer le déroulement d’un transport',
];

const ROLE_CARDS = [
  { Icon: IcoLayers, title: 'Centraliser les demandes', delay: styles.revealDelay1 },
  { Icon: IcoShare, title: 'Coordonner les missions', delay: styles.revealDelay2 },
  { Icon: IcoActivity, title: 'Partager l’information utile', delay: styles.revealDelay3 },
  { Icon: IcoList, title: 'Suivre les étapes clés', delay: styles.revealDelay4 },
  { Icon: IcoHistory, title: 'Conserver un historique structuré', delay: styles.revealDelay5 },
];

const ACTORS_TOP = [
  { Icon: IcoBuilding, label: 'Hôpitaux & cliniques', delay: styles.revealDelay1 },
  { Icon: IcoHome, label: 'EMS & hébergement', delay: styles.revealDelay2 },
  { Icon: IcoHeart, label: 'Soins à domicile', delay: styles.revealDelay3 },
  { Icon: IcoUsers, label: 'Services sociaux', delay: styles.revealDelay4 },
];

const ACTORS_BOTTOM = [
  { Icon: IcoLandmark, label: 'Institutions publiques', delay: styles.revealDelay1 },
  { Icon: IcoTruck, label: 'Entreprises de transport', delay: styles.revealDelay2 },
  { Icon: IcoActivity, label: 'Équipes coordination transport', delay: styles.revealDelay3 },
];

const DEPLOY_STEPS = [
  { num: '01', title: 'Sans rupture organisationnelle', desc: 'La plateforme s’adapte à vos flux existants.' },
  { num: '02', title: 'Déploiement progressif', desc: 'Intégration par étapes selon vos priorités et votre rythme.' },
  { num: '03', title: 'Ancrage local', desc: 'Conçue pour les pratiques de Suisse romande, avec les acteurs du terrain.' },
];

const VISION_CHECKS = [
  'Meilleure lisibilité des missions',
  'Coordination plus fluide',
  'Traçabilité exploitable au pilotage',
  'Continuité d’information garantie',
];

const TRUST_PILLARS = [
  {
    mark: 'CH',
    title: 'Ancrage Suisse romande',
    desc: 'Développée en tenant compte des pratiques organisationnelles et des besoins observés localement dans les structures coordonnant des transports médicaux.',
    delay: styles.revealDelay1,
  },
  {
    mark: 'CO',
    title: 'Co-construction terrain',
    desc: 'La plateforme évolue en collaboration directe avec les acteurs concernés : institutions, partenaires transport, équipes de coordination.',
    delay: styles.revealDelay2,
  },
  {
    mark: 'EV',
    title: 'Évolution continue',
    desc: 'Un produit qui progresse avec les besoins réels — non une solution figée imposée de l’extérieur.',
    delay: styles.revealDelay3,
  },
];

const AProposPage = () => {
  const pageRef = useRef(null);
  useScrollReveal(pageRef);

  return (
    <div ref={pageRef} className={styles.page}>
      <header className={styles.hero}>
        <div className={styles.heroGridBg} aria-hidden />
        <div className={styles.heroGlow} aria-hidden />
        <div className={styles.heroFadeBottom} aria-hidden />
        <div className={styles.heroShell}>
          <div className={styles.heroGrid}>
            <div className={styles.heroMain}>
              <div className={styles.heroBadge}>
                <IcoShield s={12} />
                <span className={styles.heroBadgeLabel}>Plateforme de coordination — transport médical &amp; institutionnel</span>
              </div>
              <h1 className={styles.heroTitle}>
                Une infrastructure de coordination
                <br />
                <span className={styles.heroTitleItalic}>dédiée aux transports médicaux</span>
              </h1>
              <p className={styles.heroLead}>
                LIRIE est une solution logicielle de coordination qui fluidifie l’organisation des transports entre
                institutions, entreprises partenaires et équipes terrain — dans un cadre clair, structuré et adapté aux
                pratiques suisses.
              </p>
              <p className={styles.heroComplement}>
                Développée en lien direct avec les réalités opérationnelles de Suisse romande, la plateforme améliore la
                lisibilité des missions et la continuité de l’information entre acteurs.
              </p>
              <div className={styles.heroCtas}>
                <Link to="/contact" className={styles.btnPrimary}>
                  Contacter l’équipe
                  <IcoChevR s={14} />
                </Link>
                <Link to="/contact/institution" className={styles.heroBtnGhost}>
                  Demander une présentation
                </Link>
              </div>
              <nav className={styles.heroQuickNav} aria-label="Pages liées">
                <Link to="/professionnel" className={styles.heroQuickLink}>
                  Professionnel
                </Link>
                <span className={styles.heroQuickDot} aria-hidden />
                <Link to="/deplacez-vous" className={styles.heroQuickLink}>
                  Déplacez-vous
                </Link>
                <span className={styles.heroQuickDot} aria-hidden />
                <Link to="/conduire" className={styles.heroQuickLink}>
                  Conduire
                </Link>
                <span className={styles.heroQuickDot} aria-hidden />
                <Link to="/" className={styles.heroQuickLink}>
                  Accueil
                </Link>
              </nav>
            </div>
            <aside className={styles.heroAside} aria-label="En bref">
              {[
                {
                  icon: <IcoMapPin s={20} />,
                  title: 'Ancrage local',
                  desc: 'Conçue pour et avec les pratiques observées en Suisse romande.',
                },
                {
                  icon: <IcoLayers s={20} />,
                  title: 'Infrastructure de coordination',
                  desc: 'Un cadre commun — non un remplacement des structures existantes.',
                },
                {
                  icon: <IcoShield s={20} />,
                  title: 'Rôles clairement définis',
                  desc: 'La plateforme organise l’information ; les partenaires exécutent les missions.',
                },
              ].map((card) => (
                <div key={card.title} className={styles.trustCard}>
                  <div className={styles.trustCardIcon}>{card.icon}</div>
                  <div className={styles.trustCardBody}>
                    <div className={styles.trustCardTitle}>{card.title}</div>
                    <div className={styles.trustCardText}>{card.desc}</div>
                  </div>
                </div>
              ))}
              <div className={styles.heroChBadge}>
                <span className={styles.heroChMark} aria-hidden>
                  CH
                </span>
                <span className={styles.heroChText}>Développé en Suisse romande</span>
              </div>
            </aside>
          </div>
        </div>
      </header>

      <section className={styles.blockWhite} aria-labelledby="mission-heading">
        <div className={styles.blockInner}>
          <div data-reveal className={styles.reveal}>
            <div className={styles.sectionEyebrow}>
              <span className={styles.eyebrowLine} aria-hidden />
              Mission
            </div>
            <h2 id="mission-heading" className={styles.sectionTitle}>
              Simplifier la coordination des transports médicaux et accompagnés
            </h2>
            <p className={styles.sectionLead}>
              LIRIE propose un environnement commun aux structures qui organisent, suivent ou exécutent des missions de
              transport — afin de fluidifier les échanges et renforcer la continuité opérationnelle. Elle soutient
              également les équipes dans l’organisation quotidienne des déplacements.
            </p>
          </div>
          <div className={styles.missionGridWrap}>
            {MISSION_CARDS.map(({ Icon, title, desc, delay }) => (
              <div key={title} data-reveal className={`${styles.reveal} ${delay}`}>
                <div className={styles.missionCard}>
                  <div className={styles.missionCardIcon}>
                    <Icon s={20} />
                  </div>
                  <h3 className={styles.missionCardTitle}>{title}</h3>
                  <p className={styles.missionCardDesc}>{desc}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className={styles.blockMuted} aria-labelledby="origine-heading">
        <div className={styles.blockInner}>
          <div className={styles.narrativeSplit}>
            <div data-reveal className={styles.reveal}>
              <div className={styles.sectionEyebrow}>
                <span className={styles.eyebrowLine} aria-hidden />
                Origine
              </div>
              <h2 id="origine-heading" className={styles.sectionTitle}>
                Né d’un constat terrain,
                <br />
                <span className={styles.sectionTitleAccent}>pas d’une hypothèse</span>
              </h2>
              <p className={styles.narrativeProse}>
                Dans de nombreuses organisations, la coordination des transports repose encore sur des échanges fragmentés
                entre services, partenaires et transporteurs. Les informations circulent par téléphone, courriel ou
                messageries informelles — sans traçabilité commune ni visibilité partagée.
              </p>
              <p className={styles.narrativeProse}>
                LIRIE est né de ce constat, avec l’objectif de proposer un cadre plus lisible et partagé —{' '}
                <strong>sans modifier l’organisation existante des acteurs</strong>.
              </p>
            </div>
            <div data-reveal className={`${styles.reveal} ${styles.revealDelay2}`}>
              <div className={styles.constatPanel}>
                <div className={styles.constatPanelHead}>
                  <span>Problèmes observés</span>
                </div>
                {CONSTAT_ITEMS.map((item) => (
                  <div key={item} className={styles.constatRow}>
                    <span className={styles.constatDot} aria-hidden />
                    <span className={styles.constatText}>{item}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className={styles.blockWhite} aria-labelledby="role-heading">
        <div className={styles.blockInner}>
          <div data-reveal className={styles.reveal} style={{ marginBottom: '2.5rem' }}>
            <div className={styles.sectionEyebrow}>
              <span className={styles.eyebrowLine} aria-hidden />
              Rôle produit
            </div>
            <h2 id="role-heading" className={styles.sectionTitle}>
              Ce que LIRIE permet concrètement
            </h2>
            <p className={styles.sectionLead} style={{ marginBottom: 0 }}>
              Une interface de coordination entre institutions et entreprises de transport — pas un transporteur, pas un
              intermédiaire de prestation.
            </p>
          </div>
          <div className={styles.roleGrid}>
            {ROLE_CARDS.map(({ Icon, title, delay }) => (
              <div key={title} data-reveal className={`${styles.reveal} ${delay}`}>
                <div className={styles.roleCard}>
                  <div className={styles.roleCardIcon}>
                    <Icon s={20} />
                  </div>
                  <p className={styles.roleCardTitle}>{title}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className={styles.legalBand} aria-labelledby="legal-heading">
        <div className={styles.legalBandInner}>
          <div data-reveal className={styles.reveal}>
            <div className={styles.sectionEyebrow}>
              <span className={styles.eyebrowLine} aria-hidden />
              Cadre légal
            </div>
            <h2 id="legal-heading" className={styles.legalBandTitle}>
              Une plateforme de coordination, pas un transporteur
            </h2>
          </div>
          <div data-reveal className={`${styles.reveal} ${styles.revealDelay2}`}>
            <p className={styles.legalBandProse}>
              LIRIE <strong>n’exécute pas</strong> elle-même les prestations de transport. Les missions sont réalisées par
              des <strong>entreprises partenaires juridiquement indépendantes</strong>, responsables de leurs activités
              conformément aux réglementations applicables.
            </p>
            <p className={styles.legalBandProse}>
              La plateforme facilite l’organisation et le suivi des transports <strong>sans modifier les responsabilités</strong>{' '}
              des acteurs concernés. Pour le détail :{' '}
              <Link to="/conditions">conditions générales</Link>, <Link to="/mentions-legales">mentions légales</Link>.
            </p>
          </div>
        </div>
      </section>

      <section className={styles.blockWhite} aria-labelledby="acteurs-heading">
        <div className={styles.blockInner}>
          <div data-reveal className={styles.reveal}>
            <div className={styles.sectionEyebrow}>
              <span className={styles.eyebrowLine} aria-hidden />
              Écosystème
            </div>
            <h2 id="acteurs-heading" className={styles.sectionTitle}>
              Conçue pour plusieurs types d’acteurs institutionnels
            </h2>
            <p className={styles.sectionLead}>
              Chaque acteur accède aux informations qui lui sont utiles selon son rôle dans la chaîne de coordination.
            </p>
          </div>
          <div className={styles.actorsGridTop}>
            {ACTORS_TOP.map(({ Icon, label, delay }) => (
              <div key={label} data-reveal className={`${styles.reveal} ${delay}`}>
                <div className={styles.actorCell}>
                  <span className={styles.actorCellIcon}>
                    <Icon s={18} />
                  </span>
                  <span className={styles.actorCellLabel}>{label}</span>
                </div>
              </div>
            ))}
          </div>
          <div className={styles.actorsGridBottom}>
            {ACTORS_BOTTOM.map(({ Icon, label, delay }) => (
              <div key={label} data-reveal className={`${styles.reveal} ${delay}`}>
                <div className={`${styles.actorCell} ${styles.actorCellMuted}`}>
                  <span className={styles.actorCellIcon}>
                    <Icon s={18} />
                  </span>
                  <span className={styles.actorCellLabel}>{label}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className={styles.collabDark} aria-labelledby="collab-heading">
        <div className={styles.collabDarkGlow} aria-hidden />
        <div className={styles.collabDarkInner}>
          <div data-reveal className={styles.reveal}>
            <div className={styles.sectionEyebrow}>
              <span className={styles.eyebrowLine} aria-hidden />
              Réseau
            </div>
            <h2 id="collab-heading" className={styles.collabDarkTitle}>
              Un modèle à trois acteurs, un cadre unique
            </h2>
            <p className={styles.collabDarkLead}>
              LIRIE repose sur un modèle de coordination entre partenaires — simple à lire, robuste en organisation.
            </p>
          </div>
          <div className={styles.collabPipeline}>
            <div data-reveal className={`${styles.reveal} ${styles.revealDelay1}`}>
              <div className={styles.collabCard}>
                <div className={styles.collabRole}>Organisent</div>
                <div className={styles.collabName}>Institutions</div>
                <p className={styles.collabDesc}>
                  Définissent les besoins, les missions et les cadres d’intervention. Accèdent à une vue structurée de
                  leurs transports.
                </p>
              </div>
            </div>
            <div className={styles.collabArrow} aria-hidden>
              <IcoCollabArrow />
            </div>
            <div data-reveal className={`${styles.reveal} ${styles.revealDelay2}`}>
              <div className={`${styles.collabCard} ${styles.collabCardHighlight}`}>
                <div className={styles.collabRole}>Coordonne</div>
                <div className={styles.collabName}>Plateforme LIRIE</div>
                <p className={styles.collabDesc}>
                  Facilite la circulation de l’information entre acteurs autorisés — sans modifier les relations
                  contractuelles existantes.
                </p>
              </div>
            </div>
            <div className={styles.collabArrow} aria-hidden>
              <IcoCollabArrow />
            </div>
            <div data-reveal className={`${styles.reveal} ${styles.revealDelay3}`}>
              <div className={styles.collabCard}>
                <div className={styles.collabRole}>Exécutent</div>
                <div className={styles.collabName}>Entreprises de transport</div>
                <p className={styles.collabDesc}>
                  Réalisent les courses conformément aux règles et autorisations applicables. Partenaires indépendants et
                  responsables.
                </p>
              </div>
            </div>
          </div>
          <p className={styles.collabFoot}>
            Ce fonctionnement permet de structurer les échanges <strong>sans modifier les relations contractuelles</strong>{' '}
            existantes lorsque votre organisation le souhaite.
          </p>
        </div>
      </section>

      <section className={styles.blockMuted} aria-labelledby="deploy-heading">
        <div className={styles.blockInner}>
          <div className={styles.deploySplit}>
            <div data-reveal className={styles.reveal}>
              <div className={styles.sectionEyebrow}>
                <span className={styles.eyebrowLine} aria-hidden />
                Déploiement
              </div>
              <h2 id="deploy-heading" className={styles.sectionTitle}>
                Une intégration adaptée aux organisations existantes
              </h2>
              <p className={styles.narrativeProse}>
                LIRIE s’intègre dans des environnements où plusieurs partenaires interviennent déjà dans l’organisation des
                transports. La plateforme n’a pas vocation à remplacer les structures existantes, mais à{' '}
                <strong>améliorer la coordination entre elles</strong>.
              </p>
              <p className={styles.narrativeProse}>
                Son déploiement peut être <strong>progressif</strong> et adapté aux pratiques locales — un point essentiel
                pour les institutions qui font évoluer leurs outils avec prudence.
              </p>
            </div>
            <ul className={styles.deploySteps}>
              {DEPLOY_STEPS.map((step, i) => (
                <li
                  key={step.num}
                  data-reveal
                  className={`${styles.reveal} ${i === 1 ? styles.revealDelay2 : i === 2 ? styles.revealDelay3 : styles.revealDelay1}`}
                >
                  <div className={styles.deployStep}>
                    <span className={styles.deployStepNum}>{step.num}</span>
                    <div>
                      <div className={styles.deployStepTitle}>{step.title}</div>
                      <p className={styles.deployStepDesc}>{step.desc}</p>
                    </div>
                  </div>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </section>

      <section className={styles.visionSection} aria-labelledby="vision-heading">
        <div className={styles.visionGrid}>
          <div data-reveal className={styles.reveal}>
            <div className={styles.sectionEyebrow}>
              <span className={styles.eyebrowLine} aria-hidden />
              Vision
            </div>
            <h2 id="vision-heading" className={styles.sectionTitle}>
              Un cadre numérique commun pour les acteurs du transport médical
            </h2>
          </div>
          <div data-reveal className={`${styles.reveal} ${styles.revealDelay2}`}>
            <p className={styles.visionLead}>
              L’organisation des transports médicaux et accompagnés repose sur la coopération entre de nombreux acteurs.
              LIRIE vise à proposer le cadre numérique qui manque : partagé, structurant, non intrusif.
            </p>
            <div className={styles.visionChecks}>
              {VISION_CHECKS.map((label) => (
                <div key={label} className={styles.visionCheck}>
                  <span className={styles.visionCheckIcon}>
                    <IcoCheckSm />
                  </span>
                  <span className={styles.visionCheckLabel}>{label}</span>
                </div>
              ))}
            </div>
            <p className={styles.visionFoot}>
              L’objectif est de soutenir les organisations dans la gestion de ces flux essentiels au parcours de leurs
              bénéficiaires — en Suisse romande et au-delà.
            </p>
          </div>
        </div>
      </section>

      <section className={styles.trustPillars} aria-labelledby="trust-heading">
        <h2 id="trust-heading" className={styles.srOnly}>
          Ancrage et confiance
        </h2>
        <div className={styles.trustPillarsGrid}>
          {TRUST_PILLARS.map(({ mark, title, desc, delay }) => (
            <article key={title} data-reveal className={`${styles.reveal} ${delay}`}>
              <div className={styles.trustPillar}>
                <span className={styles.trustPillarMark}>{mark}</span>
                <h3 className={styles.trustPillarTitle}>{title}</h3>
                <p className={styles.trustPillarText}>{desc}</p>
              </div>
            </article>
          ))}
        </div>
      </section>

      <section className={styles.finalDark} aria-labelledby="cta-heading">
        <div className={styles.finalDarkGlow} aria-hidden />
        <div className={styles.finalDarkInner}>
          <div data-reveal className={styles.reveal}>
            <p className={styles.finalKicker}>En savoir plus</p>
            <h2 id="cta-heading" className={styles.finalDarkTitle}>
              Découvrir comment LIRIE peut s’intégrer à votre organisation
            </h2>
            <p className={styles.finalDarkSub}>
              Pour toute question concernant le fonctionnement de LIRIE, ses cas d’usage ou ses modalités de
              collaboration, notre équipe vous oriente selon votre organisation.
            </p>
            <p className={styles.finalDarkMicro}>
              Développé en Suisse romande pour des organisations coordonnant des transports sensibles.
            </p>
          </div>
          <div data-reveal className={`${styles.reveal} ${styles.revealDelay2}`}>
            <div className={styles.finalDarkCt}>
              <Link to="/contact" className={styles.btnPrimary}>
                Contacter l’équipe
                <IcoChevR s={14} />
              </Link>
              <Link to="/contact/institution" className={styles.btnSecondary}>
                Demander une présentation
              </Link>
              <Link to="/professionnel" className={styles.btnGhostLight}>
                Voir la page professionnels <IcoChevR s={12} />
              </Link>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};

export default AProposPage;
