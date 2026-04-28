import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import styles from './ProfessionnelPage.module.css';

// ─── Icons (même tracés que la maquette) ─────────────────────────────────────

function IconBuilding({ s = 12 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <rect x="4" y="2" width="16" height="20" />
      <path d="M9 22V12h6v10" />
      <line x1="9" y1="7" x2="9.01" y2="7" />
      <line x1="15" y1="7" x2="15.01" y2="7" />
    </svg>
  );
}

function IconLayers() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <polygon points="12 2 2 7 12 12 22 7 12 2" />
      <polyline points="2 17 12 22 22 17" />
      <polyline points="2 12 12 17 22 12" />
    </svg>
  );
}

function IconClock() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <circle cx="12" cy="12" r="10" />
      <polyline points="12 6 12 12 16 14" />
    </svg>
  );
}

function IconUsers() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
      <circle cx="9" cy="7" r="4" />
      <path d="M23 21v-2a4 4 0 0 0-3-3.87" />
      <path d="M16 3.13a4 4 0 0 1 0 7.75" />
    </svg>
  );
}

function IconShield() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
    </svg>
  );
}

function IconActivity() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
    </svg>
  );
}

function IconCheck() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <polyline points="20 6 9 17 4 12" />
    </svg>
  );
}

function IconArrow() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <polyline points="9 18 15 12 9 6" />
    </svg>
  );
}

function IconChevron({ open }) {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={`${styles.faqChev}${open ? ` ${styles.faqChevOpen}` : ''}`}
      aria-hidden
    >
      <polyline points="6 9 12 15 18 9" />
    </svg>
  );
}

function IconFileText() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
      <polyline points="14 2 14 8 20 8" />
      <line x1="16" y1="13" x2="8" y2="13" />
      <line x1="16" y1="17" x2="8" y2="17" />
      <polyline points="10 9 9 9 8 9" />
    </svg>
  );
}

function IconRoute() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <circle cx="6" cy="19" r="3" />
      <path d="M9 19h8.5a3.5 3.5 0 0 0 0-7h-11a3.5 3.5 0 0 1 0-7H15" />
      <circle cx="18" cy="5" r="3" />
    </svg>
  );
}

function IconGrid() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <rect x="3" y="3" width="7" height="7" />
      <rect x="14" y="3" width="7" height="7" />
      <rect x="14" y="14" width="7" height="7" />
      <rect x="3" y="14" width="7" height="7" />
    </svg>
  );
}

/** Icônes section acteurs (stroke 1.75, pas d’emoji) */
function IconHospital() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M12 6v4" />
      <path d="M14 8h-4" />
      <path d="M18 2h-3a3 3 0 0 0-6 0H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V4a2 2 0 0 0-2-2Z" />
      <path d="M9 22v-4h6v4" />
    </svg>
  );
}

function IconHome() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
      <polyline points="9 22 9 12 15 12 15 22" />
    </svg>
  );
}

function IconHeart() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z" />
    </svg>
  );
}

function IconScale() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="m16 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z" />
      <path d="m2 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z" />
      <path d="M7 21h10" />
      <path d="M12 3v18" />
      <path d="M5 3h14" />
    </svg>
  );
}

function IconLandmark() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M3 21h18" />
      <path d="M4 21V10l8-5 8 5v11" />
      <path d="M9 21v-4h6v4" />
    </svg>
  );
}

// ─── Layout ─────────────────────────────────────────────────────────────────

function Shell({ children, className = '' }) {
  return <div className={`${styles.shell} ${className}`.trim()}>{children}</div>;
}

function Eyebrow({ children }) {
  return (
    <div className={styles.eyebrow}>
      <span className={styles.eyebrowLine} aria-hidden />
      <span className={styles.eyebrowLabel}>{children}</span>
    </div>
  );
}

const TRUST_ITEMS = [
  { Icon: IconShield, label: 'Cadre institutionnel', sub: 'Ton orienté organisation et continuité de service' },
  { Icon: IconActivity, label: 'Compréhension métier', sub: 'Hôpitaux, EMS, soins à domicile, services sociaux' },
  { Icon: IconLayers, label: 'Plateforme, pas transporteur', sub: 'Coordination logicielle entre acteurs existants' },
];

const CONSTAT_POINTS = [
  'Demandes dispersées entre téléphone, courriel et habitudes locales',
  "Difficulté à suivre l'état réel d'une course en cours",
  'Visibilité limitée sur les confirmations et retards',
  'Coordination complexe lorsque plusieurs transporteurs interviennent',
  'Charge administrative disproportionnée pour les équipes',
  "Traçabilité partielle en cas d'incident ou de question ultérieure",
];

const VALUE_CARDS = [
  {
    icon: <IconFileText />,
    title: 'Planification centralisée',
    desc: "Les demandes de transport sont regroupées dans un environnement unique accessible aux acteurs autorisés, quel que soit le service d'origine.",
  },
  {
    icon: <IconRoute />,
    title: 'Assignation structurée',
    desc: "Les missions sont orientées vers les transporteurs partenaires selon vos règles d'organisation et les disponibilités, sans arbitrage manuel permanent.",
  },
  {
    icon: <IconClock />,
    title: 'Suivi en temps réel',
    desc: "Vos équipes disposent d'une visibilité continue sur l'avancement des courses, sans avoir à relancer chaque prestataire individuellement.",
  },
  {
    icon: <IconActivity />,
    title: 'Communication maîtrisée',
    desc: 'Les informations utiles circulent entre institution, coordination et transporteurs dans un cadre partagé et tracé.',
  },
  {
    icon: <IconShield />,
    title: 'Historique des missions',
    desc: "Les étapes importantes sont enregistrées et consultables selon les droits d'accès, utile en cas d'audit ou de question ultérieure.",
  },
  {
    icon: <IconGrid />,
    title: "Moins d'incertitudes",
    desc: 'Moins de relances, moins de doublons — une coordination plus fluide entre les services et les prestataires.',
  },
];

const FLOW_STEPS = [
  { num: '01', title: 'Demande', desc: 'Votre équipe crée ou transmet une demande de transport depuis la plateforme.' },
  { num: '02', title: 'Organisation', desc: 'La mission est structurée selon vos règles internes et assignée aux partenaires disponibles.' },
  { num: '03', title: 'Exécution', desc: "Le transporteur reçoit les informations nécessaires à l'exécution de la mission." },
  { num: '04', title: 'Remontée des statuts', desc: 'Les étapes clés remontent dans la plateforme au fil du déroulement de la course.' },
  { num: '05', title: "Vue d'ensemble", desc: "Votre institution conserve une visibilité globale et un historique exploitable." },
];

const ACTORS = [
  { Icon: IconHospital, label: 'Hôpitaux et cliniques', sub: 'Sorties, retours à domicile, transferts inter-établissements' },
  { Icon: IconHome, label: "EMS et structures d'hébergement", sub: 'Transports récurrents, rendez-vous médicaux, admissions' },
  { Icon: IconHeart, label: 'Soins à domicile', sub: 'Coordination de transports autour des interventions à domicile' },
  { Icon: IconScale, label: 'Services sociaux et curateurs', sub: 'Accompagnements, démarches administratives, transports sensibles' },
  { Icon: IconLandmark, label: 'Institutions publiques', sub: 'Structures coordinatrices avec plusieurs prestataires partenaires' },
  { Icon: IconLayers, label: 'Organisations multi-transporteurs', sub: 'Tout acteur gérant plusieurs prestataires de transport en parallèle' },
];

const BENEFITS = [
  'Meilleure visibilité sur les missions en cours, à tout moment',
  "Moins d'appels pour vérifier l'état d'un transport",
  'Coordination renforcée lors des sorties ou retours à domicile',
  "Partage d'informations plus fluide entre services internes",
  'Continuité de suivi même lorsque plusieurs acteurs interviennent',
  "Capacité à reconstituer le déroulement complet d'une mission",
];

const TRACE_ITEMS = [
  { title: 'Enregistrement des étapes clés', desc: 'Chaque étape significative de la mission est horodatée et conservée dans la plateforme.' },
  { title: "Visibilité sur l'avancement", desc: 'Les statuts remontent en temps réel pour permettre un suivi opérationnel sans friction.' },
  { title: 'Historique consultable', desc: "Accessible selon les droits définis dans votre organisation — sans surexposition des données." },
  { title: "Appui à l'analyse organisationnelle", desc: "Les données d'historique permettent d'identifier les patterns et d'améliorer les processus." },
  { title: "Cadre en cas d'incident", desc: "La reconstitution du déroulement d'une mission est possible et documentée." },
];

const PARTNER_CAPS = [
  { title: 'Coordination multi-sites', desc: 'Un cadre unique pour gérer les transports sur plusieurs établissements ou unités.' },
  { title: 'Organisation par service ou unité', desc: "Accès différenciés et visibilité adaptée selon les rôles dans votre organisation." },
  { title: 'Multi-transporteurs partenaires', desc: 'Pilotage et suivi simultané de plusieurs prestataires de transport.' },
  { title: 'Supervision des missions en cours', desc: 'Vue en temps réel sur les courses actives, sans appels de vérification.' },
  { title: 'Déploiement progressif', desc: "Accompagnement à l'intégration adapté à votre rythme et à vos workflows." },
];

const FAQ_ITEMS = [
  {
    q: 'LIRIE remplace-t-elle nos transporteurs actuels ?',
    a: 'Non. La plateforme peut coordonner vos partenaires existants selon votre organisation. Les relations contractuelles avec vos transporteurs restent inchangées.',
  },
  {
    q: 'LIRIE est-elle un transporteur ?',
    a: 'Non. LIRIE fournit un outil de coordination entre institutions et entreprises de transport partenaires. Les prestations de transport sont réalisées par des entreprises indépendantes.',
  },
  {
    q: 'Peut-on travailler avec plusieurs transporteurs en parallèle ?',
    a: 'Oui. La coordination multi-transporteurs fait partie des usages centraux de la plateforme. LIRIE permet de piloter plusieurs prestataires depuis un cadre commun.',
  },
  {
    q: 'Qui accède aux informations de mission ?',
    a: "Les accès dépendent des rôles définis dans votre organisation. La plateforme permet une gestion différenciée des droits selon les services et les acteurs.",
  },
  {
    q: 'Est-ce adapté aux sorties hospitalières ?',
    a: "Oui. La plateforme peut être utilisée pour organiser et suivre différents types de transports planifiés — sorties hospitalières, retours à domicile, transferts inter-établissements — selon les partenaires disponibles.",
  },
  {
    q: 'Faut-il modifier nos processus internes ?',
    a: "Non. L'objectif est d'améliorer et de structurer la coordination existante, pas de la remplacer. La plateforme s'adapte à vos workflows, pas l'inverse.",
  },
];

const PILLARS = [
  { icon: <IconLayers />, title: 'Coordination centralisée', desc: "Un cadre commun pour l'ensemble des demandes, acteurs et missions." },
  { icon: <IconActivity />, title: 'Traçabilité opérationnelle', desc: 'Historique structuré des étapes, utile au pilotage et à la qualité.' },
  { icon: <IconUsers />, title: 'Multi-transporteurs', desc: 'Visibilité et pilotage sans rupture entre vos partenaires habituels.' },
];

function Hero() {
  return (
    <section className={styles.hero}>
      <div className={styles.heroDots} aria-hidden />
      <Shell>
        <div className={styles.heroOuter}>
          <div>
            <div className={styles.heroBadge}>
              <IconBuilding />
              <span className={styles.heroBadgeLabel}>Institutions &amp; coordination</span>
            </div>

            <div className={styles.heroInner}>
              <div>
                <h1 className={styles.heroTitle}>
                  La maîtrise organisationnelle
                  <br />
                  <span className={styles.heroTitleAccent}>de vos transports médicaux</span>
                </h1>

                <p className={styles.heroLead}>
                  LIRIE est une plateforme de coordination qui centralise les demandes, structure l&apos;assignation et apporte une
                  visibilité continue sur l&apos;ensemble des missions — de la demande à la dépose.
                </p>

                <p className={styles.heroSub}>
                  Conçue pour les hôpitaux, EMS, soins à domicile, services sociaux et institutions publiques qui coordonnent des
                  transports sensibles ou récurrents.
                </p>

                <div className={styles.heroCtas}>
                  <Link to="/contact/institution" className={styles.btnPrimary}>
                    Demander une présentation <IconArrow />
                  </Link>
                  <Link to="/contact/demo" className={styles.btnSecondary}>
                    Organiser une démonstration
                  </Link>
                </div>

                <div className={styles.heroQuick}>
                  <span className={styles.heroQuickLabel}>Voir aussi</span>
                  <Link to="/deplacez-vous" className={styles.heroQuickLink}>
                    Patients &amp; proches
                  </Link>
                  <span className={styles.heroQuickSep}>·</span>
                  <Link to="/conduire" className={styles.heroQuickLink}>
                    Entreprises de transport
                  </Link>
                </div>
              </div>

              <div className={styles.pillarStack}>
                {PILLARS.map((item) => (
                  <div key={item.title} className={styles.pillarCard}>
                    <div className={styles.pillarIcon}>{item.icon}</div>
                    <div>
                      <div className={styles.pillarTitle}>{item.title}</div>
                      <div className={styles.pillarDesc}>{item.desc}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </Shell>
    </section>
  );
}

function TrustBar() {
  return (
    <div className={styles.trust}>
      <Shell>
        <div className={styles.trustGrid}>
          {TRUST_ITEMS.map(({ Icon, label, sub }) => (
            <div key={label} className={styles.trustItem}>
              <div className={styles.trustIcon}>
                <Icon />
              </div>
              <div>
                <div className={styles.trustLabel}>{label}</div>
                <p className={styles.trustSub}>{sub}</p>
              </div>
            </div>
          ))}
        </div>
      </Shell>
    </div>
  );
}

function Constat() {
  return (
    <section id="constat" className={styles.sectionCream}>
      <Shell>
        <div className={styles.constatGrid}>
          <div>
            <Eyebrow>Constat opérationnel</Eyebrow>
            <h2 className={styles.h2}>
              Des transports essentiels,
              <br />
              une coordination encore fragmentée
            </h2>
            <p className={styles.lead16}>
              Dans de nombreuses structures, l&apos;organisation des transports repose encore sur des échanges multiples entre services,
              partenaires et prestataires — sans cadre partagé.
            </p>
            <div className={styles.callout}>
              <p>
                LIRIE propose un <strong>cadre de coordination commun</strong> pour structurer ces échanges sans alourdir les pratiques
                existantes.
              </p>
            </div>
          </div>

          <div>
            <div className={styles.painGrid}>
              {CONSTAT_POINTS.map((point) => (
                <div key={point} className={styles.painCard}>
                  <span className={styles.painDot}>
                    <span className={styles.painDotInner} />
                  </span>
                  <span className={styles.painText}>{point}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </Shell>
    </section>
  );
}

function ValeurOperationnelle() {
  return (
    <section id="apport" className={styles.sectionWhite}>
      <Shell>
        <div className={styles.introBlock}>
          <Eyebrow>Valeur opérationnelle</Eyebrow>
          <h2 className={`${styles.h2} ${styles.h2TightMb16}`}>Reprendre la maîtrise de l&apos;organisation transport</h2>
          <p className={styles.lead16} style={{ marginBottom: 0 }}>
            La promesse centrale n&apos;est pas «&nbsp;réservez facilement&nbsp;» — c&apos;est de redonner à vos équipes{' '}
            <strong>coordination, lisibilité et continuité</strong> sur l&apos;ensemble des missions.
          </p>
        </div>

        <div className={styles.valueGrid}>
          {VALUE_CARDS.map((item) => (
            <div key={item.title} className={styles.valueCard}>
              <div className={styles.valueIcon}>{item.icon}</div>
              <h3>{item.title}</h3>
              <p>{item.desc}</p>
            </div>
          ))}
        </div>
      </Shell>
    </section>
  );
}

function Fonctionnement() {
  return (
    <section id="fonctionnement" className={styles.sectionCream}>
      <Shell>
        <div className={styles.flowGrid}>
          <div className={styles.flowSticky}>
            <Eyebrow>Fonctionnement</Eyebrow>
            <h2 className={`${styles.h2} ${styles.h2Sm} ${styles.h2TightMb16}`}>Une logique simple pour des organisations complexes</h2>
            <p className={styles.lead15}>
              LIRIE n&apos;impose pas sa structure — elle aide à ordonner la vôtre, étape par étape.
            </p>
            <div className={`${styles.callout} ${styles.calloutSm}`}>
              <p>
                LIRIE agit comme <strong>couche de coordination</strong> entre les acteurs, sans se substituer aux responsabilités de
                chacun.
              </p>
            </div>
          </div>

          <div>
            {FLOW_STEPS.map((step, i) => (
              <div key={step.num} className={`${styles.flowStep}${i === FLOW_STEPS.length - 1 ? ` ${styles.flowStepLast}` : ''}`}>
                <div className={styles.flowRail}>
                  <div className={styles.flowNum}>{step.num}</div>
                  {i < FLOW_STEPS.length - 1 ? <div className={styles.flowConnector} aria-hidden /> : null}
                </div>
                <div className={`${styles.flowBody}${i < FLOW_STEPS.length - 1 ? ` ${styles.flowBodySpaced}` : ''}`}>
                  <h3>{step.title}</h3>
                  <p>{step.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </Shell>
    </section>
  );
}

function ActeursCibles() {
  return (
    <section id="acteurs" className={styles.sectionInk}>
      <Shell>
        <div className={styles.introBlock600}>
          <div className={styles.actIntroEyebrow}>
            <span className={styles.actIntroEyebrowLine} aria-hidden />
            <span className={styles.actIntroEyebrowLabel}>Cible institutionnelle</span>
          </div>
          <h2 className={`${styles.h2} ${styles.h2Light} ${styles.h2TightMb16}`}>
            Pensé pour les structures qui coordonnent des transports sensibles
          </h2>
          <p className={styles.leadOnDark}>
            La plateforme s&apos;adresse aux organisations qui gèrent des flux de transport récurrents, multi-acteurs et dont la
            traçabilité est un enjeu de qualité de service.
          </p>
        </div>

        <div className={styles.actorGrid}>
          {ACTORS.map(({ Icon, label, sub }) => (
            <div key={label} className={styles.actorCard}>
              <div className={styles.actorIcon} aria-hidden>
                <Icon />
              </div>
              <div>
                <div className={styles.actorLabel}>{label}</div>
                <div className={styles.actorSub}>{sub}</div>
              </div>
            </div>
          ))}
        </div>
      </Shell>
    </section>
  );
}

function BeneficesQuotidiens() {
  return (
    <section id="quotidien" className={styles.sectionWhite}>
      <Shell>
        <div className={styles.benefitGrid}>
          <div>
            <Eyebrow>Quotidien opérationnel</Eyebrow>
            <h2 className={`${styles.h2} ${styles.h2Md} ${styles.h2TightMb32}`}>Des bénéfices concrets pour vos équipes</h2>
            <ul className={styles.benefitList}>
              {BENEFITS.map((b) => (
                <li key={b} className={styles.benefitRow}>
                  <span className={styles.benefitCheck}>
                    <IconCheck />
                  </span>
                  <span className={styles.benefitText}>{b}</span>
                </li>
              ))}
            </ul>
          </div>

          <div className={styles.benefitAside}>
            <div className={styles.quoteCard}>
              <div className={styles.quoteBar} aria-hidden />
              <p className={styles.quoteText}>
                &ldquo;Moins de relances. Moins d&apos;incertitudes. Une meilleure continuité dans la coordination, même lorsque plusieurs
                prestataires interviennent sur un même parcours patient.&rdquo;
              </p>
              <p className={styles.quoteMeta}>Promesse opérationnelle LIRIE — coordination multi-acteurs</p>
            </div>
          </div>
        </div>
      </Shell>
    </section>
  );
}

function Tracabilite() {
  return (
    <section id="tracabilite" className={styles.sectionCream}>
      <Shell>
        <div className={styles.introBlock}>
          <Eyebrow>Pilotage &amp; traçabilité</Eyebrow>
          <h2 className={`${styles.h2} ${styles.h2TightMb16}`}>
            Une traçabilité utile à la coordination et à la qualité de service
          </h2>
          <p className={styles.lead16} style={{ marginBottom: 0 }}>
            LIRIE améliore la lisibilité des missions sans remplacer vos procédures internes. Nous privilégions le vocabulaire du{' '}
            <strong>pilotage</strong>, de la <strong>traçabilité</strong> et de la <strong>qualité de service</strong>.
          </p>
        </div>

        <div className={styles.traceGrid}>
          {TRACE_ITEMS.map((item) => (
            <div key={item.title} className={styles.traceCard}>
              <h3>{item.title}</h3>
              <p>{item.desc}</p>
            </div>
          ))}
        </div>
      </Shell>
    </section>
  );
}

function Partenaires() {
  return (
    <section id="partenaires" className={styles.sectionWhite}>
      <Shell>
        <div className={styles.partnerGrid}>
          <div>
            <Eyebrow>Adoption &amp; intégration</Eyebrow>
            <h2 className={`${styles.h2} ${styles.h2Md}`}>
              Vos partenaires existants,
              <br />
              dans un cadre partagé
            </h2>
            <p className={styles.lead16mb16}>
              LIRIE n&apos;impose pas de remplacer vos transporteurs habituels. Les relations contractuelles existantes restent inchangées.
            </p>
            <p className={styles.lead15mb32}>
              Selon votre organisation, la plateforme peut structurer la coordination avec vos partenaires actuels, faciliter le travail
              multi-transporteurs et renforcer la continuité d&apos;information entre services et prestataires.
            </p>
            <div className={`${styles.callout} ${styles.calloutPartner}`}>
              <p>
                <strong>Les relations contractuelles existantes restent inchangées.</strong>
                <br />
                LIRIE intervient comme couche de coordination logicielle, pas comme intermédiaire contractuel.
              </p>
            </div>
          </div>

          <div className={styles.capList}>
            {PARTNER_CAPS.map((cap) => (
              <div key={cap.title} className={styles.capRow}>
                <span className={styles.capBullet}>
                  <span className={styles.capBulletDot} />
                </span>
                <div>
                  <span className={styles.capTitle}>{cap.title}</span>
                  <span className={styles.capDesc}>{cap.desc}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </Shell>
    </section>
  );
}

function RoleLirie() {
  return (
    <section id="role-lirie" className={styles.roleSection}>
      <Shell>
        <div className={styles.roleInner}>
          <Eyebrow>Cadre juridique</Eyebrow>
          <div className={styles.roleCard}>
            <h2 className={styles.h2Role}>Une plateforme de coordination, pas un transporteur</h2>
            <p>
              LIRIE fournit une <strong>solution logicielle de coordination</strong> des transports. Les prestations de transport sont
              réalisées par des <strong>entreprises partenaires juridiquement indépendantes</strong>, conformément aux réglementations
              applicables.
            </p>
            <p>
              La plateforme facilite l&apos;organisation, le suivi et la circulation de l&apos;information entre les acteurs autorisés. Pour le
              cadre contractuel et légal, voir les{' '}
              <Link to="/conditions" className={styles.inlineLink}>
                conditions générales
              </Link>
              , les{' '}
              <Link to="/mentions-legales" className={styles.inlineLink}>
                mentions légales
              </Link>{' '}
              et la{' '}
              <Link to="/privacy" className={styles.inlineLink}>
                politique de confidentialité
              </Link>
              .
            </p>
          </div>
        </div>
      </Shell>
    </section>
  );
}

function MidCta() {
  return (
    <section id="demo-cta" className={styles.midCta}>
      <Shell>
        <div className={styles.midCtaGrid}>
          <div>
            <div className={styles.midEyebrow}>
              <span className={styles.midEyebrowLine} aria-hidden />
              <span className={styles.midEyebrowLabel}>Échange</span>
            </div>
            <h2 className={styles.midCtaTitle}>Voyons ensemble comment LIRIE peut s&apos;intégrer à votre organisation</h2>
            <p className={styles.midCtaLead}>
              Nous pouvons vous présenter le fonctionnement de la plateforme, les cas d&apos;usage et les modalités d&apos;intégration adaptées à
              votre structure.
            </p>
          </div>
          <div className={styles.midCtaBtns}>
            <Link to="/contact/demo" className={styles.midBtnSolid}>
              Organiser une démonstration <IconArrow />
            </Link>
            <Link to="/contact/institution" className={styles.midBtnGhost}>
              Demander une présentation
            </Link>
          </div>
        </div>
      </Shell>
    </section>
  );
}

function FAQ() {
  const [open, setOpen] = useState(null);

  return (
    <section id="faq" className={styles.sectionWhite}>
      <Shell>
        <div className={styles.faqGrid}>
          <div>
            <Eyebrow>FAQ</Eyebrow>
            <h2 className={`${styles.h2} ${styles.h2Md} ${styles.h2TightMb12}`}>Questions fréquentes</h2>
            <p className={styles.faqLead}>
              Réponses générales — le détail dépend de votre organisation, des conventions et de la configuration des accès.
            </p>
          </div>

          <div>
            {FAQ_ITEMS.map((item, i) => (
              <div key={item.q} className={styles.faqRow}>
                <button
                  type="button"
                  className={styles.faqTrigger}
                  aria-expanded={open === i}
                  onClick={() => setOpen(open === i ? null : i)}
                >
                  <span className={styles.faqQ}>{item.q}</span>
                  <span className={styles.faqChevWrap}>
                    <IconChevron open={open === i} />
                  </span>
                </button>
                {open === i ? (
                  <div className={styles.faqAnswer}>
                    <p>{item.a}</p>
                  </div>
                ) : null}
              </div>
            ))}
          </div>
        </div>
      </Shell>
    </section>
  );
}

function FinalCta() {
  return (
    <section className={styles.finalSection}>
      <Shell>
        <div className={styles.finalIntro}>
          <Eyebrow>Prochaine étape</Eyebrow>
          <h2 className={`${styles.h2} ${styles.h2TightMb16}`}>Discutons de votre organisation transport</h2>
          <p className={styles.lead16} style={{ marginBottom: 0 }}>
            Que vous cherchiez à structurer vos flux, améliorer la visibilité ou mieux coordonner plusieurs transporteurs, LIRIE peut vous
            aider à poser un cadre plus lisible et plus fluide.
          </p>
        </div>

        <div className={styles.finalCards}>
          <Link to="/contact/demo" className={styles.finalCard}>
            <div className={styles.finalCardKicker}>Démonstration</div>
            <span className={styles.finalCardTitle}>Organiser une démonstration</span>
            <p className={styles.finalCardDesc}>Parcours guidé selon vos enjeux et votre contexte institutionnel.</p>
            <span className={styles.finalCardCta}>
              Formulaire démo <IconArrow />
            </span>
          </Link>

          <Link to="/contact" className={styles.finalCard}>
            <div className={`${styles.finalCardKicker} ${styles.finalCardKickerMuted}`}>Contact</div>
            <span className={styles.finalCardTitle}>Contacter l&apos;équipe LIRIE</span>
            <p className={styles.finalCardDesc}>Pour toute question générale ou orientation vers le bon interlocuteur.</p>
            <span className={`${styles.finalCardCta} ${styles.finalCardCtaOutline}`}>
              Page contact <IconArrow />
            </span>
          </Link>

          <div className={`${styles.finalCard} ${styles.finalCardStatic}`}>
            <div className={`${styles.finalCardKicker} ${styles.finalCardKickerMuted}`}>Ressources</div>
            <span className={styles.finalCardTitle}>Explorer la plateforme</span>
            <p className={styles.finalCardDesc}>
              Découvrez les aspects techniques et organisationnels dans notre section aide.
            </p>
            <Link to="/aide" className={styles.finalCardLink}>
              Centre d&apos;aide <IconArrow />
            </Link>
          </div>
        </div>
      </Shell>
    </section>
  );
}

export default function ProfessionnelPage() {
  return (
    <div className={styles.page}>
      <main style={{ flex: 1 }}>
        <Hero />
        <TrustBar />
        <Constat />
        <ValeurOperationnelle />
        <Fonctionnement />
        <ActeursCibles />
        <BeneficesQuotidiens />
        <Tracabilite />
        <Partenaires />
        <RoleLirie />
        <MidCta />
        <FAQ />
        <FinalCta />
      </main>
      <div className={styles.bottomSpacer} aria-hidden />
    </div>
  );
}
