import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import styles from './TermsOfService.module.css';

const UPDATED_AT = '13 avril 2026';

function IcoFileText({ s = 20 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
      <polyline points="14 2 14 8 20 8" />
      <line x1="16" y1="13" x2="8" y2="13" />
      <line x1="16" y1="17" x2="8" y2="17" />
    </svg>
  );
}
function IcoBook({ s = 16 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
      <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" />
    </svg>
  );
}
function IcoLayers({ s = 16 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <polygon points="12 2 2 7 12 12 22 7 12 2" />
      <polyline points="2 17 12 22 22 17" />
      <polyline points="2 12 12 17 22 12" />
    </svg>
  );
}
function IcoShield({ s = 16 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
    </svg>
  );
}
function IcoLink2({ s = 16 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71" />
      <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71" />
    </svg>
  );
}
function IcoKey({ s = 16 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M21 2l-2 2m-7.61 7.61a5.5 5.5 0 1 1-7.778 7.778 5.5 5.5 0 0 1 7.777-7.777zm0 0L15.5 7.5m0 0l3 3L22 7l-3-3m-3.5 3.5L19 4" />
    </svg>
  );
}
function IcoUser({ s = 16 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
      <circle cx="12" cy="7" r="4" />
    </svg>
  );
}
function IcoTruck({ s = 16 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <rect x="1" y="3" width="15" height="13" />
      <polygon points="16 8 20 8 23 11 23 16 16 16 16 8" />
      <circle cx="5.5" cy="18.5" r="2.5" />
      <circle cx="18.5" cy="18.5" r="2.5" />
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
function IcoDollar({ s = 16 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <line x1="12" y1="1" x2="12" y2="23" />
      <path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6" />
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
function IcoXCircle({ s = 16 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <circle cx="12" cy="12" r="10" />
      <line x1="15" y1="9" x2="9" y2="15" />
      <line x1="9" y1="9" x2="15" y2="15" />
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
function IcoWifi({ s = 16 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M5 12.55a11 11 0 0 1 14.08 0" />
      <path d="M1.42 9a16 16 0 0 1 21.16 0" />
      <path d="M8.53 16.11a6 6 0 0 1 6.95 0" />
      <line x1="12" y1="20" x2="12.01" y2="20" />
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
function IcoAlertTriangle({ s = 16 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
      <line x1="12" y1="9" x2="12" y2="13" />
      <line x1="12" y1="17" x2="12.01" y2="17" />
    </svg>
  );
}
function IcoCloud({ s = 16 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M18 10h-1.26A8 8 0 1 0 9 20h9a5 5 0 0 0 0-10z" />
    </svg>
  );
}
function IcoCopyright({ s = 16 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <circle cx="12" cy="12" r="10" />
      <path d="M14.83 14.83A4 4 0 1 1 14.83 9.17" />
    </svg>
  );
}
function IcoSlash({ s = 16 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <circle cx="12" cy="12" r="10" />
      <line x1="4.93" y1="4.93" x2="19.07" y2="19.07" />
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
function IcoGavel({ s = 16 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M14 6l-1-2H5v17h2v-7h5.5l1 2H19V6h-5z" />
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

const SECTIONS = [
  { id: 'objet', num: 1, label: 'Objet', Icon: IcoFileText },
  { id: 'definitions', num: 2, label: 'Définitions', Icon: IcoBook },
  { id: 'champ-application', num: 3, label: "Champ d'application", Icon: IcoLayers },
  { id: 'description-plateforme', num: 4, label: 'Description de la plateforme', Icon: IcoLayers },
  { id: 'role-lirie', num: 5, label: 'Rôle de Lirie', Icon: IcoShield },
  { id: 'relations-contractuelles', num: 6, label: 'Relations contractuelles', Icon: IcoLink2 },
  { id: 'acces', num: 7, label: 'Accès à la plateforme', Icon: IcoKey },
  { id: 'comptes', num: 8, label: 'Comptes utilisateurs', Icon: IcoUser },
  { id: 'missions-transport', num: 9, label: 'Missions de transport', Icon: IcoTruck },
  { id: 'obligations-utilisateurs', num: 10, label: 'Obligations des utilisateurs', Icon: IcoUsers },
  { id: 'obligations-partenaires', num: 11, label: 'Obligations des partenaires', Icon: IcoTruck },
  { id: 'obligations-institutions', num: 12, label: 'Obligations des institutions', Icon: IcoUsers },
  { id: 'tarification', num: 13, label: 'Tarification', Icon: IcoDollar },
  { id: 'paiement', num: 14, label: 'Paiement', Icon: IcoCreditCard },
  { id: 'modification-annulation', num: 15, label: 'Modification et annulation', Icon: IcoXCircle },
  { id: 'donnees-tracabilite', num: 16, label: 'Données et traçabilité', Icon: IcoDatabase },
  { id: 'disponibilite', num: 17, label: 'Disponibilité du service', Icon: IcoWifi },
  { id: 'securite-usage', num: 18, label: 'Sécurité et usage conforme', Icon: IcoLock },
  { id: 'responsabilite', num: 19, label: 'Responsabilité', Icon: IcoAlertTriangle },
  { id: 'force-majeure', num: 20, label: 'Force majeure', Icon: IcoCloud },
  { id: 'propriete-intellectuelle', num: 21, label: 'Propriété intellectuelle', Icon: IcoCopyright },
  { id: 'suspension-resiliation', num: 22, label: 'Suspension et résiliation', Icon: IcoSlash },
  { id: 'modification-cgu', num: 23, label: 'Modification des CGU', Icon: IcoRefresh },
  { id: 'protection-donnees', num: 24, label: 'Protection des données', Icon: IcoShield },
  { id: 'droit-applicable', num: 25, label: 'Droit applicable', Icon: IcoGavel },
  { id: 'for-juridique', num: 26, label: 'For juridique', Icon: IcoMapPin },
  { id: 'contact', num: 27, label: 'Contact', Icon: IcoMail },
];

const DEFINITIONS = [
  {
    term: 'Plateforme',
    def: "L'application web, les interfaces mobiles et les services associés édités sous la dénomination Lirie Opérations.",
  },
  {
    term: 'Institution cliente',
    def: 'Organisme ou structure (public, privé ou associatif) qui organise ou commandite des missions de transport via la Plateforme ou en lien avec celle-ci.',
  },
  {
    term: 'Entreprise partenaire',
    def: 'Entreprise de transport ou exploitant habilité à exécuter des missions de transport.',
  },
  { term: 'Chauffeur', def: "Utilisateur professionnel assigné à l'exécution d'une mission." },
  { term: 'Mission', def: 'Prestation de transport planifiée, suivie ou documentée via la Plateforme.' },
  {
    term: 'Utilisateur',
    def: "Toute personne physique ou morale disposant d'un accès autorisé à la Plateforme (y compris administrateurs, dispatchers, contacts institutionnels ou, le cas échéant, bénéficiaires accédant à un portail dédié).",
  },
  {
    term: 'Lirie / nous',
    def: (
      <div className={styles.defRich}>
        <p className={styles.defRichLead}>
          La Plateforme et le service associés, édités et exploités par <strong>Drin Jasiqi</strong> dans le cadre du
          projet <strong>Lirie</strong>.
        </p>
        <p className={styles.defRichNote}>
          Pour l&apos;identification de l&apos;éditeur : voir les{' '}
          <Link to="/mentions-legales" className={styles.contactLink}>
            mentions légales
          </Link>
          .
        </p>
      </div>
    ),
  },
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

const TermsOfService = () => {
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
            <IcoFileText s={26} />
          </div>
          <div className={styles.heroBody}>
            <h1>Conditions générales d&apos;utilisation</h1>
            <div className={styles.heroMeta}>
              <span className={styles.badge}>CGU · Lirie Opérations</span>
              <p className={styles.subtitle}>Dernière mise à jour : {UPDATED_AT}</p>
            </div>
            <p className={styles.lead}>
              Les présentes conditions générales d&apos;utilisation («&nbsp;CGU&nbsp;») régissent l&apos;accès et
              l&apos;utilisation de la plateforme logicielle <strong>Lirie Opérations</strong>, éditée par{' '}
              <strong>Drin Jasiqi</strong>, exploitant du projet <strong>Lirie</strong>, en vue de la coordination de
              missions de transport, notamment à vocation sanitaire, médicalisée ou pour personnes à mobilité réduite
              (PMR), entre institutions, entreprises de transport partenaires et utilisateurs autorisés. Elles complètent
              les{' '}
              <Link to="/mentions-legales" className={styles.contactLink}>
                mentions légales
              </Link>{' '}
              et la{' '}
              <Link to="/privacy" className={styles.contactLink}>
                politique de confidentialité
              </Link>
              .
            </p>
            <div className={styles.heroLinks}>
              <Link to="/privacy" className={styles.heroLink}>
                <IcoShield s={14} />
                Politique de confidentialité
              </Link>
              <span className={styles.heroLinkSep}>·</span>
              <a href="mailto:info@lirie.ch" className={styles.heroLink}>
                <IcoMail s={14} />
                info@lirie.ch
              </a>
            </div>
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
                <a href="mailto:info@lirie.ch">info@lirie.ch</a>
              </div>
            </div>
          </aside>

          <article className={styles.article}>
            <section id="objet" className={styles.section}>
              <SectionHead num={1} title="Objet" Icon={IcoFileText} />
              <p>
                Les présentes CGU définissent les modalités d&apos;accès et d&apos;utilisation de la plateforme Lirie
                Opérations, solution logicielle de coordination permettant l&apos;organisation, l&apos;assignation, le
                suivi en temps réel et la communication opérationnelle entre institutions clientes, entreprises de
                transport partenaires, chauffeurs et autres utilisateurs habilités. Elles complètent, sans s&apos;y
                substituer, les contrats ou mandats conclus entre <strong>Drin Jasiqi</strong> (projet Lirie) et ses
                clients professionnels, ainsi qu&apos;entre les acteurs du transport pour l&apos;exécution des missions.
              </p>
              <div className={`${styles.callout} ${styles.calloutInfo}`}>
                Les CGU <strong>complètent</strong> les accords contractuels spécifiques ; elles ne les remplacent pas. En
                cas de divergence sur la relation commerciale exploitant–client professionnel, les stipulations écrites
                spécifiques prévalent dans la limite du droit impératif (voir encadré en fin de page).
              </div>
            </section>

            <section id="definitions" className={styles.section}>
              <SectionHead num={2} title="Définitions" Icon={IcoBook} />
              <p>Aux fins des présentes CGU, les termes suivants ont la signification suivante :</p>
              <div className={styles.defGrid}>
                {DEFINITIONS.map((row) => (
                  <div key={row.term} className={styles.defItem}>
                    <div className={styles.defTerm}>{row.term}</div>
                    <div className={styles.defDef}>{row.def}</div>
                  </div>
                ))}
              </div>
            </section>

            <section id="champ-application" className={styles.section}>
              <SectionHead num={3} title="Champ d'application" Icon={IcoLayers} />
              <p>Les présentes CGU s&apos;appliquent à tout Utilisateur de la Plateforme, notamment :</p>
              <ul className={styles.listPlain}>
                <li>institutions clientes et leurs collaborateurs habilités ;</li>
                <li>entreprises partenaires et leurs équipes (dispatch, gestion) ;</li>
                <li>chauffeurs et conducteurs autorisés ;</li>
                <li>administrateurs et personnel Lirie ;</li>
                <li>
                  utilisateurs de portails associés (par ex. réservation ou suivi pour un client final ou un
                  accompagnant), lorsque ces accès sont proposés.
                </li>
              </ul>
              <p>
                Selon les profils, certaines dispositions peuvent s&apos;appliquer de manière différenciée (obligations
                professionnelles, accès restreints). L&apos;utilisation de la Plateforme vaut acceptation des présentes
                CGU dans la mesure où elles sont opposables à la catégorie d&apos;Utilisateur concernée.
              </p>
            </section>

            <section id="description-plateforme" className={styles.section}>
              <SectionHead num={4} title="Description de la plateforme" Icon={IcoLayers} />
              <p>
                La Plateforme est une <strong>solution logicielle</strong> permettant notamment la planification, la
                coordination, l&apos;assignation, le suivi en temps réel des missions (dont suivi de position lorsque la
                fonction est activée), l&apos;historisation des statuts d&apos;exécution et la communication opérationnelle
                entre les acteurs. Elle peut inclure des modules de réservation, de paiement en ligne lorsqu&apos;ils sont
                activés, et d&apos;export ou de reporting selon configuration.
              </p>
              <div className={`${styles.callout} ${styles.calloutInfo}`}>
                Lirie <strong>n&apos;est pas une entreprise de transport</strong> au sens de l&apos;exécution physique des
                trajets sur la route ; l&apos;exécution des transports relève des entreprises partenaires et de leurs
                obligations réglementaires.
              </div>
            </section>

            <section id="role-lirie" className={styles.section}>
              <SectionHead num={5} title="Rôle de Lirie (non-transporteur)" Icon={IcoShield} />
              <div className={styles.roleCards}>
                <div className={`${styles.roleCard} ${styles.roleYes}`}>
                  <div className={styles.roleCardTitle}>Ce que couvre Lirie</div>
                  <ul className={`${styles.listPlain} ${styles.listCheck}`}>
                    <li>Outil technique de coordination et d&apos;organisation des missions</li>
                    <li>Mise à disposition et maintenance de la Plateforme</li>
                    <li>Support opérationnel des utilisateurs habilités</li>
                  </ul>
                </div>
                <div className={`${styles.roleCard} ${styles.roleNo}`}>
                  <div className={styles.roleCardTitle}>Ce que Lirie n&apos;assume pas</div>
                  <ul className={styles.listPlain}>
                    <li>Exécution des prestations de transport sur la route</li>
                    <li>Qualité de transporteur au sens réglementaire</li>
                    <li>Prise en charge médicale, diagnostic ou actes de soins</li>
                  </ul>
                </div>
              </div>
              <p>
                Les prestations de transport sont réalisées sous la responsabilité des entreprises partenaires, dans le
                respect de la réglementation qui leur est applicable. Toute question de nature sanitaire relève des
                professionnels de santé et des protocoles de l&apos;institution ou du transporteur concerné.
              </p>
            </section>

            <section id="relations-contractuelles" className={styles.section}>
              <SectionHead num={6} title="Relations contractuelles entre les parties" Icon={IcoLink2} />
              <p>
                Sauf stipulation contractuelle spécifique entre les parties concernées, la relation contractuelle relative
                à l&apos;<strong>exécution d&apos;une mission de transport</strong> est en principe établie entre
                l&apos;<strong>institution cliente</strong> (ou le commanditaire désigné) et l&apos;
                <strong>entreprise partenaire</strong> exécutant la mission. Lirie agit en qualité d&apos;
                <strong>opérateur technique</strong> de la Plateforme et, selon les cas, de prestataire de services
                numériques à l&apos;égard de ses clients contractuels (contrats-cadres, mandats ou conditions commerciales
                distinctes).
              </p>
              <p>
                Lorsqu&apos;une réservation ou une commande est passée par un <strong>utilisateur final</strong> (client,
                patient, accompagnant) via un parcours dédié, le contrat de transport ou les conditions applicables à la
                prestation sont ceux de l&apos;entreprise partenaire exécutante et/ou de l&apos;institution, selon les
                informations fournies au moment de la commande et les accords en vigueur.
              </p>
            </section>

            <section id="acces" className={styles.section}>
              <SectionHead num={7} title="Accès à la plateforme" Icon={IcoKey} />
              <p>
                L&apos;accès à la Plateforme est réservé aux Utilisateurs disposant d&apos;identifiants valides ou
                d&apos;un mécanisme d&apos;accès autorisé (invitation, lien sécurisé, authentification unique, etc.). Un
                équipement compatible, une connexion Internet et, pour les applications mobiles, un système
                d&apos;exploitation supporté sont nécessaires.
              </p>
              <div className={`${styles.callout} ${styles.calloutInfo}`}>
                L&apos;Utilisateur est responsable de la <strong>confidentialité</strong> de ses moyens
                d&apos;authentification et de toute activité effectuée depuis son compte, sauf preuve d&apos;une
                compromission imputable à l&apos;exploitant de la Plateforme.
              </div>
            </section>

            <section id="comptes" className={styles.section}>
              <SectionHead num={8} title="Comptes utilisateurs" Icon={IcoUser} />
              <p>
                L&apos;Utilisateur garantit l&apos;exactitude des informations fournies à l&apos;ouverture ou au maintien
                du compte. Il doit informer sans délai son administrateur interne ou Lirie en cas de changement de
                situation ou de suspicion d&apos;accès non autorisé.
              </p>
              <p>
                Lirie peut désactiver, restreindre ou supprimer un compte en cas de violation des présentes CGU, de
                risque de sécurité ou sur demande de l&apos;employeur / institution ayant délivré l&apos;accès. Les
                données associées peuvent faire l&apos;objet d&apos;archivage ou de suppression conformément à la{' '}
                <Link to="/privacy" className={styles.contactLink}>
                  politique de confidentialité
                </Link>{' '}
                et aux obligations légales.
              </p>
            </section>

            <section id="missions-transport" className={styles.section}>
              <SectionHead num={9} title="Gestion des missions de transport" Icon={IcoTruck} />
              <p>
                La Plateforme permet la création, la modification, l&apos;assignation et le suivi des missions. Sauf
                mention contraire dans l&apos;interface ou dans les accords entre les parties, une mission est considérée
                comme confirmée ou exécutable <strong>uniquement après acceptation ou validation</strong> par
                l&apos;entreprise partenaire habilitée.
              </p>
              <p>
                Les horaires, durées et itinéraires affichés peuvent être <strong>indicatifs</strong> et dépendre du
                trafic, des contraintes d&apos;accès ou des impératifs médicaux ou logistiques communiqués par les
                Utilisateurs. Lirie ne garantit pas un temps de parcours ou une disponibilité matérielle hors de son
                contrôle raisonnable.
              </p>
              <p>
                <strong>Localisation GPS</strong> : lorsqu&apos;elle est activée pour les chauffeurs, la localisation est
                utilisée à des fins de <strong>coordination opérationnelle</strong> (assignation, suivi, sécurité des
                missions), dans les limites prévues par la loi et la{' '}
                <Link to="/privacy" className={styles.contactLink}>
                  politique de confidentialité
                </Link>
                .
              </p>
            </section>

            <section id="obligations-utilisateurs" className={styles.section}>
              <SectionHead num={10} title="Obligations des utilisateurs" Icon={IcoUsers} />
              <p>L&apos;Utilisateur s&apos;engage notamment à :</p>
              <ul className={`${styles.listPlain} ${styles.listCheck}`}>
                <li>fournir des informations exactes, complètes et à jour nécessaires à l&apos;organisation des missions ;</li>
                <li>
                  utiliser la Plateforme conformément à sa finalité et aux instructions des responsables habilités ;
                </li>
                <li>respecter le personnel des entreprises partenaires et les règles de sécurité applicables ;</li>
                <li>
                  ne pas porter atteinte à la sécurité, à l&apos;intégrité des données ou au bon fonctionnement du
                  service ;
                </li>
                <li>respecter les droits des personnes concernées et la confidentialité des missions.</li>
              </ul>
            </section>

            <section id="obligations-partenaires" className={styles.section}>
              <SectionHead num={11} title="Obligations des entreprises partenaires" Icon={IcoTruck} />
              <p>
                Les entreprises partenaires demeurent <strong>seules responsables</strong> de la bonne exécution des
                missions de transport, du respect des autorisations, assurances, qualifications professionnelles, mise en
                conformité des véhicules et du respect du droit du travail applicable à leurs conducteurs.
              </p>
              <p>
                Elles veillent à ce que les chauffeurs disposent des habilitations nécessaires et à informer Lirie de tout
                élément susceptible d&apos;affecter la sécurité ou la conformité d&apos;une mission traitée via la
                Plateforme.
              </p>
            </section>

            <section id="obligations-institutions" className={styles.section}>
              <SectionHead num={12} title="Obligations des institutions clientes" Icon={IcoUsers} />
              <p>
                Les institutions clientes s&apos;engagent à transmettre des informations complètes, exactes et
                proportionnées pour chaque mission, notamment lorsqu&apos;existent des{' '}
                <strong>contraintes médicales ou logistiques</strong> (accès, horaires critiques, accompagnement requis,
                équipements). Elles sont responsables de la licéité et de la qualité des instructions données aux
                transporteurs via la Plateforme.
              </p>
              <p>
                Lorsque Lirie agit sur instruction documentée d&apos;une institution pour le traitement de données ou de
                missions, les rôles respectifs (responsable / sous-traitant) sont précisés dans les conventions
                applicables.
              </p>
            </section>

            <section id="tarification" className={styles.section}>
              <SectionHead num={13} title="Tarification" Icon={IcoDollar} />
              <p>
                Les conditions tarifaires et modalités de facturation applicables aux missions sont en principe définies
                dans les accords conclus entre l&apos;institution cliente et l&apos;entreprise partenaire (ou grilles
                contractuelles en vigueur). Les montants éventuellement affichés dans la Plateforme peuvent être{' '}
                <strong>indicatifs</strong> tant qu&apos;ils ne sont pas confirmés selon les règles métier ou contractuelles
                affichées à l&apos;Utilisateur.
              </p>
              <div className={`${styles.callout} ${styles.calloutInfo}`}>
                Les redevances liées à l&apos;abonnement ou à la licence d&apos;utilisation de la Plateforme par les
                professionnels font l&apos;objet de <strong>conditions commerciales distinctes</strong> entre l&apos;
                exploitant du projet Lirie et ses clients.
              </div>
            </section>

            <section id="paiement" className={styles.section}>
              <SectionHead num={14} title="Paiement" Icon={IcoCreditCard} />
              <p>
                Lorsque des paiements en ligne sont proposés, ils sont traités par un <strong>prestataire de paiement
                certifié</strong> (par exemple flux de type Saferpay dans l&apos;écosystème des solutions Worldline).
                Lirie n&apos;a pas accès aux numéros complets de carte bancaire ni au cryptogramme visuel. Les relations
                de paiement entre parties (client, institution, transporteur) suivent les termes affichés au moment de la
                transaction et les contrats applicables.
              </p>
            </section>

            <section id="modification-annulation" className={styles.section}>
              <SectionHead num={15} title="Modification et annulation des missions" Icon={IcoXCircle} />
              <p>
                Toute modification ou annulation d&apos;une mission doit être communiquée <strong>dès que possible</strong>{' '}
                via la Plateforme ou selon les canaux convenus entre les parties. Des frais, pénalités ou conditions
                particulières peuvent s&apos;appliquer conformément aux accords entre institution et entreprise
                partenaire, ou aux règles affichées dans l&apos;interface.
              </p>
              <p className={styles.note}>
                Lirie peut fournir des outils de notification ou d&apos;historisation, mais la gestion opérationnelle et
                financière des annulations relève en premier lieu des contractants de la mission.
              </p>
            </section>

            <section id="donnees-tracabilite" className={styles.section}>
              <SectionHead num={16} title="Données opérationnelles et traçabilité" Icon={IcoDatabase} />
              <p>
                La Plateforme enregistre des données opérationnelles (horodatages, statuts, localisations lorsque
                activées, échanges liés à la mission) afin d&apos;assurer la coordination, la <strong>traçabilité</strong>{' '}
                des interventions, le contrôle qualité et la résolution de litiges, dans les limites de la politique de
                confidentialité et de la loi.
              </p>
              <p>
                Ces données peuvent être utilisées comme éléments de preuve proportionnés en cas de contestation entre
                professionnels, sous réserve des droits des personnes concernées.
              </p>
            </section>

            <section id="disponibilite" className={styles.section}>
              <SectionHead num={17} title="Disponibilité du service" Icon={IcoWifi} />
              <p>
                Lirie s&apos;efforce d&apos;assurer une disponibilité élevée de la Plateforme. Toutefois, un accès{' '}
                <strong>ininterrompu</strong> ou exempt d&apos;erreurs ne peut être garanti. Des interruptions peuvent
                survenir notamment pour maintenance, mise à jour, incident technique, cas de force majeure ou
                défaillance de tiers (hébergeur, opérateur réseau).
              </p>
            </section>

            <section id="securite-usage" className={styles.section}>
              <SectionHead num={18} title="Sécurité et usage conforme" Icon={IcoLock} />
              <div className={`${styles.callout} ${styles.calloutWarn}`}>
                Toute tentative d&apos;accès non autorisé, d&apos;altération des données, de contournement des mesures de
                sécurité, de scraping abusif ou d&apos;utilisation de la Plateforme à des fins illicites ou détournées
                est <strong>interdite</strong>.
              </div>
              <p>
                Lirie peut analyser les journaux techniques et prendre les mesures nécessaires pour préserver
                l&apos;intégrité du service.
              </p>
            </section>

            <section id="responsabilite" className={styles.section}>
              <SectionHead num={19} title="Responsabilité" Icon={IcoAlertTriangle} />
              <p>
                Dans la limite autorisée par le droit applicable, la responsabilité de Lirie est{' '}
                <strong>limitée aux dommages directs</strong> prouvés résultant d&apos;une faute lourde ou intentionnelle
                dans la fourniture de la Plateforme. Lirie ne saurait être tenue responsable des retards, annulations,
                dommages ou incidents imputables aux entreprises partenaires, aux institutions, aux Utilisateurs
                (informations erronées ou incomplètes), aux conditions de circulation, météorologiques, ou à des
                circonstances indépendantes de la volonté raisonnable de Lirie.
              </p>
              <p className={styles.note}>
                Les limitations ci-dessus s&apos;appliquent dans le cadre légal, notamment lorsque le cocontractant est
                un consommateur au sens du Code des obligations suisses ou du droit de l&apos;Union européenne le cas
                échéant.
              </p>
            </section>

            <section id="force-majeure" className={styles.section}>
              <SectionHead num={20} title="Force majeure" Icon={IcoCloud} />
              <p>
                Lirie ne pourra être tenue responsable de l&apos;inexécution ou du retard dans l&apos;exécution de ses
                obligations résultant d&apos;un cas de force majeure au sens du droit applicable (notamment catastrophes,
                guerres, pandémies, grèves générales, décisions d&apos;autorités, pannes massives d&apos;Internet ou du
                cloud).
              </p>
            </section>

            <section id="propriete-intellectuelle" className={styles.section}>
              <SectionHead num={21} title="Propriété intellectuelle" Icon={IcoCopyright} />
              <p>
                Les éléments composant la Plateforme (logiciels, bases de données, design, marques, logos, documentation)
                sont protégés par les droits de propriété intellectuelle et demeurent la propriété de{' '}
                <strong>Drin Jasiqi</strong>, exploitant du projet <strong>Lirie</strong>, ou de ses concédants. Aucune
                licence autre que le droit d&apos;utiliser la Plateforme dans le cadre des présentes CGU n&apos;est
                concédée sans accord écrit.
              </p>
            </section>

            <section id="suspension-resiliation" className={styles.section}>
              <SectionHead num={22} title="Suspension et résiliation" Icon={IcoSlash} />
              <p>
                Lirie se réserve le droit de suspendre ou de résilier l&apos;accès à la Plateforme, ou certaines
                fonctionnalités, en cas de violation des présentes CGU, d&apos;usage abusif, de fraude, de risque pour la
                sécurité ou sur exigence légale. Les clients professionnels sont en outre soumis aux modalités de
                résiliation de leurs contrats cadres.
              </p>
            </section>

            <section id="modification-cgu" className={styles.section}>
              <SectionHead num={23} title="Modification des CGU" Icon={IcoRefresh} />
              <p>
                Lirie peut modifier les présentes CGU pour tenir compte de l&apos;évolution du service ou du cadre légal.
                Les modifications <strong>substantielles</strong> seront portées à la connaissance des Utilisateurs par
                des moyens appropriés (notification dans l&apos;application, courriel aux contacts enregistrés, ou avis
                sur le site). La poursuite de l&apos;utilisation après notification peut valoir acceptation lorsque la
                loi le permet ; pour les comptes institutionnels, des voies contractuelles spécifiques peuvent
                s&apos;appliquer.
              </p>
            </section>

            <section id="protection-donnees" className={styles.section}>
              <SectionHead num={24} title="Protection des données" Icon={IcoShield} />
              <p>
                Le traitement des données à caractère personnel est décrit dans la{' '}
                <Link to="/privacy" className={styles.contactLink}>
                  Politique de confidentialité
                </Link>
                .
              </p>
              <div className={`${styles.callout} ${styles.calloutBrand}`}>
                En cas de contradiction apparente entre les CGU et la Politique de confidentialité sur les données
                personnelles, la <strong>Politique de confidentialité prévaut</strong> pour ce seul objet.
              </div>
            </section>

            <section id="droit-applicable" className={styles.section}>
              <SectionHead num={25} title="Droit applicable" Icon={IcoGavel} />
              <p>
                Les présentes CGU sont régies par le <strong>droit matériel suisse</strong>, à l&apos;exclusion de ses
                règles de conflit de lois. Pour les Utilisateurs soumis à des dispositions impératives d&apos;un autre
                État (notamment au sein de l&apos;EEE), ces dispositions demeurent réservées dans la mesure où elles
                s&apos;imposent.
              </p>
            </section>

            <section id="for-juridique" className={styles.section}>
              <SectionHead num={26} title="For juridique" Icon={IcoMapPin} />
              <p>
                Sauf disposition légale impérative contraire, tout litige relatif aux présentes CGU ou à l&apos;utilisation
                de la Plateforme sera soumis aux <strong>tribunaux ordinaires du Canton de Genève</strong>, siège de
                l&apos;exploitant du projet Lirie, ou, pour les litiges relevant de la compétence matérielle, au Tribunal
                cantonal de Genève.
              </p>
            </section>

            <section id="contact" className={`${styles.section} ${styles.sectionLast}`}>
              <SectionHead num={27} title="Contact" Icon={IcoMail} />
              <p>
                Pour toute question relative aux présentes CGU ou à la page{' '}
                <Link to="/contact" className={styles.contactLink}>
                  Contact
                </Link>
                .
              </p>
              <div className={styles.contactGrid}>
                <a href="mailto:info@lirie.ch" className={styles.contactCard}>
                  <div className={styles.contactCardIcon}>
                    <IcoMail s={18} />
                  </div>
                  <div>
                    <div className={styles.contactCardTitle}>Questions générales</div>
                    <div className={styles.contactCardAddr}>info@lirie.ch</div>
                  </div>
                </a>
                <a href="mailto:privacy@lirie.ch" className={styles.contactCard}>
                  <div className={styles.contactCardIcon}>
                    <IcoShield s={18} />
                  </div>
                  <div>
                    <div className={styles.contactCardTitle}>Données personnelles</div>
                    <div className={styles.contactCardAddr}>privacy@lirie.ch</div>
                  </div>
                </a>
              </div>
              <p className={styles.note} style={{ marginTop: '0.75rem' }}>
                <strong>Drin Jasiqi</strong> — exploitant du projet Lirie — Avenue Ernest-Pictet 9, 1203 Genève, Suisse.
              </p>
            </section>

            <div className={styles.disclaimer}>
              <IcoFileText s={14} />
              <span>
                Les présentes CGU encadrent l&apos;usage de la Plateforme Lirie Opérations. Elles ne remplacent pas les
                contrats de transport, mandats, conventions-cadres ou conditions particulières conclus entre institutions,
                transporteurs et l&apos;exploitant du projet Lirie. En cas de divergence, les stipulations contractuelles
                spécifiques conclues par écrit entre l&apos;exploitant et un client professionnel prévalent pour ce qui
                concerne la relation commerciale exploitant–client, dans la limite du droit impératif.
              </span>
            </div>

            <div className={styles.footerStamp}>
              <IcoGavel s={14} />
              <span>Droit suisse applicable · For : tribunaux du Canton de Genève · CGU — {UPDATED_AT}</span>
            </div>
          </article>
        </div>
      </div>
    </div>
  );
};

export default TermsOfService;
