import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import styles from './LegalNotice.module.css';

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
function IcoInfo({ s = 16 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <circle cx="12" cy="12" r="10" />
      <line x1="12" y1="8" x2="12" y2="12" />
      <line x1="12" y1="16" x2="12.01" y2="16" />
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
function IcoServer({ s = 16 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <rect x="2" y="2" width="20" height="8" rx="2" />
      <rect x="2" y="14" width="20" height="8" rx="2" />
      <line x1="6" y1="6" x2="6.01" y2="6" />
      <line x1="6" y1="18" x2="6.01" y2="18" />
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
function IcoShield({ s = 16 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
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
function IcoAlertTriangle({ s = 16 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
      <line x1="12" y1="9" x2="12" y2="13" />
      <line x1="12" y1="17" x2="12.01" y2="17" />
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
function IcoGavel({ s = 16 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M14 6l-1-2H5v17h2v-7h5.5l1 2H19V6h-5z" />
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
function IcoBook({ s = 16 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
      <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" />
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
  { id: 'editeur', num: 1, label: 'Éditeur de la plateforme', Icon: IcoInfo },
  { id: 'responsable-publication', num: 2, label: 'Responsable de la publication', Icon: IcoUser },
  { id: 'hebergement', num: 3, label: 'Hébergement et infrastructure', Icon: IcoServer },
  { id: 'acces-plateforme', num: 4, label: 'Accès et contenus', Icon: IcoGlobe },
  { id: 'role-lirie', num: 5, label: 'Rôle de Lirie', Icon: IcoShield },
  { id: 'propriete-intellectuelle', num: 6, label: 'Propriété intellectuelle', Icon: IcoCopyright },
  { id: 'responsabilite', num: 7, label: 'Responsabilité', Icon: IcoAlertTriangle },
  { id: 'liens-externes', num: 8, label: 'Liens externes', Icon: IcoLink2 },
  { id: 'protection-donnees', num: 9, label: 'Protection des données', Icon: IcoShield },
  { id: 'disponibilite', num: 10, label: 'Disponibilité du service', Icon: IcoWifi },
  { id: 'droit-applicable', num: 11, label: 'Droit applicable', Icon: IcoGavel },
  { id: 'contact', num: 12, label: 'Contact', Icon: IcoMail },
  { id: 'version-langue', num: 13, label: 'Version du document et langue', Icon: IcoBook },
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

const LegalNotice = () => {
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
            <h1>Mentions légales</h1>
            <div className={styles.heroMeta}>
              <span className={styles.badge}>Lirie · Genève</span>
              <p className={styles.subtitle}>Dernière mise à jour : {UPDATED_AT}</p>
            </div>
            <p className={styles.lead}>
              Les présentes mentions légales identifient l&apos;éditeur de la plateforme <strong>Lirie</strong>, ses
              coordonnées et les informations essentielles relatives à l&apos;exploitation du service (hébergement,
              propriété intellectuelle, responsabilité). Elles complètent les{' '}
              <Link to="/conditions" className={styles.contactLink}>
                Conditions générales d&apos;utilisation
              </Link>{' '}
              et la{' '}
              <Link to="/privacy" className={styles.contactLink}>
                Politique de confidentialité
              </Link>
              , sans s&apos;y substituer.
            </p>
            <div className={styles.heroLinks}>
              <Link to="/conditions" className={styles.heroLink}>
                <IcoFileText s={14} />
                CGU
              </Link>
              <span className={styles.heroLinkSep}>·</span>
              <Link to="/privacy" className={styles.heroLink}>
                <IcoShield s={14} />
                Confidentialité
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
            <section id="editeur" className={styles.section}>
              <SectionHead num={1} title="Éditeur de la plateforme" Icon={IcoInfo} />
              <div className={styles.identityCard}>
                <div className={styles.identityRow}>
                  <strong>Drin Jasiqi</strong>
                </div>
                <div className={styles.identityRow}>
                  Exploitant du projet <strong>Lirie</strong> — plateforme logicielle de coordination de missions de
                  transport
                </div>
                <div className={styles.identityRow}>
                  Avenue Ernest-Pictet 9
                  <br />
                  1203 Genève — Suisse
                </div>
                <div className={styles.identityRow}>
                  Courriel :{' '}
                  <a href="mailto:info@lirie.ch" className={styles.contactLink}>
                    info@lirie.ch
                  </a>
                </div>
              </div>
              <p>
                À la date de mise à jour de cette page, l&apos;exploitation de <strong>Lirie</strong> n&apos;est pas
                structurée sous la forme d&apos;une <strong>société immatriculée au registre du commerce</strong> du canton
                de Genève. Aucune dénomination de personne morale ni numéro d&apos;<strong>IDE / UID</strong> de société
                n&apos;est donc indiqué ici. Les présentes mentions seront actualisées si la forme d&apos;exploitation ou
                l&apos;immatriculation évoluent.
              </p>
            </section>

            <section id="responsable-publication" className={styles.section}>
              <SectionHead num={2} title="Responsable de la publication" Icon={IcoUser} />
              <p>
                Responsable de la publication au sens des contenus édités sur la plateforme et le site :{' '}
                <strong>Drin Jasiqi</strong>, exploitant du projet <strong>Lirie</strong>.
              </p>
              <p>
                Pour toute question relative aux contenus ou à leur retrait :{' '}
                <a href="mailto:info@lirie.ch" className={styles.contactLink}>
                  info@lirie.ch
                </a>
                .
              </p>
            </section>

            <section id="hebergement" className={styles.section}>
              <SectionHead num={3} title="Hébergement et infrastructure" Icon={IcoServer} />
              <p>
                Les services <strong>Lirie</strong> (applications, données et traitements associés) sont hébergés et
                exploités via des <strong>prestataires techniques</strong> situés en Europe, choisis pour des niveaux de
                sécurité et de disponibilité conformes aux usages du secteur. La liste nominative des hébergeurs ou
                sous-traitants d&apos;infrastructure peut être précisée dans la documentation contractuelle ou
                communiquée sur demande raisonnable à{' '}
                <a href="mailto:privacy@lirie.ch" className={styles.contactLink}>
                  privacy@lirie.ch
                </a>
                , dans le respect des obligations de confidentialité et de sécurité.
              </p>
              <p>
                Pour le traitement des données à caractère personnel, se référer également à la{' '}
                <Link to="/privacy" className={styles.contactLink}>
                  Politique de confidentialité
                </Link>{' '}
                (transferts, sous-traitants techniques, etc.).
              </p>
              <p>
                Certains prestataires techniques peuvent traiter ou héberger des données en dehors de la Suisse ou de{' '}
                <strong>l&apos;Espace économique européen (EEE)</strong>. Le cas échéant, Lirie veille à ce que des
                garanties appropriées soient mises en place conformément à la réglementation applicable (notamment la LPD
                suisse et, le cas échéant, le RGPD), comme détaillé dans la Politique de confidentialité.
              </p>
            </section>

            <section id="acces-plateforme" className={styles.section}>
              <SectionHead num={4} title="Accès à la plateforme et exactitude des informations" Icon={IcoGlobe} />
              <p>
                Lirie s&apos;efforce d&apos;assurer l&apos;<strong>exactitude</strong> et l&apos;
                <strong>actualisation</strong> des informations affichées sur la plateforme (statuts de mission, horaires
                indicatifs, contenus d&apos;aide). Toutefois, des erreurs, retards de synchronisation ou interruptions
                temporaires peuvent survenir (maintenance, dépendance à des tiers, réseaux, intégrations partenaires).
              </p>
              <p>
                Les informations de dispatch ou de suivi sont fournies à titre <strong>opérationnel</strong> et peuvent
                être mises à jour en temps réel ; elles ne dispensent pas les utilisateurs professionnels de vérifier les
                données critiques sur le terrain et dans leurs propres systèmes lorsque la loi ou les protocoles
                l&apos;exigent.
              </p>

              <h3 className={styles.subheading}>Usage professionnel prioritaire</h3>
              <p>
                La plateforme <strong>Lirie</strong> est principalement destinée à un <strong>usage professionnel</strong>{' '}
                par des institutions, entreprises de transport, chauffeurs habilités et administrateurs. Les parcours
                ouverts à des utilisateurs finaux (réservation, suivi) le sont dans le cadre défini par les professionnels
                concernés.
              </p>
              <p>
                Certaines fonctionnalités sont <strong>réservées aux utilisateurs professionnels autorisés</strong>{' '}
                (comptes institutionnels, entreprises de transport, chauffeurs habilités, administrateurs).
              </p>

              <h3 className={styles.subheading}>Services techniques tiers</h3>
              <p>
                Certaines fonctionnalités reposent sur des <strong>services tiers</strong> (cartographie, itinéraires,
                notifications push, infrastructure cloud, outils de supervision). Leur indisponibilité, dégradation ou
                modification par le prestataire peut affecter <strong>temporairement</strong> tout ou partie de la
                plateforme sans ouvrir droit à indemnisation à l&apos;encontre de Lirie, dans les limites du droit
                applicable.
              </p>

              <h3 className={styles.subheading}>Sécurité des accès</h3>
              <p>
                Lirie met en œuvre des mesures de sécurité <strong>raisonnables</strong> pour protéger l&apos;accès à la
                plateforme et l&apos;intégrité des données, compte tenu de l&apos;état de la technique. Elle ne peut
                toutefois pas garantir une absence totale de risques inhérents à l&apos;utilisation d&apos;Internet ou à
                des attaques externes.
              </p>
            </section>

            <section id="role-lirie" className={styles.section}>
              <SectionHead num={5} title="Rôle de Lirie (plateforme de coordination)" Icon={IcoShield} />
              <p>
                Lirie fournit une <strong>solution logicielle de coordination</strong> des missions de transport
                (planification, assignation, suivi, communication) entre institutions clientes, entreprises partenaires et
                chauffeurs autorisés.
              </p>
              <div className={`${styles.callout} ${styles.calloutInfo}`}>
                Lirie <strong>n&apos;est pas un transporteur</strong> et <strong>n&apos;exécute pas</strong> elle-même les
                prestations de transport sur la voie publique. Lirie{' '}
                <strong>n&apos;intervient pas en qualité de prestataire de soins</strong> et ne participe pas à la prise en
                charge médicale des personnes transportées.
              </div>
              <p>
                Les informations affichées ou saisies dans le cadre des missions (y compris indications liées à la
                mobilité ou au contexte sanitaire){' '}
                <strong>ne constituent en aucun cas un avis médical</strong>, une recommandation thérapeutique ou une
                prescription ; elles servent uniquement à l&apos;organisation logistique des transports.
              </p>

              <h3 className={styles.subheading}>Relations contractuelles relatives aux transports</h3>
              <p>
                Les relations contractuelles relatives à l&apos;<strong>exécution des prestations de transport</strong> sont
                en principe conclues <strong>directement</strong> entre les institutions clientes, les utilisateurs
                concernés et les entreprises de transport partenaires, selon leurs accords respectifs (conditions
                tarifaires, assurance, responsabilité, etc.). Lirie intervient{' '}
                <strong>exclusivement comme opérateur technique de coordination</strong> et n&apos;est pas partie aux
                contrats de transport sauf stipulation expresse distincte.
              </p>

              <h3 className={styles.subheading}>Traçabilité des actions sur la plateforme</h3>
              <p>
                Certaines actions réalisées sur la plateforme (création ou modification de mission, validation
                d&apos;étapes, horodatages opérationnels, connexions techniques utiles au diagnostic) peuvent être{' '}
                <strong>enregistrées</strong> à des fins de <strong>sécurité</strong>, de <strong>traçabilité</strong>, de
                prévention des abus et d&apos;amélioration du service, dans les limites prévues par la{' '}
                <Link to="/privacy" className={styles.contactLink}>
                  Politique de confidentialité
                </Link>
                .
              </p>
            </section>

            <section id="propriete-intellectuelle" className={styles.section}>
              <SectionHead num={6} title="Propriété intellectuelle" Icon={IcoCopyright} />
              <p>
                L&apos;ensemble des éléments composant la plateforme <strong>Lirie</strong> (logiciels, interfaces, charte
                graphique, textes, bases de données, marques et logos «&nbsp;Lirie&nbsp;» lorsqu&apos;ils sont protégés)
                est protégé par le droit de la propriété intellectuelle et demeure, sauf mention contraire, la{' '}
                <strong>propriété exclusive de l&apos;exploitant du projet Lirie</strong> ou de ses concédants.
              </p>
              <p>
                Toute reproduction, représentation, modification ou exploitation non autorisée est interdite sous réserve
                des exceptions légales (copie privée, citation, etc.) et des droits expressément concédés par contrat.
              </p>
            </section>

            <section id="responsabilite" className={styles.section}>
              <SectionHead num={7} title="Responsabilité" Icon={IcoAlertTriangle} />
              <p>
                Lirie met à disposition un <strong>outil technique</strong>. Les prestations de transport sont réalisées
                par des <strong>entreprises partenaires juridiquement indépendantes</strong>, sous leur propre
                responsabilité réglementaire et contractuelle.
              </p>
              <p>
                Lirie ne saurait être tenue responsable de l&apos;<strong>exécution</strong> des transports (retards,
                annulations, incidents de parcours, litiges entre passagers et transporteurs), ni des contenus ou
                instructions saisis par les utilisateurs sur la plateforme, sous réserve des limitations légales
                impératives applicables.
              </p>
            </section>

            <section id="liens-externes" className={styles.section}>
              <SectionHead num={8} title="Liens externes" Icon={IcoLink2} />
              <p>
                La plateforme ou le site peuvent contenir des liens vers des sites tiers (prestataires, autorités,
                documentation). Lirie n&apos;exerce <strong>aucun contrôle</strong> sur le contenu de ces sites et décline
                toute responsabilité quant à leur accessibilité, leur exactitude ou leur politique de données.
                L&apos;activation d&apos;un lien externe est sous la responsabilité de l&apos;utilisateur.
              </p>
            </section>

            <section id="protection-donnees" className={styles.section}>
              <SectionHead num={9} title="Protection des données" Icon={IcoShield} />
              <div className={`${styles.callout} ${styles.calloutBrand}`}>
                Le traitement des données à caractère personnel est décrit dans la{' '}
                <Link to="/privacy" className={styles.contactLink}>
                  Politique de confidentialité
                </Link>
                . Pour toute demande relative aux données :{' '}
                <a href="mailto:privacy@lirie.ch" className={styles.contactLink}>
                  privacy@lirie.ch
                </a>
                .
              </div>
            </section>

            <section id="disponibilite" className={styles.section}>
              <SectionHead num={10} title="Disponibilité du service" Icon={IcoWifi} />
              <p>
                Lirie s&apos;efforce d&apos;assurer une disponibilité élevée de la plateforme. Lirie se réserve toutefois
                le droit d&apos;<strong>interrompre temporairement</strong> l&apos;accès pour maintenance, mise à jour,
                évolution technique ou mesures de sécurité, sans que cela n&apos;ouvre systématiquement droit à
                indemnisation, dans les limites du droit applicable.
              </p>
              <p>
                Lirie <strong>ne garantit pas</strong> la disponibilité immédiate d&apos;un transporteur ni{' '}
                <strong>l&apos;exécution effective</strong> d&apos;une mission planifiée : celles-ci dépendent des
                entreprises partenaires, des créneaux, du trafic, des contraintes médicales ou logistiques et des
                validations opérationnelles, indépendamment du bon fonctionnement technique de l&apos;outil.
              </p>
            </section>

            <section id="droit-applicable" className={styles.section}>
              <SectionHead num={11} title="Droit applicable" Icon={IcoGavel} />
              <p>
                Les présentes mentions légales sont régies par le <strong>droit matériel suisse</strong>, à l&apos;exclusion
                de ses règles de conflit de lois. Les dispositions impératives d&apos;un autre État demeurent réservées
                lorsqu&apos;elles s&apos;imposent à Lirie ou aux utilisateurs concernés.
              </p>
            </section>

            <section id="contact" className={styles.section}>
              <SectionHead num={12} title="Contact" Icon={IcoMail} />
              <div className={styles.contactGrid}>
                <a href="mailto:info@lirie.ch" className={styles.contactCard}>
                  <div className={styles.contactCardIcon} aria-hidden>
                    <IcoMail s={18} />
                  </div>
                  <div>
                    <div className={styles.contactCardTitle}>Questions générales</div>
                    <div className={styles.contactCardAddr}>info@lirie.ch</div>
                  </div>
                </a>
                <a href="mailto:privacy@lirie.ch" className={styles.contactCard}>
                  <div className={styles.contactCardIcon} aria-hidden>
                    <IcoShield s={18} />
                  </div>
                  <div>
                    <div className={styles.contactCardTitle}>Données personnelles</div>
                    <div className={styles.contactCardAddr}>privacy@lirie.ch</div>
                  </div>
                </a>
              </div>
              <p className={`${styles.note} ${styles.noteAfterCards}`}>
                <strong>Drin Jasiqi</strong> — projet <strong>Lirie</strong>, Avenue Ernest-Pictet 9, 1203 Genève, Suisse.
                Formulaire :{' '}
                <Link to="/contact" className={styles.contactLink}>
                  page Contact
                </Link>
                .
              </p>
            </section>

            <section id="version-langue" className={`${styles.section} ${styles.sectionLast}`}>
              <SectionHead num={13} title="Version du document et langue de référence" Icon={IcoBook} />
              <p>
                La version des présentes mentions légales <strong>applicable</strong> est celle publiée en ligne à la date
                de consultation, telle qu&apos;identifiée par la mention «&nbsp;Dernière mise à jour&nbsp;» figurant en tête
                de page.
              </p>
              <p>
                En cas de traduction dans d&apos;autres langues, la <strong>version française</strong> prévaut en cas
                d&apos;écart ou d&apos;ambiguïté d&apos;interprétation, sous réserve des dispositions impératives du droit
                applicable.
              </p>
            </section>

            <div className={styles.disclaimer}>
              <IcoInfo s={14} />
              <span>
                Les présentes mentions légales ont pour objet l&apos;information du public et la transparence sur
                l&apos;exploitation de la plateforme <strong>Lirie</strong>. Elles ne constituent pas un conseil juridique.
                Toute évolution de la forme d&apos;exploitation, d&apos;immatriculation ou d&apos;identification devant
                figurer en ligne devra se refléter dans une mise à jour de cette page.
              </span>
            </div>

            <div className={styles.footerStamp}>
              <IcoGavel s={14} />
              Droit suisse applicable · Genève, Suisse · Mentions légales v2026-04-13
            </div>
          </article>
        </div>
      </div>
    </div>
  );
};

export default LegalNotice;
