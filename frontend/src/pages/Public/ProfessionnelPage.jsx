import React, { useId, useState } from 'react';
import { Link } from 'react-router-dom';
import styles from './ProfessionnelPage.module.css';

function blockImageSave(event) {
  event.preventDefault();
}

function IcoChevR({ s = 14 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <polyline points="9 18 15 12 9 6" />
    </svg>
  );
}

const TODAY_FRICTION = [
  'Un appel pour confirmer le départ',
  'Un mail qui croise un autre canal',
  'Deux transporteurs, aucune vue commune',
  'Des relances pour savoir où en est la mission',
];

const AFTER_FLOW = [
  'Une demande, un seul point d’entrée',
  'La mission reste lisible pour chaque rôle',
  'Institution et entreprise voient la même étape',
  'Moins de charge mentale, plus de maîtrise',
];

const GAINS = [
  {
    role: 'Institution',
    items: [
      'Savoir où en est le patient',
      'Moins de relances entre services',
      'Historique consultable selon les droits',
      'Plusieurs transporteurs, un même repère',
    ],
  },
  {
    role: 'Entreprise',
    items: [
      'Recevoir une mission claire',
      'Consignes utiles avant le départ',
      'Moins d’allers-retours d’information',
      'Un cadre stable avec les institutions',
    ],
  },
  {
    role: 'Patient',
    items: [
      'Trajet confirmé',
      'Informations au bon moment',
      'Moins d’incertitude sur le départ',
      'Continuité entre les interlocuteurs',
    ],
  },
];

const CONVERGENCE = [
  { role: 'Institution', text: 'Patient pris en charge' },
  { role: 'Entreprise', text: 'Mission claire' },
  { role: 'Patient', text: 'Trajet confirmé' },
];

const FAQ = [
  {
    q: 'Lirie remplace-t-elle nos transporteurs ?',
    a: 'Non. Lirie accompagne la mission : vos relations contractuelles avec vos transporteurs restent inchangées.',
  },
  {
    q: 'Lirie est-elle un transporteur ?',
    a: 'Non. L’exécution sur la voie publique relève des entreprises partenaires. Lirie rend la mission lisible pour tous les acteurs habilités.',
  },
  {
    q: 'Une entreprise peut-elle rejoindre le réseau ?',
    a: 'Oui. Le parcours partenaire couvre les critères, la configuration des accès et l’activation des missions.',
  },
  {
    q: 'Faut-il modifier nos processus internes ?',
    a: 'Non. Chacun garde son rôle. L’objectif est de rendre la mission lisible, pas de remplacer votre organisation.',
  },
  {
    q: 'Qui voit quoi sur une mission ?',
    a: 'Les accès dépendent des rôles définis. Institution, entreprise et patient ne voient que ce qui leur est utile.',
  },
];

const ProfessionnelPage = () => {
  const [openFaq, setOpenFaq] = useState(null);
  const faqBaseId = useId();

  return (
    <div className={styles.page}>
      {/* ── Hero — structure / gouvernance (≠ Déplacez-vous émotion) ── */}
      <header className={styles.hero}>
        <div className={styles.shell}>
          <div className={styles.heroVisual}>
            <div className={styles.heroFrame} onContextMenu={blockImageSave}>
              <img
                src="/images/lirie-coordination-institution-transporteur.webp"
                alt="Coordination d’une mission entre une institution et un transporteur."
                className={styles.heroImg}
                width={1448}
                height={1086}
                decoding="async"
                fetchPriority="high"
                draggable={false}
                onDragStart={blockImageSave}
                onContextMenu={blockImageSave}
              />
              <span className={styles.imgShield} aria-hidden="true" onContextMenu={blockImageSave} />
            </div>
          </div>
          <div className={styles.heroInner}>
            <p className={styles.heroMeta}>Institutions · Entreprises de transport</p>
            <h1 className={styles.heroTitle}>
              <span className={styles.heroLine}>Vos transports ne sont pas complexes.</span>
              <span className={`${styles.heroLine} ${styles.heroLineAccent}`}>Leur coordination l’est.</span>
            </h1>
            <p className={styles.heroLead}>
              Une mission traverse plusieurs organisations. Lirie permet de la garder lisible pour chaque acteur — sans
              devenir le centre de l’histoire.
            </p>
            <div className={styles.heroCtas}>
              <Link to="/contact/demo" className={styles.btnPrimary}>
                Organiser une démonstration
                <IcoChevR s={14} />
              </Link>
              <Link to="/contact/transport" className={styles.heroLinkSecondary}>
                Parcours entreprise
              </Link>
            </div>
          </div>
        </div>
      </header>

      {/* ── Pourquoi est-ce compliqué ? ── */}
      <section id="friction" className={styles.actFriction} aria-labelledby="friction-heading">
        <div className={styles.shell}>
          <h2 id="friction-heading" className={styles.actTitle}>
            Pourquoi est-ce compliqué ?
          </h2>
          <p className={styles.actLead}>
            Ce n’est pas le trajet. C’est la dispersion des informations entre ceux qui doivent rester alignés.
          </p>
          <div className={styles.contrast} role="group" aria-label="Contraste entre la situation actuelle et un fil commun">
            <div className={styles.contrastCol}>
              <h3 className={styles.contrastLabel}>Aujourd’hui</h3>
              <p className={styles.contrastHint}>Informations dispersées</p>
              <ul className={`${styles.contrastList} ${styles.contrastListBefore}`}>
                {TODAY_FRICTION.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </div>
            <div className={styles.contrastDivider} aria-hidden />
            <div className={`${styles.contrastCol} ${styles.contrastColAfter}`}>
              <h3 className={styles.contrastLabel}>Avec un fil commun</h3>
              <p className={styles.contrastHint}>Une mission lisible</p>
              <ul className={`${styles.contrastList} ${styles.contrastListAfter}`}>
                {AFTER_FLOW.map((item, i) => (
                  <li key={item}>
                    <span className={styles.contrastStep} aria-hidden>
                      {i + 1}
                    </span>
                    <span>{item}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* ── Un fil commun ── */}
      <section id="fil" className={styles.actFil} aria-labelledby="fil-heading">
        <div className={styles.shell}>
          <div className={styles.filLayout}>
            <div className={styles.filCopy}>
              <h2 id="fil-heading" className={styles.actTitle}>
                Un fil commun
              </h2>
              <p className={styles.filLead}>
                La mission naît chez l’institution, circule vers les transporteurs, et reste suivie jusqu’à
                l’historique — un même objet, plusieurs regards.
              </p>
              <ol className={styles.filChain}>
                <li>Institution</li>
                <li>Plateforme</li>
                <li>Transporteurs</li>
                <li>Suivi</li>
                <li>Historique</li>
              </ol>
            </div>
            <div className={styles.filVisual}>
              <div className={styles.filFrame} onContextMenu={blockImageSave}>
                <img
                  src="/images/lirie-fil-commun-mission.webp"
                  alt="Fil commun d’une mission partagée entre institution, plateforme et transporteur."
                  className={styles.filImg}
                  width={1448}
                  height={1086}
                  decoding="async"
                  loading="lazy"
                  draggable={false}
                  onDragStart={blockImageSave}
                  onContextMenu={blockImageSave}
                />
                <span className={styles.imgShield} aria-hidden="true" onContextMenu={blockImageSave} />
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ── Ce qui change pour chacun ── */}
      <section id="gains" className={styles.actGains} aria-labelledby="gains-heading">
        <div className={styles.shell}>
          <h2 id="gains-heading" className={styles.actTitle}>
            Ce qui change pour chacun
          </h2>
          <div className={styles.gainsRow}>
            {GAINS.map(({ role, items }) => (
              <div key={role} className={styles.gainsCol}>
                <h3 className={styles.gainsRole}>{role}</h3>
                <ul className={styles.gainsList}>
                  {items.map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Moment de convergence ── */}
      <section id="convergence" className={styles.actConvergence} aria-labelledby="convergence-heading">
        <div className={styles.shell}>
          <header className={styles.convergenceHeader}>
            <h2 id="convergence-heading" className={styles.convergenceTitle}>
              Un même transport. Trois réalités.
            </h2>
            <p className={styles.convergenceBeat}>
              <time className={styles.convergenceTime} dateTime="09:42">
                09:42
              </time>
              <span className={styles.convergenceBeatLabel}>Départ</span>
            </p>
          </header>
          <div className={styles.convergenceRail} aria-hidden />
          <div className={styles.convergenceGrid}>
            {CONVERGENCE.map(({ role, text }) => (
              <div key={role} className={styles.convergenceCol}>
                <h3 className={styles.convergenceRole}>{role}</h3>
                <p className={styles.convergenceText}>{text}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Chacun garde son rôle ── */}
      <section id="role" className={styles.actRole} aria-labelledby="role-heading">
        <div className={styles.shell}>
          <h2 id="role-heading" className={styles.actTitle}>
            Chacun garde son rôle.
          </h2>
          <p className={styles.roleLead}>
            Lirie n’est pas un transporteur et ne remplace pas vos processus. Elle accompagne la mission pour que
            chaque acteur conserve sa place — avec moins de charge mentale.
          </p>
          <div className={styles.roleCols}>
            <div>
              <h3 className={styles.roleSub}>Ce qui reste le vôtre</h3>
              <ul className={styles.roleList}>
                <li>Processus et responsabilités internes</li>
                <li>Contrats avec vos transporteurs</li>
                <li>Exécution du transport sur la voie publique</li>
              </ul>
            </div>
            <div>
              <h3 className={styles.roleSub}>Ce que fait le cadre commun</h3>
              <ul className={styles.roleList}>
                <li>Rendre la mission lisible pour les rôles habilités</li>
                <li>Réduire les relances et les zones d’ombre</li>
                <li>Conserver un historique selon les droits</li>
              </ul>
            </div>
          </div>
          <p className={styles.roleLinks}>
            <Link to="/conditions" className={styles.inlineLink}>
              Conditions générales
            </Link>
            {' · '}
            <Link to="/mentions-legales" className={styles.inlineLink}>
              Mentions légales
            </Link>
            {' · '}
            <Link to="/privacy" className={styles.inlineLink}>
              Confidentialité
            </Link>
          </p>
        </div>
      </section>

      {/* ── Comment rejoindre — deux parcours ── */}
      <section id="rejoindre" className={styles.actJoin} aria-labelledby="rejoindre-heading">
        <div className={styles.shell}>
          <h2 id="rejoindre-heading" className={styles.actTitle}>
            Comment rejoindre
          </h2>
          <div className={styles.joinRow}>
            <article className={styles.joinPath}>
              <h3 className={styles.joinPathTitle}>Institution</h3>
              <ol className={styles.joinSteps}>
                <li>Découvrir la plateforme</li>
                <li>Organiser une démonstration</li>
                <li>Préparer un déploiement</li>
              </ol>
              <Link to="/contact/demo" className={styles.btnPrimary}>
                Organiser une démonstration
                <IcoChevR s={13} />
              </Link>
            </article>
            <article className={styles.joinPath}>
              <h3 className={styles.joinPathTitle}>Entreprise</h3>
              <ol className={styles.joinSteps}>
                <li>Devenir partenaire</li>
                <li>Comprendre les critères</li>
                <li>Rejoindre le réseau</li>
              </ol>
              <Link to="/contact/transport" className={styles.btnPrimary}>
                Contacter l’équipe transport
                <IcoChevR s={13} />
              </Link>
            </article>
          </div>
        </div>
      </section>

      {/* ── FAQ ── */}
      <section id="faq" className={styles.actFaq} aria-labelledby="faq-heading">
        <div className={styles.shell}>
          <h2 id="faq-heading" className={styles.actTitle}>
            Questions fréquentes
          </h2>
          <div className={styles.faq}>
            {FAQ.map((item, i) => {
              const open = openFaq === i;
              const hId = `${faqBaseId}-h-${i}`;
              const pId = `${faqBaseId}-p-${i}`;
              return (
                <div key={item.q} className={styles.faqItem}>
                  <h3 className={styles.faqHeading}>
                    <button
                      type="button"
                      id={hId}
                      className={styles.faqTrigger}
                      aria-expanded={open}
                      aria-controls={pId}
                      onClick={() => setOpenFaq(open ? null : i)}
                    >
                      <span>{item.q}</span>
                      <span className={`${styles.faqIcon} ${open ? styles.faqIconOpen : ''}`} aria-hidden>
                        +
                      </span>
                    </button>
                  </h3>
                  <div id={pId} role="region" className={styles.faqPanel} aria-labelledby={hId} hidden={!open}>
                    <p>{item.a}</p>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* ── CTA dual ── */}
      <div className={styles.shell}>
        <section className={styles.finalBand} aria-labelledby="final-cta-heading">
          <h2 id="final-cta-heading" className={styles.finalTitle}>
            Une mission lisible pour tous
          </h2>
          <p className={styles.finalSub}>Choisissez le parcours qui correspond à votre organisation.</p>
          <div className={styles.finalActions}>
            <Link to="/contact/demo" className={styles.btnFinal}>
              Démonstration institution
              <IcoChevR s={14} />
            </Link>
            <Link to="/contact/transport" className={styles.btnFinalGhost}>
              Rejoindre en tant qu’entreprise
            </Link>
          </div>
        </section>
      </div>

      <div className={styles.bottomSpacer} aria-hidden />
    </div>
  );
};

export default ProfessionnelPage;
