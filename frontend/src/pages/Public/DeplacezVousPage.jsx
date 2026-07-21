import React, { useId, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  Section,
  SectionBody,
  SectionEyebrow,
  SectionFooter,
  SectionHeader,
  SectionLead,
  SectionTitle,
} from '../../brand/layout';
import styles from './DeplacezVousPage.module.css';

const LOGIN_BOOK = '/login?next=%2Fbook%2Fnew';

function blockImageSave(event) {
  event.preventDefault();
}

function IcoShield({ s = 20 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
    </svg>
  );
}
function IcoBuilding({ s = 20 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <rect x="4" y="2" width="16" height="20" />
      <line x1="9" y1="22" x2="9" y2="12" />
      <line x1="15" y1="22" x2="15" y2="12" />
      <rect x="9" y="12" width="6" height="10" />
      <line x1="9" y1="7" x2="9.01" y2="7" />
      <line x1="15" y1="7" x2="15.01" y2="7" />
      <line x1="9" y1="2" x2="9.01" y2="2" />
      <line x1="15" y1="2" x2="15.01" y2="2" />
    </svg>
  );
}
function IcoUser({ s = 20 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
      <circle cx="12" cy="7" r="4" />
    </svg>
  );
}
function IcoCheck({ s = 18 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <polyline points="20 6 9 17 4 12" />
    </svg>
  );
}
function IcoWheelchair({ s = 20 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <circle cx="12" cy="4" r="2" />
      <path d="M9 10h6l-1 5H9z" />
      <path d="M9 20a4 4 0 1 0 0-8" />
      <line x1="15" y1="15" x2="19" y2="15" />
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
function IcoHome({ s = 20 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
      <polyline points="9 22 9 12 15 12 15 22" />
    </svg>
  );
}
function IcoCalendar({ s = 20 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <rect x="3" y="4" width="18" height="18" rx="2" />
      <line x1="16" y1="2" x2="16" y2="6" />
      <line x1="8" y1="2" x2="8" y2="6" />
      <line x1="3" y1="10" x2="21" y2="10" />
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

const TRANSPORT_TYPES = [
  { Icon: IcoUser, title: 'Transport assis', desc: 'Déplacements vers des soins externes sans matériel médical spécifique.', meta: 'Véhicule standard' },
  { Icon: IcoWheelchair, title: 'Transport en fauteuil roulant', desc: 'Véhicule adapté, aide à la montée et descente, espace pour fauteuil.', meta: 'Véhicule adapté' },
  { Icon: IcoHeart, title: 'Transport avec accompagnement', desc: 'Présence d’un accompagnant, consignes d’accueil et besoins spécifiques.', meta: 'Accompagnant possible' },
  { Icon: IcoHome, title: 'Sortie d’hospitalisation', desc: 'Organisation du retour après une hospitalisation ou une journée de soins.', meta: 'Retour coordonné' },
  { Icon: IcoCalendar, title: 'Rendez-vous réguliers', desc: 'Dialyse, rééducation, consultations chroniques — planification récurrente.', meta: 'Planification récurrente' },
  { Icon: IcoBuilding, title: 'Transport organisé par une institution', desc: 'Flux coordonnés par un hôpital, un EMS, une clinique ou un service social.', meta: 'Gestion institutionnelle' },
];

const CRED_ITEMS = [
  { title: 'Entreprises habilitées', text: 'Les missions sont confiées à des entreprises répondant aux exigences applicables à leur activité.' },
  { title: 'Protection des données', text: 'Les données personnelles sont traitées selon les règles applicables, notamment la LPD suisse.' },
  { title: 'Responsabilités claires', text: 'Lirie facilite la coordination sans remplacer le jugement clinique ni les protocoles des professionnels de santé.' },
];

const FAQ = [
  {
    q: 'Mon transport est-il remboursé par la caisse maladie ?',
    a: 'Selon les conditions de votre assurance, une participation peut être envisagée lorsqu’un transport est médicalement nécessaire et prescrit. Lirie facilite la coordination — les questions de remboursement restent du ressort de votre assureur et de votre institution.',
  },
  {
    q: 'Que faire si je ne peux pas réserver moi-même ?',
    a: (
      <>
        Votre proche, votre aidant ou un professionnel de votre institution peut organiser le transport à votre place,
        dès lors qu’il dispose des accès appropriés. Si vous avez des doutes,{' '}
        <Link to="/contact" className={styles.inlineLink}>
          contactez-nous
        </Link>{' '}
        — nous vous orienterons vers le bon canal.
      </>
    ),
  },
  {
    q: "Combien de temps à l'avance faut-il réserver ?",
    a: "De préférence au moins 24 h à l'avance pour les courses programmées. Pour les situations urgentes, contactez directement votre institution ou utilisez le formulaire adapté sur la page contact. Les délais peuvent varier selon les partenaires de transport disponibles dans votre zone.",
  },
];

const DeplacezVousPage = () => {
  const [openFaq, setOpenFaq] = useState(null);
  const faqBaseId = useId();

  return (
    <div className={styles.page}>
      {/* ── Hero ── */}
      <header className={styles.hero}>
        <div className={styles.shell}>
          <div className={styles.heroGrid}>
            <div className={styles.heroMain}>
              <p className={styles.heroEyebrow}>Transport médical &amp; PMR · Suisse romande</p>
              <h1 className={styles.heroTitle}>
                Votre transport médical,
                <br />
                <span className={styles.heroTitleAccent}>coordonné avec soin.</span>
              </h1>
              <p className={styles.heroLead}>
                Lirie centralise votre demande, transmet les informations utiles et facilite la coordination avec une
                entreprise de transport habilitée. Lorsque le suivi est activé, les personnes autorisées restent informées
                de la mission.
              </p>
              <div className={styles.heroCtas}>
                <Link to={LOGIN_BOOK} className={styles.btnPrimary}>
                  Organiser mon transport
                  <IcoChevR s={14} />
                </Link>
                <Link to="/contact/family" className={styles.btnSecondary}>
                  Être accompagné
                </Link>
              </div>
            </div>
            <div className={styles.heroVisual}>
              <div className={styles.heroArtworkStack}>
                <div
                  className={styles.heroArtworkPrimaryFrame}
                  onContextMenu={blockImageSave}
                >
                  <img
                    src="/images/brand/artwork/care-accompany.png"
                    alt="Accompagnement d’une personne en fauteuil roulant vers un véhicule adapté"
                    className={styles.heroArtworkPrimary}
                    width={960}
                    height={720}
                    decoding="async"
                    draggable={false}
                    onDragStart={blockImageSave}
                    onContextMenu={blockImageSave}
                  />
                  <span
                    className={styles.heroArtworkShield}
                    aria-hidden="true"
                    onContextMenu={blockImageSave}
                  />
                </div>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* ── Timeline ── */}
      <div className={styles.shell}>
        <Section aria-labelledby="how-heading" className={styles.block}>
          <SectionHeader>
            <SectionEyebrow>Parcours</SectionEyebrow>
            <SectionTitle id="how-heading">Votre transport en quatre étapes</SectionTitle>
            <SectionLead>Lirie coordonne la mission. Une entreprise habilitée réalise le transport.</SectionLead>
          </SectionHeader>
          <SectionBody>
            <ol className={styles.timeline}>
              <li className={styles.step}>
                <span className={styles.stepNum} aria-hidden>
                  1
                </span>
                <div className={styles.stepTitle}>Votre demande est enregistrée</div>
                <p className={styles.stepMeta}>Saisie estimée : environ 2 minutes</p>
                <p className={styles.stepDesc}>
                  Lieux, horaire, besoins spécifiques (PMR, accompagnement, équipement). Via votre institution ou votre
                  compte lorsqu’il est activé.
                </p>
              </li>
              <li className={styles.step}>
                <span className={styles.stepNum} aria-hidden>
                  2
                </span>
                <div className={styles.stepTitle}>Un transporteur est assigné</div>
                <p className={styles.stepDesc}>
                  Selon la disponibilité, la zone et le type de véhicule requis. Le transporteur désigné reçoit la mission
                  en temps réel.
                </p>
              </li>
              <li className={styles.step}>
                <span className={styles.stepNum} aria-hidden>
                  3
                </span>
                <div className={styles.stepTitle}>Le transport est réalisé</div>
                <p className={styles.stepDesc}>
                  Prise en charge, trajet et dépose conformément aux consignes. Suivi pour les parties habilitées.
                </p>
              </li>
              <li className={styles.step}>
                <span className={styles.stepNum} aria-hidden>
                  4
                </span>
                <div className={styles.stepTitle}>La mission est confirmée</div>
                <p className={styles.stepDesc}>
                  Statut final, horodatages et éléments de mission accessibles à votre institution et aux parties
                  autorisées.
                </p>
              </li>
            </ol>
          </SectionBody>
          <SectionFooter>
            <p className={styles.notice}>
              Les modalités de validation, facturation et d’annulation sont définies avec votre institution ou dans vos
              accords avec le transporteur. Voir les{' '}
              <Link to="/conditions" className={styles.inlineLink}>
                conditions générales d’utilisation
              </Link>
              .
            </p>
          </SectionFooter>
        </Section>
      </div>

      {/* ── Types ── */}
      <Section tone="tint" aria-labelledby="types-heading" className={styles.typesSection}>
        <div className={styles.shell}>
          <SectionHeader>
            <SectionEyebrow>Types de transport</SectionEyebrow>
            <SectionTitle id="types-heading">Des situations concrètes, coordonnées via Lirie</SectionTitle>
            <SectionLead>
              Selon les partenaires disponibles dans votre région, Lirie peut coordonner différents types de transports
              médicaux et accompagnés.
            </SectionLead>
          </SectionHeader>
          <SectionBody>
            <div className={styles.typesGrid}>
              {TRANSPORT_TYPES.map(({ Icon, title, desc, meta }) => (
                <article key={title} className={styles.typeCard}>
                  <div className={styles.typeIcon}>
                    <Icon s={22} />
                  </div>
                  <h3 className={styles.typeTitle}>{title}</h3>
                  <p className={styles.typeDesc}>{desc}</p>
                  <span className={styles.typeMeta}>{meta}</span>
                </article>
              ))}
            </div>
          </SectionBody>
        </div>
      </Section>

      {/* ── Respiration unique ── */}
      <section className={styles.breath} aria-label="Coordination">
        <div className={styles.shell}>
          <div className={styles.breathGrid}>
            <div className={styles.breathArtworkFrame} onContextMenu={blockImageSave}>
              <img
                src="/images/brand/artwork/connection-coordinate.png"
                alt="Coordination du transport : suivi, trajet et accompagnement"
                className={styles.breathArtwork}
                width={960}
                height={540}
                decoding="async"
                loading="lazy"
                draggable={false}
                onDragStart={blockImageSave}
                onContextMenu={blockImageSave}
              />
              <span className={styles.breathArtworkShield} aria-hidden="true" onContextMenu={blockImageSave} />
            </div>
            <p className={styles.breathText}>
              Une seule coordination — du premier échange jusqu’à l’arrivée — pour que chacun sache où en est la mission.
            </p>
          </div>
        </div>
      </section>

      {/* ── Confiance ── */}
      <div className={styles.shell}>
        <Section aria-labelledby="cred-heading" className={styles.block}>
          <SectionHeader>
            <SectionEyebrow>Confiance</SectionEyebrow>
            <SectionTitle id="cred-heading">Une coordination encadrée et transparente</SectionTitle>
          </SectionHeader>
          <SectionBody>
            <div className={styles.credGrid}>
              <div className={styles.credPanel}>
                <h3 className={styles.credBlockTitle}>Protection et responsabilités</h3>
                <ul className={styles.credList}>
                  {CRED_ITEMS.map((item) => (
                    <li key={item.title} className={styles.credRow}>
                      <span className={styles.credRowIcon} aria-hidden>
                        <IcoCheck s={13} />
                      </span>
                      <div>
                        <div className={styles.credRowTitle}>{item.title}</div>
                        <div className={styles.credRowText}>{item.text}</div>
                      </div>
                    </li>
                  ))}
                </ul>
                <div className={styles.credLegal}>
                  <IcoShield s={14} />
                  <span>
                    <Link to="/privacy">Politique de confidentialité</Link>
                    <span className={styles.credSep}>·</span>
                    <Link to="/conditions">CGU</Link>
                    <span className={styles.credSep}>·</span>
                    <Link to="/mentions-legales">Mentions légales</Link>
                  </span>
                </div>
              </div>
              <div className={styles.credPanelTint}>
                <h3 className={styles.credBlockTitle}>Institutions</h3>
                <p className={styles.credNetworkIntro}>Le réseau institutionnel Lirie est en cours de développement.</p>
                <p className={styles.credNetworkIntro}>
                  Vous représentez un établissement de santé ? Échangez avec nous sur un projet pilote ou une
                  collaboration.
                </p>
                <Link to="/contact/institution" className={styles.credCta}>
                  Échanger sur un projet pilote
                  <IcoChevR s={14} />
                </Link>
              </div>
            </div>
          </SectionBody>
        </Section>

        {/* ── FAQ ── */}
        <Section aria-labelledby="faq-heading" className={styles.block}>
          <SectionHeader>
            <SectionTitle id="faq-heading">Avant d’organiser votre transport</SectionTitle>
            <SectionLead>Les réponses aux questions les plus fréquentes.</SectionLead>
          </SectionHeader>
          <SectionBody>
            <div className={styles.faq}>
              {FAQ.map((item, i) => {
                const open = openFaq === i;
                const headingId = `${faqBaseId}-h-${i}`;
                const panelId = `${faqBaseId}-p-${i}`;
                return (
                  <div key={item.q} className={`${styles.faqItem}${open ? ` ${styles.faqItemOpen}` : ''}`}>
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
                    <div
                      id={panelId}
                      role="region"
                      className={styles.faqPanel}
                      aria-labelledby={headingId}
                      hidden={!open}
                    >
                      {typeof item.a === 'string' ? <p>{item.a}</p> : <div>{item.a}</div>}
                    </div>
                  </div>
                );
              })}
            </div>
          </SectionBody>
        </Section>
      </div>

      {/* ── CTA final ── */}
      <div className={styles.shell}>
        <section className={styles.finalBand} aria-labelledby="final-cta-heading">
          <h2 id="final-cta-heading" className={styles.finalTitle}>
            Prêt à organiser votre transport ?
          </h2>
          <p className={styles.finalSub}>
            Déposez une demande, demandez de l’aide ou découvrez l’espace destiné aux institutions.
          </p>
          <div className={styles.finalActions}>
            <Link to={LOGIN_BOOK} className={styles.btnPrimary}>
              Organiser mon transport
              <IcoChevR s={14} />
            </Link>
            <Link to="/contact/family" className={styles.btnSecondary}>
              Être accompagné
            </Link>
          </div>
          <p className={styles.finalSecondary}>
            Vous représentez une institution ?{' '}
            <Link to="/professionnel" className={styles.inlineLink}>
              Découvrir l’espace professionnel
            </Link>
          </p>
        </section>
      </div>

      <div className={styles.bottomSpacer} aria-hidden />
    </div>
  );
};

export default DeplacezVousPage;
