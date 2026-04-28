import React, { useId, useState } from 'react';
import { Link } from 'react-router-dom';
import styles from './DeplacezVousPage.module.css';

const LOGIN_BOOK = '/login?next=%2Fbook%2Fnew';

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
function IcoClipboard({ s = 18 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2" />
      <rect x="8" y="2" width="8" height="4" rx="1" />
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
function IcoActivity({ s = 20 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
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
  { Icon: IcoUser, title: 'Transport assis simple', desc: 'Déplacements vers des soins externes sans matériel médical spécifique.', badge: 'Fréquent' },
  { Icon: IcoWheelchair, title: 'Transport PMR', desc: 'Véhicule adapté, aide à la montée et descente, espace pour fauteuil roulant.', badge: 'Adapté' },
  { Icon: IcoHeart, title: 'Transport accompagné', desc: 'Présence d’un accompagnant, consignes d’accueil et besoins spécifiques.', badge: 'Sur mesure' },
  { Icon: IcoBuilding, title: 'Transport institutionnel', desc: 'Flux coordonnés par un hôpital, un EMS, une clinique ou un service social.', badge: 'Via institution' },
  { Icon: IcoHome, title: 'Retour à domicile', desc: 'Organisation du retour après une hospitalisation ou une journée de soins.', badge: 'Fréquent' },
  { Icon: IcoCalendar, title: 'Courses programmées', desc: 'Rendez-vous réguliers (dialyse, rééducation, consultations chroniques).', badge: 'Récurrent' },
];

const CRED_ITEMS = [
  'La plateforme est conçue pour que les missions soient confiées à des entreprises de transport habilitées et assurées, dans le respect des exigences cantonales.',
  'Lirie prévoit la vérification des habilitations et la traçabilité des missions lorsque des transporteurs sont rattachés au service.',
  'Les données médicales et personnelles sont traitées conformément à la LPD suisse.',
  'Lirie ne remplace ni le jugement clinique ni les protocoles des établissements de soins ou des structures impliquées.',
  'Plateforme hébergée en Europe, avec garanties adaptées au RGPD / LPD.',
];

const FAQ = [
  {
    q: 'Mon transport est-il remboursé par la caisse maladie ?',
    a: "Cela dépend de votre prise en charge médicale et de la prescription de votre médecin. Pour les transports médicalement justifiés (LAMal), votre institution ou votre médecin traitant peut vous renseigner sur les démarches à entreprendre. Lirie facilite la coordination du transport — les questions de remboursement restent du ressort de votre assureur et de votre institution.",
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
      <header className={styles.hero}>
        <div className={styles.heroShell}>
          <div className={styles.heroGrid}>
            <div className={styles.heroMain}>
              <div className={styles.heroBadge}>
                <IcoMapPin s={12} />
                <span className={styles.heroBadgeLabel}>Transport médical &amp; PMR · Suisse romande</span>
              </div>
              <h1 className={styles.heroTitle}>
                Votre transport médical,
                <br />
                <span className={styles.heroTitleAccent}>coordonné avec soin.</span>
              </h1>
              <p className={styles.heroLead}>
                Lirie vise à faciliter la coordination entre votre demande et des entreprises de transport habilitées et
                assurées. Vous organisez, vos proches sont informés lorsque c’est prévu ; le chauffeur suit les consignes
                transmises sur la plateforme lorsque le service est disponible sur votre zone.
              </p>
              <div className={styles.heroCtas}>
                <Link to={LOGIN_BOOK} className={styles.btnPrimary}>
                  Organiser mon transport
                  <IcoChevR s={14} />
                </Link>
                <Link to="/contact/institution" className={styles.btnSecondary}>
                  Vous êtes une institution ?
                </Link>
              </div>
              <div className={styles.heroProof}>
                <div className={styles.heroProofDots} aria-hidden>
                  <span className={styles.heroProofDot} />
                  <span className={styles.heroProofDot} />
                  <span className={styles.heroProofDot} />
                </div>
                <span>
                  Projet développé en Suisse romande — ouvert aux institutions et aux entreprises de transport qui souhaitent
                  explorer une collaboration.
                </span>
              </div>
            </div>

            <aside className={styles.heroAside} aria-label="Points forts">
              <div className={styles.statCard}>
                <div className={styles.statIcon}>
                  <IcoShield s={20} />
                </div>
                <div>
                  <div className={styles.statVal}>Habilités</div>
                  <div className={styles.statLabel}>Chauffeurs et entreprises soumis aux règles cantonales</div>
                </div>
              </div>
              <div className={styles.statCard}>
                <div className={styles.statIcon}>
                  <IcoActivity s={20} />
                </div>
                <div>
                  <div className={styles.statVal}>Temps réel</div>
                  <div className={styles.statLabel}>Suivi de mission pour les acteurs autorisés</div>
                </div>
              </div>
              <div className={styles.statCard}>
                <div className={styles.statIcon}>
                  <IcoBuilding s={20} />
                </div>
                <div>
                  <div className={styles.statVal}>Suisse</div>
                  <div className={styles.statLabel}>Conçu pour les flux institutionnels romands</div>
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
                <div className={styles.reassuranceTitle}>Chauffeurs habilités &amp; assurés</div>
                <p className={styles.reassuranceDesc}>
                  Les entreprises de transport rattachées au service devront respecter les autorisations cantonales et les
                  assurances requises pour leur activité.
                </p>
              </div>
            </li>
            <li className={styles.reassuranceItem}>
              <div className={styles.reassuranceIcon}>
                <IcoMapPin s={20} />
              </div>
              <div>
                <div className={styles.reassuranceTitle}>Suivi en temps réel</div>
                <p className={styles.reassuranceDesc}>
                  Lorsque la fonctionnalité est activée, les acteurs autorisés (institution, proches, etc.) peuvent suivre la
                  mission selon les droits configurés sur la plateforme.
                </p>
              </div>
            </li>
            <li className={styles.reassuranceItem}>
              <div className={styles.reassuranceIcon}>
                <IcoBuilding s={20} />
              </div>
              <div>
                <div className={styles.reassuranceTitle}>Pensé pour les institutions</div>
                <p className={styles.reassuranceDesc}>
                  Conçu pour pouvoir s’intégrer aux processus des hôpitaux, EMS et services sociaux — coordination et
                  traçabilité.
                </p>
              </div>
            </li>
          </ul>
        </div>
      </div>

      <main className={styles.main}>
        <section className={styles.section} aria-labelledby="audience-heading">
          <div className={styles.sectionEyebrow}>
            <span className={styles.eyebrowLine} aria-hidden />
            Pour vous
          </div>
          <h2 id="audience-heading" className={styles.sectionTitle}>
            Ce service est fait pour vous
          </h2>
          <p className={styles.sectionLead}>
            Que vous soyez patient, proche aidant ou coordinateur institutionnel — Lirie s’adapte à votre situation.
          </p>
          <div className={styles.profiles}>
            <Link to={LOGIN_BOOK} className={styles.profileCard}>
              <div className={styles.profileIconWrap}>
                <IcoUser s={24} />
              </div>
              <div className={styles.profileTitle}>Patient ou personne à mobilité réduite</div>
              <p className={styles.profileQuote}>
                « Je dois me rendre à un rendez-vous médical et j’ai besoin d’un véhicule adapté ou d’accompagnement. »
              </p>
              <span className={styles.profileCta}>
                Démarrer une demande <IcoChevR s={12} />
              </span>
            </Link>
            <Link to="/contact/family" className={styles.profileCard}>
              <div className={styles.profileIconWrap}>
                <IcoUsers s={24} />
              </div>
              <div className={styles.profileTitle}>Proche ou aidant</div>
              <p className={styles.profileQuote}>
                « J’organise le transport pour un membre de ma famille et je veux être informé lorsque c’est activé. »
              </p>
              <span className={styles.profileCta}>
                Voir comment ça fonctionne <IcoChevR s={12} />
              </span>
            </Link>
            <Link to="/contact/institution" className={styles.profileCard}>
              <div className={styles.profileIconWrap}>
                <IcoBuilding s={24} />
              </div>
              <div className={styles.profileTitle}>Institution ou professionnel de santé</div>
              <p className={styles.profileQuote}>
                « Je coordonne les sorties et les transports de mon service — j’ai besoin d’un outil fiable et traçable. »
              </p>
              <span className={styles.profileCta}>
                Espace institution <IcoChevR s={12} />
              </span>
            </Link>
          </div>
        </section>

        <section className={styles.section} aria-labelledby="how-heading">
          <div className={styles.sectionEyebrow}>
            <span className={styles.eyebrowLine} aria-hidden />
            Processus
          </div>
          <h2 id="how-heading" className={styles.sectionTitle}>
            Votre demande, prise en charge de A à Z
          </h2>
          <p className={styles.sectionLead}>
            Lirie coordonne la mission — les transports sont assurés par des{' '}
            <strong>entreprises de transport habilitées</strong>, une fois le service et les accords opérationnels sur votre
            zone.
          </p>

          <ol className={styles.timeline}>
            <li className={styles.step}>
              <span className={styles.stepNum} aria-hidden>
                1
              </span>
              <div className={styles.stepIconWrap}>
                <IcoClipboard s={18} />
              </div>
              <div className={styles.stepTitle}>La demande est enregistrée</div>
              <p className={styles.stepDesc}>
                Lieux, horaire, besoins spécifiques (PMR, accompagnement, équipement). Via votre institution ou votre
                compte lorsqu’il est activé.
              </p>
              <span className={styles.stepBadge}>≈ 2 min</span>
            </li>
            <li className={styles.step}>
              <span className={styles.stepNum} aria-hidden>
                2
              </span>
              <div className={styles.stepIconWrap}>
                <IcoTruck s={18} />
              </div>
              <div className={styles.stepTitle}>Un transporteur est assigné</div>
              <p className={styles.stepDesc}>
                Selon la disponibilité, la zone et le type de véhicule requis. Le transporteur désigné reçoit la mission en
                temps réel.
              </p>
            </li>
            <li className={styles.step}>
              <span className={styles.stepNum} aria-hidden>
                3
              </span>
              <div className={styles.stepIconWrap}>
                <IcoMapPin s={18} />
              </div>
              <div className={styles.stepTitle}>La mission est réalisée</div>
              <p className={styles.stepDesc}>
                Prise en charge, trajet et dépose conformément aux consignes. Suivi pour les parties habilitées.
              </p>
            </li>
            <li className={styles.step}>
              <span className={styles.stepNum} aria-hidden>
                4
              </span>
              <div className={styles.stepIconWrap}>
                <IcoCheck s={18} />
              </div>
              <div className={styles.stepTitle}>Confirmation &amp; traçabilité</div>
              <p className={styles.stepDesc}>
                Statut final, horodatages et éléments de mission accessibles à votre institution et aux parties autorisées.
              </p>
            </li>
          </ol>

          <p className={styles.notice}>
            Les modalités de validation, facturation et d’annulation sont définies avec votre institution ou dans vos accords
            avec le transporteur. Voir les{' '}
            <Link to="/conditions" className={styles.inlineLink}>
              conditions générales d’utilisation
            </Link>
            .
          </p>
        </section>

        <section className={styles.section} aria-labelledby="types-heading">
          <div className={styles.sectionEyebrow}>
            <span className={styles.eyebrowLine} aria-hidden />
            Prestations
          </div>
          <h2 id="types-heading" className={styles.sectionTitle}>
            Types de transports coordonnés via Lirie
          </h2>
          <p className={styles.sectionLead}>
            La liste effective dépendra des transporteurs rattachés et des conventions locales. Voici les cas les plus
            courants côtés besoins.
          </p>
          <div className={styles.typesGrid}>
            {TRANSPORT_TYPES.map(({ Icon, title, desc, badge }) => (
              <article key={title} className={styles.typeCard}>
                <div className={styles.typeIcon}>
                  <Icon s={22} />
                </div>
                <h3 className={styles.typeTitle}>{title}</h3>
                <p className={styles.typeDesc}>{desc}</p>
                <span className={styles.typeBadge}>{badge}</span>
              </article>
            ))}
          </div>
        </section>

        <section className={styles.section} aria-labelledby="cred-heading">
          <div className={styles.sectionEyebrow}>
            <span className={styles.eyebrowLine} aria-hidden />
            Sécurité &amp; conformité
          </div>
          <h2 id="cred-heading" className={styles.sectionTitle}>
            Un cadre rigoureux pour la coordination
          </h2>
          <div className={styles.cred}>
            <div className={styles.credCol}>
              <h3 className={styles.credBlockTitle}>Nos engagements</h3>
              <ul className={styles.credList}>
                {CRED_ITEMS.map((item) => (
                  <li key={item} className={styles.credRow}>
                    <span className={styles.credRowIcon} aria-hidden>
                      <IcoCheck s={13} />
                    </span>
                    <span>{item}</span>
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
            <div className={styles.credCol}>
              <h3 className={styles.credBlockTitle}>Institutions</h3>
              <p className={styles.credNetworkIntro}>
                À ce jour, <strong>aucune institution n’est référencée comme partenaire contractuel</strong> sur cette page
                et aucune liste d’établissements équipés n’est publiée ici. Le réseau institutionnel est en construction.
              </p>
              <p className={styles.credNetworkIntro}>
                Vous représentez un hôpital, un EMS, une clinique ou un service social ?{' '}
                <Link to="/contact/institution" className={styles.inlineLink}>
                  Écrivez-nous
                </Link>{' '}
                pour discuter d’une mise en relation ou d’un pilote.
              </p>
              <p className={styles.credPartnerNote}>
                Lorsque des conventions ou déploiements pourront être communiqués publiquement, cette section sera mise à
                jour en conséquence.
              </p>
            </div>
          </div>
        </section>

        <section className={styles.section} aria-labelledby="faq-heading">
          <div className={styles.sectionEyebrow}>
            <span className={styles.eyebrowLine} aria-hidden />
            Questions fréquentes
          </div>
          <h2 id="faq-heading" className={styles.sectionTitle}>
            Vos questions, répondues
          </h2>
          <p className={styles.sectionLead}>Tout ce qu’il faut savoir avant d’organiser un transport via Lirie.</p>
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
        </section>
      </main>

      <div className={styles.shell}>
        <section className={styles.finalBand} aria-labelledby="final-cta-heading">
          <h2 id="final-cta-heading" className={styles.finalTitle}>
            Par quel chemin souhaitez-vous démarrer ?
          </h2>
          <p className={styles.finalSub}>Chaque situation est différente. Choisissez le parcours adapté à votre besoin.</p>
          <div className={styles.finalCards}>
            <Link to={LOGIN_BOOK} className={`${styles.finalCard} ${styles.finalCardPrimary}`}>
              <span className={styles.finalCardLabel}>Je suis patient ou PMR</span>
              <span className={styles.finalCardTitle}>Organiser mon transport</span>
              <span className={styles.finalCardDesc}>
                Connectez-vous pour déposer une demande, choisir les horaires et suivre votre mission.
              </span>
              <span className={styles.finalCardBtn}>
                Démarrer <IcoChevR s={13} />
              </span>
            </Link>
            <Link to="/contact/family" className={styles.finalCard}>
              <span className={styles.finalCardLabel}>Je suis un proche ou aidant</span>
              <span className={styles.finalCardTitle}>Aide &amp; accompagnement</span>
              <span className={styles.finalCardDesc}>
                Nous vous guidons dans la prise en main et la coordination avec l’institution responsable.
              </span>
              <span className={`${styles.finalCardBtn} ${styles.finalCardBtnOutline}`}>
                Être guidé <IcoChevR s={13} />
              </span>
            </Link>
            <Link to="/professionnel" className={styles.finalCard}>
              <span className={styles.finalCardLabel}>Je représente une institution</span>
              <span className={styles.finalCardTitle}>Page Professionnel</span>
              <span className={styles.finalCardDesc}>
                Coordination institutionnelle, traçabilité et démonstration : vue d’ensemble pour décideurs.
              </span>
              <span className={`${styles.finalCardBtn} ${styles.finalCardBtnOutline}`}>
                Découvrir <IcoChevR s={13} />
              </span>
            </Link>
          </div>
        </section>
      </div>

      <div className={styles.bottomSpacer} aria-hidden />
    </div>
  );
};

export default DeplacezVousPage;
