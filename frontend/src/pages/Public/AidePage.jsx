import React, { useEffect, useId, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import styles from './AidePage.module.css';

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
  }, [containerRef]);
}

function IcoChevR({ s = 14 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <polyline points="9 18 15 12 9 6" />
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

function IcoSteering({ s = 20 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <circle cx="12" cy="12" r="10" />
      <circle cx="12" cy="12" r="3" />
      <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41" />
    </svg>
  );
}

function IcoTruck({ s = 20 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M1 3h15v13H1z" />
      <path d="M16 8h4l3 3v5h-7V8z" />
      <circle cx="5.5" cy="18.5" r="2.5" />
      <circle cx="18.5" cy="18.5" r="2.5" />
    </svg>
  );
}

function IcoPhone({ s = 18 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07A19.5 19.5 0 0 1 4.69 12 19.79 19.79 0 0 1 1.61 3.32C1.6 2.16 2.37 1.14 3.5 1h3a2 2 0 0 1 2 1.72c.127.96.361 1.903.7 2.81a2 2 0 0 1-.45 2.11L7.91 8.54a16 16 0 0 0 6.07 6.07l.91-.91a2 2 0 0 1 2.11-.45c.907.339 1.85.573 2.81.7A2 2 0 0 1 22 16.92z" />
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

function IcoHelpCircle({ s = 12 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <circle cx="12" cy="12" r="10" />
      <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" />
      <line x1="12" y1="17" x2="12.01" y2="17" />
    </svg>
  );
}

function IcoMail({ s = 18 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z" />
      <polyline points="22,6 12,13 2,6" />
    </svg>
  );
}

function IcoPresentation({ s = 18 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M2 3h20" />
      <path d="M21 3v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V3" />
      <path d="M7 21l5-5 5 5" />
    </svg>
  );
}

const FAQ_PATIENTS = [
  {
    q: 'Comment demander un transport ?',
    a: 'Selon votre situation, la demande peut être organisée par votre institution, votre professionnel de santé ou via votre accès personnel lorsqu’il est disponible. L’équipe LIRIE peut vous orienter si vous n’êtes pas certain du circuit à emprunter.',
  },
  {
    q: 'Puis-je réserver directement sur LIRIE ?',
    a: 'Cela dépend des conventions locales et de l’organisation mise en place avec votre institution ou le transporteur partenaire. Dans la plupart des cas, c’est votre institution qui initie la demande.',
  },
  {
    q: 'Qui réalise le transport ?',
    a: 'Les missions sont effectuées par des entreprises de transport partenaires juridiquement indépendantes. LIRIE coordonne l’attribution — elle ne conduit pas les véhicules.',
  },
  {
    q: 'Puis-je modifier ou annuler une course ?',
    a: 'Contactez en priorité l’acteur ayant organisé le transport : institution, service coordinateur ou transporteur partenaire. Ils disposent des outils pour gérer les modifications.',
  },
  {
    q: 'Comment savoir si le transport est confirmé ?',
    a: 'Le statut dépend de la validation par les acteurs responsables. En cas de doute, votre institution ou le transporteur pourra vous confirmer la mission.',
  },
];

const FAQ_INSTITUTIONS = [
  {
    q: 'À quoi sert LIRIE pour une institution ?',
    a: 'LIRIE facilite la coordination des transports entre institutions et entreprises partenaires dans un environnement partagé. Vous disposez d’un tableau de bord pour planifier, suivre et gérer les missions en temps réel.',
  },
  {
    q: 'LIRIE remplace-t-elle nos transporteurs actuels ?',
    a: 'Non. La plateforme coordonne vos partenaires existants. Vous conservez vos relations contractuelles — LIRIE apporte la couche technologique de coordination.',
  },
  {
    q: 'Peut-on travailler avec plusieurs transporteurs ?',
    a: 'Oui. LIRIE est conçu nativement pour la coordination multi-transporteurs. Vous pouvez configurer vos priorités et règles d’attribution selon vos besoins.',
  },
  {
    q: 'Qui accède aux informations de mission ?',
    a: 'Les accès sont configurés selon les rôles définis dans votre organisation. Chaque profil ne voit que les données correspondant à son périmètre.',
  },
  {
    q: 'Comment organiser une présentation de la plateforme ?',
    a: 'Vous pouvez contacter l’équipe LIRIE pour planifier une démonstration adaptée à votre structure et à vos besoins spécifiques.',
  },
];

const FAQ_CHAUFFEURS = [
  {
    q: 'Puis-je travailler directement pour LIRIE ?',
    a: 'Non. LIRIE n’est pas un employeur. Les missions sont réalisées via des entreprises de transport partenaires juridiquement indépendantes.',
  },
  {
    q: 'Puis-je travailler comme indépendant ?',
    a: 'Oui, si vous exercez dans une structure enregistrée disposant d’un numéro IDE et des autorisations nécessaires pour le transport concerné.',
  },
  {
    q: 'Puis-je utiliser mon véhicule personnel ?',
    a: 'Uniquement si celui-ci respecte la réglementation applicable au type de transport concerné (homologation, assurance, équipements requis).',
  },
  {
    q: 'Qui me rémunère ?',
    a: 'Votre employeur ou votre entreprise indépendante. LIRIE n’est en aucun cas l’employeur des chauffeurs opérant sur la plateforme.',
  },
  {
    q: 'Comment rejoindre le réseau ?',
    a: 'Contactez l’équipe partenaires LIRIE pour l’étude de votre situation. L’intégration se fait au travers d’une entreprise de transport enregistrée.',
  },
];

const FAQ_ENTREPRISES = [
  {
    q: 'Comment intégrer le réseau LIRIE ?',
    a: 'L’intégration se fait progressivement après un échange avec l’équipe partenaires. Un accompagnement est prévu pour la mise en place technique et opérationnelle.',
  },
  {
    q: 'Peut-on connecter plusieurs chauffeurs ?',
    a: 'Oui. L’accès est configurable selon votre organisation interne — flottes et équipes de toute taille peuvent être intégrées.',
  },
  {
    q: 'Peut-on publier des offres chauffeurs ?',
    a: 'Oui. Les entreprises partenaires peuvent diffuser des opportunités de recrutement via le réseau LIRIE.',
  },
  {
    q: 'LIRIE modifie-t-elle la relation avec nos clients ?',
    a: 'Non. La plateforme facilite la coordination opérationnelle sans toucher à vos relations contractuelles existantes avec vos clients institutionnels.',
  },
];

const FAQ_SITUATIONS = [
  {
    q: 'Je ne vois pas ma course dans l’application',
    a: 'Contactez l’acteur ayant organisé le transport (institution ou transporteur). La visibilité d’une course dépend de votre rôle et de la configuration de votre accès.',
  },
  {
    q: 'Le chauffeur est en retard',
    a: 'Adressez-vous directement au transporteur responsable de la mission. Ils ont les outils pour localiser le véhicule et vous informer du délai.',
  },
  {
    q: 'Le statut de la course ne se met pas à jour',
    a: 'Les mises à jour de statut dépendent des étapes validées par les intervenants terrain. Si le problème persiste, contactez l’organisateur du transport.',
  },
  {
    q: 'Je dois modifier ou annuler une réservation',
    a: 'Contactez l’institution ou le transporteur ayant organisé la mission. Les modifications doivent être traitées par l’acteur responsable de la course.',
  },
  {
    q: 'Je ne sais pas qui contacter',
    a: 'Utilisez la page contact LIRIE — notre équipe vous orientera vers le bon interlocuteur selon votre situation.',
  },
];

function FaqAccordion({ items, sectionSlug, openKey, setOpenKey, idPrefix }) {
  return (
    <div className={styles.faqList}>
      {items.map((item, i) => {
        const key = `${sectionSlug}:${i}`;
        const open = openKey === key;
        const headingId = `${idPrefix}-h-${sectionSlug}-${i}`;
        const panelId = `${idPrefix}-p-${sectionSlug}-${i}`;
        return (
          <div key={`${sectionSlug}-${i}`} className={styles.faqRow}>
            <h3 className={styles.faqHeading}>
              <button
                type="button"
                id={headingId}
                className={`${styles.faqTrigger}${open ? ` ${styles.faqTriggerOpen}` : ''}`}
                aria-expanded={open}
                aria-controls={panelId}
                onClick={() => setOpenKey(open ? null : key)}
              >
                <span>{item.q}</span>
                <span className={`${styles.faqIcon}${open ? ` ${styles.faqIconOpen}` : ''}`} aria-hidden>
                  {open ? '\u00D7' : '+'}
                </span>
              </button>
            </h3>
            <div
              className={`${styles.faqPanelGrid} ${open ? styles.faqPanelGridOpen : ''}`}
              aria-hidden={!open}
            >
              <div className={styles.faqPanelInner}>
                <div id={panelId} role="region" className={styles.faqPanel} aria-labelledby={headingId}>
                  <p>{item.a}</p>
                </div>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

const AidePage = () => {
  const [openKey, setOpenKey] = useState(null);
  const faqBaseId = useId();
  const pageRef = useRef(null);
  useScrollReveal(pageRef);

  return (
    <div ref={pageRef} className={styles.page}>
      <header className={styles.hero}>
        <div className={styles.heroShell}>
          <div className={styles.heroStack}>
            <div className={`${styles.heroAnim} ${styles.heroAnim0}`}>
              <div className={styles.heroBadge}>
                <IcoHelpCircle s={12} />
                <span className={styles.heroBadgeLabel}>Centre d’aide LIRIE</span>
              </div>
            </div>
            <h1 className={`${styles.heroTitle} ${styles.heroAnim} ${styles.heroAnim1}`}>
              Comment pouvons-nous <span className={styles.heroTitleAccent}>vous aider ?</span>
            </h1>
            <p className={`${styles.heroLead} ${styles.heroAnim} ${styles.heroAnim2}`}>
              Trouvez rapidement des réponses selon votre profil, ou contactez l’équipe LIRIE pour être orienté vers le bon interlocuteur.
            </p>
            <p className={`${styles.heroMicro} ${styles.heroAnim} ${styles.heroAnim3}`}>
              LIRIE est une plateforme de coordination — pas un transporteur ni un service d’urgence. Pour la vision du projet,{' '}
              <Link to="/a-propos" className={styles.inlineLink}>
                voir À propos
              </Link>
              .
            </p>
            <nav className={`${styles.quickPills} ${styles.heroAnim} ${styles.heroAnim4}`} aria-label="Navigation rapide dans la page">
              <a href="#profils" className={styles.quickPill}>
                Votre situation
                <IcoChevR s={11} />
              </a>
              <a href="#situations" className={styles.quickPill}>
                Situations courantes
                <IcoChevR s={11} />
              </a>
              <a href="#urgence" className={styles.quickPill}>
                Urgence
                <IcoChevR s={11} />
              </a>
              <a href="#contact-support" className={styles.quickPill}>
                Contact support
                <IcoChevR s={11} />
              </a>
            </nav>
          </div>
        </div>
      </header>

      <section id="profils" className={styles.sectionBandWhite} aria-labelledby="choix-situation-heading">
        <div className={styles.sectionInner}>
          <div data-reveal className={`${styles.reveal} ${styles.revealDelay1}`}>
            <div className={styles.sectionEyebrow}>
              <span className={styles.eyebrowLine} aria-hidden />
              Orientation
            </div>
            <h2 id="choix-situation-heading" className={styles.sectionTitle}>
              Choisissez votre profil
            </h2>
            <p className={styles.sectionLeadLarge}>
              Chaque profil dispose d’une section dédiée avec des réponses aux questions fréquentes.
            </p>
            <div className={styles.situationGrid}>
              <a href="#patients" className={styles.situationCard}>
                <div className={styles.situationCardIcon} aria-hidden>
                  <IcoHeart s={22} />
                </div>
                <h3 className={styles.situationCardTitle}>Patient ou proche</h3>
                <p className={styles.situationCardDesc}>
                  Organiser un transport, comprendre une course, savoir qui contacter.
                </p>
                <span className={styles.situationCardCta}>
                  Aide patients <IcoChevR s={12} />
                </span>
              </a>
              <a href="#institutions" className={styles.situationCard}>
                <div className={styles.situationCardIcon} aria-hidden>
                  <IcoBuilding s={22} />
                </div>
                <h3 className={styles.situationCardTitle}>Institution ou service coordinateur</h3>
                <p className={styles.situationCardDesc}>
                  Coordination multi-transporteurs, suivi des missions, accès à la plateforme.
                </p>
                <span className={styles.situationCardCta}>
                  Aide institutions <IcoChevR s={12} />
                </span>
              </a>
              <a href="#chauffeurs" className={styles.situationCard}>
                <div className={styles.situationCardIcon} aria-hidden>
                  <IcoSteering s={22} />
                </div>
                <h3 className={styles.situationCardTitle}>Chauffeur</h3>
                <p className={styles.situationCardDesc}>
                  Conditions d’accès, missions, fonctionnement de l’application terrain.
                </p>
                <span className={styles.situationCardCta}>
                  Aide chauffeurs <IcoChevR s={12} />
                </span>
              </a>
              <a href="#entreprises" className={styles.situationCard}>
                <div className={styles.situationCardIcon} aria-hidden>
                  <IcoTruck s={22} />
                </div>
                <h3 className={styles.situationCardTitle}>Entreprise de transport</h3>
                <p className={styles.situationCardDesc}>
                  Rejoindre le réseau, intégrer vos équipes, publier des offres chauffeurs.
                </p>
                <span className={styles.situationCardCta}>
                  Aide entreprises <IcoChevR s={12} />
                </span>
              </a>
            </div>
          </div>
        </div>
      </section>

      <section id="patients" className={styles.sectionBandMuted} aria-labelledby="patients-heading">
        <div className={styles.sectionInner}>
          <div data-reveal className={styles.reveal}>
            <div className={styles.sectionEyebrow}>
              <span className={styles.eyebrowLine} aria-hidden />
              Patients &amp; proches
            </div>
            <h2 id="patients-heading" className={styles.sectionTitle}>
              Aide pour les patients et leurs proches
            </h2>
            <p className={styles.sectionCallout}>
              LIRIE est une plateforme de coordination — l’organisation concrète du trajet relève de votre institution ou du transporteur partenaire selon les cas.
            </p>
            <FaqAccordion
              idPrefix={faqBaseId}
              sectionSlug="patients"
              items={FAQ_PATIENTS}
              openKey={openKey}
              setOpenKey={setOpenKey}
            />
          </div>
        </div>
      </section>

      <section id="institutions" className={styles.sectionBandWhite} aria-labelledby="institutions-heading">
        <div className={styles.sectionInner}>
          <div data-reveal className={`${styles.reveal} ${styles.revealDelay1}`}>
            <div className={styles.sectionEyebrow}>
              <span className={styles.eyebrowLine} aria-hidden />
              Institutions
            </div>
            <h2 id="institutions-heading" className={styles.sectionTitle}>
              Institutions et services coordinateurs
            </h2>
            <p className={styles.sectionLead}>
              Pour les EMS, hôpitaux, services sociaux et équipes qui planifient ou suivent des missions sur la plateforme.
            </p>
            <FaqAccordion
              idPrefix={faqBaseId}
              sectionSlug="institutions"
              items={FAQ_INSTITUTIONS}
              openKey={openKey}
              setOpenKey={setOpenKey}
            />
            <div className={styles.sectionCtaRow}>
              <Link to="/contact/demo" className={styles.btnPrimary}>
                <IcoPresentation s={18} />
                Demander une présentation
                <IcoChevR s={14} />
              </Link>
              <Link to="/contact/institution" className={styles.btnSecondary}>
                Contacter le pôle institutions
              </Link>
            </div>
          </div>
        </div>
      </section>

      <section id="chauffeurs" className={styles.sectionBandMuted} aria-labelledby="chauffeurs-heading">
        <div className={styles.sectionInner}>
          <div data-reveal className={styles.reveal}>
            <div className={styles.sectionEyebrow}>
              <span className={styles.eyebrowLine} aria-hidden />
              Chauffeurs
            </div>
            <h2 id="chauffeurs-heading" className={styles.sectionTitle}>
              Aide pour les chauffeurs
            </h2>
            <p className={styles.sectionCallout}>
              Les missions sont confiées par les entreprises partenaires. LIRIE ne recrute pas directement en tant qu’employeur.
            </p>
            <FaqAccordion
              idPrefix={faqBaseId}
              sectionSlug="chauffeurs"
              items={FAQ_CHAUFFEURS}
              openKey={openKey}
              setOpenKey={setOpenKey}
            />
            <div className={styles.sectionCtaRow}>
              <Link to="/contact/transport" className={styles.btnPrimary}>
                Rejoindre le réseau transport
                <IcoChevR s={14} />
              </Link>
              <Link to="/conduire" className={styles.btnSecondary}>
                Page Conduire
              </Link>
            </div>
          </div>
        </div>
      </section>

      <section id="entreprises" className={styles.sectionBandWhite} aria-labelledby="entreprises-heading">
        <div className={styles.sectionInner}>
          <div data-reveal className={`${styles.reveal} ${styles.revealDelay1}`}>
            <div className={styles.sectionEyebrow}>
              <span className={styles.eyebrowLine} aria-hidden />
              Partenaires transport
            </div>
            <h2 id="entreprises-heading" className={styles.sectionTitle}>
              Aide entreprises partenaires
            </h2>
            <p className={styles.sectionLead}>
              Intégration progressive, multi-chauffeurs et respect de vos relations contractuelles existantes.
            </p>
            <FaqAccordion
              idPrefix={faqBaseId}
              sectionSlug="entreprises"
              items={FAQ_ENTREPRISES}
              openKey={openKey}
              setOpenKey={setOpenKey}
            />
            <div className={styles.sectionCtaRow}>
              <Link to="/contact/transport" className={styles.btnPrimary}>
                Contacter l’équipe partenaires
                <IcoChevR s={14} />
              </Link>
              <Link to="/conduire" className={styles.btnSecondary}>
                Page Conduire
              </Link>
            </div>
          </div>
        </div>
      </section>

      <section id="paiement" className={styles.sectionBandMuted} aria-labelledby="paiement-heading">
        <div className={styles.sectionInner}>
          <div data-reveal className={styles.reveal}>
            <div className={styles.sectionEyebrow}>
              <span className={styles.eyebrowLine} aria-hidden />
              Facturation
            </div>
            <h2 id="paiement-heading" className={styles.sectionTitle}>
              Paiement et facturation
            </h2>
            <p className={styles.sectionLead}>
              Les modalités dépendent du cadre convenu entre les acteurs — la plateforme reste un outil de coordination.
            </p>
            <div className={styles.paymentBox}>
              <p>Les modalités de facturation dépendent notamment :</p>
              <ul className={styles.paymentList}>
                <li>
                  <span className={styles.paymentDot} aria-hidden />
                  de l’institution organisatrice ;
                </li>
                <li>
                  <span className={styles.paymentDot} aria-hidden />
                  du transporteur partenaire ;
                </li>
                <li>
                  <span className={styles.paymentDot} aria-hidden />
                  du cadre administratif applicable.
                </li>
              </ul>
              <p>
                LIRIE agit comme plateforme de coordination et ne facture pas directement les prestations de transport, sauf indication spécifique dans votre contexte.
              </p>
              <p>
                Pour une question précise, utilisez le{' '}
                <Link to="/contact/billing" className={styles.inlineLink}>
                  contact facturation
                </Link>{' '}
                ou la page contact générale.
              </p>
            </div>
          </div>
        </div>
      </section>

      <section id="situations" className={styles.sectionBandWhite} aria-labelledby="situations-faq-heading">
        <div className={styles.sectionInner}>
          <div data-reveal className={`${styles.reveal} ${styles.revealDelay1}`}>
            <div className={styles.sectionEyebrow}>
              <span className={styles.eyebrowLine} aria-hidden />
              Dépannage
            </div>
            <h2 id="situations-faq-heading" className={styles.sectionTitle}>
              Situations courantes
            </h2>
            <p className={styles.sectionLead}>
              Dans la majorité des cas, l’organisateur du transport (institution ou transporteur) est le bon interlocuteur opérationnel.
            </p>
            <div className={styles.hintBand}>
              <IcoInfo s={18} />
              <p>
                <strong>Mission en cours :</strong> si votre demande concerne une course déjà planifiée ou en route, contactez en priorité l’organisateur du transport (institution ou entreprise de transport responsable).
              </p>
            </div>
            <FaqAccordion
              idPrefix={faqBaseId}
              sectionSlug="situations"
              items={FAQ_SITUATIONS}
              openKey={openKey}
              setOpenKey={setOpenKey}
            />
            <div className={styles.sectionCtaRow}>
              <Link to="/contact" className={styles.btnSecondary}>
                Accéder à la page contact
              </Link>
            </div>
          </div>
        </div>
      </section>

      <section id="urgence" className={styles.sectionBandMuted} aria-labelledby="urgence-heading">
        <div className={styles.sectionInner}>
          <div data-reveal className={styles.reveal}>
            <div className={styles.sectionEyebrow}>
              <span className={styles.eyebrowLine} aria-hidden />
              Urgence
            </div>
            <h2 id="urgence-heading" className={styles.sectionTitle}>
              Situation urgente pendant un transport
            </h2>
            <div className={styles.emergencyGrid}>
              <div className={`${styles.emergencyCard} ${styles.emergencyCardUrgent}`}>
                <div className={styles.emergencyCardHead}>
                  <div className={`${styles.emergencyIcon} ${styles.emergencyIconUrgent}`} aria-hidden>
                    <IcoPhone s={18} />
                  </div>
                  <h3 className={styles.emergencyCardTitle}>Urgence médicale</h3>
                </div>
                <p>En cas d’urgence médicale, contactez immédiatement les services compétents.</p>
                <a
                  href="tel:144"
                  className={styles.emergencyBadge}
                  aria-label="Appeler le 144 — Urgences Suisse"
                >
                  <span className={styles.emergencyBadgeNum}>144</span>
                  <span className={styles.emergencyBadgeLabel}>Urgences — Suisse</span>
                </a>
              </div>
              <div className={`${styles.emergencyCard} ${styles.emergencyCardMission}`}>
                <div className={styles.emergencyCardHead}>
                  <div className={`${styles.emergencyIcon} ${styles.emergencyIconMission}`} aria-hidden>
                    <IcoTruck s={18} />
                  </div>
                  <h3 className={styles.emergencyCardTitle}>Mission en cours</h3>
                </div>
                <p>Pour toute question liée à une mission en route, contactez directement :</p>
                <ul className={styles.emergencyList}>
                  <li>
                    <span className={styles.emergencyListDot} aria-hidden />
                    votre institution ou le service coordinateur ;
                  </li>
                  <li>
                    <span className={styles.emergencyListDot} aria-hidden />
                    l’entreprise de transport responsable de la mission.
                  </li>
                </ul>
                <p className={styles.emergencyFoot}>
                  LIRIE ne remplace pas les services d’urgence ni le pilotage terrain par les acteurs habilités.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      <div className={styles.finalWrap}>
        <section id="contact-support" className={styles.finalBand} aria-labelledby="contact-support-heading">
          <div className={styles.heroShell}>
            <div data-reveal className={`${styles.reveal} ${styles.revealDelay2}`}>
              <p className={styles.finalKicker}>Support LIRIE</p>
              <h2 id="contact-support-heading" className={styles.finalTitle}>
                Vous ne trouvez pas votre réponse ?
              </h2>
              <p className={styles.finalSub}>
                Notre équipe vous orientera selon votre situation et vous mettra en relation avec l’interlocuteur approprié.
              </p>
              <p className={styles.finalMicro}>Nous répondons selon votre rôle et votre organisation.</p>
              <div className={styles.ctaRow3}>
                <Link to="/contact/support" className={styles.btnOnLight}>
                  <IcoMail s={18} />
                  Contacter le support
                  <IcoChevR s={13} />
                </Link>
                <Link to="/contact/transport" className={styles.btnGhostLight}>
                  Équipe partenaires
                </Link>
                <Link to="/contact/demo" className={styles.btnGhostLight}>
                  <IcoPresentation s={18} />
                  Demander une présentation
                </Link>
              </div>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
};

export default AidePage;
