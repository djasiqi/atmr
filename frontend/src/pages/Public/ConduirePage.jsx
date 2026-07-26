import React, { useId, useState } from 'react';
import { Link } from 'react-router-dom';
import styles from './ConduirePage.module.css';

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

const DAY_STEPS = [
  {
    title: 'Mission reçue',
    text: 'Les consignes utiles sont déjà là — lieux, horaire, besoins.',
  },
  {
    title: 'Je pars',
    text: 'Vous quittez le point de départ avec un trajet clair.',
  },
  {
    title: 'J’arrive',
    text: 'Prise en charge selon les indications de la mission.',
  },
  {
    title: 'Je transporte',
    text: 'Le trajet se déroule ; le suivi reste disponible si besoin.',
  },
  {
    title: 'Mission terminée',
    text: 'Clôture simple : l’essentiel est déjà tracé.',
  },
];

const VALUE_PROOFS = [
  'Moins d’allers-retours au téléphone',
  'Informations utiles avant le départ',
  'Suivi en temps réel pour les acteurs habilités',
  'Historique complet de la mission',
  'Toutes vos courses au même endroit',
];

const MISSION_TYPES_LINE =
  'Transport planifié · PMR · Retour à domicile · Trajets réguliers · Inter-établissements';

const MODE_PATHS = [
  {
    title: 'Salarié partenaire',
    text: 'Vous restez employé de votre entreprise. Les missions, le contrat et les accès passent par votre structure.',
  },
  {
    title: 'Indépendant structuré',
    text: 'Vous exercez via votre propre structure enregistrée : IDE, autorisations cantonales, assurances et véhicule conformes.',
  },
];

const JOIN_PROCESS_LINE = 'Contact · Vérification · Configuration · Activation';

const EXIGENCES = [
  'Permis et catégories adaptés au transport',
  'Autorisation professionnelle si la loi l’exige',
  'Véhicule conforme (PMR si besoin)',
  'Assurances alignées sur l’activité',
  'Affiliation partenaire ou structure enregistrée',
];

const FAQ = [
  {
    q: 'Puis-je travailler directement pour Lirie comme employé ?',
    a: 'Non. Vous intervenez via une entreprise de transport partenaire ou, en indépendant, via votre propre structure enregistrée et autorisée.',
  },
  {
    q: 'Puis-je travailler comme indépendant ?',
    a: 'Oui, si vous exercez dans une entreprise légalement constituée (IDE, autorisations et assurances selon le canton). La plateforme ne remplace pas ces obligations.',
  },
  {
    q: 'Qui me rémunère ?',
    a: 'Votre employeur partenaire, ou votre structure si vous êtes indépendant. Lirie coordonne les missions ; elle n’est pas votre employeur.',
  },
  {
    q: 'Comment mon entreprise rejoint-elle le réseau ?',
    a: 'Via le formulaire transport. Un interlocuteur accompagne la conformité et la configuration des accès.',
  },
];

const ConduirePage = () => {
  const [openFaq, setOpenFaq] = useState(null);
  const faqBaseId = useId();

  return (
    <div className={styles.page}>
      {/* ── 1. Hero — émotion ── */}
      <header className={styles.hero}>
        <div className={styles.shell}>
          <div className={styles.heroGrid}>
            <div className={styles.heroMain}>
              <h1 className={styles.heroTitle}>Concentrez-vous sur la route.</h1>
              <p className={styles.heroSub}>Vos missions restent claires, du départ à l’arrivée.</p>
              <p className={styles.heroLead}>
                Moins d’appels, des consignes au bon endroit, un fil de mission lisible — pour les professionnels du
                transport médical et accompagné.
              </p>
              <div className={styles.heroCtas}>
                <Link to="/contact/transport" className={styles.btnPrimary}>
                  Rejoindre le réseau
                  <IcoChevR s={14} />
                </Link>
              </div>
            </div>
            <div className={styles.heroVisual}>
              <div className={styles.heroFrame} onContextMenu={blockImageSave}>
                <img
                  src="/images/lirie-chauffeur-consulte-mission-transport.webp"
                  alt="Chauffeur consultant une mission de transport sur smartphone près du véhicule."
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
          </div>
        </div>
      </header>

      {/* ── 2. Une journée — immersion ── */}
      <section id="journee" className={styles.actDay} aria-labelledby="journee-heading">
        <div className={styles.shell}>
          <h2 id="journee-heading" className={styles.actTitle}>
            Une journée sur le terrain
          </h2>
          <ol className={styles.dayTimeline}>
            {DAY_STEPS.map((step, i) => (
              <li key={step.title} className={styles.dayStep}>
                <span className={styles.dayNum} aria-hidden>
                  {i + 1}
                </span>
                <div className={styles.dayStepTitle}>{step.title}</div>
                <p className={styles.dayStepText}>{step.text}</p>
              </li>
            ))}
          </ol>
        </div>
      </section>

      {/* ── 3. Ce qui change — démonstration ── */}
      <section id="valeur" className={styles.actValue} aria-labelledby="valeur-heading">
        <div className={styles.shell}>
          <div className={styles.valueSplit}>
            <div className={styles.valueMain}>
              <h2 id="valeur-heading" className={styles.valueHeadline}>
                Ce qui change concrètement
              </h2>
              <p className={styles.valueLead}>
                Plus de temps au volant, moins à chercher l’information. Vos missions sont centralisées, documentées et
                suivies — selon les droits de votre entreprise.
              </p>
            </div>
            <ul className={styles.valueList}>
              {VALUE_PROOFS.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </div>
        </div>
      </section>

      {/* ── 4. Une mission, sans rupture — respiration ── */}
      <section className={styles.actMission} aria-labelledby="mission-heading">
        <div className={styles.shell}>
          <div className={styles.missionSplit}>
            <div className={styles.missionVisual}>
              <div className={styles.missionFrame} onContextMenu={blockImageSave}>
                <img
                  src="/images/lirie-chauffeur-mission-transport-pmr.webp"
                  alt="Chauffeur en mission de transport PMR, de la prise en charge à la destination."
                  className={styles.missionImg}
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
            <div className={styles.missionCopy}>
              <h2 id="mission-heading" className={styles.missionTitle}>
                Une mission, sans rupture
              </h2>
              <p className={styles.missionCaption}>Du premier signal à l’arrivée — un fil continu.</p>
              <p className={styles.typesLine}>{MISSION_TYPES_LINE}</p>
            </div>
          </div>
        </div>
      </section>

      {/* ── 5. Comment rejoindre — information ── */}
      <section id="rejoindre" className={styles.actJoin} aria-labelledby="rejoindre-heading">
        <div className={styles.shell}>
          <h2 id="rejoindre-heading" className={styles.actTitle}>
            Comment rejoindre le réseau
          </h2>
          <p className={styles.joinLead}>
            Un formulaire pour les salariés partenaires, les entreprises et les indépendants autorisés.
          </p>
          <p className={styles.joinDisclaimer}>
            Les chauffeurs interviennent par l’intermédiaire d’une entreprise partenaire ou de leur propre structure
            autorisée.
          </p>

          <div className={styles.modesRow}>
            {MODE_PATHS.map(({ title, text }) => (
              <article key={title} className={styles.modeCol}>
                <h3 className={styles.modeTitle}>{title}</h3>
                <p className={styles.modeText}>{text}</p>
              </article>
            ))}
          </div>

          <div className={styles.joinFooter}>
            <div className={styles.joinFooterMeta}>
              <p className={styles.joinProcess}>{JOIN_PROCESS_LINE}</p>
              <p className={styles.joinNote}>
                Pas d’offres en ligne pour l’instant — le formulaire oriente candidature ou offre entreprise.{' '}
                <a href="#cadre" className={styles.inlineLink}>
                  Cadre et responsabilités
                </a>
              </p>
            </div>
            <Link to="/contact/transport" className={styles.btnPrimary}>
              Contacter l’équipe transport
              <IcoChevR s={13} />
            </Link>
          </div>
        </div>
      </section>

      {/* ── 6. Cadre — rassurance ── */}
      <section id="cadre" className={styles.actCadre} aria-labelledby="cadre-heading">
        <div className={styles.shell}>
          <h2 id="cadre-heading" className={styles.actTitle}>
            Cadre et responsabilités
          </h2>
          <p className={styles.cadreIntro}>
            La plateforme coordonne les missions. L’exécution sur la voie publique relève des entreprises partenaires
            juridiquement indépendantes. Lirie n’emploie pas les chauffeurs.
          </p>

          <div className={styles.cadreCols}>
            <div className={styles.cadreCol}>
              <h3 className={styles.cadreColTitle}>De votre côté</h3>
              <ul className={styles.cadreList}>
                <li>Relation d’emploi ou d’exploitation avec votre structure</li>
                <li>Conformité véhicule, assurances, autorisations</li>
                <li>Jugement terrain et consignes de mission</li>
              </ul>
            </div>
            <div className={styles.cadreCol}>
              <h3 className={styles.cadreColTitle}>La plateforme ne fait pas</h3>
              <ul className={styles.cadreList}>
                <li>Exécuter le transport sur la voie publique</li>
                <li>Se substituer au transporteur</li>
                <li>Remplacer les protocoles médicaux</li>
              </ul>
            </div>
          </div>

          <div className={styles.cadreFooter}>
            <p className={styles.cadreSub}>Exigences fréquentes</p>
            <p className={styles.exigenceLine}>{EXIGENCES.join(' · ')}</p>
            <p className={styles.cadreLinks}>
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
        </div>
      </section>

      {/* ── 7. FAQ — calme ── */}
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

      {/* ── 8. CTA — conclusion ── */}
      <div className={styles.shell}>
        <section className={styles.finalBand} aria-labelledby="final-cta-heading">
          <h2 id="final-cta-heading" className={styles.finalTitle}>
            Prêt à rejoindre le réseau ?
          </h2>
          <p className={styles.finalSub}>Un formulaire. L’équipe transport oriente la suite.</p>
          <Link to="/contact/transport" className={styles.btnFinal}>
            Ouvrir le formulaire
            <IcoChevR s={14} />
          </Link>
          <p className={styles.finalSecondary}>
            <Link to="/contact" className={styles.finalSecondaryLink}>
              Autre demande
            </Link>
          </p>
        </section>
      </div>

      <div className={styles.bottomSpacer} aria-hidden />
    </div>
  );
};

export default ConduirePage;
