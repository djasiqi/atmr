import { Link } from 'react-router-dom';
import styles from './AProposPage.module.css';

function IcoChevR({ s = 14 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <polyline points="9 18 15 12 9 6" />
    </svg>
  );
}

const AProposPage = () => {
  return (
    <div className={styles.page}>
      <header className={styles.hero}>
        <div className={styles.heroText}>
          <h1 className={styles.heroTitle}>Une mission paraît simple.</h1>

          <p className={styles.heroHook}>Pourtant, elle traverse déjà bien davantage qu’un trajet.</p>

          <p className={styles.heroLead}>
            Un départ, un trajet, une arrivée. Entre ces trois moments, plusieurs organisations se transmettent une même
            responsabilité.
          </p>
        </div>

        <div className={styles.heroVisual}>
          <img
            src="/images/lirie-mission-transport-pmr-hopital.webp"
            alt="Plusieurs professionnels coordonnent une mission de transport PMR à la sortie d’un hôpital."
            className={styles.heroImage}
            width={1448}
            height={1086}
            decoding="async"
            loading="eager"
            fetchPriority="high"
            draggable={false}
          />
        </div>
      </header>

      <section id="dispersion" className={styles.dispersion} aria-labelledby="dispersion-heading">
        <div className={styles.shell}>
          <h2 id="dispersion-heading" className={styles.actTitle}>
            L’information se disperse.
          </h2>

          <p className={styles.actLead}>
            Elle passe d’un appel à un courriel, d’un service à un autre, d’un interlocuteur au suivant. La mission reste
            la même, mais chacun n’en voit qu’une partie.
          </p>

          <p className={styles.dispersionClosing}>
            Avec le temps, ce qui était évident au départ devient difficile à reconstituer.
          </p>
        </div>
      </section>

      <section id="bascule" className={styles.actBascule} aria-labelledby="bascule-heading">
        <div className={styles.shell}>
          <h2 id="bascule-heading" className={styles.basculeTitle}>
            Plus personne ne possède toute la mission.
          </h2>
          <p className={styles.basculeLead}>
            Un moment arrive où reconstituer naturellement ce qui s’est passé n’est plus possible. Le véritable défi
            n’est pas le transport. C’est la <strong>continuité de l’information</strong> autour d’une même mission.
          </p>
        </div>
      </section>

      <section id="cadre-commun" className={styles.act} aria-labelledby="cadre-commun-heading">
        <div className={styles.shell}>
          <h2 id="cadre-commun-heading" className={styles.actTitle}>
            Un même fil.
          </h2>

          <p className={styles.actLead}>
            LIRIE donne à cette mission une forme commune, lisible et opérationnelle, sans remplacer les organisations
            qui la portent.
          </p>

          <p className={styles.actLead}>
            LIRIE est une plateforme suisse de coordination des transports. Elle permet aux patients, aux établissements
            de santé et aux entreprises de transport de partager une même demande, de suivre son avancement et de
            conserver un historique des échanges. LIRIE fournit l&apos;outil de coordination, mais n&apos;exécute pas
            elle-même les prestations de transport. Le déploiement principal concerne Genève et la Suisse romande.
          </p>

          <p className={styles.actSecondary}>
            Tant que plusieurs organisations coopéreront autour d’une même mission, celle-ci aura besoin de rester une
            seule mission.
          </p>
        </div>
      </section>

      <section id="cadre" className={styles.actCadre} aria-labelledby="cadre-heading">
        <div className={styles.shell}>
          <h2 id="cadre-heading" className={styles.actTitle}>
            Chacun garde sa responsabilité.
          </h2>
          <p className={styles.cadreLead}>
            Lirie n’exécute pas les prestations de transport. Les missions sont réalisées par des entreprises
            partenaires juridiquement indépendantes. La plateforme facilite l’organisation et le suivi sans modifier les
            responsabilités des acteurs.
          </p>
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
      </section>

      <div className={styles.shell}>
        <section className={styles.finalBand} aria-labelledby="final-cta-heading">
          <h2 id="final-cta-heading" className={styles.finalTitle}>
            Voir comment cette approche s’intègre à votre organisation
          </h2>
          <p className={styles.finalSub}>Une présentation adaptée à votre contexte.</p>
          <div className={styles.finalActions}>
            <Link to="/contact/institution" className={styles.btnFinal}>
              Demander une présentation
              <IcoChevR s={14} />
            </Link>
            <Link to="/contact" className={styles.finalSecondary}>
              Contacter l’équipe
            </Link>
          </div>
        </section>
      </div>

      <div className={styles.bottomSpacer} aria-hidden />
    </div>
  );
};

export default AProposPage;
