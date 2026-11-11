import styles from './BillPeriodModal.module.css';

/** Affiche les acomptes déjà calculés par le serveur. Aucun bouton d'émission. */
export default function CashPreviewSummary({ summary }) {
  if (!summary) return null;
  return (
    <section
      className={styles.cashPreview}
      data-testid="cash-preview-summary"
      aria-label="Acomptes déjà encaissés"
    >
      <div className={styles.cashPreviewRow}>
        <span>Montant des prestations</span>
        <strong>{summary.services}</strong>
      </div>
      <div className={styles.cashPreviewRow}>
        <span>Déjà encaissé par le chauffeur</span>
        <strong>{summary.collected}</strong>
      </div>
      <div className={styles.cashPreviewRow}>
        <span>Reste à payer</span>
        <strong data-testid="cash-preview-remaining">{summary.remaining}</strong>
      </div>
      {summary.legs.length > 1 ? (
        <ul className={styles.cashPreviewLegs}>
          {summary.legs.map((leg) => (
            <li key={leg.bookingId} data-testid={`cash-preview-leg-${leg.bookingId}`}>
              Prestation {leg.contractual} · Déjà encaissé {leg.collected} · Reste{' '}
              {leg.remaining}
            </li>
          ))}
        </ul>
      ) : null}
    </section>
  );
}
