import React from 'react';
import styles from './InvoiceSituation.module.css';
import { formatServerAmount } from '../invoiceSituation';

const LINES = [
  ['Montant facturé', 'prestations_chf'],
  ['Avoirs', 'avoirs_chf'],
  ['Net facturé', 'net_facture_chf'],
  ['Acomptes reçus', 'acomptes_chf'],
  ['Paiements complémentaires', 'paiements_complementaires_chf'],
  ['Remboursements confirmés', 'remboursements_confirmes_chf'],
  ['Reste à payer', 'reste_a_payer_chf'],
  ['Montant à rembourser', 'a_rembourser_chf'],
];

function DocumentLink({ document, onOpenPdf }) {
  if (document?.disponible && document.pdf_url) {
    const name = document.label === 'Avoir' ? "Consulter l'avoir" : 'Consulter la facture';
    return (
      <button type="button" onClick={() => onOpenPdf?.(document.pdf_url)}>
        {name}
        {document.reference ? ` ${document.reference}` : ''}
      </button>
    );
  }
  return <span className={styles.missing}>Document indisponible</span>;
}

export default function InvoiceSituation({
  statement,
  error,
  loading,
  onClose,
  onOpenPdf,
}) {
  if (loading) {
    return <p className={styles.panel}>Consultation de la facture…</p>;
  }
  if (error) {
    return (
      <section className={styles.panel} role="alert">
        <p className={styles.error}>{error}</p>
        {onClose && (
          <button type="button" onClick={onClose}>Fermer</button>
        )}
      </section>
    );
  }
  if (!statement) return null;

  return (
    <section className={styles.panel} aria-label="Situation de la facture">
      <h2 className={styles.title}>Situation de la facture</h2>
      <dl className={styles.lines}>
        {LINES.map(([label, key]) => (
          <React.Fragment key={key}>
            <dt>{label}</dt>
            <dd>{formatServerAmount(statement[key])}</dd>
          </React.Fragment>
        ))}
      </dl>
      <ol className={styles.history}>
        {(statement.events || []).map((event, index) => (
          <li key={`${event.label}-${event.at || 'sans-date'}-${index}`}>
            <span>{event.label}</span>
            <span>{event.reference || 'Référence non indiquée'}</span>
            <span>{formatServerAmount(event.montant)}</span>
            <time>{event.at || 'Date non indiquée'}</time>
          </li>
        ))}
      </ol>
      <div className={styles.documents}>
        {(statement.documents || []).map((document, index) => (
          <div key={`${document.label}-${document.reference || index}`}>
            <DocumentLink document={document} onOpenPdf={onOpenPdf} />
          </div>
        ))}
      </div>
      {onClose && (
        <button type="button" onClick={onClose}>Fermer</button>
      )}
    </section>
  );
}
