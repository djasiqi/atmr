import React from 'react';
import styles from '../InvoicesRegistry.module.css';
import sk from './InvoicesTableSkeleton.module.css';

const Bar = ({ className }) => <div className={`${sk.bar} ${className || ''}`} aria-hidden />;

/**
 * Squelette aligné sur le tableau du registre des factures (9 colonnes : sélection, N°, client, échéance, etc.).
 */
function InvoicesTableSkeleton({ rowCount = 7 }) {
  return (
    <div className={styles.tableContainer}>
      <div className={sk.srOnly}>Chargement du tableau des factures…</div>
      <table className={styles.table} data-testid="invoices-table-skeleton" aria-hidden>
        <thead>
          <tr>
            <th className={styles.thCheckbox} />
            <th>N&deg; facture</th>
            <th>Client</th>
            <th>Echeance</th>
            <th>Montant</th>
            <th>Paiement</th>
            <th>Statut</th>
            <th>Rappel</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {Array.from({ length: rowCount }, (_, i) => (
            <tr key={i} className={sk.row}>
              <td className={styles.tdCheckbox}>
                <div className={`${sk.bar} ${sk.wCheckbox}`} />
              </td>
              <td>
                <div className={sk.invoiceStack}>
                  <Bar className={sk.wNumber} />
                  <Bar className={sk.wPeriod} />
                </div>
              </td>
              <td>
                <Bar className={sk.wClient} />
              </td>
              <td>
                <Bar className={sk.wDate} />
              </td>
              <td className={styles.cellAmount}>
                <Bar className={sk.wAmount} />
              </td>
              <td>
                <div className={sk.paymentCol}>
                  <Bar className={sk.wPayLine} />
                  <Bar className={sk.wPayLine} />
                </div>
              </td>
              <td>
                <Bar className={sk.wBadge} />
              </td>
              <td>
                <Bar className={sk.wReminder} />
              </td>
              <td>
                <Bar className={sk.wMenu} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default React.memo(InvoicesTableSkeleton);
