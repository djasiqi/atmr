// src/pages/company/Dashboard/components/ReservationTableSkeleton.jsx
import React from 'react';
import tableStyles from './ReservationTable.module.css';
import sk from './ReservationTableSkeleton.module.css';

const COLS = ['Passager', 'Date / Heure', 'Trajet', 'Montant', 'Statut', 'Actions'];

const SkeletonBar = ({ className }) => (
  <div className={`${sk.skeletonBlock} ${className || ''}`} aria-hidden />
);

function ReservationTableSkeleton({ rowCount = 8 }) {
  return (
    <div className={tableStyles.tableContainer} data-testid="reservation-table-skeleton">
      <div className={sk.srOnly}>Chargement du tableau des réservations…</div>
      <table className={tableStyles.table} aria-hidden>
        <thead>
          <tr>
            {COLS.map((label) => (
              <th key={label} className={label === 'Actions' ? tableStyles.actionsCell : undefined}>
                {label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {Array.from({ length: rowCount }, (_, i) => (
            <tr key={i} className={sk.skeletonRow}>
              <td className={tableStyles.clientCell}>
                <div className={sk.clientStack}>
                  <SkeletonBar className={sk.wClientName} />
                  <SkeletonBar className={sk.wClientSub} />
                </div>
              </td>
              <td>
                <SkeletonBar className={sk.wDate} />
              </td>
              <td>
                <div className={sk.locStack}>
                  <SkeletonBar className={sk.wLocation} />
                  <SkeletonBar className={sk.wLocationShort} />
                </div>
              </td>
              <td>
                <SkeletonBar className={sk.wAmount} />
              </td>
              <td>
                <SkeletonBar className={sk.wStatus} />
              </td>
              <td className={tableStyles.actionsCell}>
                <SkeletonBar className={sk.wAction} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default React.memo(ReservationTableSkeleton);
