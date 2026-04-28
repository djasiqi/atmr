// src/pages/company/components/CompanyDriverTableSkeleton.jsx
import React from 'react';
import s from './CompanyDriverTable.module.css';
import sk from './CompanyDriverTableSkeleton.module.css';

const COLS = ['Chauffeur', 'Vehicule', 'Disponibilite', 'Fraicheur', 'Compte', 'Actions'];

const Bar = ({ className }) => <div className={`${sk.bar} ${className || ''}`} aria-hidden />;

function CompanyDriverTableSkeleton({ rowCount = 6 }) {
  return (
    <div className={s.tableContainer} data-testid="company-driver-table-skeleton">
      <div className={sk.srOnly}>Chargement du tableau des chauffeurs…</div>
      <table className={s.table} aria-hidden>
        <thead>
          <tr>
            {COLS.map((label) => (
              <th key={label} className={label === 'Actions' ? s.thActions : undefined}>
                {label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {Array.from({ length: rowCount }, (_, i) => (
            <tr key={i} className={sk.row}>
              <td>
                <div className={s.driverCell}>
                  <div className={sk.avatar} />
                  <div className={s.driverInfo}>
                    <Bar className={sk.name} />
                    <Bar className={sk.email} />
                  </div>
                </div>
              </td>
              <td>
                <Bar className={sk.veh} />
              </td>
              <td>
                <Bar className={sk.pill} />
              </td>
              <td>
                <Bar className={sk.fresh} />
              </td>
              <td>
                <Bar className={sk.account} />
              </td>
              <td className={s.actionsCell}>
                <div className={sk.actions}>
                  <Bar className={sk.btn} />
                  <Bar className={sk.btn} />
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default React.memo(CompanyDriverTableSkeleton);
