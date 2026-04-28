// src/pages/company/Clients/components/ClientsTableSkeleton.jsx
import React from 'react';
import s from './ClientsTable.module.css';
import sk from './ClientsTableSkeleton.module.css';

const Bar = ({ className }) => <div className={`${sk.bar} ${className || ''}`} aria-hidden />;

function ClientsTableSkeleton({ rowCount = 8 }) {
  return (
    <div className={s.tableContainer}>
      <div className={s.tableScroll} data-testid="clients-table-skeleton">
        <div className={sk.srOnly}>Chargement du tableau des clients…</div>
        <table className={s.table} aria-hidden>
          <thead>
            <tr>
              <th>Client</th>
              <th>Contact</th>
              <th>Adresse</th>
              <th>Statut</th>
              <th>Cree le</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {Array.from({ length: rowCount }, (_, i) => (
              <tr key={i} className={sk.row}>
                <td>
                  <div className={s.clientName}>
                    <div className={sk.nameStack}>
                      <Bar className={sk.wName} />
                      <Bar className={sk.wId} />
                    </div>
                  </div>
                </td>
                <td>
                  <Bar className={sk.wContact} />
                </td>
                <td>
                  <Bar className={sk.wAddress} />
                </td>
                <td>
                  <Bar className={sk.wStatus} />
                </td>
                <td className={s.dateCell}>
                  <Bar className={sk.wDate} />
                </td>
                <td className={s.tdActions}>
                  <Bar className={sk.wMenu} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default React.memo(ClientsTableSkeleton);
