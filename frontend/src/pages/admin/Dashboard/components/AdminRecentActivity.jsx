import React from 'react';
import { Link } from 'react-router-dom';
import styles from './AdminRecentActivity.module.css';

const formatActivityTime = (iso) => {
  if (!iso) return '—';
  try {
    return new Date(iso).toLocaleString('fr-CH', {
      day: '2-digit',
      month: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
    });
  } catch {
    return String(iso);
  }
};

/**
 * Activité récente (max 5) + lien vers les transports.
 * @param {{ items: Array<{ entity_id?: number|string, label: string, status?: string, occurred_at?: string, action?: string }>, resolveItemTo?: (item) => string|null, listTo: string, emptyLabel?: string }} props
 */
export default function AdminRecentActivity({
  items = [],
  resolveItemTo,
  listTo,
  emptyLabel = 'Aucun événement récent à afficher.',
}) {
  return (
    <section className={styles.section} aria-labelledby="admin-dash-activity-title">
      <div className={styles.head}>
        <h2 id="admin-dash-activity-title" className={styles.title}>
          Activité récente
        </h2>
        <Link to={listTo} className={styles.allLink}>
          Voir tous les transports
        </Link>
      </div>
      {items.length === 0 ? (
        <p className={styles.empty}>{emptyLabel}</p>
      ) : (
        <ul className={styles.list}>
          {items.map((item, idx) => {
            const to = resolveItemTo?.(item) || null;
            const key = `${item.action || 'item'}-${item.entity_id ?? idx}-${item.occurred_at}`;
            const content = (
              <>
                <span className={styles.time}>{formatActivityTime(item.occurred_at)}</span>
                <span className={styles.label}>{item.label}</span>
              </>
            );
            return (
              <li key={key} className={styles.item}>
                {to ? (
                  <Link to={to} className={styles.itemLink}>
                    {content}
                  </Link>
                ) : (
                  <div className={styles.itemStatic}>{content}</div>
                )}
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}
