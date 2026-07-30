import React, { useCallback, useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { toast } from 'sonner';
import { executeCompanyActionQueueItem } from '../../../../services/companyService';
import { lirieKeys } from '../../../../queryKeys/lirie';
import { getCurrentAuthEnv } from '../../../../utils/apiClient';
import styles from './ActionQueueNow.module.css';

const KIND_LABELS = {
  pending_decision: 'À décider',
  unassigned: 'Sans chauffeur',
  critical_delay: 'Urgence critique',
};

const ACTION_LABELS = {
  accept: 'Accepter',
  reject: 'Refuser',
  assign: 'Assigner',
  acknowledge: 'Accuser réception',
};

function kindClass(kind) {
  if (kind === 'critical_delay') return styles.kindCritical;
  if (kind === 'unassigned') return styles.kindUnassigned;
  return styles.kindPending;
}

function shortPickup(bookingSummary) {
  const raw = bookingSummary?.pickup_location || '';
  const text = String(raw).trim();
  if (!text) return `Course #${bookingSummary?.id ?? '?'}`;
  return text.length > 48 ? `${text.slice(0, 45)}…` : text;
}

function randomIdempotencyKey() {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID();
  }
  return `idem-${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

export default function ActionQueueNow({
  companyId,
  day,
  actionQueue = [],
  actionQueueTotal = 0,
  truncated = false,
  toHandle = null,
  onAssign,
  onActionComplete,
}) {
  const queryClient = useQueryClient();
  const [busyActionId, setBusyActionId] = useState(null);

  const total = typeof toHandle === 'number' ? toHandle : actionQueueTotal;

  const invalidateBootstrap = useCallback(async () => {
    if (!companyId || !day) return;
    const authEnv = getCurrentAuthEnv();
    await queryClient.invalidateQueries({
      queryKey: lirieKeys.companyDashboardBootstrap(authEnv, companyId, day),
    });
    onActionComplete?.();
  }, [companyId, day, queryClient, onActionComplete]);

  const handleExecute = useCallback(
    async (item, action) => {
      if (!item?.action_id) return;
      setBusyActionId(item.action_id);
      try {
        await executeCompanyActionQueueItem(item.action_id, {
          action,
          expectedVersion: item.version,
          idempotencyKey: randomIdempotencyKey(),
        });
        toast.success('Action enregistrée');
        await invalidateBootstrap();
      } catch (err) {
        const msg =
          err?.response?.data?.message ||
          err?.response?.data?.error ||
          "Impossible d'exécuter l'action";
        toast.error(msg);
      } finally {
        setBusyActionId(null);
      }
    },
    [invalidateBootstrap]
  );

  const handleActionClick = useCallback(
    (item, action) => {
      if (action === 'assign') {
        const bookingId = item.entity_id ?? item.booking_summary?.id;
        if (bookingId != null) onAssign?.(bookingId);
        return;
      }
      void handleExecute(item, action);
    },
    [handleExecute, onAssign]
  );

  return (
    <section className={styles.section} aria-labelledby="action-queue-now-heading">
      <div className={styles.header}>
        <h2 id="action-queue-now-heading" className={styles.title}>
          À traiter maintenant
        </h2>
        <span className={styles.countBadge} aria-label={`${total} action${total !== 1 ? 's' : ''} à traiter`}>
          {total}
        </span>
      </div>

      {truncated ? (
        <p className={styles.truncationNote} role="status">
          Affichage limité — {actionQueue.length} sur {total} actions visibles.
        </p>
      ) : null}

      {total === 0 ? (
        <p className={styles.empty} role="status">
          Aucune action urgente pour le moment.
        </p>
      ) : (
        <ul className={styles.list}>
          {actionQueue.map((item) => {
            const kind = item.kind || 'pending_decision';
            const kindLabel = KIND_LABELS[kind] || kind;
            const pickup = shortPickup(item.booking_summary);
            const allowed = Array.isArray(item.allowed_actions) ? item.allowed_actions : [];
            const isBusy = busyActionId === item.action_id;

            return (
              <li key={item.action_id || `${kind}-${item.entity_id}`} className={styles.item}>
                <span className={`${styles.kindLabel} ${kindClass(kind)}`}>{kindLabel}</span>
                <span className={styles.pickup} title={pickup}>
                  {pickup}
                </span>
                <div className={styles.actions}>
                  {allowed.map((action) => {
                    const label = ACTION_LABELS[action] || action;
                    const accent =
                      action === 'accept'
                        ? styles.actionAccept
                        : action === 'reject'
                          ? styles.actionReject
                          : '';
                    return (
                      <button
                        key={`${item.action_id}-${action}`}
                        type="button"
                        className={`${styles.actionBtn} ${accent}`}
                        aria-label={`${label} — ${kindLabel} — ${pickup}`}
                        disabled={isBusy}
                        onClick={() => handleActionClick(item, action)}
                      >
                        {label}
                      </button>
                    );
                  })}
                </div>
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}
