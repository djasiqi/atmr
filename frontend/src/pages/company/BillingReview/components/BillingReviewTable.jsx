// frontend/src/pages/company/BillingReview/components/BillingReviewTable.jsx
import React from 'react';
import styles from './BillingReviewTable.module.css';
import {
  getRecipientLabel,
  getRecipientSourceLabel,
  getRecipientStatus,
  getRecipientWarningText,
} from '../../../../utils/billingRecipient';

const BillingReviewTable = ({
  bookings,
  loading,
  onSetPayer,
  onLock,
  onUnlock,
  isAdmin,
  selectedIds = [],
  onSelectionChange,
  onBatchAction,
  onRowClick,
  emptyMessage,
}) => {
  const getStatusBadge = (status) => {
    const statusMap = {
      draft: { label: 'Brouillon', className: styles.badgeDraft },
      needs_review: { label: 'À vérifier', className: styles.badgeNeedsReview },
      ready: { label: 'Prêt', className: styles.badgeReady },
      locked: { label: 'Verrouillé', className: styles.badgeLocked },
    };
    const statusInfo = statusMap[status] || { label: status, className: styles.badgeDefault };
    return (
      <span className={`${styles.badge} ${statusInfo.className}`}>{statusInfo.label}</span>
    );
  };

  const getRecipientBadgeClass = (recipientStatus) => {
    const map = {
      ok: styles.recipientBadgeOk,
      review: styles.recipientBadgeReview,
      conflict: styles.recipientBadgeConflict,
      voucher: styles.recipientBadgeVoucher,
      missing: styles.recipientBadgeMissing,
      unknown: styles.recipientBadgeUnknown,
    };
    return map[recipientStatus] || styles.recipientBadgeUnknown;
  };

  if (loading) {
    return <div className={styles.loading}>Chargement...</div>;
  }

  if (bookings.length === 0) {
    return <div className={styles.empty}>{emptyMessage || 'Aucun résultat.'}</div>;
  }

  const handleSelectAll = (e) => {
    if (e.target.checked) {
      const allIds = bookings
        .filter((b) => b.status !== 'locked')
        .map((b) => b.booking_id);
      onSelectionChange(allIds);
    } else {
      onSelectionChange([]);
    }
  };

  const handleSelectOne = (bookingId, checked) => {
    if (checked) {
      onSelectionChange([...selectedIds, bookingId]);
    } else {
      onSelectionChange(selectedIds.filter((id) => id !== bookingId));
    }
  };

  const isAllSelected =
    bookings.filter((b) => b.status !== 'locked').length > 0 &&
    bookings
      .filter((b) => b.status !== 'locked')
      .every((b) => selectedIds.includes(b.booking_id));

  const hasSelection = selectedIds.length > 0;

  return (
    <div className={styles.tableContainer}>
      {hasSelection && (
        <div className={styles.batchActions}>
          <span className={styles.selectionCount}>
            {selectedIds.length} booking(s) sélectionné(s)
          </span>
          <button
            className={styles.btnBatch}
            onClick={() => onBatchAction(selectedIds)}
            title="Modifier le payeur en batch"
          >
            ✏️ Modifier payeur ({selectedIds.length})
          </button>
        </div>
      )}
      <table className={styles.table}>
        <thead>
          <tr>
            <th>
              <input
                type="checkbox"
                checked={isAllSelected}
                onChange={handleSelectAll}
                title="Sélectionner tout"
              />
            </th>
            <th>ID</th>
            <th>Date</th>
            <th>Patient</th>
            <th>Payeur</th>
            <th>Source</th>
            <th>Statut</th>
            <th>Montant</th>
            <th>Alertes</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {bookings.map((booking) => {
            const isSelected = selectedIds.includes(booking.booking_id);
            const isLocked = booking.status === 'locked';
            const warningText = getRecipientWarningText(booking);
            const amount = Number(booking.amount || 0);
            const sourceLabel = getRecipientSourceLabel(booking);
            const isSourceUnknown = !booking?.billing_source;
            const reviewStatus = booking?.billing_review_status || booking?.status;
            const tooltipParts = [
              `Source: ${booking?.billing_source || 'inconnue'}`,
              booking?.billing_source_ref ? `Référence: ${booking.billing_source_ref}` : null,
              `Statut review: ${reviewStatus || 'inconnu'}`,
            ].filter(Boolean);
            return (
              <tr
                key={booking.booking_id}
                className={`${styles.rowClickable} ${isSelected ? styles.selectedRow : ''}`}
                onClick={() => onRowClick?.(booking)}
                role="button"
                tabIndex={0}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    onRowClick?.(booking);
                  }
                }}
              >
                <td>
                  <input
                    type="checkbox"
                    checked={isSelected}
            onChange={(e) => handleSelectOne(booking.booking_id, e.target.checked)}
                    disabled={isLocked}
                    title={isLocked ? 'Booking verrouillé' : 'Sélectionner'}
            onClick={(e) => e.stopPropagation()}
                  />
                </td>
        <td>{booking.booking_id}</td>
        <td>{booking.date}</td>
        <td>{booking.patient_name}</td>
        <td>
          <span
            className={`${styles.recipientBadge} ${getRecipientBadgeClass(
              getRecipientStatus(booking)
            )}`}
          >
            {getRecipientLabel(booking)}
          </span>
        </td>
        <td>
          <div className={styles.sourceCell}>
            <span
              className={`${styles.sourceChip} ${
                isSourceUnknown ? styles.sourceChipMuted : ''
              }`}
            >
              {sourceLabel}
            </span>
            <button
              type="button"
              className={styles.sourceTooltip}
              title={tooltipParts.join(' • ')}
              onClick={(e) => e.stopPropagation()}
            >
              Pourquoi ?
            </button>
          </div>
          {booking.billing_source_ref && (
            <div className={styles.sourceRef}>{booking.billing_source_ref}</div>
          )}
        </td>
        <td>{getStatusBadge(booking.status)}</td>
        <td>{amount.toFixed(2)} CHF</td>
        <td>
          {warningText ? (
            <div className={styles.alertChip}>
              <span className={styles.alertChipLabel}>Alerte</span>
              <span className={styles.alertChipText}>{warningText}</span>
            </div>
          ) : (
            <span className={styles.alertChipEmpty}>—</span>
          )}
        </td>
        <td>
          <div className={styles.actions}>
            {booking.status !== 'locked' && (
              <>
                <button
                  className={styles.btnSetPayer}
                  onClick={(e) => {
                    e.stopPropagation();
                    onSetPayer(booking);
                  }}
                  title="Modifier le payeur"
                >
                  ✏️
                </button>
                <button
                  className={styles.btnLock}
                  onClick={(e) => {
                    e.stopPropagation();
                    onLock(booking);
                  }}
                  title="Verrouiller"
                >
                  🔒
                </button>
              </>
            )}
            {booking.status === 'locked' && isAdmin && (
              <button
                className={styles.btnUnlock}
                onClick={(e) => {
                  e.stopPropagation();
                  onUnlock(booking);
                }}
                title="Déverrouiller (admin)"
              >
                🔓
              </button>
            )}
          </div>
        </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
};

export default BillingReviewTable;
