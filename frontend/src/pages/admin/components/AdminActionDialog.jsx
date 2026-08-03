import React, { useEffect, useId, useRef, useState } from 'react';
import styles from './AdminActionDialog.module.css';

/**
 * @typedef {object} AdminActionDialogProps
 * @property {boolean} open
 * @property {string} title
 * @property {import('react').ReactNode} description
 * @property {import('react').ReactNode} [impact]
 * @property {string} confirmationLabel
 * @property {string} [confirmText]
 * @property {{ required: boolean, label: string, minLength?: number }} [reason]
 * @property {boolean} [loading]
 * @property {boolean} [danger]
 * @property {(payload: { reason?: string }) => Promise<void>} onConfirm
 * @property {() => void} onClose
 */

/**
 * Dialogue d’action sensible admin — contrat strict (pas d’objet métier libre).
 * @param {AdminActionDialogProps} props
 */
export default function AdminActionDialog({
  open,
  title,
  description,
  impact,
  confirmationLabel,
  confirmText,
  reason,
  loading = false,
  danger = false,
  onConfirm,
  onClose,
}) {
  const titleId = useId();
  const [reasonValue, setReasonValue] = useState('');
  const [confirmValue, setConfirmValue] = useState('');
  const [error, setError] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const inFlightRef = useRef(false);

  useEffect(() => {
    if (!open) return undefined;
    setReasonValue('');
    setConfirmValue('');
    setError(null);
    setSubmitting(false);
    inFlightRef.current = false;
    const onKey = (e) => {
      if (e.key === 'Escape' && !loading && !inFlightRef.current) onClose();
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [open, loading, onClose]);

  if (!open) return null;

  const busy = loading || submitting;

  const reasonOk =
    !reason?.required ||
    (reasonValue.trim().length >= (reason.minLength ?? 1));
  const confirmOk = !confirmText || confirmValue.trim() === confirmText;
  const canSubmit = reasonOk && confirmOk && !busy;

  const handleClose = () => {
    if (busy || inFlightRef.current) return;
    onClose();
  };

  const handleConfirm = async () => {
    if (!reasonOk || !confirmOk || inFlightRef.current || loading || submitting) {
      return;
    }
    inFlightRef.current = true;
    setError(null);
    setSubmitting(true);
    try {
      await onConfirm({
        reason: reason ? reasonValue.trim() : undefined,
      });
    } catch (e) {
      const msg =
        e?.response?.data?.error ||
        e?.response?.data?.message ||
        e?.message ||
        'Une erreur est survenue.';
      setError(String(msg));
      inFlightRef.current = false;
      setSubmitting(false);
      return;
    }
    inFlightRef.current = false;
    setSubmitting(false);
  };

  return (
    <div className={styles.overlay} role="presentation" onClick={handleClose}>
      <div
        className={styles.dialog}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        onClick={(e) => e.stopPropagation()}
      >
        <h2 id={titleId} className={styles.title}>
          {title}
        </h2>
        <div className={styles.description}>{description}</div>
        {impact ? <div className={styles.impact}>{impact}</div> : null}

        {reason ? (
          <label className={styles.field}>
            <span>
              {reason.label}
              {reason.required ? ' *' : ''}
            </span>
            <textarea
              value={reasonValue}
              onChange={(e) => setReasonValue(e.target.value)}
              disabled={busy}
              rows={3}
            />
          </label>
        ) : null}

        {confirmText ? (
          <label className={styles.field}>
            <span>Saisir « {confirmText} » pour confirmer</span>
            <input
              type="text"
              value={confirmValue}
              onChange={(e) => setConfirmValue(e.target.value)}
              disabled={busy}
              autoComplete="off"
            />
          </label>
        ) : null}

        {error ? (
          <div className={styles.error} role="alert">
            {error}
          </div>
        ) : null}

        <div className={styles.actions}>
          <button
            type="button"
            className={styles.btnSecondary}
            onClick={handleClose}
            disabled={busy}
          >
            Annuler
          </button>
          <button
            type="button"
            className={danger ? styles.btnDanger : styles.btnPrimary}
            onClick={handleConfirm}
            disabled={!canSubmit}
          >
            {busy ? 'Traitement…' : confirmationLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
