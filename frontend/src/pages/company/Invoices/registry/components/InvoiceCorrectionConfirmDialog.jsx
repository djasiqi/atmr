/**
 * Confirmation comptable ponctuelle — pas un formulaire permanent.
 */
import React from 'react';
import { FiAlertTriangle } from 'react-icons/fi';
import { money } from '../correctionPreparation';
import { formatServerAmount } from '../invoiceSituation';
import styles from './InvoiceDraftEditModal.module.css';

function chf(v) {
  return formatServerAmount(money(v));
}

const InvoiceCorrectionConfirmDialog = ({
  open,
  originalNumber,
  preview,
  warnings = [],
  unsupported = [],
  reason,
  onReasonChange,
  prepareCorrective,
  onPrepareCorrectiveChange,
  confirmed,
  onConfirmedChange,
  submitting,
  error,
  onCancel,
  onConfirm,
}) => {
  if (!open) return null;

  const amount = money(preview?.proposed_amount_ttc);
  const originalTotal = money(preview?.original_total);
  const alreadyCredited = money(preview?.credit_notes_total_before);
  const creditCeiling = Math.max(0, originalTotal - alreadyCredited);
  const canConfirm =
    confirmed &&
    amount > 0 &&
    String(reason || '').trim().length > 0 &&
    unsupported.length === 0 &&
    !submitting &&
    Boolean(preview);

  return (
    <div className={styles.correctionConfirmOverlay} role="presentation" onClick={onCancel}>
      <div
        className={styles.correctionConfirmDialog}
        role="dialog"
        aria-modal="true"
        aria-labelledby="correction-confirm-title"
        data-testid="invoice-correction-confirm"
        onClick={(e) => e.stopPropagation()}
      >
        <div className={styles.correctionConfirmHead}>
          <FiAlertTriangle size={18} aria-hidden />
          <h3 id="correction-confirm-title">Confirmer la correction</h3>
        </div>
        <p className={styles.correctionWorkbenchHint}>
          L’original <strong>{originalNumber}</strong> (PDF, numéro, paiements) reste
          inchangé. Les écritures ci-dessous seront créées après confirmation.
        </p>

        {preview ? (
          <dl className={styles.correctionSnapshot}>
            <div>
              <dt>Avoir HT</dt>
              <dd>{chf(preview.proposed_amount_ht)}</dd>
            </div>
            <div>
              <dt>Avoir TVA</dt>
              <dd>{chf(preview.proposed_amount_vat)}</dd>
            </div>
            <div>
              <dt>Avoir TTC</dt>
              <dd>
                <strong>{chf(amount)}</strong>
              </dd>
            </div>
            <div>
              <dt>Plafond créditable</dt>
              <dd>{chf(creditCeiling)}</dd>
            </div>
            <div>
              <dt>Net facturé après</dt>
              <dd>{chf(preview.net_invoiced_total)}</dd>
            </div>
            <div>
              <dt>Solde après</dt>
              <dd>{chf(preview.balance_due)}</dd>
            </div>
            <div>
              <dt>Trop-perçu</dt>
              <dd>{chf(preview.overpayment_amount)}</dd>
            </div>
          </dl>
        ) : (
          <p className={styles.muted}>Aucun aperçu serveur disponible.</p>
        )}

        {warnings.length > 0 ? (
          <ul className={styles.correctionWarningList}>
            {warnings.map((w) => (
              <li key={w}>{w}</li>
            ))}
          </ul>
        ) : null}

        {unsupported.length > 0 ? (
          <ul className={styles.correctionErrorList} data-testid="correction-unsupported">
            {unsupported.map((u) => (
              <li key={u.invoice_line_id}>{u.reason}</li>
            ))}
          </ul>
        ) : null}

        <label className={styles.correctionField}>
          Motif de correction
          <textarea
            value={reason}
            onChange={(e) => onReasonChange(e.target.value)}
            rows={3}
            maxLength={2000}
            disabled={submitting}
            placeholder="Ex. : prestation facturée à tort, montant erroné…"
          />
        </label>

        <label className={styles.correctionConfirm}>
          <input
            type="checkbox"
            checked={prepareCorrective}
            onChange={(e) => onPrepareCorrectiveChange(e.target.checked)}
            disabled={submitting || amount <= 0}
          />
          Préparer une facture rectificative liée (brouillon, sans déplacer les paiements)
        </label>

        <label className={styles.correctionConfirm}>
          <input
            type="checkbox"
            checked={confirmed}
            onChange={(e) => onConfirmedChange(e.target.checked)}
            disabled={submitting || amount <= 0 || unsupported.length > 0}
          />
          Je confirme l’émission d’un avoir de {chf(amount)} sur {originalNumber}.
        </label>

        {error ? <p className={styles.err}>{error}</p> : null}

        <div className={styles.footer}>
          <div className={styles.footerGroup}>
            <button
              type="button"
              className={styles.btnMuted}
              onClick={onCancel}
              disabled={submitting}
            >
              Annuler
            </button>
            <button
              type="button"
              className={`${styles.btn} ${styles.btnPrimary}`}
              onClick={onConfirm}
              disabled={!canConfirm}
              data-testid="correction-confirm-submit"
            >
              {submitting
                ? 'Émission…'
                : prepareCorrective
                  ? 'Confirmer l’avoir et préparer la rectificative'
                  : 'Confirmer l’avoir'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default InvoiceCorrectionConfirmDialog;
