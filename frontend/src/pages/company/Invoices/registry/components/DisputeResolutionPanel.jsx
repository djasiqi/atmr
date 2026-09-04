import React, { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { FiX } from 'react-icons/fi';
import { invoiceService, formatCurrencyCHF } from '../../../../../services/invoiceService';
import { getApiErrorMessage } from '../../../../../utils/apiErrorMessage';
import { formatPreviewDayMonth } from '../../../../../utils/invoiceLinesPreviewUi';
import {
  CARRIER_STANCES,
  EVIDENCE_KINDS,
  EXCLUSION_REASONS,
  hasUploadedEvidence,
  presentInstitutionDisputeReason,
  systemFactsLines,
  unwrapDisputePayload,
} from '../../../../../utils/bookingDisputeUi';
import styles from './DisputeResolutionPanel.module.css';

const DisputeResolutionPanel = ({
  companyId,
  row,
  onClose,
  onChanged,
}) => {
  const dialogRef = useRef(null);
  const [dispute, setDispute] = useState(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [stance, setStance] = useState('');
  const [step, setStep] = useState('choose');
  const [exclusionReason, setExclusionReason] = useState('created_by_error');
  const [note, setNote] = useState('');
  const [evidenceKind, setEvidenceKind] = useState('signed_transport_sheet');
  const [evidenceNote, setEvidenceNote] = useState('');
  const [proposedAmount, setProposedAmount] = useState('');
  const [proposedPayer, setProposedPayer] = useState('clinic');

  useEffect(() => {
    setStance('');
    setStep('choose');
    setError('');
    setDispute(null);
    let cancelled = false;
    const load = async () => {
      setLoading(true);
      try {
        const res = await invoiceService.getBookingDispute(companyId, row.bookingId);
        if (cancelled) return;
        const payload = unwrapDisputePayload(res);
        setDispute(payload);
        if (payload?.carrier_stance) setStance(payload.carrier_stance);
      } catch (err) {
        if (!cancelled) {
          setError(getApiErrorMessage(err) || 'Impossible de charger la contestation.');
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    void load();
    return () => {
      cancelled = true;
    };
  }, [companyId, row.bookingId]);

  useEffect(() => {
    const node = dialogRef.current;
    if (node) node.focus();
  }, [row.bookingId]);

  useEffect(() => {
    const onKey = (event) => {
      if (event.key !== 'Escape') return;
      event.preventDefault();
      event.stopPropagation();
      onClose();
    };
    document.addEventListener('keydown', onKey, true);
    return () => document.removeEventListener('keydown', onKey, true);
  }, [onClose]);

  const afterChange = async (res) => {
    const payload = unwrapDisputePayload(res);
    if (payload) setDispute(payload);
    if (typeof onChanged === 'function') onChanged();
  };

  const run = async (fn) => {
    setBusy(true);
    setError('');
    try {
      const res = await fn();
      await afterChange(res);
    } catch (err) {
      setError(getApiErrorMessage(err) || 'Action impossible.');
    } finally {
      setBusy(false);
    }
  };

  const who = row.patientName || dispute?.patient_name || '';
  const when = formatPreviewDayMonth(row.scheduledAt || dispute?.scheduled_at);
  const amount = formatCurrencyCHF(row.amountHt || dispute?.amount_ht || 0);
  const status = String(dispute?.status || '');
  const waitingValidation = status === 'evidence_submitted';
  const resolved = status === 'resolved_institution' || status === 'resolved_carrier';
  const reason = presentInstitutionDisputeReason(
    dispute?.institution_reason_code,
    dispute?.institution_reason_text
  );

  const dialog = (
    <div
      className={styles.overlay}
      data-testid="dispute-resolution-overlay"
      data-placement="viewport-fixed"
      onClick={(event) => {
        event.stopPropagation();
        onClose();
      }}
    >
      <div
        ref={dialogRef}
        className={styles.dialog}
        data-testid="dispute-resolution-panel"
        role="dialog"
        aria-modal="true"
        aria-label={`Contestation${who ? ` — ${who}` : ''}`}
        tabIndex={-1}
        onClick={(event) => event.stopPropagation()}
      >
        <div className={styles.header}>
          <h3>Contestation{who ? ` — ${who}` : ''}</h3>
          <button
            type="button"
            className={styles.close}
            onClick={onClose}
            aria-label="Fermer la contestation"
          >
            <FiX size={16} aria-hidden />
          </button>
        </div>
        <div className={styles.body}>
          <p className={styles.meta}>{[when, amount].filter(Boolean).join(' · ')}</p>
          {loading ? (
            <p className={styles.hint}>Chargement…</p>
          ) : (
            <>
              <p className={styles.reasonLabel}>Motif signalé par l'institution</p>
              <p className={styles.reasonCategory}>{reason.category}</p>
              {reason.comment ? (
                <p className={styles.reasonComment}>{reason.comment}</p>
              ) : null}
              {error ? (
                <p className={styles.error} role="alert">{error}</p>
              ) : null}

              {resolved ? (
                <p className={styles.hint}>
                  Résolue le {dispute?.resolved_at || '—'}. Décision :{' '}
                  {status === 'resolved_institution' ? 'institution' : 'transporteur'}.
                  {dispute?.resolution_note ? ` Motif : ${dispute.resolution_note}` : ''}
                </p>
              ) : waitingValidation ? (
                <p className={styles.hint}>
                  Justificatif soumis. La course reste bloquée jusqu'à validation
                  par l'institution ou un opérateur LIRIE.
                </p>
              ) : (
                <>
                  <fieldset className={styles.fieldset}>
                    <legend>Que s'est-il passé ?</legend>
                    {CARRIER_STANCES.map((option) => (
                      <label key={option.value} className={styles.radio}>
                        <input
                          type="radio"
                          name="dispute-stance"
                          value={option.value}
                          checked={stance === option.value}
                          disabled={busy}
                          onChange={() => setStance(option.value)}
                        />
                        {option.label}
                      </label>
                    ))}
                  </fieldset>

                  {step === 'choose' ? (
                    <div className={styles.actions}>
                      <button
                        type="button"
                        className={styles.primary}
                        disabled={!stance || busy}
                        data-testid="dispute-continue"
                        onClick={() => setStep('act')}
                      >
                        Continuer
                      </button>
                    </div>
                  ) : null}

                  {step === 'act' && stance === 'institution_right' ? (
                    <div className={styles.block}>
                      <p>Cette prestation ne doit pas être facturée.</p>
                      <label className={styles.label} htmlFor="dispute-exclusion">
                        Motif
                      </label>
                      <select
                        id="dispute-exclusion"
                        value={exclusionReason}
                        disabled={busy}
                        onChange={(e) => setExclusionReason(e.target.value)}
                      >
                        {EXCLUSION_REASONS.map((option) => (
                          <option key={option.value} value={option.value}>
                            {option.label}
                          </option>
                        ))}
                      </select>
                      <textarea
                        value={note}
                        disabled={busy}
                        onChange={(e) => setNote(e.target.value)}
                        placeholder="Note interne (optionnel)"
                      />
                      <button
                        type="button"
                        className={styles.primary}
                        disabled={busy}
                        data-testid="dispute-confirm-exclusion"
                        onClick={() =>
                          run(() =>
                            invoiceService.respondBookingDispute(companyId, row.bookingId, {
                              stance: 'institution_right',
                              exclusion_reason: exclusionReason,
                              note,
                            })
                          )
                        }
                      >
                        Confirmer l'exclusion définitive
                      </button>
                    </div>
                  ) : null}

                  {step === 'act' && stance === 'needs_correction' ? (
                    <div className={styles.block}>
                      <label className={styles.label} htmlFor="dispute-amount">
                        Montant HT proposé
                      </label>
                      <input
                        id="dispute-amount"
                        type="number"
                        min="0"
                        step="0.05"
                        value={proposedAmount}
                        disabled={busy}
                        onChange={(e) => setProposedAmount(e.target.value)}
                      />
                      <label className={styles.label} htmlFor="dispute-payer">
                        Payeur proposé
                      </label>
                      <select
                        id="dispute-payer"
                        value={proposedPayer}
                        disabled={busy}
                        onChange={(e) => setProposedPayer(e.target.value)}
                      >
                        <option value="clinic">Institution</option>
                        <option value="patient">Patient</option>
                      </select>
                    </div>
                  ) : null}

                  {step === 'act' && (stance === 'mission_done' || stance === 'needs_correction') ? (
                    <div className={styles.block}>
                      <p className={styles.reasonLabel}>Éléments déjà connus du système</p>
                      <ul className={styles.facts}>
                        {systemFactsLines(dispute?.system_facts).map((line) => (
                          <li key={line}>{line}</li>
                        ))}
                      </ul>
                      <p>Ajouter un justificatif (obligatoire — le snapshot système ne suffit pas).</p>
                      <select
                        value={evidenceKind}
                        disabled={busy}
                        onChange={(e) => setEvidenceKind(e.target.value)}
                        aria-label="Type de preuve"
                      >
                        {EVIDENCE_KINDS.map((option) => (
                          <option key={option.value} value={option.value}>
                            {option.label}
                          </option>
                        ))}
                      </select>
                      <textarea
                        value={evidenceNote}
                        disabled={busy}
                        onChange={(e) => setEvidenceNote(e.target.value)}
                        placeholder="Référence, horaires, précisions…"
                      />
                      <button
                        type="button"
                        className={styles.linkBtn}
                        disabled={busy}
                        data-testid="dispute-add-evidence"
                        onClick={() =>
                          run(() =>
                            invoiceService.addBookingDisputeEvidence(companyId, row.bookingId, {
                              kind: evidenceKind,
                              note: evidenceNote,
                            })
                          )
                        }
                      >
                        + Ajouter une preuve
                      </button>
                      {(dispute?.evidence || []).filter((e) => e.source === 'uploaded').length ? (
                        <ul className={styles.facts}>
                          {dispute.evidence
                            .filter((e) => e.source === 'uploaded')
                            .map((e) => (
                              <li key={e.id}>{e.kind}{e.note ? ` — ${e.note}` : ''}</li>
                            ))}
                        </ul>
                      ) : null}
                      <button
                        type="button"
                        className={styles.primary}
                        disabled={busy || !hasUploadedEvidence(dispute)}
                        data-testid="dispute-submit-validation"
                        onClick={() =>
                          run(async () => {
                            if (stance && stance !== dispute?.carrier_stance) {
                              await invoiceService.respondBookingDispute(companyId, row.bookingId, {
                                stance,
                                note,
                                proposed_amount_ht: proposedAmount ? Number(proposedAmount) : null,
                                proposed_payer_type: proposedPayer,
                                proposed_correction_note: note,
                              });
                            }
                            return invoiceService.submitBookingDispute(companyId, row.bookingId);
                          })
                        }
                      >
                        Soumettre pour validation
                      </button>
                    </div>
                  ) : null}
                </>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );

  if (typeof document === 'undefined' || !document.body) return null;
  return createPortal(dialog, document.body);
};

export default DisputeResolutionPanel;
