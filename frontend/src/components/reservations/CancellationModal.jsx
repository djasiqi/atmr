import React, { useState, useCallback, useMemo, useRef, useEffect } from 'react';
import { FiAlertTriangle, FiX, FiInfo, FiAlertCircle, FiCreditCard, FiLoader } from 'react-icons/fi';
import Modal from '../common/Modal';
import { CANCELLATION_REASONS } from '../../constants/cancellationReasons';
import s from './CancellationModal.module.css';

const CancellationModal = ({ isOpen, reservation, onConfirm, onClose }) => {
  const [selectedCode, setSelectedCode] = useState(null);
  const [reasonText, setReasonText] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const reasonTextRef = useRef(null);

  useEffect(() => {
    if (selectedCode === 'OTHER' && reasonTextRef.current) {
      reasonTextRef.current.focus();
    }
  }, [selectedCode]);

  const status = useMemo(() => {
    const raw = reservation?.status;
    if (!raw) return '';
    return (typeof raw === 'object' ? raw.value || raw.name || '' : String(raw)).toLowerCase();
  }, [reservation]);

  const resetState = useCallback(() => {
    setSelectedCode(null);
    setReasonText('');
    setLoading(false);
    setError(null);
  }, []);

  const handleClose = useCallback(() => {
    if (loading) return;
    resetState();
    onClose();
  }, [onClose, resetState, loading]);

  const isSimpleDelete = status === 'pending' || status === 'accepted';
  const isBlocked = status === 'in_progress';

  const selectedReason = useMemo(
    () => CANCELLATION_REASONS.find((r) => r.code === selectedCode),
    [selectedCode],
  );

  const isValid = useMemo(() => {
    if (isSimpleDelete) return true;
    if (!selectedCode) return false;
    if (selectedCode === 'OTHER' && !reasonText.trim()) return false;
    return true;
  }, [isSimpleDelete, selectedCode, reasonText]);

  const handleConfirm = useCallback(async () => {
    if (!reservation || loading) return;
    const id = reservation.id;

    setLoading(true);
    setError(null);

    try {
      if (isSimpleDelete) {
        await onConfirm(id, null, null);
      } else {
        const rt = reasonText.trim() || null;
        await onConfirm(id, selectedCode, rt);
      }
      resetState();
    } catch (err) {
      const msg =
        err?.response?.data?.error ||
        err?.response?.data?.message ||
        err?.message ||
        "Erreur lors de l'annulation. Veuillez réessayer.";
      setError(msg);
      setLoading(false);
    }
  }, [reservation, loading, isSimpleDelete, selectedCode, reasonText, onConfirm, resetState]);

  if (!isOpen || !reservation) return null;

  const amount = reservation.amount || reservation.price || 0;
  const clientName = reservation.customer_name || reservation.client?.full_name || '';

  if (isBlocked) {
    return (
      <div className={s.dialogOverlay} onClick={handleClose}>
        <div className={s.dialogModal} onClick={(e) => e.stopPropagation()}>
          <h3 className={s.dialogTitle}>Course en cours</h3>
          <p className={s.dialogMessage}>
            Le patient est à bord. L'annulation n'est pas disponible depuis l'interface web.
          </p>
          <div className={s.dialogActions}>
            <button type="button" className={s.dialogBtnFull} onClick={handleClose}>
              Fermer
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (isSimpleDelete) {
    return (
      <div className={s.dialogOverlay} onClick={handleClose}>
        <div className={s.dialogModal} onClick={(e) => e.stopPropagation()}>
          <h3 className={s.dialogTitle}>Supprimer la réservation</h3>
          <p className={s.dialogMessage}>
            La réservation{clientName ? <> pour <strong>{clientName}</strong></> : ''} sera définitivement supprimée.
          </p>
          {error && <div className={s.dialogError}><FiAlertCircle size={12} /> {error}</div>}
          <div className={s.dialogActions}>
            <button type="button" className={s.dialogCancel} onClick={handleClose} disabled={loading}>
              Annuler
            </button>
            <button
              type="button"
              className={`${s.dialogConfirm} ${s.dialogConfirmDanger}`}
              onClick={handleConfirm}
              disabled={loading}
            >
              {loading ? 'Suppression...' : 'Supprimer'}
            </button>
          </div>
        </div>
      </div>
    );
  }

  const isEnRoute = status === 'en_route';
  const modalTitle = isEnRoute ? 'Annuler la course (chauffeur en route)' : 'Annuler la course';
  const showAmount = selectedReason?.isClientFault && amount > 0;

  return (
    <Modal onClose={handleClose} size="md">
      <div className={s.modal}>
        <div className={s.header}>
          <h3 className={s.title}>{modalTitle}</h3>
          <button type="button" className={s.closeBtn} onClick={handleClose} aria-label="Fermer">
            <FiX size={16} />
          </button>
        </div>

        {isEnRoute ? (
          <div className={s.alertWarning}>
            <FiAlertTriangle size={14} />
            <span>Le chauffeur est en route. La facturation dépend du motif sélectionné.</span>
          </div>
        ) : (
          <div className={s.alertInfo}>
            <FiInfo size={14} />
            <span>Le chauffeur assigné sera automatiquement libéré.</span>
          </div>
        )}

        <div className={s.reasonsList}>
          {CANCELLATION_REASONS.map((reason) => {
            const isSelected = selectedCode === reason.code;
            return (
              <div
                key={reason.code}
                className={`${s.reasonCard} ${isSelected ? s.reasonCardSelected : ''} ${reason.isClientFault ? s.reasonCardClientFault : ''}`}
                onClick={() => { if (!loading) setSelectedCode(reason.code); }}
                role="radio"
                aria-checked={isSelected}
                tabIndex={isSelected ? 0 : -1}
                onKeyDown={(e) => { if (!loading && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); setSelectedCode(reason.code); } }}
              >
                <div className={`${s.reasonRadio} ${isSelected ? s.reasonRadioSelected : ''}`}>
                  {isSelected && <div className={s.reasonRadioDot} />}
                </div>
                <div className={s.reasonContent}>
                  <div className={s.reasonHeader}>
                    <span className={s.reasonLabel}>{reason.label}</span>
                    {reason.isClientFault ? (
                      <span className={s.billingBadge}><FiCreditCard size={9} /> Facturée</span>
                    ) : (
                      <span className={s.noBillingBadge}>Non facturée</span>
                    )}
                  </div>
                  <div className={s.reasonDesc}>{reason.description}</div>
                </div>
              </div>
            );
          })}
        </div>

        {selectedCode === 'OTHER' && (
          <div className={s.reasonTextGroup}>
            <label className={s.reasonTextLabel} htmlFor="cancel-reason-text">Justification</label>
            <textarea
              ref={reasonTextRef}
              id="cancel-reason-text"
              className={s.reasonTextArea}
              placeholder="Décrivez la raison de l'annulation..."
              value={reasonText}
              onChange={(e) => setReasonText(e.target.value)}
              maxLength={500}
              disabled={loading}
            />
          </div>
        )}

        {showAmount && (
          <div className={s.amountPreview}>
            <FiCreditCard size={14} />
            <span>Montant facturé : {Number(amount).toFixed(2)} CHF</span>
          </div>
        )}

        {error && <div className={s.errorBanner}><FiAlertCircle size={13} /> {error}</div>}

        <div className={s.footer}>
          <button type="button" className={s.cancelBtn} onClick={handleClose} disabled={loading}>Annuler</button>
          <button
            type="button"
            className={`${s.dangerBtn} ${(!isValid || loading) ? s.dangerBtnDisabled : ''}`}
            onClick={handleConfirm}
            disabled={!isValid || loading}
          >
            {loading ? <><FiLoader size={13} className={s.spinner} /> Annulation...</> : 'Annuler la course'}
          </button>
        </div>
      </div>
    </Modal>
  );
};

export default CancellationModal;
