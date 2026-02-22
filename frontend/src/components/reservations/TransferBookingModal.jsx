import React, { useState, useEffect } from 'react';
import { createPortal } from 'react-dom';
import Modal from '../common/Modal';
import { fetchPartnershipsForTransfer, proposeTransfer } from '../../services/partnershipService';
import { FiSend, FiUser, FiX, FiAlertCircle, FiInfo, FiClock, FiChevronDown } from 'react-icons/fi';
import styles from './TransferBookingModal.module.css';

function PartnerChipDropdown({ partnerships, selected, onSelect, getModelLabel }) {
  const [open, setOpen] = React.useState(false);
  const btnRef = React.useRef(null);
  const menuRef = React.useRef(null);
  const [pos, setPos] = React.useState({ top: 0, left: 0, width: 0 });

  React.useEffect(() => {
    if (!open) return;
    const onClick = (e) => {
      if (btnRef.current?.contains(e.target) || menuRef.current?.contains(e.target)) return;
      setOpen(false);
    };
    const onKey = (e) => { if (e.key === 'Escape') setOpen(false); };
    document.addEventListener('mousedown', onClick);
    document.addEventListener('keydown', onKey);
    return () => { document.removeEventListener('mousedown', onClick); document.removeEventListener('keydown', onKey); };
  }, [open]);

  const reposition = React.useCallback(() => {
    if (!btnRef.current) return;
    const r = btnRef.current.getBoundingClientRect();
    setPos({ top: r.bottom + 4, left: r.left, width: r.width });
  }, []);

  React.useEffect(() => {
    if (!open) return;
    reposition();
    window.addEventListener('scroll', reposition, true);
    window.addEventListener('resize', reposition);
    return () => { window.removeEventListener('scroll', reposition, true); window.removeEventListener('resize', reposition); };
  }, [open, reposition]);

  const label = selected
    ? (selected.partner_company_name || `Partenaire #${selected.id}`)
    : 'Sélectionner un partenaire';

  return (
    <div className={styles.chipDrop}>
      <button
        ref={btnRef}
        type="button"
        className={`${styles.chipBtn} ${selected ? styles.chipBtnActive : ''}`}
        onClick={() => setOpen((p) => !p)}
      >
        <span className={styles.chipText}>{label}</span>
        <FiChevronDown size={11} className={`${styles.chipArrow} ${open ? styles.chipArrowOpen : ''}`} />
      </button>
      {open && createPortal(
        <div
          ref={menuRef}
          className={styles.chipMenu}
          style={{ position: 'fixed', top: pos.top, left: pos.left, width: pos.width, zIndex: 10000 }}
        >
          <button
            type="button"
            className={`${styles.chipOption} ${!selected ? styles.chipOptionActive : ''}`}
            onClick={() => { onSelect(null); setOpen(false); }}
          >
            Aucun
          </button>
          {partnerships.map((p) => (
            <button
              key={p.id}
              type="button"
              className={`${styles.chipOption} ${selected?.id === p.id ? styles.chipOptionActive : ''}`}
              onClick={() => { onSelect(p); setOpen(false); }}
            >
              {p.partner_company_name || `Partenaire #${p.id}`}
              <span className={styles.chipOptionHint}>{getModelLabel(p.default_transfer_model)}</span>
            </button>
          ))}
        </div>,
        document.body
      )}
    </div>
  );
}

const TransferBookingModal = ({ isOpen, onClose, reservation, onSuccess }) => {
  const [partnerships, setPartnerships] = useState([]);
  const [loading, setLoading] = useState(false);
  const [selectedPartnership, setSelectedPartnership] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    if (isOpen) {
      loadPartnerships();
    }
  }, [isOpen]);

  const loadPartnerships = async () => {
    try {
      setLoading(true);
      const data = await fetchPartnershipsForTransfer();
      setPartnerships(data || []);
    } catch (err) {
      console.error('Erreur lors du chargement des partenariats:', err);
      setError('Impossible de charger les partenariats');
    } finally {
      setLoading(false);
    }
  };

  const handleTransfer = async () => {
    if (!selectedPartnership || !reservation) {
      setError('Veuillez selectionner un partenaire');
      return;
    }

    try {
      setLoading(true);
      setError('');
      await proposeTransfer(selectedPartnership.id, reservation.id);
      onSuccess?.();
      onClose();
    } catch (err) {
      console.error('Erreur lors du transfert:', err);
      setError(err?.response?.data?.error || err?.message || 'Erreur lors du transfert');
    } finally {
      setLoading(false);
    }
  };

  if (!isOpen) return null;

  const getModelLabel = (model) => {
    if (model === 'SUBCONTRACT') return 'Sous-traitance';
    if (model === 'ASSIGN_TO_PARTNER') return 'Assignation';
    return 'Marketplace';
  };

  const getModelDesc = (model) => {
    if (model === 'SUBCONTRACT') return 'Vous facturez le client, le partenaire vous facture';
    if (model === 'ASSIGN_TO_PARTNER') return 'Course assignee directement au partenaire';
    return 'Publication sur le marketplace';
  };

  const scheduledTime = reservation?.scheduled_time
    ? new Date(reservation.scheduled_time).toLocaleString('fr-CH', {
        day: '2-digit', month: '2-digit', year: 'numeric',
        hour: '2-digit', minute: '2-digit',
      })
    : null;

  return (
    <Modal onClose={onClose} size="md">
      <div className={styles.modal}>
        <div className={styles.header}>
          <div className={styles.headerLeft}>
            <FiSend size={16} className={styles.headerIcon} />
            <div>
              <h2 className={styles.headerTitle}>Transferer la course</h2>
              <p className={styles.headerHint}>Proposer cette course a un partenaire</p>
            </div>
          </div>
          <button type="button" className={styles.closeBtn} onClick={onClose} aria-label="Fermer">
            <FiX size={18} />
          </button>
        </div>

        <div className={styles.body}>
          {reservation && (
            <div className={styles.courseCard}>
              <div className={styles.courseRow}>
                <div className={styles.courseClient}>
                  <FiUser size={13} />
                  <span className={styles.clientName}>
                    {reservation.client?.full_name || reservation.client_name}
                  </span>
                </div>
                <span className={styles.courseAmount}>
                  {Number(reservation.amount || 0).toFixed(2)} CHF
                </span>
              </div>

              <div className={styles.courseRoute}>
                <div className={styles.routeLine}>
                  <span className={styles.dot} data-type="pickup" />
                  <span className={styles.routeText}>{reservation.pickup_location}</span>
                </div>
                <div className={styles.routeConnector} />
                <div className={styles.routeLine}>
                  <span className={styles.dot} data-type="dropoff" />
                  <span className={styles.routeText}>{reservation.dropoff_location}</span>
                </div>
              </div>

              {scheduledTime && (
                <div className={styles.courseMeta}>
                  <FiClock size={11} />
                  <span>{scheduledTime}</span>
                </div>
              )}
            </div>
          )}

          <div className={styles.fieldGroup}>
            <label htmlFor="partnership-select" className={styles.fieldLabel}>
              Partenaire destinataire
            </label>
            {loading && partnerships.length === 0 ? (
              <div className={styles.loadingText}>Chargement des partenariats...</div>
            ) : partnerships.length === 0 ? (
              <div className={styles.warningBox}>
                <FiAlertCircle size={14} />
                <span>Aucun partenariat actif. Creez-en un depuis Parametres &gt; Partenariats.</span>
              </div>
            ) : (
              <PartnerChipDropdown
                partnerships={partnerships}
                selected={selectedPartnership}
                onSelect={(p) => { setSelectedPartnership(p); setError(''); }}
                getModelLabel={getModelLabel}
              />
            )}
          </div>

          {selectedPartnership && (
            <div className={styles.infoBox}>
              <FiInfo size={13} className={styles.infoIcon} />
              <div className={styles.infoContent}>
                <span className={styles.infoTitle}>
                  {getModelLabel(selectedPartnership.default_transfer_model)}
                </span>
                <span className={styles.infoDesc}>
                  {getModelDesc(selectedPartnership.default_transfer_model)}
                </span>
                {(selectedPartnership.default_margin_percent || selectedPartnership.default_partner_tariff_percent) && (
                  <div className={styles.infoMeta}>
                    {selectedPartnership.default_margin_percent && (
                      <span className={styles.metaTag}>Marge {selectedPartnership.default_margin_percent}%</span>
                    )}
                    {selectedPartnership.default_partner_tariff_percent && (
                      <span className={styles.metaTag}>Tarif {selectedPartnership.default_partner_tariff_percent}%</span>
                    )}
                  </div>
                )}
              </div>
            </div>
          )}

          {error && <div className={styles.errorBox}>{error}</div>}
        </div>

        <div className={styles.footer}>
          <button
            type="button"
            onClick={onClose}
            className={styles.cancelBtn}
            disabled={loading}
          >
            Annuler
          </button>
          <button
            type="button"
            onClick={handleTransfer}
            className={styles.submitBtn}
            disabled={loading || !selectedPartnership || partnerships.length === 0}
          >
            <FiSend size={13} />
            {loading ? 'Transfert en cours...' : 'Transferer'}
          </button>
        </div>
      </div>
    </Modal>
  );
};

export default TransferBookingModal;

