// src/components/reservations/TransferBookingModal.jsx
import React, { useState, useEffect } from 'react';
import Modal from '../common/Modal';
import { fetchPartnershipsForTransfer, proposeTransfer } from '../../services/partnershipService';
import styles from './TransferBookingModal.module.css';

/**
 * Modal pour transférer une course à un partenaire
 */
const TransferBookingModal = ({ isOpen, onClose, reservation, onSuccess }) => {
  const [partnerships, setPartnerships] = useState([]);
  const [loading, setLoading] = useState(false);
  const [selectedPartnership, setSelectedPartnership] = useState(null);
  const [error, setError] = useState('');

  // Charger les partenariats acceptés
  useEffect(() => {
    if (isOpen) {
      loadPartnerships();
    }
  }, [isOpen]);

  const loadPartnerships = async () => {
    try {
      setLoading(true);
      // Utiliser la route spécifique pour les transferts (uniquement partenariats où l'entreprise est propriétaire)
      const data = await fetchPartnershipsForTransfer();
      // #region agent log
      fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'TransferBookingModal.jsx:loadPartnerships:raw_data',message:'Raw partnerships for transfer data received',data:{count:data?.length || 0,partnerships:data?.map(p=>({id:p.id,status:p.status,is_active:p.is_active,owner_company_id:p.owner_company_id,partner_company_id:p.partner_company_id,owner_company_name:p.owner_company_name,partner_company_name:p.partner_company_name})) || []},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'A'})}).catch(()=>{});
      // #endregion
      // Les partenariats retournés sont déjà filtrés (acceptés, actifs, et où l'entreprise est propriétaire)
      setPartnerships(data || []);
    } catch (err) {
      console.error('Erreur lors du chargement des partenariats:', err);
      setError('Impossible de charger les partenariats');
      // #region agent log
      fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'TransferBookingModal.jsx:loadPartnerships:error',message:'Error loading partnerships',data:{error:err?.message || String(err),response:err?.response?.data},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'A'})}).catch(()=>{});
      // #endregion
    } finally {
      setLoading(false);
    }
  };

  const handleTransfer = async () => {
    if (!selectedPartnership || !reservation) {
      setError('Veuillez sélectionner un partenaire');
      return;
    }

    try {
      setLoading(true);
      setError('');
      await proposeTransfer(selectedPartnership.id, reservation.id);
      onSuccess?.();
      onClose();
      // #region agent log
      fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'TransferBookingModal.jsx:handleTransfer:success',message:'Transfer proposed successfully',data:{booking_id:reservation.id,partnership_id:selectedPartnership.id},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'A'})}).catch(()=>{});
      // #endregion
    } catch (err) {
      console.error('Erreur lors du transfert:', err);
      setError(err?.response?.data?.error || err?.message || 'Erreur lors du transfert');
      // #region agent log
      fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'TransferBookingModal.jsx:handleTransfer:error',message:'Transfer failed',data:{booking_id:reservation?.id,partnership_id:selectedPartnership?.id,error:err?.response?.data?.error || err?.message},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'A'})}).catch(()=>{});
      // #endregion
    } finally {
      setLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      title="🔄 Transférer la course à un partenaire"
      size="medium"
    >
      <div className={styles.transferModal}>
        {reservation && (
          <div className={styles.bookingInfo}>
            <h3>Course à transférer</h3>
            <p>
              <strong>Client:</strong> {reservation.client?.full_name || reservation.customer_name}
            </p>
            <p>
              <strong>Départ:</strong> {reservation.pickup_location}
            </p>
            <p>
              <strong>Arrivée:</strong> {reservation.dropoff_location}
            </p>
            <p>
              <strong>Montant:</strong> {Number(reservation.amount || 0).toFixed(2)} CHF
            </p>
          </div>
        )}

        <div className={styles.partnershipSelection}>
          <label htmlFor="partnership-select">
            <strong>Sélectionner un partenaire *</strong>
          </label>
          {loading && partnerships.length === 0 ? (
            <p>Chargement des partenariats...</p>
          ) : partnerships.length === 0 ? (
            <p className={styles.noPartnerships}>
              Aucun partenariat actif disponible. Créez un partenariat depuis la page
              "Partenariats et sous-traitance".
            </p>
          ) : (
            <select
              id="partnership-select"
              value={selectedPartnership?.id || ''}
              onChange={(e) => {
                const partnership = partnerships.find(
                  (p) => p.id === parseInt(e.target.value, 10)
                );
                setSelectedPartnership(partnership);
                setError('');
              }}
              className={styles.select}
            >
              <option value="">-- Choisir un partenaire --</option>
              {partnerships.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.partner_company_name || `Partenaire #${p.id}`} -{' '}
                  {p.default_transfer_model === 'SUBCONTRACT'
                    ? 'Sous-traitance'
                    : p.default_transfer_model === 'ASSIGN_TO_PARTNER'
                      ? 'Assignation'
                      : 'Marketplace'}
                </option>
              ))}
            </select>
          )}
        </div>

        {selectedPartnership && (
          <div className={styles.transferInfo}>
            <h4>Informations du transfert</h4>
            <p>
              <strong>Modèle:</strong>{' '}
              {selectedPartnership.default_transfer_model === 'SUBCONTRACT'
                ? 'Sous-traitance (A facture client, B facture A)'
                : selectedPartnership.default_transfer_model === 'ASSIGN_TO_PARTNER'
                  ? 'Assignation au partenaire'
                  : 'Marketplace'}
            </p>
            {selectedPartnership.default_margin_percent && (
              <p>
                <strong>Marge:</strong> {selectedPartnership.default_margin_percent}%
              </p>
            )}
            {selectedPartnership.default_partner_tariff_percent && (
              <p>
                <strong>Tarif partenaire:</strong>{' '}
                {selectedPartnership.default_partner_tariff_percent}% du prix client
              </p>
            )}
          </div>
        )}

        {error && <div className={styles.error}>{error}</div>}

        <div className={styles.actions}>
          <button
            type="button"
            onClick={onClose}
            className={styles.cancelButton}
            disabled={loading}
          >
            Annuler
          </button>
          <button
            type="button"
            onClick={handleTransfer}
            className={styles.transferButton}
            disabled={loading || !selectedPartnership || partnerships.length === 0}
          >
            {loading ? 'Transfert en cours...' : 'Transférer'}
          </button>
        </div>
      </div>
    </Modal>
  );
};

export default TransferBookingModal;

