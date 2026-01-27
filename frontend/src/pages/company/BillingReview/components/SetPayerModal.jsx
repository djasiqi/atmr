// frontend/src/pages/company/BillingReview/components/SetPayerModal.jsx
import React, { useState, useEffect } from 'react';
import { fetchBillingParties } from '../../../../services/settingsService';
import styles from './SetPayerModal.module.css';

const SetPayerModal = ({ booking, companyId, onClose, onSave }) => {
  const [form, setForm] = useState({
    billed_to_type: 'patient',
    billing_party_id: null,
    billed_to_company_id: null,
    reason: '',
  });
  const [billingParties, setBillingParties] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (companyId) {
      loadBillingParties();
    }
  }, [companyId]);

  const loadBillingParties = async () => {
    try {
      const response = await fetchBillingParties();
      setBillingParties(response.data || []);
    } catch (err) {
      console.error('Erreur lors du chargement des tiers payeurs:', err);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!form.reason.trim()) {
      setError('Le motif est obligatoire');
      return;
    }

    try {
      setLoading(true);
      setError(null);
      await onSave(booking.booking_id, form);
    } catch (err) {
      setError(err.response?.data?.error || 'Erreur lors de la modification');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={styles.modalOverlay} onClick={onClose}>
      <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
        <div className={styles.modalHeader}>
          <h2>
            {booking.isBatch
              ? `Modifier le payeur (${booking.booking_ids.length} bookings)`
              : 'Modifier le payeur'}
          </h2>
          <button className={styles.closeBtn} onClick={onClose}>
            ×
          </button>
        </div>

        <form onSubmit={handleSubmit} className={styles.form}>
          <div className={styles.formGroup}>
            <label htmlFor="billed_to_type">Type de payeur *</label>
            <select
              id="billed_to_type"
              value={form.billed_to_type}
              onChange={(e) =>
                setForm({ ...form, billed_to_type: e.target.value, billing_party_id: null, billed_to_company_id: null })
              }
              required
            >
              <option value="patient">Patient</option>
              <option value="billing_party">Tiers payeur</option>
              <option value="company">Entreprise</option>
            </select>
          </div>

          {form.billed_to_type === 'billing_party' && (
            <div className={styles.formGroup}>
              <label htmlFor="billing_party_id">Tiers payeur *</label>
              <select
                id="billing_party_id"
                value={form.billing_party_id || ''}
                onChange={(e) =>
                  setForm({ ...form, billing_party_id: e.target.value ? parseInt(e.target.value) : null })
                }
                required
              >
                <option value="">Sélectionner un tiers payeur</option>
                {billingParties.map((party) => (
                  <option key={party.id} value={party.id}>
                    {party.display_name}
                  </option>
                ))}
              </select>
            </div>
          )}

          {form.billed_to_type === 'company' && (
            <div className={styles.formGroup}>
              <label htmlFor="billed_to_company_id">Entreprise *</label>
              <input
                id="billed_to_company_id"
                type="number"
                value={form.billed_to_company_id || ''}
                onChange={(e) =>
                  setForm({ ...form, billed_to_company_id: e.target.value ? parseInt(e.target.value) : null })
                }
                required
                placeholder="ID de l'entreprise"
              />
            </div>
          )}

          <div className={styles.formGroup}>
            <label htmlFor="reason">Motif de la modification *</label>
            <textarea
              id="reason"
              value={form.reason}
              onChange={(e) => setForm({ ...form, reason: e.target.value })}
              required
              rows={4}
              placeholder="Expliquez pourquoi vous modifiez le payeur..."
            />
          </div>

          {error && <div className={styles.error}>{error}</div>}

          <div className={styles.modalActions}>
            <button type="button" onClick={onClose} className={styles.btnCancel}>
              Annuler
            </button>
            <button type="submit" disabled={loading} className={styles.btnSave}>
              {loading ? 'Enregistrement...' : 'Enregistrer'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default SetPayerModal;
