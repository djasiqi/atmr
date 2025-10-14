import React, { useState } from 'react';
import styles from './PaymentModal.module.css';

const PaymentModal = ({ open, invoice, onClose, onPayment }) => {
  const [formData, setFormData] = useState({
    amount: '',
    paid_at: new Date().toISOString().split('T')[0],
    method: 'bank_transfer',
    reference: ''
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const paymentMethods = [
    { value: 'bank_transfer', label: 'Virement bancaire' },
    { value: 'cash', label: 'Espèces' },
    { value: 'card', label: 'Carte' },
    { value: 'adjustment', label: 'Ajustement' }
  ];

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!formData.amount || parseFloat(formData.amount) <= 0) {
      setError('Le montant doit être positif');
      return;
    }

    if (parseFloat(formData.amount) > invoice.balance_due) {
      setError('Le montant ne peut pas dépasser le solde dû');
      return;
    }

    try {
      setLoading(true);
      setError(null);
      
      const paymentData = {
        ...formData,
        amount: parseFloat(formData.amount),
        paid_at: new Date(formData.paid_at).toISOString()
      };
      
      await onPayment(invoice.id, paymentData);
    } catch (err) {
      setError(err.message || 'Erreur lors de l\'enregistrement du paiement');
    } finally {
      setLoading(false);
    }
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
    
    // Réinitialiser l'erreur quand l'utilisateur tape
    if (error) setError(null);
  };

  const handleClose = () => {
    setFormData({
      amount: '',
      paid_at: new Date().toISOString().split('T')[0],
      method: 'bank_transfer',
      reference: ''
    });
    setError(null);
    onClose();
  };

  if (!open || !invoice) return null;

  return (
    <div className={styles.modalOverlay}>
      <div className={styles.modal}>
        <div className={styles.modalHeader}>
          <h2>Enregistrer un paiement</h2>
          <button className={styles.closeBtn} onClick={handleClose}>
            ✕
          </button>
        </div>

        <div className={styles.modalBody}>
          <div className={styles.invoiceInfo}>
            <h3>Facture {invoice.invoice_number}</h3>
            <p>Client: {invoice.client ? 
              `${invoice.client.first_name || ''} ${invoice.client.last_name || ''}`.trim() || invoice.client.username
              : 'Client inconnu'
            }</p>
            <p>Montant total: {invoice.total_amount.toFixed(2)} CHF</p>
            <p>Déjà payé: {invoice.amount_paid.toFixed(2)} CHF</p>
            <p className={styles.balanceDue}>Solde dû: {invoice.balance_due.toFixed(2)} CHF</p>
          </div>

          <form onSubmit={handleSubmit} className={styles.form}>
            <div className={styles.formGroup}>
              <label htmlFor="amount" className={styles.label}>
                Montant du paiement *
              </label>
              <input
                type="number"
                id="amount"
                name="amount"
                value={formData.amount}
                onChange={handleInputChange}
                className={styles.input}
                step="0.01"
                min="0.01"
                max={invoice.balance_due}
                placeholder="0.00"
                required
              />
            </div>

            <div className={styles.formGroup}>
              <label htmlFor="paid_at" className={styles.label}>
                Date du paiement *
              </label>
              <input
                type="date"
                id="paid_at"
                name="paid_at"
                value={formData.paid_at}
                onChange={handleInputChange}
                className={styles.input}
                required
              />
            </div>

            <div className={styles.formGroup}>
              <label htmlFor="method" className={styles.label}>
                Mode de paiement *
              </label>
              <select
                id="method"
                name="method"
                value={formData.method}
                onChange={handleInputChange}
                className={styles.select}
                required
              >
                {paymentMethods.map(method => (
                  <option key={method.value} value={method.value}>
                    {method.label}
                  </option>
                ))}
              </select>
            </div>

            <div className={styles.formGroup}>
              <label htmlFor="reference" className={styles.label}>
                Référence (optionnel)
              </label>
              <input
                type="text"
                id="reference"
                name="reference"
                value={formData.reference}
                onChange={handleInputChange}
                className={styles.input}
                placeholder="Référence bancaire, n° transaction..."
              />
            </div>

            {error && (
              <div className={styles.error}>
                {error}
              </div>
            )}

            <div className={styles.formActions}>
              <button
                type="button"
                className={styles.cancelBtn}
                onClick={handleClose}
                disabled={loading}
              >
                Annuler
              </button>
              <button
                type="submit"
                className={styles.submitBtn}
                disabled={loading}
              >
                {loading ? 'Enregistrement...' : 'Enregistrer le paiement'}
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};

export default PaymentModal;
