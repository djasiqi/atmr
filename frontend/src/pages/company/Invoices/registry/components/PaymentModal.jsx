import React, { useState } from 'react';
import styles from './PaymentModal.module.css';

const PaymentModal = ({ open, invoice, onClose, onPayment }) => {
  const [formData, setFormData] = useState({
    amount: '',
    paid_at: new Date().toISOString().split('T')[0],
    method: 'bank_transfer',
    reference: '',
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const paymentMethods = [
    { value: 'bank_transfer', label: 'Virement bancaire' },
    { value: 'cash', label: 'Espèces' },
    { value: 'card', label: 'Carte' },
    { value: 'adjustment', label: 'Ajustement' },
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
        paid_at: new Date(formData.paid_at).toISOString(),
      };

      await onPayment(invoice.id, paymentData);
    } catch (err) {
      setError(err.message || "Erreur lors de l'enregistrement du paiement");
    } finally {
      setLoading(false);
    }
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));

    // Réinitialiser l'erreur quand l'utilisateur tape
    if (error) setError(null);
  };

  const handleClose = () => {
    setFormData({
      amount: '',
      paid_at: new Date().toISOString().split('T')[0],
      method: 'bank_transfer',
      reference: '',
    });
    setError(null);
    onClose();
  };

  if (!open || !invoice) return null;

  return (
    <div className="modal-overlay">
      <div className="modal-content modal-md">
        <div className="modal-header">
          <h2 className="modal-title">Enregistrer un paiement</h2>
          <button className="modal-close" onClick={handleClose}>
            ✕
          </button>
        </div>

        <div className="modal-body">
          <div className={styles.invoiceInfo}>
            <h3>Facture {invoice.invoice_number}</h3>
            <p>
              Client:{' '}
              {invoice.client
                ? `${invoice.client.first_name || ''} ${invoice.client.last_name || ''}`.trim() ||
                  invoice.client.username
                : 'Client inconnu'}
            </p>
            <p>Montant total: {invoice.total_amount.toFixed(2)} CHF</p>
            <p>Déjà payé: {invoice.amount_paid.toFixed(2)} CHF</p>
            <p className={styles.balanceDue}>Solde dû: {invoice.balance_due.toFixed(2)} CHF</p>
          </div>

          <form onSubmit={handleSubmit}>
            <div className="form-group">
              <label htmlFor="amount" className="form-label required">
                Montant du paiement
              </label>
              <input
                type="number"
                id="amount"
                name="amount"
                value={formData.amount}
                onChange={handleInputChange}
                className="form-input"
                step="0.01"
                min="0.01"
                max={invoice.balance_due}
                placeholder="0.00"
                required
              />
            </div>

            <div className="form-group">
              <label htmlFor="paid_at" className="form-label required">
                Date du paiement
              </label>
              <input
                type="date"
                id="paid_at"
                name="paid_at"
                value={formData.paid_at}
                onChange={handleInputChange}
                className="form-input"
                required
              />
            </div>

            <div className="form-group">
              <label htmlFor="method" className="form-label required">
                Mode de paiement
              </label>
              <select
                id="method"
                name="method"
                value={formData.method}
                onChange={handleInputChange}
                className="form-select"
                required
              >
                {paymentMethods.map((method) => (
                  <option key={method.value} value={method.value}>
                    {method.label}
                  </option>
                ))}
              </select>
            </div>

            <div className="form-group">
              <label htmlFor="reference" className="form-label">
                Référence (optionnel)
              </label>
              <input
                type="text"
                id="reference"
                name="reference"
                value={formData.reference}
                onChange={handleInputChange}
                className="form-input"
                placeholder="Référence bancaire, n° transaction..."
              />
            </div>

            {error && <div className="alert alert-error mb-md">{error}</div>}

            <div className="modal-footer">
              <button
                type="button"
                className="btn btn-secondary"
                onClick={handleClose}
                disabled={loading}
              >
                Annuler
              </button>
              <button type="submit" className="btn btn-primary" disabled={loading}>
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
