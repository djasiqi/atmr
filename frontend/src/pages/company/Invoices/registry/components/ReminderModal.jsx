import React, { useState } from 'react';
import styles from './ReminderModal.module.css';
import { getNextReminderLevel } from '../../../../../services/invoiceService';

const ReminderModal = ({ open, invoice, onClose, onReminder }) => {
  const [level, setLevel] = useState(1);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const reminderLevels = [
    { value: 1, label: '1er rappel', description: 'Rappel amiable sans frais' },
    { value: 2, label: '2e rappel', description: 'Rappel avec frais supplémentaires' },
    { value: 3, label: 'Dernier rappel', description: 'Dernier rappel avant mise en demeure' },
  ];

  const handleSubmit = async (e) => {
    e.preventDefault();

    try {
      setLoading(true);
      setError(null);

      await onReminder(invoice.id, level);
    } catch (err) {
      setError(err.message || 'Erreur lors de la génération du rappel');
    } finally {
      setLoading(false);
    }
  };

  const handleClose = () => {
    setLevel(1);
    setError(null);
    onClose();
  };

  if (!open || !invoice) return null;

  const nextLevel = getNextReminderLevel(invoice);
  const availableLevels = reminderLevels.filter((rl) => rl.value >= nextLevel);

  return (
    <div className="modal-overlay">
      <div className="modal-content modal-md">
        <div className="modal-header">
          <h2 className="modal-title">Générer un rappel</h2>
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
            <p>
              Solde dû: <strong>{invoice.balance_due.toFixed(2)} CHF</strong>
            </p>
            <p>
              Niveau actuel:{' '}
              {invoice.reminder_level === 0 ? 'Aucun' : `Rappel ${invoice.reminder_level}`}
            </p>
            <p>
              Dernier rappel:{' '}
              {invoice.last_reminder_at
                ? new Date(invoice.last_reminder_at).toLocaleDateString('fr-FR')
                : 'Jamais'}
            </p>
          </div>

          <form onSubmit={handleSubmit}>
            <div className="form-group">
              <label className="form-label required">Niveau du rappel</label>
              <div className={styles.levelOptions}>
                {availableLevels.map((reminderLevel) => (
                  <div key={reminderLevel.value} className={styles.levelOption}>
                    <input
                      type="radio"
                      id={`level-${reminderLevel.value}`}
                      name="level"
                      value={reminderLevel.value}
                      checked={level === reminderLevel.value}
                      onChange={(e) => setLevel(parseInt(e.target.value))}
                      className={styles.radio}
                    />
                    <label htmlFor={`level-${reminderLevel.value}`} className={styles.levelLabel}>
                      <div className={styles.levelTitle}>{reminderLevel.label}</div>
                      <div className={styles.levelDescription}>{reminderLevel.description}</div>
                    </label>
                  </div>
                ))}
              </div>
            </div>

            {level > 0 && (
              <div className={styles.feeInfo}>
                <h4>Frais associés</h4>
                <p>
                  {level === 1 && 'Aucun frais pour le 1er rappel'}
                  {level === 2 && 'Frais de rappel de niveau 2 seront ajoutés à la facture'}
                  {level === 3 && 'Aucun frais supplémentaire pour le dernier rappel'}
                </p>
              </div>
            )}

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
              <button
                type="submit"
                className="btn btn-primary"
                disabled={loading || availableLevels.length === 0}
              >
                {loading ? 'Génération...' : `Générer le rappel niveau ${level}`}
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};

export default ReminderModal;
