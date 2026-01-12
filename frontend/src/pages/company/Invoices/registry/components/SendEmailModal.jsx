import React, { useState, useEffect } from 'react';
import styles from './SendEmailModal.module.css';

const SendEmailModal = ({ invoice, onClose, onSend, isReminder = false, reminderId = null }) => {
  const [email, setEmail] = useState('');
  const [forceRegenerate, setForceRegenerate] = useState(false);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState('');

  // Pré-remplir avec l'email du client
  useEffect(() => {
    if (invoice?.client?.contact_email) {
      setEmail(invoice.client.contact_email);
    }
  }, [invoice]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');

    // Validation basique de l'email
    if (!email || !email.trim()) {
      setError('Veuillez saisir une adresse email');
      return;
    }

    if (!email.match(/^[^\s@]+@[^\s@]+\.[^\s@]+$/)) {
      setError('Adresse email invalide');
      return;
    }

    setSending(true);
    try {
      await onSend({
        recipient_email: email,
        force_regenerate_pdf: forceRegenerate,
        reminder_id: reminderId,
      });
      onClose();
    } catch (err) {
      console.error('Erreur lors de l\'envoi:', err);
      setError(err?.response?.data?.error || 'Erreur lors de l\'envoi de l\'email');
    } finally {
      setSending(false);
    }
  };

  const clientName = invoice?.client
    ? invoice.client.institution_name ||
      `${invoice.client.first_name || ''} ${invoice.client.last_name || ''}`.trim() ||
      invoice.client.username
    : 'Client inconnu';

  return (
    <div className={styles.modalOverlay} onClick={onClose}>
      <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
        <div className={styles.header}>
          <h2>
            {isReminder
              ? `📧 Envoyer le rappel par email`
              : `📧 Envoyer la facture par email`}
          </h2>
          <button className={styles.closeBtn} onClick={onClose} title="Fermer">
            ✕
          </button>
        </div>

        <form onSubmit={handleSubmit}>
          <div className={styles.content}>
            <div className={styles.infoBox}>
              <div className={styles.infoRow}>
                <span className={styles.infoLabel}>
                  {isReminder ? 'Rappel pour :' : 'Facture :'}
                </span>
                <span className={styles.infoValue}>
                  {invoice?.invoice_number || 'N/A'}
                </span>
              </div>
              <div className={styles.infoRow}>
                <span className={styles.infoLabel}>Client :</span>
                <span className={styles.infoValue}>{clientName}</span>
              </div>
              <div className={styles.infoRow}>
                <span className={styles.infoLabel}>Montant :</span>
                <span className={styles.infoValue}>
                  {invoice?.total_amount?.toFixed(2) || '0.00'} CHF
                </span>
              </div>
            </div>

            <div className={styles.formGroup}>
              <label htmlFor="email">
                Email du destinataire <span className={styles.required}>*</span>
              </label>
              <input
                type="email"
                id="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="client@example.com"
                className={styles.input}
                autoFocus
                required
              />
              <small className={styles.hint}>
                Par défaut : email de contact du client. Vous pouvez modifier si nécessaire.
              </small>
            </div>

            <div className={styles.checkboxGroup}>
              <label>
                <input
                  type="checkbox"
                  checked={forceRegenerate}
                  onChange={(e) => setForceRegenerate(e.target.checked)}
                />
                <span>Régénérer le PDF avant envoi</span>
              </label>
              <small className={styles.hint}>
                Utile si des modifications ont été apportées depuis la dernière génération.
              </small>
            </div>

            {error && <div className={styles.error}>{error}</div>}

            <div className={styles.warningBox}>
              <span className={styles.warningIcon}>ℹ️</span>
              <div className={styles.warningContent}>
                <strong>Configuration SMTP :</strong>
                <p>
                  L'email sera envoyé depuis la configuration SMTP de votre entreprise. Si vous
                  n'avez pas encore configuré votre SMTP, l'email sera envoyé depuis la
                  configuration globale du système.
                </p>
                <a href="/dashboard/company/settings?tab=billing" className={styles.link}>
                  → Configurer mon SMTP
                </a>
              </div>
            </div>
          </div>

          <div className={styles.footer}>
            <button
              type="button"
              onClick={onClose}
              className={`${styles.btn} ${styles.btnSecondary}`}
              disabled={sending}
            >
              Annuler
            </button>
            <button
              type="submit"
              className={`${styles.btn} ${styles.btnPrimary}`}
              disabled={sending || !email}
            >
              {sending ? '📤 Envoi en cours...' : '📧 Envoyer par email'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default SendEmailModal;
