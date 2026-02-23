import React, { useState, useEffect } from 'react';
import { FiInfo, FiMail, FiRefreshCw, FiSend, FiX } from 'react-icons/fi';
import styles from './SendEmailModal.module.css';

const SendEmailModal = ({ invoice, onClose, onSend, isReminder = false, reminderId = null }) => {
  const [email, setEmail] = useState('');
  const [forceRegenerate, setForceRegenerate] = useState(false);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState('');

  const resolveDefaultRecipientEmail = (inv) => {
    if (!inv) return '';
    // Priorité au destinataire facturé réel
    if (inv.billing_party?.contact_email) return inv.billing_party.contact_email;
    if (inv.bill_to_client?.contact_email) return inv.bill_to_client.contact_email;
    if (inv.billed_to_company?.billing_email) return inv.billed_to_company.billing_email;
    if (inv.billed_to_company?.contact_email) return inv.billed_to_company.contact_email;
    // Fallback client
    if (inv.client?.contact_email) return inv.client.contact_email;
    return '';
  };

  // Pré-remplir avec l'email du destinataire enregistré
  useEffect(() => {
    setEmail(resolveDefaultRecipientEmail(invoice));
  }, [invoice]);

  useEffect(() => {
    const onKeyDown = (event) => {
      if (event.key === 'Escape' && !sending) {
        onClose();
      }
    };
    document.addEventListener('keydown', onKeyDown);
    return () => document.removeEventListener('keydown', onKeyDown);
  }, [onClose, sending]);

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
          <div className={styles.headerTitleWrap}>
            <div className={styles.headerIconWrap}>
              <FiMail size={16} />
            </div>
            <div>
              <h2>
                {isReminder ? 'Envoyer le rappel par email' : 'Envoyer la facture par email'}
              </h2>
              <p className={styles.headerSubtitle}>
                Vérifiez le destinataire avant l’envoi.
              </p>
            </div>
          </div>
          <button className={styles.closeBtn} onClick={onClose} title="Fermer" aria-label="Fermer">
            <FiX size={18} />
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
              <label htmlFor="email" className={styles.fieldLabel}>
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
              <label className={styles.checkboxLabel}>
                <input
                  type="checkbox"
                  checked={forceRegenerate}
                  onChange={(e) => setForceRegenerate(e.target.checked)}
                />
                <span>
                  <FiRefreshCw size={14} />
                  Régénérer le PDF avant envoi
                </span>
              </label>
              <small className={styles.hint}>
                Utile si des modifications ont été apportées depuis la dernière génération.
              </small>
            </div>

            {error && <div className={styles.error}>{error}</div>}

            <div className={styles.warningBox}>
              <span className={styles.warningIcon}>
                <FiInfo size={16} />
              </span>
              <div className={styles.warningContent}>
                <strong>Configuration SMTP :</strong>
                <p>
                  L'email sera envoyé depuis la configuration SMTP de votre entreprise. Si vous
                  n'avez pas encore configuré votre SMTP, l'email sera envoyé depuis la
                  configuration globale du système.
                </p>
                <a href="/dashboard/company/settings?section=emailConfig#billing" className={styles.link}>
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
              disabled={sending || !email.trim()}
            >
              {sending ? 'Envoi en cours...' : (
                <>
                  <FiSend size={14} />
                  Envoyer par email
                </>
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default SendEmailModal;
