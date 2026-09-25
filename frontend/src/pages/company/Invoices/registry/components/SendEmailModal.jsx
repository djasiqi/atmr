import React, { useState, useEffect } from 'react';
import { createPortal } from 'react-dom';
import { FiInfo, FiMail, FiRefreshCw, FiSend, FiX } from 'react-icons/fi';
import { invoiceService } from '../../../../../services/invoiceService';
import { resolveDefaultRecipientEmail } from '../../../../../utils/portalInvoiceRecipientEmail';
import styles from './SendEmailModal.module.css';

const SendEmailModal = ({
  invoice,
  companyId = null,
  onClose,
  onSend,
  isReminder = false,
  reminderId = null,
}) => {
  const [email, setEmail] = useState('');
  const [forceRegenerate, setForceRegenerate] = useState(false);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState('');

  // Pré-remplir : stub local, sinon rechargement facture (user.email PORTAL)
  useEffect(() => {
    let cancelled = false;
    const run = async () => {
      let resolved = resolveDefaultRecipientEmail(invoice);
      const cid = companyId || invoice?.company_id;
      const iid = invoice?.id;
      if (!resolved && cid && iid) {
        try {
          const fresh = await invoiceService.getInvoice(cid, iid, { cacheBust: true });
          const data = fresh?.data ?? fresh;
          resolved = resolveDefaultRecipientEmail(data);
          if (!resolved && data?.default_recipient_email) {
            resolved = String(data.default_recipient_email).trim();
          }
        } catch {
          /* garde le champ vide — saisie manuelle */
        }
      }
      if (!cancelled) {
        setEmail(resolved || '');
      }
    };
    void run();
    return () => {
      cancelled = true;
    };
  }, [invoice, companyId]);

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

  const clientName = invoice?.billing_party?.display_name
    || (invoice?.client
      ? invoice.client.institution_name ||
        invoice.client.patient_display_name ||
        `${invoice.client.first_name || ''} ${invoice.client.last_name || ''}`.trim() ||
        invoice.client.username
      : null)
    || 'Client inconnu';

  return createPortal(
    <div className={styles.modalOverlay} onClick={onClose} role="presentation">
      <div
        className={styles.modal}
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-labelledby="send-email-modal-title"
      >
        <div className={styles.header}>
          <div className={styles.headerTitleWrap}>
            <div className={styles.headerIconWrap}>
              <FiMail size={16} />
            </div>
            <div>
              <h2 id="send-email-modal-title">
                {isReminder ? 'Envoyer le rappel par email' : 'Envoyer la facture par email'}
              </h2>
              <p className={styles.headerSubtitle}>Vérifiez le destinataire avant l’envoi.</p>
            </div>
          </div>
          <button
            type="button"
            className={styles.closeBtn}
            onClick={onClose}
            title="Fermer"
            aria-label="Fermer"
            disabled={sending}
          >
            <FiX size={18} />
          </button>
        </div>

        <form onSubmit={handleSubmit}>
          <div className={styles.content}>
            <div className={styles.infoBox}>
              <div className={styles.infoRow}>
                <span className={styles.infoLabel}>Facture :</span>
                <span className={styles.infoValue}>{invoice?.invoice_number || '—'}</span>
              </div>
              <div className={styles.infoRow}>
                <span className={styles.infoLabel}>Client :</span>
                <span className={styles.infoValue}>{clientName}</span>
              </div>
              <div className={styles.infoRow}>
                <span className={styles.infoLabel}>Montant :</span>
                <span className={styles.infoValue}>
                  {invoice?.total_amount != null
                    ? `${Number(invoice.total_amount).toFixed(2)} CHF`
                    : '—'}
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
                placeholder="client@example.com"
                className={styles.input}
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                disabled={sending}
                autoComplete="email"
              />
              <small className={styles.hint}>
                Par défaut : email du compte client (PORTAL) ou contact facturation. Vous pouvez modifier si nécessaire.
              </small>
            </div>

            <div className={styles.checkboxGroup}>
              <label className={styles.checkboxLabel}>
                <input
                  type="checkbox"
                  checked={forceRegenerate}
                  onChange={(e) => setForceRegenerate(e.target.checked)}
                  disabled={sending}
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

            {error ? <div className={styles.error}>{error}</div> : null}

            <div className={styles.warningBox}>
              <span className={styles.warningIcon}>
                <FiInfo size={16} />
              </span>
              <div className={styles.warningContent}>
                <strong>Configuration SMTP :</strong>
                <p>
                  L&apos;email sera envoyé depuis la configuration SMTP de votre entreprise. Si vous
                  n&apos;avez pas encore configuré votre SMTP, l&apos;email sera envoyé depuis la
                  configuration globale du système.
                </p>
                <a
                  href="/dashboard/company/settings?section=emailConfig#billing"
                  className={styles.link}
                >
                  → Configurer mon SMTP
                </a>
              </div>
            </div>
          </div>

          <div className={styles.footer}>
            <button
              type="button"
              className={`${styles.btn} ${styles.btnSecondary}`}
              onClick={onClose}
              disabled={sending}
            >
              Annuler
            </button>
            <button
              type="submit"
              className={`${styles.btn} ${styles.btnPrimary}`}
              disabled={sending || !email.trim()}
            >
              <FiSend size={14} />
              {sending ? 'Envoi…' : 'Envoyer par email'}
            </button>
          </div>
        </form>
      </div>
    </div>,
    document.body
  );
};

export default SendEmailModal;
