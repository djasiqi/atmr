import React from 'react';
import { FiX } from 'react-icons/fi';
import DraftInvoiceEditorPanel from './DraftInvoiceEditorPanel';
import styles from './InvoiceDraftEditModal.module.css';

/**
 * Édition facture (brouillon ou envoyée / en encaissement) depuis le registre : overlay + DraftInvoiceEditorPanel.
 */
const InvoiceDraftEditModal = ({
  open,
  initialInvoice,
  companyId,
  onClose,
  onUpdated,
  onOpenSendEmail,
  onMarkAsSent,
}) => {
  if (!open || !initialInvoice) return null;

  return (
    <div className={styles.overlay} onClick={onClose}>
      <div className={styles.panel} onClick={(e) => e.stopPropagation()}>
        <div className={styles.headCloseOnly}>
          <button type="button" className={styles.close} onClick={onClose} aria-label="Fermer">
            <FiX size={18} />
          </button>
        </div>
        <DraftInvoiceEditorPanel
          key={initialInvoice?.id ?? 'draft'}
          open={open}
          initialInvoice={initialInvoice}
          companyId={companyId}
          onUpdated={onUpdated}
          onOpenSendEmail={onOpenSendEmail}
          onMarkAsSent={onMarkAsSent}
        />
      </div>
    </div>
  );
};

export default InvoiceDraftEditModal;
