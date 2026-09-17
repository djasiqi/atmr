import React, { useEffect, useState } from 'react';
import { createPortal } from 'react-dom';
import { FiX } from 'react-icons/fi';
import DraftInvoiceEditorPanel from './DraftInvoiceEditorPanel';
import PartnerInvoiceDraftEditModal from './PartnerInvoiceDraftEditModal';
import { INVOICE_CATALOG, resolveInvoiceResource } from '../../../../../utils/invoiceCatalog';
import styles from './InvoiceDraftEditModal.module.css';

/**
 * Édition facture depuis le registre.
 * Catalogue partenaire → éditeur partenaire natif (jamais GET /invoices/{id}).
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
  const [portalTarget, setPortalTarget] = useState(null);

  useEffect(() => {
    if (typeof document === 'undefined' || !document.body) return undefined;
    const el = document.createElement('div');
    el.setAttribute('data-portal', 'invoice-draft-edit-modal');
    document.body.appendChild(el);
    setPortalTarget(el);
    return () => {
      setPortalTarget(null);
      if (el.parentNode) el.parentNode.removeChild(el);
    };
  }, []);

  if (!open || !initialInvoice) return null;

  const resource = resolveInvoiceResource(initialInvoice, companyId);
  if (resource.type === INVOICE_CATALOG.PARTNER) {
    return (
      <PartnerInvoiceDraftEditModal
        open={open}
        initialInvoice={initialInvoice}
        companyId={companyId}
        onClose={onClose}
        onUpdated={onUpdated}
      />
    );
  }

  if (!portalTarget) return null;

  return createPortal(
    <div className={styles.overlay} onClick={onClose} role="presentation">
      <div
        className={styles.panel}
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-label="Édition de la facture"
      >
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
    </div>,
    portalTarget
  );
};

export default InvoiceDraftEditModal;
