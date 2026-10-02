import React, { useEffect, useState } from 'react';
import { createPortal } from 'react-dom';
import { FiX } from 'react-icons/fi';
import DraftInvoiceEditorPanel from './DraftInvoiceEditorPanel';
import { invoiceCatalogRowKey } from '../../../../../utils/invoiceCatalog';
import styles from './InvoiceDraftEditModal.module.css';

/**
 * Éditeur unique (patient, clinique, partenaire).
 * Le catalogue est porté par l'adaptateur, pas par un second formulaire.
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

  if (!open || !initialInvoice || !portalTarget) return null;

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
          key={invoiceCatalogRowKey(initialInvoice, companyId) || 'draft'}
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
