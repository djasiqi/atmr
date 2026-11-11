import React, { useEffect, useState } from 'react';
import { createPortal } from 'react-dom';
import { FiX } from 'react-icons/fi';
import DraftInvoiceEditorPanel from './DraftInvoiceEditorPanel';
import { invoiceCatalogRowKey } from '../../../../../utils/invoiceCatalog';
import styles from './InvoiceDraftEditModal.module.css';

/**
 * Éditeur unique (patient, clinique, partenaire).
 * Brouillon = édition ; facture émise = mode correctif (même coque visuelle).
 */
const InvoiceDraftEditModal = ({
  open,
  initialInvoice,
  companyId,
  onClose,
  onUpdated,
  onOpenSendEmail,
  onMarkAsSent,
  /** `edit` | `correct` — déduit du statut si omis. */
  editorIntent,
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

  const status = String(initialInvoice?.status || '').toLowerCase();
  const intent =
    editorIntent ||
    (status === 'draft' ? 'edit' : 'correct');
  const ariaLabel =
    intent === 'correct' ? 'Corriger la facture' : 'Éditer la facture';

  return createPortal(
    <div className={styles.overlay} onClick={onClose} role="presentation">
      <div
        className={styles.panel}
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-label={ariaLabel}
        data-editor-intent={intent}
      >
        <div className={styles.headCloseOnly}>
          <button
            type="button"
            className={styles.close}
            onClick={(event) => {
              event.stopPropagation();
              onClose();
            }}
            aria-label="Fermer"
          >
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
          editorIntent={intent}
        />
      </div>
    </div>,
    portalTarget
  );
};

export default InvoiceDraftEditModal;
