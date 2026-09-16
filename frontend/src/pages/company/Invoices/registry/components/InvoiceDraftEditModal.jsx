import React, { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { FiX } from 'react-icons/fi';
import DraftInvoiceEditorPanel from './DraftInvoiceEditorPanel';
import { INVOICE_CATALOG, resolveInvoiceResource } from '../../../../../utils/invoiceCatalog';
import { openProtectedPdfInNewTab } from '../../../../../utils/protectedPdf';
import { buildInvoicePdfDownloadFilename } from '../../../../../utils/invoicePdfFilename';
import styles from './InvoiceDraftEditModal.module.css';

/**
 * Édition facture (brouillon ou envoyée / en encaissement) depuis le registre : overlay + DraftInvoiceEditorPanel.
 * Portal sur body pour passer au-dessus de la sidebar company (stacking context).
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
  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;

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

  useEffect(() => {
    if (!open || !initialInvoice) return;
    const resource = resolveInvoiceResource(initialInvoice, companyId);
    if (resource.type !== INVOICE_CATALOG.PARTNER) return;
    if (resource.pdfApiUrl) {
      void openProtectedPdfInNewTab(resource.pdfApiUrl, null, {
        filename: buildInvoicePdfDownloadFilename(resource.invoice),
      });
    }
    onCloseRef.current?.();
  }, [open, initialInvoice, companyId]);

  if (!open || !initialInvoice || !portalTarget) return null;
  if (resolveInvoiceResource(initialInvoice, companyId).type === INVOICE_CATALOG.PARTNER) {
    return null;
  }

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
