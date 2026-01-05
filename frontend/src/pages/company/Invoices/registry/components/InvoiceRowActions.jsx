import React, { useState, useRef } from 'react';
import styles from './InvoiceRowActions.module.css';
import {
  canSendInvoice,
  canAddPayment,
  canGenerateReminder,
  canRegeneratePdf,
  canCancelInvoice,
  canDuplicateInvoice,
  getNextReminderLevel,
} from '../../../../../services/invoiceService';

const InvoiceRowActions = ({
  invoice,
  onSend,
  onPayment,
  onReminder,
  onRegeneratePdf,
  onCancel,
  onViewPdf,
  onDuplicate,
}) => {
  const [showMenu, setShowMenu] = useState(false);
  const [menuPosition, setMenuPosition] = useState({ top: 0, left: 0 });
  const buttonRef = useRef(null);

  const handleAction = (action) => {
    setShowMenu(false);
    action();
  };

  const actions = [
    {
      key: 'view',
      label: 'Voir PDF',
      icon: '👁️',
      onClick: onViewPdf,
      className: styles.actionBtnSecondary,
      show: !!invoice.pdf_url,
    },
    {
      key: 'send',
      label: 'Marquer envoyée',
      icon: '📧',
      onClick: onSend,
      className: styles.actionBtnPrimary,
      show: canSendInvoice(invoice),
    },
    {
      key: 'payment',
      label: 'Enregistrer paiement',
      icon: '💰',
      onClick: onPayment,
      className: styles.actionBtnSuccess,
      show: canAddPayment(invoice),
    },
    {
      key: 'reminder',
      label: `Générer rappel ${getNextReminderLevel(invoice)}`,
      icon: '⚠️',
      onClick: onReminder,
      className: styles.actionBtnWarning,
      show: canGenerateReminder(invoice),
    },
    {
      key: 'regenerate',
      label: 'Régénérer PDF',
      icon: '🔄',
      onClick: onRegeneratePdf,
      className: styles.actionBtnSecondary,
      show: canRegeneratePdf(invoice),
    },
    {
      key: 'duplicate',
      label: 'Créer un correctif',
      icon: '📝',
      onClick: onDuplicate,
      className: styles.actionBtnSecondary,
      show: canDuplicateInvoice(invoice),
    },
    {
      key: 'cancel',
      label: 'Annuler',
      icon: '❌',
      onClick: onCancel,
      className: styles.actionBtnDanger,
      show: canCancelInvoice(invoice),
    },
  ];

  const visibleActions = actions.filter((action) => action.show);

  const handleToggleMenu = () => {
    if (!showMenu && buttonRef.current) {
      const rect = buttonRef.current.getBoundingClientRect();
      
      // Estimer la hauteur du menu : nombre d'actions * hauteur approximative d'un élément (~48px)
      const estimatedMenuHeight = visibleActions.length * 48;
      const spaceBelow = window.innerHeight - rect.bottom;
      const spaceAbove = rect.top;
      
      // Si pas assez d'espace en bas mais assez en haut, afficher au-dessus
      const showAbove = spaceBelow < estimatedMenuHeight && spaceAbove > estimatedMenuHeight;
      
      setMenuPosition({
        top: showAbove 
          ? rect.top + window.scrollY - estimatedMenuHeight 
          : rect.bottom + window.scrollY,
        left: rect.right - 220 + window.scrollX, // 220px = min-width du menu
      });
    }
    setShowMenu(!showMenu);
  };

  if (visibleActions.length === 0) {
    return <span className={styles.noActions}>Aucune action</span>;
  }

  // Toujours afficher le menu déroulant pour toutes les factures
  return (
    <div className={styles.actionMenu}>
      <button
        ref={buttonRef}
        className={`${styles.actionBtn} ${styles.actionBtnSecondary}`}
        onClick={handleToggleMenu}
        title="Actions"
      >
        <span className={styles.actionIcon}>⚙️</span>
        <span className={styles.actionLabel}>Actions</span>
        <span className={styles.actionIcon}>▼</span>
      </button>

      {showMenu && (
        <>
          <div className={styles.menuOverlay} onClick={() => setShowMenu(false)} />
          <div
            className={styles.menu}
            style={{
              position: 'fixed',
              top: `${menuPosition.top}px`,
              left: `${menuPosition.left}px`,
            }}
          >
            {visibleActions.map((action) => (
              <button
                key={action.key}
                className={`${styles.menuItem} ${action.className}`}
                onClick={() => handleAction(action.onClick)}
              >
                <span className={styles.actionIcon}>{action.icon}</span>
                <span className={styles.actionLabel}>{action.label}</span>
              </button>
            ))}
          </div>
        </>
      )}
    </div>
  );
};

export default InvoiceRowActions;
