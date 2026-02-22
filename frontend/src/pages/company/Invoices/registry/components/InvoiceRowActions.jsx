import React, { useState, useRef, useEffect } from 'react';
import {
  FiFileText,
  FiBell,
  FiMail,
  FiDollarSign,
  FiClock,
  FiRefreshCw,
  FiEdit,
  FiXCircle,
  FiMoreHorizontal,
  FiSend,
} from 'react-icons/fi';
import styles from './InvoiceRowActions.module.css';
import { ensurePdfUrlWorksInDev } from '../../../../../utils/pdfUrlFallback';
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
  onSendEmail,
  onPayment,
  onReminder,
  onSendReminderEmail,
  onRegeneratePdf,
  onCancel,
  onViewPdf,
  onDuplicate,
}) => {
  const [showMenu, setShowMenu] = useState(false);
  const [menuPosition, setMenuPosition] = useState({ top: 0, left: 0, direction: 'below' });
  const buttonRef = useRef(null);
  const menuRef = useRef(null);

  // Fermer le menu lors du scroll ou resize
  useEffect(() => {
    const handleCloseMenu = () => {
      if (showMenu) {
        setShowMenu(false);
      }
    };

    if (showMenu) {
      // Écouter le scroll de tous les conteneurs parents
      window.addEventListener('scroll', handleCloseMenu, true);
      window.addEventListener('resize', handleCloseMenu);
      
      // Détecter le scroll dans les conteneurs parents scrollables
      let parent = buttonRef.current?.parentElement;
      const scrollableParents = [];
      
      while (parent) {
        const { overflow, overflowY } = window.getComputedStyle(parent);
        if (overflow === 'auto' || overflow === 'scroll' || overflowY === 'auto' || overflowY === 'scroll') {
          scrollableParents.push(parent);
          parent.addEventListener('scroll', handleCloseMenu);
        }
        parent = parent.parentElement;
      }

      return () => {
        window.removeEventListener('scroll', handleCloseMenu, true);
        window.removeEventListener('resize', handleCloseMenu);
        scrollableParents.forEach(p => p.removeEventListener('scroll', handleCloseMenu));
      };
    }
  }, [showMenu]);

  const handleAction = (action) => {
    setShowMenu(false);
    action();
  };

  // Trouver le rappel le plus recent (OPEN ou PAID)
  const latestReminder = invoice.reminders?.length > 0
    ? invoice.reminders
        .sort((a, b) => new Date(b.generated_at || 0) - new Date(a.generated_at || 0))[0]
    : null;
  const hasReminder = latestReminder && latestReminder.pdf_url;

  const actions = [
    {
      key: 'viewInitial',
      label: 'Voir facture initiale',
      icon: <FiFileText size={14} />,
      onClick: () => onViewPdf(ensurePdfUrlWorksInDev(invoice.pdf_url)),
      className: styles.actionBtnSecondary,
      show: !!invoice.pdf_url,
    },
    {
      key: 'viewReminder',
      label: 'Voir rappel (PDF)',
      icon: <FiBell size={14} />,
      onClick: () => {
        if (latestReminder?.pdf_url) {
          window.open(ensurePdfUrlWorksInDev(latestReminder.pdf_url), '_blank');
        }
      },
      className: styles.actionBtnSecondary,
      show: hasReminder,
    },
    {
      key: 'sendEmail',
      label: 'Envoyer par email',
      icon: <FiMail size={14} />,
      onClick: onSendEmail,
      className: styles.actionBtnPrimary,
      show: canSendInvoice(invoice),
    },
    {
      key: 'send',
      label: 'Marquer envoyee (papier)',
      icon: <FiSend size={14} />,
      onClick: onSend,
      className: styles.actionBtnSecondary,
      show: canSendInvoice(invoice),
    },
    {
      key: 'payment',
      label: 'Enregistrer paiement',
      icon: <FiDollarSign size={14} />,
      onClick: onPayment,
      className: styles.actionBtnSuccess,
      show: canAddPayment(invoice),
    },
    {
      key: 'reminder',
      label: `Generer rappel suivant ${getNextReminderLevel(invoice)}`,
      icon: <FiClock size={14} />,
      onClick: onReminder,
      className: styles.actionBtnWarning,
      show: canGenerateReminder(invoice),
    },
    {
      key: 'sendReminderEmail',
      label: 'Envoyer rappel par email',
      icon: <FiMail size={14} />,
      onClick: onSendReminderEmail,
      className: styles.actionBtnPrimary,
      show: invoice.reminder_level > 0 && invoice.status !== 'paid',
    },
    {
      key: 'regenerate',
      label: 'Regenerer PDF',
      icon: <FiRefreshCw size={14} />,
      onClick: onRegeneratePdf,
      className: styles.actionBtnSecondary,
      show: canRegeneratePdf(invoice),
    },
    {
      key: 'duplicate',
      label: 'Creer un correctif',
      icon: <FiEdit size={14} />,
      onClick: onDuplicate,
      className: styles.actionBtnSecondary,
      show: canDuplicateInvoice(invoice),
    },
    {
      key: 'cancel',
      label: 'Annuler',
      icon: <FiXCircle size={14} />,
      onClick: onCancel,
      className: styles.actionBtnDanger,
      show: canCancelInvoice(invoice),
    },
  ];

  const visibleActions = actions.filter((action) => action.show);

  const calculateMenuPosition = () => {
    if (!buttonRef.current) return null;

    const rect = buttonRef.current.getBoundingClientRect();
    
    // Constantes
    const MENU_WIDTH = 240;
    const ITEM_HEIGHT = 58;
    const GAP = 4; // Petit gap entre le bouton et le menu
    const VIEWPORT_MARGIN = 16; // Marge de sécurité par rapport au viewport
    const estimatedMenuHeight = visibleActions.length * ITEM_HEIGHT + 16;
    
    // Espaces disponibles (en pixels depuis les bords du viewport)
    const spaceBelow = window.innerHeight - rect.bottom - VIEWPORT_MARGIN;
    const spaceAbove = rect.top - VIEWPORT_MARGIN;
    const spaceRight = window.innerWidth - rect.left - VIEWPORT_MARGIN;
    const spaceLeft = rect.right - VIEWPORT_MARGIN;
    
    // Position verticale
    let positioning = {}; // Peut contenir { top, bottom, maxHeight }
    let direction = 'below';
    
    // Décider si on affiche en dessous ou au-dessus
    if (spaceBelow >= estimatedMenuHeight) {
      // Cas 1 : assez d'espace en dessous
      // Le haut du menu part du bas du bouton + GAP
      positioning.top = rect.bottom + GAP;
      direction = 'below';
      
    } else if (spaceAbove >= estimatedMenuHeight) {
      // Cas 2 : assez d'espace au-dessus
      // Le BAS du menu doit être collé au haut du bouton - GAP
      // On utilise 'bottom' pour positionner depuis le bas du viewport
      positioning.bottom = window.innerHeight - rect.top + GAP;
      direction = 'above';
      
    } else {
      // Cas 3 : pas assez d'espace ni en haut ni en bas
      if (spaceAbove > spaceBelow) {
        // Plus d'espace au-dessus - utiliser bottom
        positioning.bottom = window.innerHeight - rect.top + GAP;
        positioning.maxHeight = rect.top - VIEWPORT_MARGIN - GAP;
        direction = 'above';
      } else {
        // Plus d'espace en dessous - utiliser top
        positioning.top = rect.bottom + GAP;
        positioning.maxHeight = window.innerHeight - rect.bottom - GAP - VIEWPORT_MARGIN;
        direction = 'below';
      }
    }
    
    // Position horizontale (identique pour les deux directions)
    let left;
    let horizontalAlignment = 'left';
    
    if (spaceRight >= MENU_WIDTH) {
      left = rect.left;
      horizontalAlignment = 'left';
    } else if (spaceLeft >= MENU_WIDTH) {
      left = rect.right - MENU_WIDTH;
      horizontalAlignment = 'right';
    } else {
      const idealLeft = rect.left + (rect.width - MENU_WIDTH) / 2;
      left = Math.max(
        VIEWPORT_MARGIN,
        Math.min(idealLeft, window.innerWidth - MENU_WIDTH - VIEWPORT_MARGIN)
      );
      horizontalAlignment = 'center';
    }
    
    // Sécurités horizontales
    left = Math.min(left, window.innerWidth - MENU_WIDTH - VIEWPORT_MARGIN);
    left = Math.max(VIEWPORT_MARGIN, left);
    
    return {
      ...positioning,
      left: Math.round(left),
      maxHeight: positioning.maxHeight ? Math.round(positioning.maxHeight) : null,
      direction,
      horizontalAlignment,
    };
  };

  const handleToggleMenu = () => {
    if (!showMenu) {
      const position = calculateMenuPosition();
      if (position) {
        setMenuPosition(position);
      }
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
        <FiMoreHorizontal size={16} />
      </button>

      {showMenu && (
        <>
          <div 
            className={styles.menuOverlay} 
            onClick={() => setShowMenu(false)}
            onContextMenu={(e) => {
              e.preventDefault();
              setShowMenu(false);
            }}
          />
          <div
            ref={menuRef}
            className={`${styles.menu} ${styles[`menu${menuPosition.direction === 'above' ? 'Above' : 'Below'}`]}`}
            style={{
              position: 'fixed',
              ...(menuPosition.top !== undefined && { top: `${menuPosition.top}px` }),
              ...(menuPosition.bottom !== undefined && { bottom: `${menuPosition.bottom}px` }),
              left: `${menuPosition.left}px`,
              maxHeight: menuPosition.maxHeight ? `${menuPosition.maxHeight}px` : 'none',
              overflowY: menuPosition.maxHeight ? 'auto' : 'visible',
            }}
            role="menu"
            aria-orientation="vertical"
          >
            {visibleActions.map((action) => (
              <button
                key={action.key}
                className={`${styles.menuItem} ${action.className}`}
                onClick={() => handleAction(action.onClick)}
                role="menuitem"
                tabIndex={showMenu ? 0 : -1}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    handleAction(action.onClick);
                  } else if (e.key === 'Escape') {
                    setShowMenu(false);
                    buttonRef.current?.focus();
                  } else if (e.key === 'ArrowDown') {
                    e.preventDefault();
                    const nextButton = e.currentTarget.nextElementSibling;
                    if (nextButton) nextButton.focus();
                  } else if (e.key === 'ArrowUp') {
                    e.preventDefault();
                    const prevButton = e.currentTarget.previousElementSibling;
                    if (prevButton) prevButton.focus();
                  }
                }}
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
