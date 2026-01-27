import React, { useState, useRef, useEffect } from 'react';
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

  // ✅ Trouver le rappel le plus récent (OPEN ou PAID)
  const latestReminder = invoice.reminders?.length > 0
    ? invoice.reminders
        .sort((a, b) => new Date(b.generated_at || 0) - new Date(a.generated_at || 0))[0]
    : null;
  const hasReminder = latestReminder && latestReminder.pdf_url;

  // ✅ Helper pour supprimer les emojis des labels (protection contre double icône)
  const stripEmojis = (text) => {
    // Regex pour détecter les emojis (plages Unicode principales)
    return text.replace(/[\u{1F300}-\u{1F9FF}]|[\u{2600}-\u{26FF}]|[\u{2700}-\u{27BF}]|[\u{1F900}-\u{1F9FF}]|[\u{1F1E0}-\u{1F1FF}]/gu, '').trim();
  };

  const actions = [
    // ✅ NOUVEAU : "Voir facture initiale" (toujours disponible si PDF existe)
    // Même après un rappel, la facture initiale reste accessible
    {
      key: 'viewInitial',
      label: 'Voir facture initiale',
      icon: '📄',
      onClick: onViewPdf,
      className: styles.actionBtnSecondary,
      show: !!invoice.pdf_url,
    },
    // ✅ NOUVEAU : "Voir rappel (PDF)" (si rappel existe avec PDF)
    {
      key: 'viewReminder',
      label: hasReminder && latestReminder.status === 'PAID'
        ? 'Voir rappel (PDF)'
        : 'Voir rappel (PDF)',
      icon: '🔔',
      onClick: () => {
        if (latestReminder?.pdf_url) {
          window.open(latestReminder.pdf_url, '_blank');
        }
      },
      className: styles.actionBtnSecondary,
      show: hasReminder,
    },
    {
      key: 'sendEmail',
      label: 'Envoyer par email',
      icon: '📧',
      onClick: onSendEmail,
      className: styles.actionBtnPrimary,
      show: canSendInvoice(invoice),
    },
    {
      key: 'send',
      label: 'Marquer envoyée (papier)',
      icon: '📄',
      onClick: onSend,
      className: styles.actionBtnSecondary,
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
      label: `Générer rappel suivant ${getNextReminderLevel(invoice)}`,
      icon: '⏰',
      onClick: onReminder,
      className: styles.actionBtnWarning,
      show: canGenerateReminder(invoice),
    },
    {
      key: 'sendReminderEmail',
      label: 'Envoyer rappel par email',
      icon: '📧',
      onClick: onSendReminderEmail,
      className: styles.actionBtnPrimary,
      show: invoice.reminder_level > 0 && invoice.status !== 'paid',
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
        // Debug détaillé
        const rect = buttonRef.current?.getBoundingClientRect();
        console.log(`🎯 Menu "${invoice.invoice_number}" - Direction: ${position.direction}`, {
          bouton: {
            top: Math.round(rect.top),
            bottom: Math.round(rect.bottom),
            left: Math.round(rect.left),
            height: Math.round(rect.height),
          },
          espaces: {
            dessous: Math.round(window.innerHeight - rect.bottom),
            dessus: Math.round(rect.top),
          },
          menuCalculé: position.direction === 'above' 
            ? {
                bottom: position.bottom,
                left: position.left,
                direction: position.direction,
                maxHeight: position.maxHeight,
                note: `Bas du menu = ${Math.round(window.innerHeight - position.bottom)}px depuis le haut`,
              }
            : {
                top: position.top,
                left: position.left,
                direction: position.direction,
                maxHeight: position.maxHeight,
              },
          viewport: {
            height: window.innerHeight,
            width: window.innerWidth,
          },
        });
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
        <span className={styles.actionIcon}>⚙️</span>
        <span className={styles.actionLabel}>Actions</span>
        <span className={styles.actionIcon}>▼</span>
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
            {visibleActions.map((action) => {
              // ✅ Protection: supprimer les emojis du label si présents (évite double icône)
              const cleanLabel = stripEmojis(action.label);
              
              return (
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
                  <span className={styles.actionLabel}>{cleanLabel}</span>
                </button>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
};

export default InvoiceRowActions;
