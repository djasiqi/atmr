// frontend/src/pages/company/Clients/components/ClientDrawer.jsx
import React, { useEffect, useRef } from 'react';
import ClientReadView from './ClientReadView';
import ClientEditForm from './ClientEditForm';
import styles from './ClientDrawer.module.css';

/**
 * Side Drawer pour afficher les détails d'un client
 * Pattern Master-Detail : la liste reste visible, le drawer s'ouvre à droite
 */
const ClientDrawer = ({
  client,
  isOpen,
  isEditMode,
  onClose,
  onEdit,
  onSave,
  onCancelEdit,
  loading = false,
  hasUnsavedChanges: externalHasUnsavedChanges = false,
  onUnsavedChangesChange,
  onReloadClient,
}) => {
  const [hasUnsavedChanges, setHasUnsavedChanges] = React.useState(externalHasUnsavedChanges);
  const drawerRef = useRef(null);
  const overlayRef = useRef(null);

  // Gestion ESC pour fermer
  useEffect(() => {
    if (!isOpen) return;

    const handleEscape = (e) => {
      if (e.key === 'Escape' && !hasUnsavedChanges) {
        onClose();
      } else if (e.key === 'Escape' && hasUnsavedChanges) {
        const confirmed = window.confirm(
          'Modifications non sauvegardées. Voulez-vous vraiment fermer ?'
        );
        if (confirmed) {
          onCancelEdit();
          onClose();
        }
      }
    };

    document.addEventListener('keydown', handleEscape);
    return () => document.removeEventListener('keydown', handleEscape);
  }, [isOpen, hasUnsavedChanges, onClose, onCancelEdit]);

  // Focus management et focus trap
  useEffect(() => {
    if (!isOpen || !drawerRef.current) return;

    const drawer = drawerRef.current;
    const focusableElements = drawer.querySelectorAll(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );
    const firstFocusable = focusableElements[0];
    const lastFocusable = focusableElements[focusableElements.length - 1];

    // Focus initial sur le bouton de fermeture
    const closeButton = drawer.querySelector('button[aria-label="Fermer"], button[title*="Fermer"]');
    if (closeButton) {
      closeButton.focus();
    } else if (firstFocusable) {
      firstFocusable.focus();
    }

    // Focus trap : garder le focus dans le drawer
    const handleTabKey = (e) => {
      if (e.key !== 'Tab') return;

      if (e.shiftKey) {
        // Shift + Tab
        if (document.activeElement === firstFocusable) {
          e.preventDefault();
          lastFocusable?.focus();
        }
      } else {
        // Tab
        if (document.activeElement === lastFocusable) {
          e.preventDefault();
          firstFocusable?.focus();
        }
      }
    };

    drawer.addEventListener('keydown', handleTabKey);
    return () => {
      drawer.removeEventListener('keydown', handleTabKey);
    };
  }, [isOpen]);

  // Empêcher scroll du body quand drawer ouvert
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }
    return () => {
      document.body.style.overflow = '';
    };
  }, [isOpen]);

  if (!isOpen || !client) return null;

  const handleOverlayClick = (e) => {
    if (e.target === overlayRef.current) {
      if (hasUnsavedChanges) {
        const confirmed = window.confirm(
          'Modifications non sauvegardées. Voulez-vous vraiment fermer ?'
        );
        if (confirmed) {
          onCancelEdit();
          onClose();
        }
      } else {
        onClose();
      }
    }
  };

  return (
    <>
      {/* Overlay */}
      <div
        ref={overlayRef}
        className={`${styles.overlay} ${isOpen ? styles.visible : ''}`}
        onClick={handleOverlayClick}
        aria-hidden="true"
      />

      {/* Drawer */}
      <aside
        ref={drawerRef}
        className={`${styles.drawer} ${isOpen ? styles.open : ''}`}
        aria-label="Détails du client"
        aria-hidden={!isOpen}
        tabIndex={-1}
      >
        {isEditMode ? (
          <ClientEditForm
            client={client}
            onSave={onSave}
            onCancel={onCancelEdit}
            onClose={onClose}
            loading={loading}
            hasUnsavedChanges={hasUnsavedChanges}
            onUnsavedChangesChange={(hasChanges) => {
              setHasUnsavedChanges(hasChanges);
              onUnsavedChangesChange?.(hasChanges);
            }}
            onReloadClient={onReloadClient}
          />
        ) : (
          <ClientReadView
            client={client}
            onEdit={onEdit}
            onClose={onClose}
            loading={loading}
          />
        )}
      </aside>
    </>
  );
};

export default ClientDrawer;
