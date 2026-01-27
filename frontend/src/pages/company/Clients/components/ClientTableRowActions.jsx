// frontend/src/pages/company/Clients/components/ClientTableRowActions.jsx
import React, { useState, useRef, useEffect } from 'react';
import styles from './ClientTableRowActions.module.css';

/**
 * Menu d'actions pour une ligne de tableau (pattern ⋯)
 * Évite d'avoir des boutons visibles sur chaque ligne
 */
const ClientTableRowActions = ({ client, onEdit, onDelete, onView }) => {
  const [isOpen, setIsOpen] = useState(false);
  const menuRef = useRef(null);
  const buttonRef = useRef(null);

  // Fermer le menu si clic en dehors
  useEffect(() => {
    if (!isOpen) return;

    const handleClickOutside = (e) => {
      if (
        menuRef.current &&
        !menuRef.current.contains(e.target) &&
        buttonRef.current &&
        !buttonRef.current.contains(e.target)
      ) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [isOpen]);

  const handleEdit = (e) => {
    e.stopPropagation();
    setIsOpen(false);
    onEdit(client);
  };

  const handleDelete = (e) => {
    e.stopPropagation();
    setIsOpen(false);
    onDelete(client);
  };

  const handleView = (e) => {
    e.stopPropagation();
    setIsOpen(false);
    if (onView) {
      onView(client);
    }
  };

  return (
    <div className={styles.actionsContainer}>
      <button
        ref={buttonRef}
        type="button"
        onClick={(e) => {
          e.stopPropagation();
          setIsOpen(!isOpen);
        }}
        className={styles.menuButton}
        aria-label="Actions"
        aria-expanded={isOpen}
        title="Actions"
      >
        ⋯
      </button>

      {isOpen && (
        <div ref={menuRef} className={styles.menu} role="menu">
          {onView && (
            <button
              type="button"
              onClick={handleView}
              className={styles.menuItem}
              role="menuitem"
            >
              Voir détails
            </button>
          )}
          <button
            type="button"
            onClick={handleEdit}
            className={styles.menuItem}
            role="menuitem"
          >
            Modifier
          </button>
          <div className={styles.menuDivider} />
          <button
            type="button"
            onClick={handleDelete}
            className={`${styles.menuItem} ${styles.menuItemDanger}`}
            role="menuitem"
          >
            Supprimer
          </button>
        </div>
      )}
    </div>
  );
};

export default ClientTableRowActions;
