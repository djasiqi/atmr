import React, { useState, useRef, useEffect } from 'react';
import { FiMoreHorizontal, FiEye, FiEdit2, FiTrash2 } from 'react-icons/fi';
import styles from './ClientTableRowActions.module.css';

const ClientTableRowActions = ({ client, onEdit, onDelete, onView }) => {
  const [isOpen, setIsOpen] = useState(false);
  const menuRef = useRef(null);
  const buttonRef = useRef(null);

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
        <FiMoreHorizontal size={16} />
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
              <FiEye size={14} />
              Voir details
            </button>
          )}
          <button
            type="button"
            onClick={handleEdit}
            className={styles.menuItem}
            role="menuitem"
          >
            <FiEdit2 size={14} />
            Modifier
          </button>
          <div className={styles.menuDivider} />
          <button
            type="button"
            onClick={handleDelete}
            className={`${styles.menuItem} ${styles.menuItemDanger}`}
            role="menuitem"
          >
            <FiTrash2 size={14} />
            Supprimer
          </button>
        </div>
      )}
    </div>
  );
};

export default ClientTableRowActions;
