// src/components/common/Modal.jsx
import React, { useEffect, useRef } from 'react';

const FOCUSABLE_SELECTORS =
  'a[href], button:not([disabled]), textarea, input, select, [tabindex]:not([tabindex="-1"])';

const Modal = ({ children, onClose, size = 'lg', className = '', ariaLabel = 'Fenêtre modale' }) => {
  const contentRef = useRef(null);
  const previousActiveRef = useRef(null);

  useEffect(() => {
    previousActiveRef.current = document.activeElement;
    const content = contentRef.current;
    if (!content) return undefined;

    const focusables = content.querySelectorAll(FOCUSABLE_SELECTORS);
    if (focusables.length > 0) {
      focusables[0].focus();
    } else {
      content.focus();
    }

    const onKeyDown = (e) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        onClose?.();
        return;
      }
      if (e.key !== 'Tab') return;
      const nodes = Array.from(content.querySelectorAll(FOCUSABLE_SELECTORS));
      if (nodes.length === 0) {
        e.preventDefault();
        content.focus();
        return;
      }
      const first = nodes[0];
      const last = nodes[nodes.length - 1];
      const active = document.activeElement;
      if (e.shiftKey && active === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && active === last) {
        e.preventDefault();
        first.focus();
      }
    };

    document.addEventListener('keydown', onKeyDown);
    return () => {
      document.removeEventListener('keydown', onKeyDown);
      const prev = previousActiveRef.current;
      if (prev && typeof prev.focus === 'function') {
        prev.focus();
      }
    };
  }, [onClose]);

  const handleClickOutside = (e) => {
    // Ferme le modal si on clique sur l'overlay (pas sur le contenu)
    if (e.target.classList.contains('modal-overlay')) {
      onClose();
    }
  };

  const overlayClasses = ['modal-overlay', `modal-${size}`, className].filter(Boolean).join(' ');

  return (
    <div className={overlayClasses} onClick={handleClickOutside} role="presentation">
      <div
        className="modal-content"
        ref={contentRef}
        role="dialog"
        aria-modal="true"
        aria-label={ariaLabel}
        tabIndex={-1}
      >
        {children}
      </div>
    </div>
  );
};

export default Modal;
