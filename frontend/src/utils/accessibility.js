import React from 'react';

/**
 * Utilitaires pour l'accessibilité (A11y)
 */

/**
 * Gère le focus trap dans un élément (utile pour les modals)
 * @param {HTMLElement} element - L'élément dans lequel trapper le focus
 * @returns {Function} - Fonction de nettoyage
 */
export function trapFocus(element) {
  const focusableElements = element.querySelectorAll(
    'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
  );

  const firstFocusable = focusableElements[0];
  const lastFocusable = focusableElements[focusableElements.length - 1];

  const handleTabKey = (e) => {
    if (e.key !== 'Tab') return;

    if (e.shiftKey) {
      // Shift + Tab
      if (document.activeElement === firstFocusable) {
        e.preventDefault();
        lastFocusable.focus();
      }
    } else {
      // Tab
      if (document.activeElement === lastFocusable) {
        e.preventDefault();
        firstFocusable.focus();
      }
    }
  };

  element.addEventListener('keydown', handleTabKey);

  // Focus le premier élément
  if (firstFocusable) {
    firstFocusable.focus();
  }

  // Retourner la fonction de nettoyage
  return () => {
    element.removeEventListener('keydown', handleTabKey);
  };
}

/**
 * Annonce un message aux lecteurs d'écran
 * @param {string} message - Le message à annoncer
 * @param {string} priority - 'polite' ou 'assertive'
 */
export function announceToScreenReader(message, priority = 'polite') {
  const announcement = document.createElement('div');
  announcement.setAttribute('role', 'status');
  announcement.setAttribute('aria-live', priority);
  announcement.setAttribute('aria-atomic', 'true');
  announcement.className = 'sr-only';
  announcement.textContent = message;

  document.body.appendChild(announcement);

  // Retirer après 1 seconde
  setTimeout(() => {
    document.body.removeChild(announcement);
  }, 1000);
}

/**
 * Génère un ID unique pour aria-describedby
 * @param {string} prefix - Préfixe pour l'ID
 * @returns {string} - ID unique
 */
export function generateAriaId(prefix = 'aria') {
  return `${prefix}-${Math.random().toString(36).substr(2, 9)}`;
}

/**
 * Hook React pour gérer le focus trap dans un composant
 * Usage dans un composant fonctionnel:
 * const modalRef = useFocusTrap(isOpen);
 */
export function useFocusTrap(isActive) {
  const elementRef = React.useRef(null);

  React.useEffect(() => {
    if (isActive && elementRef.current) {
      return trapFocus(elementRef.current);
    }
  }, [isActive]);

  return elementRef;
}

/**
 * Vérifie si un élément est focusable
 * @param {HTMLElement} element - L'élément à vérifier
 * @returns {boolean}
 */
export function isFocusable(element) {
  if (!element) return false;

  const focusableSelectors = [
    'a[href]',
    'button:not([disabled])',
    'input:not([disabled])',
    'select:not([disabled])',
    'textarea:not([disabled])',
    '[tabindex]:not([tabindex="-1"])',
  ];

  return focusableSelectors.some((selector) => element.matches(selector));
}

/**
 * Gère la navigation au clavier dans une liste
 * @param {KeyboardEvent} event - L'événement clavier
 * @param {Array} items - Les éléments de la liste
 * @param {number} currentIndex - L'index actuel
 * @param {Function} onSelect - Callback appelé lors de la sélection
 * @returns {number} - Le nouvel index
 */
export function handleKeyboardNavigation(event, items, currentIndex, onSelect) {
  let newIndex = currentIndex;

  switch (event.key) {
    case 'ArrowDown':
      event.preventDefault();
      newIndex = currentIndex < items.length - 1 ? currentIndex + 1 : 0;
      break;

    case 'ArrowUp':
      event.preventDefault();
      newIndex = currentIndex > 0 ? currentIndex - 1 : items.length - 1;
      break;

    case 'Home':
      event.preventDefault();
      newIndex = 0;
      break;

    case 'End':
      event.preventDefault();
      newIndex = items.length - 1;
      break;

    case 'Enter':
    case ' ':
      event.preventDefault();
      if (onSelect) {
        onSelect(items[currentIndex]);
      }
      break;

    default:
      return currentIndex;
  }

  return newIndex;
}

/**
 * Ajoute des attributs ARIA standards à un élément de bouton
 * @param {Object} options - Options
 * @returns {Object} - Attributs ARIA
 */
export function getButtonAriaProps(options = {}) {
  const { label, pressed, expanded, controls, describedBy, disabled } = options;

  const props = {};

  if (label) props['aria-label'] = label;
  if (typeof pressed === 'boolean') props['aria-pressed'] = pressed;
  if (typeof expanded === 'boolean') props['aria-expanded'] = expanded;
  if (controls) props['aria-controls'] = controls;
  if (describedBy) props['aria-describedby'] = describedBy;
  if (disabled) props['aria-disabled'] = true;

  return props;
}

/**
 * Ajoute des attributs ARIA standards à un élément de formulaire
 * @param {Object} options - Options
 * @returns {Object} - Attributs ARIA
 */
export function getFormAriaProps(options = {}) {
  const { label, required, invalid, describedBy, errorMessage } = options;

  const props = {};

  if (label) props['aria-label'] = label;
  if (required) props['aria-required'] = true;
  if (invalid) {
    props['aria-invalid'] = true;
    if (errorMessage) {
      const errorId = generateAriaId('error');
      props['aria-describedby'] = errorId;
      props['data-error-id'] = errorId;
      props['data-error-message'] = errorMessage;
    }
  }
  if (describedBy) props['aria-describedby'] = describedBy;

  return props;
}

const accessibilityUtils = {
  trapFocus,
  announceToScreenReader,
  generateAriaId,
  useFocusTrap,
  isFocusable,
  handleKeyboardNavigation,
  getButtonAriaProps,
  getFormAriaProps,
};

export default accessibilityUtils;
