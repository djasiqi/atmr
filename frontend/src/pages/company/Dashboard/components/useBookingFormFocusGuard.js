import { useCallback, useEffect, useLayoutEffect, useRef } from 'react';

const TYPING_SELECTOR =
  'input:not(#client-select), textarea, [contenteditable="true"]';
const CLIENT_FOCUS_GRACE_MS = 120;

function isClientSelectFocused(el) {
  if (!el || !(el instanceof HTMLElement)) return false;
  if (el.id === 'client-select') return true;
  const clientZone = el.closest('[data-tour-id="booking-client"]');
  if (!clientZone) return false;
  return Boolean(el.closest('.react-select__input, .react-select__control'));
}

function restoreTypingFocus(intended) {
  if (!(intended instanceof HTMLElement) || !intended.isConnected) return false;
  if (intended.id === 'client-select') return false;

  const selectionStart = intended.selectionStart;
  const selectionEnd = intended.selectionEnd;
  intended.focus({ preventScroll: true });
  if (
    (intended instanceof HTMLInputElement || intended instanceof HTMLTextAreaElement) &&
    typeof selectionStart === 'number' &&
    typeof selectionEnd === 'number'
  ) {
    intended.setSelectionRange(selectionStart, selectionEnd);
  }
  return true;
}

function resolveTypingTarget(form, typingTargetRef, typingFieldIdRef) {
  const current = typingTargetRef.current;
  if (current instanceof HTMLElement && current.isConnected) {
    return current;
  }
  const fieldId = typingFieldIdRef.current;
  if (!fieldId || !form) return null;
  const found = form.querySelector(`#${CSS.escape(fieldId)}`);
  if (found instanceof HTMLElement) {
    typingTargetRef.current = found;
    return found;
  }
  return null;
}

/**
 * Empêche les re-renders async (pricing, OSRM, react-select…) de voler le focus
 * vers le sélecteur client pendant la saisie dans les autres champs.
 */
export function useBookingFormFocusGuard(formRef) {
  const typingTargetRef = useRef(null);
  const typingFieldIdRef = useRef(null);
  const clientIntentAtRef = useRef(0);

  const tryRestoreFocus = useCallback(() => {
    const form = formRef.current;
    if (!form) return;
    if (!isClientSelectFocused(document.activeElement)) return;
    if (Date.now() - clientIntentAtRef.current < CLIENT_FOCUS_GRACE_MS) return;

    const intended = resolveTypingTarget(form, typingTargetRef, typingFieldIdRef);
    if (!intended || intended.id === 'client-select') return;
    restoreTypingFocus(intended);
  }, [formRef]);

  useEffect(() => {
    const onFocusIn = (event) => {
      const form = formRef.current;
      if (!form) return;

      const target = event.target;
      if (!(target instanceof HTMLElement) || !form.contains(target)) return;

      if (isClientSelectFocused(target)) {
        if (Date.now() - clientIntentAtRef.current < CLIENT_FOCUS_GRACE_MS) return;
        const intended = resolveTypingTarget(form, typingTargetRef, typingFieldIdRef);
        if (intended && intended.id !== 'client-select') {
          restoreTypingFocus(intended);
        }
        return;
      }

      if (target.id === 'client-select') return;
      if (target.matches(TYPING_SELECTOR)) {
        typingTargetRef.current = target;
        typingFieldIdRef.current = target.id || target.getAttribute('name') || null;
      }
    };

    const onPointerDown = (event) => {
      const form = formRef.current;
      if (!form) return;

      const target = event.target;
      if (!(target instanceof HTMLElement) || !form.contains(target)) return;

      if (target.closest('[data-tour-id="booking-client"]')) {
        clientIntentAtRef.current = Date.now();
        typingTargetRef.current = null;
        typingFieldIdRef.current = null;
        return;
      }

      if (
        target.matches(TYPING_SELECTOR) ||
        target.closest('[data-tour-id="booking-addresses"]')
      ) {
        clientIntentAtRef.current = 0;
      }
    };

    const onKeyDown = (event) => {
      if (event.key !== 'Tab') return;
      typingTargetRef.current = null;
      typingFieldIdRef.current = null;
    };

    document.addEventListener('focusin', onFocusIn, true);
    document.addEventListener('pointerdown', onPointerDown, true);
    document.addEventListener('keydown', onKeyDown, true);
    return () => {
      document.removeEventListener('focusin', onFocusIn, true);
      document.removeEventListener('pointerdown', onPointerDown, true);
      document.removeEventListener('keydown', onKeyDown, true);
    };
  }, [formRef]);

  useLayoutEffect(() => {
    tryRestoreFocus();
  });

  useEffect(() => {
    tryRestoreFocus();
    const rafId = requestAnimationFrame(() => {
      tryRestoreFocus();
    });
    return () => cancelAnimationFrame(rafId);
  });
}
