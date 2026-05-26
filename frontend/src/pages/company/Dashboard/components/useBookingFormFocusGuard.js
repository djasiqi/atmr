import { useEffect, useLayoutEffect, useRef } from 'react';

const TYPING_SELECTOR =
  'input:not(#client-select), textarea, [contenteditable="true"]';

/**
 * Empêche les re-renders async (pricing, OSRM…) de voler le focus vers react-select.
 */
export function useBookingFormFocusGuard(formRef) {
  const typingTargetRef = useRef(null);

  useEffect(() => {
    const onFocusIn = (event) => {
      const form = formRef.current;
      if (!form) return;

      const target = event.target;
      if (!(target instanceof HTMLElement) || !form.contains(target)) return;
      if (target.id === 'client-select') return;
      if (target.matches(TYPING_SELECTOR)) {
        typingTargetRef.current = target;
      }
    };

    const onPointerDown = (event) => {
      const form = formRef.current;
      if (!form) return;

      const target = event.target;
      if (!(target instanceof HTMLElement) || !form.contains(target)) return;
      if (target.closest('[data-tour-id="booking-client"]')) {
        typingTargetRef.current = null;
      }
    };

    document.addEventListener('focusin', onFocusIn, true);
    document.addEventListener('pointerdown', onPointerDown, true);
    return () => {
      document.removeEventListener('focusin', onFocusIn, true);
      document.removeEventListener('pointerdown', onPointerDown, true);
    };
  }, [formRef]);

  useLayoutEffect(() => {
    const intended = typingTargetRef.current;
    if (!(intended instanceof HTMLElement) || !intended.isConnected) return;
    if (document.activeElement?.id !== 'client-select') return;
    if (intended.id === 'client-select') return;

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
  });
}
