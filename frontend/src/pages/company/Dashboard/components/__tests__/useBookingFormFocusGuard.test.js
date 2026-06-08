import { renderHook, act, fireEvent } from '@testing-library/react';
import { useRef } from 'react';
import { useBookingFormFocusGuard } from '../useBookingFormFocusGuard';

function setupForm() {
  const form = document.createElement('form');
  const clientZone = document.createElement('div');
  clientZone.setAttribute('data-tour-id', 'booking-client');

  const medicalInput = document.createElement('input');
  medicalInput.id = 'pickup_location';

  const clientControl = document.createElement('div');
  clientControl.className = 'react-select__control';

  const clientInput = document.createElement('input');
  clientInput.id = 'client-select';

  clientControl.append(clientInput);
  clientZone.append(clientControl);
  form.append(medicalInput, clientZone);
  document.body.appendChild(form);

  return { form, medicalInput, clientInput };
}

describe('useBookingFormFocusGuard', () => {
  afterEach(() => {
    document.body.innerHTML = '';
  });

  it('restaure le focus si react-select le vole après un re-render', () => {
    const { form, medicalInput, clientInput } = setupForm();

    const { rerender, unmount } = renderHook(() => {
      const formRef = useRef(form);
      useBookingFormFocusGuard(formRef);
    });

    act(() => {
      fireEvent.focusIn(medicalInput);
    });

    act(() => {
      clientInput.focus();
      rerender();
    });

    expect(document.activeElement).toBe(medicalInput);

    unmount();
    form.remove();
  });

  it('restaure le focus de façon asynchrone (requestAnimationFrame)', async () => {
    const { form, medicalInput, clientInput } = setupForm();

    const { unmount } = renderHook(() => {
      const formRef = useRef(form);
      useBookingFormFocusGuard(formRef);
    });

    act(() => {
      fireEvent.focusIn(medicalInput);
    });

    await act(async () => {
      clientInput.focus();
      await new Promise((resolve) => requestAnimationFrame(resolve));
    });

    expect(document.activeElement).toBe(medicalInput);

    unmount();
    form.remove();
  });

  it('laisse le focus sur le client après un clic intentionnel', () => {
    const { form, medicalInput, clientInput } = setupForm();
    const clientZoneEl = form.querySelector('[data-tour-id="booking-client"]');

    const { unmount } = renderHook(() => {
      const formRef = useRef(form);
      useBookingFormFocusGuard(formRef);
    });

    act(() => {
      fireEvent.focusIn(medicalInput);
    });

    act(() => {
      fireEvent.pointerDown(clientZoneEl);
      clientInput.focus();
    });

    expect(document.activeElement).toBe(clientInput);

    unmount();
    form.remove();
  });

  it('retrouve le champ via son id si le noeud DOM a été recréé', () => {
    const { form, medicalInput, clientInput } = setupForm();

    const { rerender, unmount } = renderHook(() => {
      const formRef = useRef(form);
      useBookingFormFocusGuard(formRef);
    });

    act(() => {
      fireEvent.focusIn(medicalInput);
    });

    const replacement = document.createElement('input');
    replacement.id = 'pickup_location';
    medicalInput.replaceWith(replacement);

    act(() => {
      clientInput.focus();
      rerender();
    });

    expect(document.activeElement).toBe(replacement);

    unmount();
    form.remove();
  });
});
