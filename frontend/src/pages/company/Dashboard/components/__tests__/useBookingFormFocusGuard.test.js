import { renderHook, act, fireEvent } from '@testing-library/react';
import { useRef } from 'react';
import { useBookingFormFocusGuard } from '../useBookingFormFocusGuard';

describe('useBookingFormFocusGuard', () => {
  it('restaure le focus si react-select le vole après un re-render', () => {
    const form = document.createElement('form');
    const medicalInput = document.createElement('input');
    medicalInput.id = 'hospital_service';
    const clientInput = document.createElement('input');
    clientInput.id = 'client-select';
    form.append(medicalInput, clientInput);
    document.body.appendChild(form);

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
});
