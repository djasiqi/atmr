import React, { useState } from 'react';
import AdminActionDialog from '../AdminActionDialog';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';

describe('AdminActionDialog', () => {
  it('bloque la double soumission et conserve l’erreur API', async () => {
    const onConfirm = jest.fn().mockRejectedValue(new Error('Échec serveur'));
    const onClose = jest.fn();

    render(
      <AdminActionDialog
        open
        title="Action test"
        description="Description"
        confirmationLabel="Confirmer"
        onConfirm={onConfirm}
        onClose={onClose}
      />
    );

    const btn = screen.getByRole('button', { name: 'Confirmer' });
    fireEvent.click(btn);
    fireEvent.click(btn);

    await waitFor(() => {
      expect(onConfirm).toHaveBeenCalledTimes(1);
      expect(screen.getByRole('alert')).toHaveTextContent('Échec serveur');
    });
    expect(onClose).not.toHaveBeenCalled();
  });

  it('ne réinitialise pas l’erreur quand loading externe change', async () => {
    const onConfirm = jest.fn().mockRejectedValue(new Error('Échec serveur'));
    let setLoadingExt;

    function Harness() {
      const [loading, setLoading] = useState(false);
      setLoadingExt = setLoading;
      return (
        <AdminActionDialog
          open
          title="Action test"
          description="Description"
          confirmationLabel="Confirmer"
          loading={loading}
          onConfirm={onConfirm}
          onClose={() => {}}
        />
      );
    }

    render(<Harness />);
    fireEvent.click(screen.getByRole('button', { name: 'Confirmer' }));

    await waitFor(() => {
      expect(screen.getByRole('alert')).toHaveTextContent('Échec serveur');
    });

    await act(async () => {
      setLoadingExt(true);
    });
    await act(async () => {
      setLoadingExt(false);
    });

    expect(screen.getByRole('alert')).toHaveTextContent('Échec serveur');
  });
});
