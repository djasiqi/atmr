import AdminActionDialog from '../AdminActionDialog';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';

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
});
