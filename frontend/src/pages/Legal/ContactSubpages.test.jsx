import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';

import ContactSupport from './ContactSupport';
import { submitContactRequest } from '../../services/contactService';

jest.mock('../../services/contactService', () => ({
  submitContactRequest: jest.fn(),
}));

const renderPage = () =>
  render(
    <BrowserRouter>
      <ContactSupport />
    </BrowserRouter>
  );

describe('Contact subpages', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('soumet avec la categorie correcte', async () => {
    submitContactRequest.mockResolvedValue({ ok: true, trace_id: 'ct_123' });
    renderPage();

    fireEvent.change(screen.getByLabelText(/nom/i), { target: { value: 'Marie Curie' } });
    fireEvent.change(screen.getByLabelText(/email/i), { target: { value: 'marie@example.com' } });
    fireEvent.change(screen.getByLabelText(/sujet \*/i), { target: { value: 'bug' } });
    fireEvent.change(screen.getByLabelText(/description du probleme/i), { target: { value: 'Erreur reccurente.' } });
    fireEvent.click(screen.getByRole('checkbox'));
    fireEvent.click(screen.getByRole('button', { name: /transmettre la demande|envoyer la demande/i }));

    await waitFor(() => {
      expect(submitContactRequest).toHaveBeenCalledWith(
        expect.objectContaining({
          category: 'support',
        })
      );
      expect(screen.getByText(/votre demande a ete transmise/i)).toBeInTheDocument();
    });
  });
});
