import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';

import DemoRequest from './DemoRequest';
import { submitDemoRequest } from '../../services/demoRequestService';

jest.mock('../../services/demoRequestService', () => ({
  DEMO_ORGANIZATION_TYPES: ['transport_company', 'ems', 'clinic', 'hospital', 'curatorship', 'other'],
  DEMO_USE_CASES: ['planning_dispatch', 'billing', 'transport_tracking', 'multi_company_coordination', 'reporting', 'si_integration', 'other'],
  submitDemoRequest: jest.fn(),
}));

const renderPage = () =>
  render(
    <BrowserRouter>
      <DemoRequest />
    </BrowserRouter>
  );

describe('DemoRequest page', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('affiche l entete et la premiere etape', () => {
    renderPage();
    expect(screen.getByRole('heading', { name: /demande de demonstration/i })).toBeInTheDocument();
    expect(screen.getByText(/etape 1 \/ 3/i)).toBeInTheDocument();
  });

  it('bloque la progression si champs requis manquants', async () => {
    renderPage();
    fireEvent.click(screen.getByRole('button', { name: /continuer/i }));

    await waitFor(() => {
      expect(screen.getByText(/le nom est requis/i)).toBeInTheDocument();
    });
  });

  it('soumet la demande apres les 3 etapes', async () => {
    submitDemoRequest.mockResolvedValue({ ok: true });
    renderPage();

    fireEvent.change(screen.getByLabelText(/nom et prenom/i), { target: { value: 'Marie Curie' } });
    fireEvent.change(screen.getByLabelText(/email professionnel/i), { target: { value: 'marie@example.com' } });
    fireEvent.change(screen.getByLabelText(/organisation \/ institution/i), { target: { value: 'Clinique Test' } });
    fireEvent.click(screen.getByLabelText(/clinique/i));
    fireEvent.click(screen.getByRole('button', { name: /continuer/i }));

    fireEvent.change(screen.getByLabelText(/cas d'usage principal/i), { target: { value: 'planning_dispatch' } });
    fireEvent.click(screen.getByLabelText(/^oui$/i));
    fireEvent.change(screen.getByLabelText(/avec quel systeme principal/i), { target: { value: 'ERP X' } });
    fireEvent.click(screen.getByLabelText(/immediat/i));
    fireEvent.click(screen.getByRole('button', { name: /continuer/i }));

    fireEvent.change(screen.getByLabelText(/creneau souhaite/i), { target: { value: 'this_week' } });
    fireEvent.change(screen.getByLabelText(/plage horaire preferee/i), { target: { value: 'morning' } });
    fireEvent.click(screen.getByRole('checkbox'));
    fireEvent.click(screen.getByRole('button', { name: /envoyer la demande/i }));

    await waitFor(() => {
      expect(submitDemoRequest).toHaveBeenCalled();
      expect(screen.getByText(/un membre de l'equipe lirie vous contacte/i)).toBeInTheDocument();
    });
  });
});
