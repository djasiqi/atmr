import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import DriverInvitationPage from './DriverInvitationPage';
import { ACCEPT_NETWORK_MESSAGE } from '../../../services/driverInvitationService';

jest.mock('../../../services/driverInvitationService', () => {
  const actual = jest.requireActual('../../../services/driverInvitationService');
  return {
    ...actual,
    openDriverInvitation: jest.fn(),
    acceptDriverInvitation: jest.fn(),
  };
});

const service = require('../../../services/driverInvitationService');

const WORST_COMPANY = 'Transports médico-sociaux Aleksandra Wiśniewska-Kowalczyk & Fils';

function renderAt(path) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <Routes>
        <Route path="/invitation/chauffeur" element={<DriverInvitationPage />} />
      </Routes>
    </MemoryRouter>
  );
}

describe('invitation chauffeur', () => {
  beforeEach(() => {
    service.openDriverInvitation.mockReset();
    service.acceptDriverInvitation.mockReset();
  });

  it('n’ouvre pas d’inscription publique sans lien', () => {
    renderAt('/invitation/chauffeur');
    expect(screen.getByText(/pas d’inscription chauffeur publique/i)).toBeInTheDocument();
    expect(screen.queryByRole('link', { name: /créer un compte chauffeur/i })).not.toBeInTheDocument();
  });

  it('présente l’entreprise et le verrou fermé', async () => {
    service.openDriverInvitation.mockResolvedValue({
      status: 'pending',
      company_name: WORST_COMPANY,
      first_name: 'Aleksandra',
      email_hint: 'a***@example.test',
      identity: 'new',
      activation: { enabled: false },
    });
    renderAt('/invitation/chauffeur?token=secret');
    expect(await screen.findByText(WORST_COMPANY)).toBeInTheDocument();
    expect(screen.getByText(/n’est pas ouverte/i)).toBeInTheDocument();
    expect(screen.getByLabelText('Choisissez votre mot de passe')).toBeInTheDocument();
  });

  it('affiche un lien expiré, annulé ou déjà utilisé', async () => {
    service.openDriverInvitation.mockResolvedValue({
      status: 'expired',
      company_name: 'Lac',
      first_name: 'Nora',
      email_hint: 'n***@example.test',
      identity: 'new',
      activation: { enabled: false },
    });
    const { unmount } = renderAt('/invitation/chauffeur?token=ancien');
    expect(await screen.findByRole('heading', { name: 'Lien expiré' })).toBeInTheDocument();
    unmount();

    service.openDriverInvitation.mockResolvedValue({
      status: 'cancelled',
      company_name: 'Lac',
      first_name: 'Nora',
      email_hint: 'n***@example.test',
      identity: 'new',
      activation: { enabled: false },
    });
    const cancelled = renderAt('/invitation/chauffeur?token=annule');
    expect(await screen.findByRole('heading', { name: 'Invitation annulée' })).toBeInTheDocument();
    cancelled.unmount();

    service.openDriverInvitation.mockResolvedValue({
      status: 'accepted',
      company_name: 'Lac',
      first_name: 'Nora',
      email_hint: 'n***@example.test',
      identity: 'new',
      activation: { enabled: false },
    });
    renderAt('/invitation/chauffeur?token=utilise');
    expect(
      await screen.findByRole('heading', { name: 'Invitation déjà utilisée' })
    ).toBeInTheDocument();
  });

  it('demande une connexion pour un compte existant et explique l’échec du verrou', async () => {
    service.openDriverInvitation.mockResolvedValue({
      status: 'pending',
      company_name: 'Lac',
      first_name: 'Nora',
      email_hint: 'n***@example.test',
      identity: 'account_present',
      activation: { enabled: false },
    });
    service.acceptDriverInvitation.mockRejectedValue({
      response: {
        status: 403,
        data: {
          message: 'L’activation des chauffeurs n’est pas ouverte. L’invitation reste en attente.',
        },
      },
    });
    const user = userEvent.setup();
    renderAt('/invitation/chauffeur?token=compte');
    expect(await screen.findByRole('link', { name: 'Se connecter' })).toHaveAttribute('href', '/login');
    await user.click(screen.getByRole('checkbox'));
    await user.click(screen.getByRole('button', { name: 'Accepter l’invitation' }));
    expect(await screen.findByRole('alert')).toHaveTextContent(/n’est pas ouverte/);
    expect(service.acceptDriverInvitation).toHaveBeenCalledWith(
      expect.objectContaining({ accept: true, token: 'compte' })
    );
  });

  it('conserve une phrase sûre si la réponse se perd', async () => {
    service.openDriverInvitation.mockRejectedValue(new Error('network'));
    renderAt('/invitation/chauffeur?token=perdu');
    await waitFor(() => {
      expect(screen.getByRole('alert')).toHaveTextContent(ACCEPT_NETWORK_MESSAGE);
    });
  });
});
