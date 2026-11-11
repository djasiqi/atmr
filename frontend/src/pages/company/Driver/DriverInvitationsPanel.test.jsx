import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import DriverInvitationsPanel from './DriverInvitationsPanel';
import { INVITE_NETWORK_MESSAGE } from '../../../services/driverInvitationService';

jest.mock('../../../services/driverInvitationService', () => {
  const actual = jest.requireActual('../../../services/driverInvitationService');
  return {
    ...actual,
    listDriverInvitations: jest.fn(),
    createDriverInvitation: jest.fn(),
    resendDriverInvitation: jest.fn(),
    cancelDriverInvitation: jest.fn(),
  };
});

const service = require('../../../services/driverInvitationService');

describe('liste des invitations chauffeur', () => {
  beforeEach(() => {
    service.listDriverInvitations.mockReset();
    service.createDriverInvitation.mockReset();
    service.resendDriverInvitation.mockReset();
    service.cancelDriverInvitation.mockReset();
  });

  it('affiche les statuts, le renvoi et l’annulation', async () => {
    service.listDriverInvitations.mockResolvedValue({
      invitations: [
        {
          public_id: 'inv-1',
          first_name: 'Aleksandra',
          last_name: 'Wiśniewska-Kowalczyk',
          email: 'aleksandra.wisniewska-kowalczyk@example.test',
          status: 'pending',
          status_label: 'En attente',
        },
        {
          public_id: 'inv-2',
          first_name: 'Nora',
          last_name: 'Bernard',
          email: 'nora@example.test',
          status: 'accepted',
          status_label: 'Acceptée',
        },
      ],
    });
    service.resendDriverInvitation.mockResolvedValue({
      notification: { state: 'queued', delivered: false },
    });
    const user = userEvent.setup();
    render(<DriverInvitationsPanel formOpen={false} onCloseForm={() => {}} />);
    expect(await screen.findByText(/Wiśniewska-Kowalczyk/)).toBeInTheDocument();
    expect(screen.getByText(/En attente/)).toBeInTheDocument();
    expect(screen.getByText(/Acceptée/)).toBeInTheDocument();
    expect(screen.getByText(/ne choisit pas le mot de passe/i)).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Renvoyer' }));
    expect(service.resendDriverInvitation).toHaveBeenCalledWith('inv-1');
    expect(await screen.findByRole('status')).toHaveTextContent(/mis en file/);
  });

  it('conserve le formulaire si l’envoi ne répond pas', async () => {
    service.listDriverInvitations.mockResolvedValue({ invitations: [] });
    service.createDriverInvitation.mockRejectedValue(new Error('network'));
    const user = userEvent.setup();
    render(<DriverInvitationsPanel formOpen onCloseForm={() => {}} />);
    await user.type(screen.getByRole('textbox', { name: 'Prénom' }), 'Nora');
    await user.type(screen.getByRole('textbox', { name: 'Nom' }), 'Bernard');
    await user.type(screen.getByRole('textbox', { name: 'E-mail' }), 'nora@example.test');
    await user.click(screen.getByRole('button', { name: 'Envoyer l’invitation' }));
    expect(await screen.findByRole('alert')).toHaveTextContent(INVITE_NETWORK_MESSAGE);
    expect(screen.getByRole('textbox', { name: 'Prénom' })).toHaveValue('Nora');
    expect(screen.queryByLabelText(/mot de passe/i)).not.toBeInTheDocument();
  });
});
