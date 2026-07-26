import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import SignupActivation from './SignupActivation';
import apiClient from '../../utils/apiClient';
import {
  removePendingActivationByEmail,
  removePendingActivationBySessionId,
  setPendingActivationSession,
} from '../../utils/activationSessionStore';

jest.mock('../../utils/apiClient', () => ({
  __esModule: true,
  default: {
    get: jest.fn(),
    post: jest.fn(),
  },
}));

jest.mock('../../utils/activationSessionStore', () => ({
  setPendingActivationSession: jest.fn(),
  removePendingActivationByEmail: jest.fn(),
  removePendingActivationBySessionId: jest.fn(),
}));

const mockNavigate = jest.fn();
jest.mock('react-router-dom', () => ({
  ...jest.requireActual('react-router-dom'),
  useNavigate: () => mockNavigate,
}));

const renderActivation = (initialEntry) =>
  render(
    <MemoryRouter initialEntries={[initialEntry]}>
      <Routes>
        <Route path="/activate-account" element={<SignupActivation />} />
      </Routes>
    </MemoryRouter>
  );

describe('SignupActivation', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    apiClient.get.mockReset();
    apiClient.post.mockReset();
    setPendingActivationSession.mockReset();
    removePendingActivationByEmail.mockReset();
    removePendingActivationBySessionId.mockReset();
    mockNavigate.mockClear();
  });

  it('verifie email via token puis charge le statut', async () => {
    apiClient.post.mockResolvedValueOnce({
      data: { activation_session_id: 'sess-123' },
    });
    apiClient.get.mockResolvedValueOnce({
      data: {
        activation_status: {
          email_verified: true,
          phone_verified: false,
          is_complete: false,
          is_finalized: false,
        },
      },
    });

    renderActivation('/activate-account?token=email-token-abc');

    await waitFor(() => {
      expect(apiClient.post).toHaveBeenCalledWith('/auth/activation/verify-email', {
        token: 'email-token-abc',
      });
    });
    await waitFor(() => {
      expect(apiClient.get).toHaveBeenCalledWith(
        '/auth/activation/status?activation_session_id=sess-123'
      );
    });
    expect(screen.getByText(/email confirm[eé] avec succ[eè]s/i)).toBeInTheDocument();
    await waitFor(() => {
      expect(mockNavigate).toHaveBeenCalledWith(
        '/activate-account?activation_session_id=sess-123',
        expect.objectContaining({ replace: true })
      );
    });
    await waitFor(() => {
      expect(setPendingActivationSession).toHaveBeenCalled();
    });
  });

  it('valide le code SMS', async () => {
    apiClient.get.mockResolvedValueOnce({
      data: {
        activation_status: {
          email_verified: true,
          phone_verified: false,
          is_complete: false,
          is_finalized: false,
        },
      },
    });
    apiClient.post.mockResolvedValueOnce({
      data: {
        activation_status: {
          email_verified: true,
          phone_verified: true,
          is_complete: true,
          is_finalized: false,
        },
      },
    });

    renderActivation('/activate-account?activation_session_id=sess-456');

    await waitFor(() => {
      expect(apiClient.get).toHaveBeenCalledTimes(1);
    });
    await waitFor(() => {
      expect(screen.queryByText(/traitement en cours/i)).not.toBeInTheDocument();
    });
    await waitFor(() => {
      expect(screen.getByRole('button', { name: /valider le code/i })).toBeEnabled();
    });

    fireEvent.change(screen.getByPlaceholderText(/code [aà] 6 chiffres/i), {
      target: { value: '123456' },
    });
    fireEvent.click(screen.getByRole('button', { name: /valider le code/i }));

    await waitFor(() => {
      expect(apiClient.post).toHaveBeenCalledWith('/auth/activation/verify-sms', {
        activation_session_id: 'sess-456',
        code: '123456',
      });
    });
    await waitFor(() => {
      expect(screen.getAllByText(/t[eé]l[eé]phone confirm[eé]/i).length).toBeGreaterThan(0);
    });
  });

  it('affiche la page de fin guidee apres finalize', async () => {
    apiClient.get.mockResolvedValueOnce({
      data: {
        activation_status: {
          email_verified: true,
          phone_verified: true,
          is_complete: true,
          is_finalized: false,
        },
      },
    });
    apiClient.post.mockResolvedValueOnce({
      data: {
        activation_status: {
          email_verified: true,
          phone_verified: true,
          is_complete: true,
          is_finalized: true,
        },
      },
    });

    renderActivation({
      pathname: '/activate-account',
      search: '?activation_session_id=sess-789',
      state: { prefillEmail: 'user@example.com' },
    });

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /activer mon compte/i })).toBeEnabled();
    });
    fireEvent.click(screen.getByRole('button', { name: /activer mon compte/i }));

    await waitFor(() => {
      expect(apiClient.post).toHaveBeenCalledWith('/auth/activation/finalize', {
        activation_session_id: 'sess-789',
      });
    });
    await screen.findByText(/activation terminee/i);
    expect(screen.getByRole('button', { name: /se connecter/i })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /activer mon compte/i })).not.toBeInTheDocument();
    expect(removePendingActivationBySessionId).toHaveBeenCalledWith('sess-789');
    expect(removePendingActivationByEmail).toHaveBeenCalledWith('user@example.com');
  });

  it('active le cooldown apres erreur rate-limit sur renvoi email', async () => {
    apiClient.get.mockResolvedValueOnce({
      data: {
        activation_status: {
          email_verified: false,
          phone_verified: false,
          is_complete: false,
          is_finalized: false,
        },
      },
    });
    apiClient.post.mockRejectedValueOnce({
      response: {
        data: {
          message: 'Trop de requetes',
          details: { retry_after_seconds: 12 },
        },
      },
    });

    renderActivation('/activate-account?activation_session_id=sess-cool');
    await waitFor(() => {
      expect(apiClient.get).toHaveBeenCalledTimes(1);
    });
    await waitFor(() => {
      expect(screen.queryByText(/traitement en cours/i)).not.toBeInTheDocument();
    });
    await waitFor(() => {
      expect(screen.getByRole('button', { name: /renvoyer l'email/i })).toBeEnabled();
    });

    fireEvent.click(screen.getByRole('button', { name: /renvoyer l'email/i }));

    await waitFor(() => {
      expect(apiClient.post).toHaveBeenCalledWith('/auth/activation/resend-email', {
        activation_session_id: 'sess-cool',
      });
    });
    await waitFor(() => {
      expect(screen.getByText(/renvoyer \(12s\)/i)).toBeInTheDocument();
    });
  });

  it('affiche le lien de secours si resend-email renvoie debug_activation_link', async () => {
    apiClient.get.mockResolvedValueOnce({
      data: {
        activation_status: {
          email_verified: false,
          phone_verified: false,
          is_complete: false,
          is_finalized: false,
        },
      },
    });
    apiClient.post.mockResolvedValueOnce({
      data: {
        message: 'Service email indisponible en local.',
        email_sent: false,
        debug_activation_link:
          'http://localhost:3000/activate-account?token=debug-token-123',
      },
    });

    renderActivation('/activate-account?activation_session_id=sess-debug-link');

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /renvoyer l'email/i })).toBeEnabled();
    });
    fireEvent.click(screen.getByRole('button', { name: /renvoyer l'email/i }));

    await waitFor(() => {
      expect(apiClient.post).toHaveBeenCalledWith('/auth/activation/resend-email', {
        activation_session_id: 'sess-debug-link',
      });
    });
    await waitFor(() => {
      const link = screen.getByRole('link', { name: /ouvrir le lien d'activation/i });
      expect(link).toHaveAttribute(
        'href',
        'http://localhost:3000/activate-account?token=debug-token-123'
      );
    });
  });

  it('permet de mettre a jour le numero puis renvoie un nouveau SMS', async () => {
    apiClient.get.mockResolvedValueOnce({
      data: {
        masked_phone: '+** *** *** 41',
        activation_status: {
          email_verified: true,
          phone_verified: false,
          is_complete: false,
          is_finalized: false,
        },
      },
    });
    apiClient.post.mockResolvedValueOnce({
      data: {
        message: 'Numéro mis à jour. Nouveau code SMS envoyé.',
        masked_phone: '+** *** *** 99',
        activation_status: {
          email_verified: true,
          phone_verified: false,
          is_complete: false,
          is_finalized: false,
        },
      },
    });

    renderActivation('/activate-account?activation_session_id=sess-phone');
    await waitFor(() => {
      expect(apiClient.get).toHaveBeenCalledTimes(1);
    });
    await waitFor(() => {
      expect(screen.getByRole('button', { name: /mauvais num[eé]ro/i })).toBeEnabled();
    });

    fireEvent.click(screen.getByRole('button', { name: /mauvais num[eé]ro/i }));
    fireEvent.change(screen.getByPlaceholderText(/nouveau num[eé]ro/i), {
      target: { value: '+41791230099' },
    });
    fireEvent.click(screen.getByRole('button', { name: /mettre [aà] jour/i }));

    await waitFor(() => {
      expect(apiClient.post).toHaveBeenCalledWith('/auth/activation/update-phone', {
        activation_session_id: 'sess-phone',
        phone: '+41791230099',
      });
    });
    await waitFor(() => {
      expect(screen.getByText(/\+\*\* \*\*\* \*\*\* 99/i)).toBeInTheDocument();
    });
  });

  it('affiche le code de secours si resend-sms renvoie debug_sms_code', async () => {
    apiClient.get.mockResolvedValueOnce({
      data: {
        activation_status: {
          email_verified: true,
          phone_verified: false,
          is_complete: false,
          is_finalized: false,
        },
      },
    });
    apiClient.post.mockResolvedValueOnce({
      data: {
        message: 'Service SMS indisponible en local.',
        sms_sent: false,
        debug_sms_code: '654321',
      },
    });

    renderActivation('/activate-account?activation_session_id=sess-sms-debug');
    await waitFor(() => {
      expect(apiClient.get).toHaveBeenCalledTimes(1);
    });
    await waitFor(() => {
      expect(screen.getByRole('button', { name: /renvoyer le code/i })).toBeEnabled();
    });

    fireEvent.click(screen.getByRole('button', { name: /renvoyer le code/i }));

    await waitFor(() => {
      expect(apiClient.post).toHaveBeenCalledWith('/auth/activation/resend-sms', {
        activation_session_id: 'sess-sms-debug',
      });
    });
    await waitFor(() => {
      expect(screen.getByText(/code sms de secours/i)).toBeInTheDocument();
      expect(screen.getByText('654321')).toBeInTheDocument();
    });
  });
});
