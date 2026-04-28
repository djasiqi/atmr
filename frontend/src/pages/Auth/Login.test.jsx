import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { BrowserRouter, MemoryRouter } from 'react-router-dom';
import { Provider } from 'react-redux';
import configureStore from 'redux-mock-store';
import Login from './Login';
import apiClient, { setCurrentAuthEnv } from '../../utils/apiClient';
import { jwtDecode } from 'jwt-decode';
import { setPendingActivationSession } from '../../utils/activationSessionStore';

jest.mock('../../utils/apiClient', () => ({
  __esModule: true,
  default: { post: jest.fn() },
  cleanLocalSession: jest.fn(),
  setCurrentAuthEnv: jest.fn((env) => env || 'app'),
}));
jest.mock('jwt-decode');
jest.mock('../../utils/activationSessionStore', () => ({
  getPendingActivationByEmail: jest.fn(() => null),
  removePendingActivationByEmail: jest.fn(),
  setPendingActivationSession: jest.fn(),
}));

const mockNavigate = jest.fn();
jest.mock('react-router-dom', () => ({
  ...jest.requireActual('react-router-dom'),
  useNavigate: () => mockNavigate,
}));

const mockStore = configureStore([]);

describe('Login Page', () => {
  let store;

  beforeEach(() => {
    store = mockStore({
      auth: { user: null, token: null, loading: false },
    });
    localStorage.clear();
    jest.clearAllMocks();
    mockNavigate.mockClear();
  });

  const renderLogin = () => {
    return render(
      <Provider store={store}>
        <BrowserRouter>
          <Login />
        </BrowserRouter>
      </Provider>
    );
  };

  it('renders login form', () => {
    renderLogin();

    expect(screen.getByLabelText(/email/i)).toBeInTheDocument();
    expect(screen.getByPlaceholderText(/entrez votre mot de passe/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /se connecter/i })).toBeInTheDocument();
  });

  it('submits login with valid credentials', async () => {
    const mockToken = 'fake-jwt-token';
    const mockUser = {
      public_id: 'user-123',
      role: 'company',
      first_name: 'Test',
      last_name: 'User',
    };

    apiClient.post.mockResolvedValue({
      data: {
        token: mockToken,
        user: mockUser,
        target_env: 'app',
        redirect_to: '/app/dashboard/company/user-123',
      },
    });

    jwtDecode.mockReturnValue({
      sub: 'user-123',
      role: 'company',
    });

    renderLogin();

    fireEvent.change(screen.getByLabelText(/email/i), {
      target: { value: 'test@test.com' },
    });
    fireEvent.change(screen.getByPlaceholderText(/entrez votre mot de passe/i), {
      target: { value: 'password123' },
    });

    fireEvent.click(screen.getByRole('button', { name: /se connecter/i }));

    await waitFor(() => {
      expect(apiClient.post).toHaveBeenCalledWith(
        '/auth/login',
        {
          email: 'test@test.com',
          password: 'password123',
        },
        expect.objectContaining({ skipCsrf: true })
      );
    });
    expect(setCurrentAuthEnv).toHaveBeenCalledWith('app');
    expect(mockNavigate).toHaveBeenCalledWith('/app/dashboard/company/user-123', { replace: true });
  });

  it('redirige vers ?next= après connexion (chemin interne)', async () => {
    const mockToken = 'fake-jwt-token';
    const mockUser = {
      public_id: 'cli-1',
      role: 'client',
      first_name: 'C',
      last_name: 'L',
    };

    apiClient.post.mockResolvedValue({
      data: {
        token: mockToken,
        user: mockUser,
        target_env: 'app',
      },
    });

    jwtDecode.mockReturnValue({
      sub: 'cli-1',
      role: 'client',
    });

    const nextPath = '/client/payment/worldline/return?bookingId=9';
    render(
      <Provider store={store}>
        <MemoryRouter initialEntries={[`/login?next=${encodeURIComponent(nextPath)}`]}>
          <Login />
        </MemoryRouter>
      </Provider>
    );

    fireEvent.change(screen.getByLabelText(/email/i), {
      target: { value: 'c@example.com' },
    });
    fireEvent.change(screen.getByPlaceholderText(/entrez votre mot de passe/i), {
      target: { value: 'password123' },
    });
    fireEvent.click(screen.getByRole('button', { name: /se connecter/i }));

    await waitFor(() => {
      expect(mockNavigate).toHaveBeenCalledWith(nextPath, { replace: true });
    });
  });

  it('shows error message on invalid credentials', async () => {
    apiClient.post.mockRejectedValue({
      response: { data: { error: 'Invalid credentials' } },
    });

    renderLogin();

    fireEvent.change(screen.getByLabelText(/email/i), {
      target: { value: 'wrong@test.com' },
    });
    fireEvent.change(screen.getByPlaceholderText(/entrez votre mot de passe/i), {
      target: { value: 'wrongpass' },
    });

    fireEvent.click(screen.getByRole('button', { name: /se connecter/i }));

    await waitFor(() => {
      expect(screen.getByText(/invalid credentials/i)).toBeInTheDocument();
    }, { timeout: 3000 });
  });

  it('redirige vers activation si compte pending_activation', async () => {
    apiClient.post.mockRejectedValue({
      response: {
        status: 403,
        data: {
          error: 'Compte en attente de validation email/SMS.',
          reason: 'account_pending_activation',
          activation_session_id: 'sess-pending-123',
          masked_email: 'u***@m***.com',
          masked_phone: '+** *** *** 12',
        },
      },
    });

    renderLogin();

    fireEvent.change(screen.getByLabelText(/email/i), {
      target: { value: 'user@example.com' },
    });
    fireEvent.change(screen.getByPlaceholderText(/entrez votre mot de passe/i), {
      target: { value: 'password123' },
    });

    fireEvent.click(screen.getByRole('button', { name: /se connecter/i }));

    await waitFor(() => {
      expect(setPendingActivationSession).toHaveBeenCalledWith({
        email: 'user@example.com',
        activation_session_id: 'sess-pending-123',
        masked_email: 'u***@m***.com',
        masked_phone: '+** *** *** 12',
      });
    });
    expect(mockNavigate).toHaveBeenCalledWith(
      '/activate-account?activation_session_id=sess-pending-123',
      expect.objectContaining({
        replace: true,
        state: expect.objectContaining({
          prefillEmail: 'user@example.com',
          maskedEmail: 'u***@m***.com',
          maskedPhone: '+** *** *** 12',
        }),
      })
    );
  });
});
