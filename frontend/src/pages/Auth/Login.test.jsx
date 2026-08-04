import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { BrowserRouter, MemoryRouter } from 'react-router-dom';
import { Provider } from 'react-redux';
import configureStore from 'redux-mock-store';
import Login from './Login';
import apiClient from '../../utils/apiClient';
import { jwtDecode } from 'jwt-decode';
import { setPendingActivationSession } from '../../utils/activationSessionStore';

const mockHydrateFromLogin = jest.fn();
const mockNavigate = jest.fn();

jest.mock('../../utils/apiClient', () => ({
  __esModule: true,
  default: {
    post: jest.fn(),
    // Stub minimal nécessaire pour les modules qui installent un intercepteur
    // au chargement (ex: companyDashboardApiTiming) — sans cela le test suite
    // refuse de démarrer.
    interceptors: {
      request: { use: jest.fn() },
      response: { use: jest.fn() },
    },
  },
  cleanLocalSession: jest.fn(),
  setCurrentAuthEnv: jest.fn((env) => env || 'app'),
}));
jest.mock('jwt-decode');
jest.mock('../../utils/activationSessionStore', () => ({
  getPendingActivationByEmail: jest.fn(() => null),
  removePendingActivationByEmail: jest.fn(),
  setPendingActivationSession: jest.fn(),
}));
jest.mock('../../contexts/SessionBootstrapContext', () => ({
  useSessionBootstrap: () => ({
    status: 'anonymous',
    isAuthenticated: false,
    user: null,
    refreshBootstrap: jest.fn(),
    hydrateFromLogin: (...args) => mockHydrateFromLogin(...args),
  }),
}));
jest.mock('../../services/companySocket', () => ({
  disconnectCompanySocket: jest.fn(),
}));

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
          remember_me: false,
        },
        expect.objectContaining({ skipCsrf: true })
      );
    });
    expect(mockHydrateFromLogin).toHaveBeenCalled();
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

  it('ignore un next dashboard incompatible avec le rôle connecté', async () => {
    const mockToken = 'fake-institution-jwt';
    const mockUser = {
      public_id: 'inst-1',
      role: 'institution',
      first_name: 'I',
      last_name: 'User',
    };

    apiClient.post.mockResolvedValue({
      data: {
        token: mockToken,
        user: mockUser,
        target_env: 'app',
      },
    });

    jwtDecode.mockReturnValue({
      sub: 'inst-1',
      role: 'institution',
    });

    const staleCompanyNext = '/dashboard/company/company-1/settings';
    render(
      <Provider store={store}>
        <MemoryRouter initialEntries={[`/login?next=${encodeURIComponent(staleCompanyNext)}`]}>
          <Login />
        </MemoryRouter>
      </Provider>
    );

    fireEvent.change(screen.getByLabelText(/email/i), {
      target: { value: 'institution@example.com' },
    });
    fireEvent.change(screen.getByPlaceholderText(/entrez votre mot de passe/i), {
      target: { value: 'password123' },
    });
    fireEvent.click(screen.getByRole('button', { name: /se connecter/i }));

    await waitFor(() => {
      expect(mockNavigate).toHaveBeenCalledWith('/dashboard/institution/inst-1', {
        replace: true,
      });
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

  it('affiche le message support si le serveur est indisponible (404 proxy)', async () => {
    apiClient.post.mockRejectedValue({
      message: 'Request failed with status code 404',
      response: {
        status: 404,
        data: '404 page not found',
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
      expect(screen.getByRole('alert')).toHaveTextContent(/momentanément indisponible/i);
      expect(screen.getByRole('alert')).toHaveTextContent(/022 512 02 03/);
    });
    expect(screen.queryByText(/404 page not found/i)).not.toBeInTheDocument();
  });

  it('envoie remember_me=true et ne stocke que l\'email quand la case est cochée', async () => {
    apiClient.post.mockResolvedValue({
      data: {
        token: 'jwt-remember',
        user: { public_id: 'user-rm', role: 'company' },
        target_env: 'app',
        redirect_to: '/app/dashboard/company/user-rm',
      },
    });
    jwtDecode.mockReturnValue({ sub: 'user-rm', role: 'company' });

    renderLogin();

    fireEvent.change(screen.getByLabelText(/email/i), {
      target: { value: 'remember@example.com' },
    });
    fireEvent.change(screen.getByPlaceholderText(/entrez votre mot de passe/i), {
      target: { value: 'sup3rs3cret' },
    });
    fireEvent.click(screen.getByLabelText(/se souvenir de moi/i));
    fireEvent.click(screen.getByRole('button', { name: /se connecter/i }));

    await waitFor(() => {
      expect(apiClient.post).toHaveBeenCalledWith(
        '/auth/login',
        {
          email: 'remember@example.com',
          password: 'sup3rs3cret',
          remember_me: true,
        },
        expect.objectContaining({ skipCsrf: true })
      );
    });

    const stored = JSON.parse(localStorage.getItem('lirie_remember_me') || '{}');
    expect(stored).toEqual({ email: 'remember@example.com', version: 2 });
    expect(stored).not.toHaveProperty('password');
  });

  it('purge l\'ancien format { email, password } et ne pré-remplit que l\'email', async () => {
    localStorage.setItem(
      'lirie_remember_me',
      JSON.stringify({ email: 'legacy@example.com', password: 'plaintext-leak' }),
    );

    renderLogin();

    expect(screen.getByLabelText(/email/i)).toHaveValue('legacy@example.com');
    expect(screen.getByPlaceholderText(/entrez votre mot de passe/i)).toHaveValue('');

    const stored = JSON.parse(localStorage.getItem('lirie_remember_me') || '{}');
    expect(stored).toEqual({ email: 'legacy@example.com', version: 2 });
    expect(stored).not.toHaveProperty('password');
  });

  it('supprime REMEMBER_KEY si la case est décochée', async () => {
    localStorage.setItem(
      'lirie_remember_me',
      JSON.stringify({ email: 'old@example.com', version: 2 }),
    );

    apiClient.post.mockResolvedValue({
      data: {
        token: 'jwt-token',
        user: { public_id: 'user-x', role: 'client' },
        target_env: 'app',
      },
    });
    jwtDecode.mockReturnValue({ sub: 'user-x', role: 'client' });

    renderLogin();

    fireEvent.click(screen.getByLabelText(/se souvenir de moi/i));
    fireEvent.change(screen.getByLabelText(/email/i), {
      target: { value: 'new@example.com' },
    });
    fireEvent.change(screen.getByPlaceholderText(/entrez votre mot de passe/i), {
      target: { value: 'password123' },
    });
    fireEvent.click(screen.getByRole('button', { name: /se connecter/i }));

    await waitFor(() => {
      expect(apiClient.post).toHaveBeenCalledWith(
        '/auth/login',
        expect.objectContaining({ remember_me: false }),
        expect.objectContaining({ skipCsrf: true }),
      );
    });
    expect(localStorage.getItem('lirie_remember_me')).toBeNull();
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
