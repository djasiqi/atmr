import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { Provider } from 'react-redux';
import configureStore from 'redux-mock-store';
import Login from './Login';
import apiClient from '../../utils/apiClient';
import { jwtDecode } from 'jwt-decode';

jest.mock('../../utils/apiClient');
jest.mock('jwt-decode');

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
    expect(screen.getByLabelText(/mot de passe/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /connexion/i })).toBeInTheDocument();
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
    fireEvent.change(screen.getByLabelText(/mot de passe/i), {
      target: { value: 'password123' },
    });

    fireEvent.click(screen.getByRole('button', { name: /connexion/i }));

    await waitFor(() => {
      expect(apiClient.post).toHaveBeenCalledWith('/auth/login', {
        email: 'test@test.com',
        password: 'password123',
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
    fireEvent.change(screen.getByLabelText(/mot de passe/i), {
      target: { value: 'wrongpass' },
    });

    fireEvent.click(screen.getByRole('button', { name: /connexion/i }));

    await waitFor(() => {
      expect(screen.getByText(/invalid credentials/i)).toBeInTheDocument();
    }, { timeout: 3000 });
  });
});
