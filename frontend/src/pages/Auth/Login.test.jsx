import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { Provider } from 'react-redux';
import configureStore from 'redux-mock-store';
import Login from './Login';
import authService from '../../services/authService';

jest.mock('../../services/authService');

const mockStore = configureStore([]);

describe('Login Page', () => {
  let store;
  
  beforeEach(() => {
    store = mockStore({
      auth: { user: null, token: null, loading: false }
    });
    localStorage.clear();
    jest.clearAllMocks();
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
    authService.login.mockResolvedValue({
      token: 'fake-token',
      refresh_token: 'fake-refresh',
      user: { id: 1, email: 'test@test.com', role: 'CLIENT' }
    });
    
    renderLogin();
    
    fireEvent.change(screen.getByLabelText(/email/i), {
      target: { value: 'test@test.com' }
    });
    fireEvent.change(screen.getByLabelText(/mot de passe/i), {
      target: { value: 'password123' }
    });
    
    fireEvent.click(screen.getByRole('button', { name: /connexion/i }));
    
    await waitFor(() => {
      expect(authService.login).toHaveBeenCalledWith({
        email: 'test@test.com',
        password: 'password123'
      });
    });
  });
  
  it('shows error message on invalid credentials', async () => {
    authService.login.mockRejectedValue({
      response: { data: { error: 'Invalid credentials' } }
    });
    
    renderLogin();
    
    fireEvent.change(screen.getByLabelText(/email/i), {
      target: { value: 'wrong@test.com' }
    });
    fireEvent.change(screen.getByLabelText(/mot de passe/i), {
      target: { value: 'wrongpass' }
    });
    
    fireEvent.click(screen.getByRole('button', { name: /connexion/i }));
    
    await waitFor(() => {
      expect(screen.getByText(/invalid credentials/i)).toBeInTheDocument();
    });
  });
});

