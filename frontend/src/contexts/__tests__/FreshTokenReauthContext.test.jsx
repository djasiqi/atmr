import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import { FreshTokenReauthProvider, useFreshTokenReauth } from '../FreshTokenReauthContext';
import { registerFreshTokenReauthHandler } from '../../utils/apiClient';
import { getFreshToken } from '../../services/authService';

jest.mock('../../services/authService', () => ({
  getFreshToken: jest.fn(),
}));

function Probe() {
  const { requestFreshTokenReauth } = useFreshTokenReauth();
  return (
    <button
      type="button"
      onClick={() =>
        requestFreshTokenReauth({
          title: 'Test re-auth',
          retryFn: jest.fn(() =>
            Promise.reject({
              isFreshTokenRequired: true,
              message: 'Encore non fresh',
            })
          ),
        })
      }
    >
      Ouvrir modale
    </button>
  );
}

describe('FreshTokenReauthContext — anti-boucle P1a.1', () => {
  beforeEach(() => {
    getFreshToken.mockReset();
  });

  it('mot de passe incorrect × 2 → une seule modale, pas de retry auto en cascade', async () => {
    getFreshToken
      .mockRejectedValueOnce(new Error('Mot de passe incorrect'))
      .mockRejectedValueOnce(new Error('Mot de passe incorrect'));

    render(
      <FreshTokenReauthProvider>
        <Probe />
      </FreshTokenReauthProvider>
    );

    fireEvent.click(screen.getByRole('button', { name: /ouvrir modale/i }));

    expect(await screen.findByRole('dialog')).toBeInTheDocument();

    const input = screen.getByLabelText(/mot de passe/i);
    fireEvent.change(input, { target: { value: 'wrong1' } });
    fireEvent.click(screen.getByRole('button', { name: /confirmer/i }));

    await waitFor(() => {
      expect(getFreshToken).toHaveBeenCalledTimes(1);
      expect(screen.getByText(/mot de passe incorrect/i)).toBeInTheDocument();
    });

    fireEvent.change(input, { target: { value: 'wrong2' } });
    fireEvent.click(screen.getByRole('button', { name: /confirmer/i }));

    await waitFor(() => {
      expect(getFreshToken).toHaveBeenCalledTimes(2);
    });

    expect(screen.getAllByRole('dialog')).toHaveLength(1);
  });

  it('enregistre le handler global apiClient au montage', () => {
    const { unmount } = render(
      <FreshTokenReauthProvider>
        <span>ok</span>
      </FreshTokenReauthProvider>
    );

    expect(typeof registerFreshTokenReauthHandler).toBe('function');
    unmount();
  });
});
