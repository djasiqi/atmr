import React from 'react';
import { MemoryRouter } from 'react-router-dom';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import DemoHome from './DemoHome';
import useAuthToken from '../../hooks/useAuthToken';
import { trackDemoEvent } from '../../services/demoAnalyticsService';
import { setDemoPassword } from '../../services/demoAccessService';
import { writeAuthSession } from '../../utils/webAuthSession';

const mockNavigate = jest.fn();

jest.mock('../../hooks/useAuthToken');
jest.mock('../../services/demoAnalyticsService', () => ({
  trackDemoEvent: jest.fn(),
}));
jest.mock('../../services/demoAccessService', () => ({
  setDemoPassword: jest.fn(),
}));
jest.mock('../../utils/webAuthSession', () => {
  const actual = jest.requireActual('../../utils/webAuthSession');
  return {
    ...actual,
    writeAuthSession: jest.fn((...args) => actual.writeAuthSession(...args)),
  };
});
jest.mock('react-router-dom', () => {
  const actual = jest.requireActual('react-router-dom');
  return {
    ...actual,
    useNavigate: () => mockNavigate,
  };
});

const renderDemoHome = () =>
  render(
    <MemoryRouter>
      <DemoHome />
    </MemoryRouter>
  );

describe('DemoHome', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    useAuthToken.mockReturnValue({
      role: 'COMPANY',
      public_id: 'cmp_demo_1',
    });
  });

  it('envoie demo_session_start avec rôle normalisé', () => {
    renderDemoHome();

    expect(trackDemoEvent).toHaveBeenCalledWith('demo_session_start', {
      role: 'company',
    });
  });

  it('navigue vers le parcours transporteur guidé', async () => {
    const user = userEvent.setup();
    renderDemoHome();

    const startButtons = screen.getAllByRole('button', {
      name: /Commencer ce parcours/i,
    });
    await user.click(startButtons[0]);

    expect(mockNavigate).toHaveBeenCalledWith(
      '/demo/dashboard/company/cmp_demo_1?demo_mission=transporteur'
    );
  });

  it('navigue vers le parcours institution guidé', async () => {
    const user = userEvent.setup();
    renderDemoHome();

    const startButtons = screen.getAllByRole('button', {
      name: /Commencer ce parcours/i,
    });
    await user.click(startButtons[1]);

    expect(mockNavigate).toHaveBeenCalledWith(
      '/demo/dashboard/institution/cmp_demo_1?demo_mission=institution'
    );
  });

  it('explore librement côté institution', async () => {
    const user = userEvent.setup();
    useAuthToken.mockReturnValue({
      role: 'INSTITUTION',
      public_id: 'inst_demo_1',
    });
    renderDemoHome();

    await user.click(screen.getByRole('button', { name: /Explorer/i }));

    expect(mockNavigate).toHaveBeenCalledWith('/demo/dashboard/institution/inst_demo_1');
  });

  it('après setDemoPassword, ne persiste aucun JWT de la réponse', async () => {
    const user = userEvent.setup();
    useAuthToken.mockReturnValue({
      role: 'company',
      public_id: 'cmp_demo_1',
      force_password_change: true,
    });
    localStorage.setItem(
      'demo_user',
      JSON.stringify({ role: 'company', public_id: 'cmp_demo_1', force_password_change: true })
    );
    localStorage.setItem('lirie_auth_env', 'demo');
    setDemoPassword.mockResolvedValue({
      token: 'jwt-from-set-demo-password',
      refresh_token: 'refresh-from-set-demo-password',
      user: { public_id: 'cmp_demo_1', role: 'company', token: 'nested-jwt' },
    });

    renderDemoHome();
    await user.type(screen.getByLabelText(/Nouveau mot de passe/i), 'password1');
    await user.type(screen.getByLabelText(/Confirmer le mot de passe/i), 'password1');
    await user.click(screen.getByRole('button', { name: /Demarrer la demo/i }));

    expect(setDemoPassword).toHaveBeenCalledWith('password1');
    expect(writeAuthSession).toHaveBeenCalledWith(
      expect.not.objectContaining({
        accessToken: expect.anything(),
        refreshToken: expect.anything(),
      })
    );
    expect(localStorage.getItem('demo_access_token')).toBeNull();
    expect(localStorage.getItem('authToken')).toBeNull();
    expect(localStorage.getItem('demo_user')).not.toMatch(/jwt-from-set-demo-password/);
  });
});
