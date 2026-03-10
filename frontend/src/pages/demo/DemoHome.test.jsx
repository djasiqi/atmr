import React from 'react';
import { MemoryRouter } from 'react-router-dom';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import DemoHome from './DemoHome';
import useAuthToken from '../../hooks/useAuthToken';
import { trackDemoEvent } from '../../services/demoAnalyticsService';

const mockNavigate = jest.fn();

jest.mock('../../hooks/useAuthToken');
jest.mock('../../services/demoAnalyticsService', () => ({
  trackDemoEvent: jest.fn(),
}));
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
});
