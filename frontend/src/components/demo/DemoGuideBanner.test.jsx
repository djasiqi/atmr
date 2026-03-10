import React from 'react';
import { MemoryRouter } from 'react-router-dom';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import DemoGuideBanner from './DemoGuideBanner';
import { trackDemoEvent } from '../../services/demoAnalyticsService';

const mockNavigate = jest.fn();

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

const renderBanner = (role = 'transporteur') =>
  render(
    <MemoryRouter>
      <DemoGuideBanner role={role} />
    </MemoryRouter>
  );

describe('DemoGuideBanner', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('affiche les étapes transporteur et trace la progression', async () => {
    const user = userEvent.setup();
    renderBanner('transporteur');

    expect(screen.getByText(/Mission transporteur/i)).toBeInTheDocument();
    expect(screen.getAllByRole('button', { name: /Étape faite/i })).toHaveLength(5);

    await user.click(screen.getAllByRole('button', { name: /Étape faite/i })[0]);

    expect(trackDemoEvent).toHaveBeenCalledWith('demo_step_reached', {
      role: 'transporteur',
      stepIndex: 1,
    });
  });

  it('affiche les étapes institution et permet de terminer la mission', async () => {
    const user = userEvent.setup();
    renderBanner('institution');

    expect(screen.getByText(/Mission institution/i)).toBeInTheDocument();
    expect(screen.getAllByRole('button', { name: /Étape faite/i })).toHaveLength(3);

    await user.click(screen.getByRole('button', { name: /Terminer et contacter LIRIE/i }));

    expect(trackDemoEvent).toHaveBeenCalledWith('demo_completed', {
      role: 'institution',
    });
    expect(mockNavigate).toHaveBeenCalledWith('/contact/demo');
  });
});
