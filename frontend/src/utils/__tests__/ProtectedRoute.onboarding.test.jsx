import React from 'react';
import { render, screen } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { jwtDecode } from 'jwt-decode';
import ProtectedRoute, { resolveOnboardingRedirect } from '../ProtectedRoute';

jest.mock('jwt-decode');

describe('ProtectedRoute onboarding', () => {
  beforeEach(() => {
    localStorage.clear();
    jest.spyOn(console, 'warn').mockImplementation(() => {});
    jwtDecode.mockReturnValue({
      role: 'company',
      exp: Math.floor(Date.now() / 1000) + 3600,
    });
  });

  afterEach(() => {
    console.warn.mockRestore();
    jest.clearAllMocks();
  });

  const renderProtected = (user, initialPath = '/dashboard') => {
    localStorage.setItem('company_access_token', 'fake-token');
    localStorage.setItem(
      'company_user',
      JSON.stringify({ ...user, role: user.role || 'company' })
    );

    return render(
      <MemoryRouter initialEntries={[initialPath]}>
        <Routes>
          <Route
            path="/dashboard"
            element={
              <ProtectedRoute allowedRoles={['company']}>
                <div>Dashboard content</div>
              </ProtectedRoute>
            }
          />
          <Route
            path="/force-reset-password/:publicId"
            element={<div>Reset password page</div>}
          />
        </Routes>
      </MemoryRouter>
    );
  };

  it('redirige vers force-reset-password quand force_password_change est true', () => {
    renderProtected({
      public_id: 'user-fpc',
      force_password_change: true,
      must_complete_onboarding: true,
      onboarding_reasons: ['force_password_change'],
    });

    expect(screen.getByText('Reset password page')).toBeInTheDocument();
  });

  it('ne redirige pas quand must_complete_onboarding=true sans force_password_change (invited)', () => {
    renderProtected({
      public_id: 'user-invited',
      force_password_change: false,
      must_complete_onboarding: true,
      onboarding_reasons: ['invited'],
    });

    expect(screen.getByText('Dashboard content')).toBeInTheDocument();
    expect(console.warn).toHaveBeenCalledWith(
      'must_complete_onboarding=true mais aucune destination configuree',
      { reasons: ['invited'] }
    );
  });

  it('resolveOnboardingRedirect retourne null pour invited sans force_password_change', () => {
    const destination = resolveOnboardingRedirect(
      {
        public_id: 'user-invited',
        force_password_change: false,
        must_complete_onboarding: true,
        onboarding_reasons: ['invited'],
      },
      '/dashboard'
    );
    expect(destination).toBeNull();
  });
});
