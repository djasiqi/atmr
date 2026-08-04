import React from 'react';
import { render, screen } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import ProtectedRoute, { resolveOnboardingRedirect } from '../ProtectedRoute';

const bootstrapState = {
  status: 'authenticated',
  isAuthenticated: true,
  user: null,
};

jest.mock('../../contexts/SessionBootstrapContext', () => ({
  useSessionBootstrap: () => bootstrapState,
}));

describe('ProtectedRoute onboarding', () => {
  beforeEach(() => {
    localStorage.clear();
    bootstrapState.status = 'authenticated';
    bootstrapState.isAuthenticated = true;
    bootstrapState.user = null;
    jest.spyOn(console, 'warn').mockImplementation(() => {});
  });

  afterEach(() => {
    console.warn.mockRestore();
    jest.clearAllMocks();
  });

  const renderProtected = (user, initialPath = '/dashboard') => {
    localStorage.setItem('company_user', JSON.stringify({ ...user, role: user.role || 'company' }));
    localStorage.setItem('app_user', JSON.stringify({ ...user, role: user.role || 'company' }));

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
          <Route path="/unauthorized" element={<div>Unauthorized page</div>} />
          <Route
            path="/force-reset-password"
            element={<div>Reset password page</div>}
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

  it('utilise company_user si le bootstrap est encore sur un ancien rôle admin', () => {
    bootstrapState.user = { public_id: 'admin-1', role: 'admin' };
    renderProtected({
      public_id: 'company-1',
      role: 'company',
      force_password_change: false,
    });

    expect(screen.getByText('Dashboard content')).toBeInTheDocument();
    expect(screen.queryByText('Unauthorized page')).not.toBeInTheDocument();
  });

  it('bloque un admin pollué (même public_id en company_user)', () => {
    bootstrapState.user = { public_id: 'admin-1', role: 'admin' };
    localStorage.setItem(
      'company_user',
      JSON.stringify({ public_id: 'admin-1', role: 'company' })
    );
    localStorage.setItem(
      'app_user',
      JSON.stringify({ public_id: 'admin-1', role: 'admin' })
    );

    render(
      <MemoryRouter initialEntries={['/dashboard']}>
        <Routes>
          <Route
            path="/dashboard"
            element={
              <ProtectedRoute allowedRoles={['company']}>
                <div>Dashboard content</div>
              </ProtectedRoute>
            }
          />
          <Route path="/unauthorized" element={<div>Unauthorized page</div>} />
        </Routes>
      </MemoryRouter>
    );

    expect(screen.getByText('Unauthorized page')).toBeInTheDocument();
  });

  it('bloque les routes company si admin_user est présent', () => {
    bootstrapState.user = { public_id: 'company-1', role: 'company' };
    localStorage.setItem(
      'admin_user',
      JSON.stringify({ public_id: 'admin-1', role: 'admin' })
    );
    localStorage.setItem(
      'company_user',
      JSON.stringify({ public_id: 'company-1', role: 'company' })
    );
    localStorage.setItem(
      'app_user',
      JSON.stringify({ public_id: 'company-1', role: 'company' })
    );

    render(
      <MemoryRouter initialEntries={['/dashboard']}>
        <Routes>
          <Route
            path="/dashboard"
            element={
              <ProtectedRoute allowedRoles={['company']}>
                <div>Dashboard content</div>
              </ProtectedRoute>
            }
          />
          <Route path="/unauthorized" element={<div>Unauthorized page</div>} />
        </Routes>
      </MemoryRouter>
    );

    expect(screen.getByText('Unauthorized page')).toBeInTheDocument();
  });

  it('redirige unauthorized si ni bootstrap ni storage ne matchent company', () => {
    bootstrapState.user = { public_id: 'admin-1', role: 'admin' };
    localStorage.setItem('company_user', JSON.stringify({ public_id: 'admin-1', role: 'admin' }));
    localStorage.setItem('app_user', JSON.stringify({ public_id: 'admin-1', role: 'admin' }));

    render(
      <MemoryRouter initialEntries={['/dashboard']}>
        <Routes>
          <Route
            path="/dashboard"
            element={
              <ProtectedRoute allowedRoles={['company']}>
                <div>Dashboard content</div>
              </ProtectedRoute>
            }
          />
          <Route path="/unauthorized" element={<div>Unauthorized page</div>} />
        </Routes>
      </MemoryRouter>
    );

    expect(screen.getByText('Unauthorized page')).toBeInTheDocument();
  });
});
