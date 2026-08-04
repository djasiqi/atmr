import React from 'react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import AdminAccountManageDrawer from '../AdminAccountManageDrawer';

jest.mock('../../../../services/adminService', () => ({
  fetchAccountManageContext: jest.fn(),
  previewUserRoleTransition: jest.fn(),
  resetUserPassword: jest.fn(),
  revokeUserSessions: jest.fn(),
  setCompanyBillingAccess: jest.fn(),
  pauseCompanyDunning: jest.fn(),
  resumeCompanyDunning: jest.fn(),
  setCompanyApproval: jest.fn(),
  setCompanyDispatchStatus: jest.fn(),
  fetchCompanyDispatchDisablePreview: jest.fn(),
  updateDriverStatus: jest.fn(),
  updateUserRole: jest.fn(),
}));

jest.mock('../../../../hooks/useAdminCapabilities', () => ({
  useAdminCapabilities: () => ({
    canUsersManage: true,
    canUsersSecurity: true,
    canBillingLock: true,
    canOrganizationsRead: true,
    canAccountsRead: true,
  }),
}));

const {
  fetchAccountManageContext,
  previewUserRoleTransition,
} = require('../../../../services/adminService');

const baseCtx = {
  account: {
    id: 42,
    username: 'alice',
    email: 'alice@example.ch',
    role: 'CLIENT',
    account_status: 'active',
    created_at: '2026-01-01T00:00:00Z',
    force_password_change: false,
  },
  legacy_context: {
    company_id: null,
    driver_id: null,
    institution_id: null,
    institution_role: null,
  },
  driver_profile: null,
  company_profile: null,
  commercial_restriction: null,
  memberships: [],
  commercial_access: null,
  security: { active_sessions: 1, password_temporary: false },
  diagnostic: { checks: [{ code: 'ok', label: 'OK', status: 'passed' }] },
  allowed_actions: {
    reset_password: true,
    revoke_sessions: true,
    change_role: true,
    change_driver_status: false,
    manage_billing_access: false,
    manage_commercial_restriction: false,
    pause_dunning: false,
    manage_operational_flags: false,
    open_billing_configuration: false,
    open_platform_operations: false,
  },
  role_transition_options: {
    transport_tenants: [{ id: 7, name: 'Transport SA' }],
    institutions: [{ id: 3, name: 'Clinique X' }],
    institution_roles: [
      'institution_admin',
      'institution_requester',
      'institution_reader',
      'institution_billing',
      'institution_curator',
      'institution_reception',
    ],
  },
};

const renderDrawer = (ui) =>
  render(
    <MemoryRouter initialEntries={['/dashboard/admin/pub-1/partners/users']}>
      <Routes>
        <Route path="/dashboard/admin/:public_id/*" element={ui} />
      </Routes>
    </MemoryRouter>
  );

describe('AdminAccountManageDrawer', () => {
  beforeEach(() => {
    fetchAccountManageContext.mockReset();
    previewUserRoleTransition.mockReset();
    fetchAccountManageContext.mockResolvedValue(baseCtx);
  });

  it('charge une seule fois le manage-context à l’ouverture', async () => {
    renderDrawer(
      <AdminAccountManageDrawer
        isOpen
        accountId={42}
        onClose={() => {}}
      />
    );

    expect(await screen.findByText('alice')).toBeInTheDocument();
    expect(fetchAccountManageContext).toHaveBeenCalledTimes(1);
    expect(fetchAccountManageContext).toHaveBeenCalledWith(42);
  });

  it('affiche 6 rôles institution dans le sélecteur', async () => {
    renderDrawer(
      <AdminAccountManageDrawer isOpen accountId={42} onClose={() => {}} />
    );
    await screen.findByText('alice');

    const roleSelect = screen.getByLabelText(/Nouveau rôle/i);
    await userEvent.selectOptions(roleSelect, 'institution');

    const irole = screen.getByLabelText(/Rôle institutionnel/i);
    const options = Array.from(irole.querySelectorAll('option')).map((o) => o.value);
    expect(options).toHaveLength(6);
    expect(options).toContain('institution_reception');
  });

  it('affiche le profil chauffeur sans restriction commerciale', async () => {
    fetchAccountManageContext.mockResolvedValue({
      ...baseCtx,
      account: {
        ...baseCtx.account,
        role: 'DRIVER',
        username: 'drv1',
        force_password_change: true,
      },
      security: { active_sessions: 0, password_temporary: true },
      driver_profile: {
        driver_id: 9,
        company_id: 1,
        company_name: 'Emmenez-moi',
        is_active: true,
        is_available: false,
        driver_type: 'REGULAR',
      },
      allowed_actions: {
        ...baseCtx.allowed_actions,
        change_driver_status: true,
        manage_billing_access: false,
      },
    });

    renderDrawer(
      <AdminAccountManageDrawer isOpen accountId={42} onClose={() => {}} />
    );
    await screen.findByText('drv1');

    expect(screen.getByText('Profil chauffeur')).toBeInTheDocument();
    expect(screen.getAllByText('Emmenez-moi').length).toBeGreaterThan(0);
    expect(screen.getByText(/Mot de passe temporaire/i)).toBeInTheDocument();
    expect(
      screen.getByRole('button', { name: /Désactiver le chauffeur/i })
    ).toBeInTheDocument();
    expect(
      screen.getByRole('button', { name: /Révoquer les sessions/i })
    ).toBeInTheDocument();
    expect(screen.queryByText(/Accès commercial/i)).not.toBeInTheDocument();
    expect(
      screen.queryByText(/Restriction commerciale LIRIE/i)
    ).not.toBeInTheDocument();
  });

  it('affiche la fiche COMPANY avec restriction commerciale LIRIE', async () => {
    fetchAccountManageContext.mockResolvedValue({
      ...baseCtx,
      account: {
        ...baseCtx.account,
        role: 'COMPANY',
        username: 'owner1',
        email: 'contact@emmenez-moi.ch',
      },
      company_profile: {
        company_id: 1,
        name: 'Emmenez-moi',
        contact_email: 'contact@emmenez-moi.ch',
        is_approved: true,
        dispatch_enabled: true,
        platform_suspended: false,
        active_drivers_count: 12,
        total_drivers_count: 15,
        inactive_drivers_count: 3,
      },
      commercial_restriction: {
        company_id: 1,
        state: 'active',
        source: 'admin_manual',
        reason_code: null,
        since: null,
        dunning_paused_until: null,
        dunning_pause_reason: null,
      },
      detected_services: {
        decision_mode: 'shadow',
        notice:
          'Ces services sont détectés depuis la configuration legacy. Ils n’autorisent ni ne bloquent encore les fonctions de l’entreprise.',
        services: [
          {
            service_key: 'company.own_portfolio',
            label: 'Portefeuille propre',
            detected: true,
            enforcement_mode: 'shadow',
          },
        ],
      },
      allowed_actions: {
        ...baseCtx.allowed_actions,
        manage_commercial_restriction: true,
        pause_dunning: true,
        manage_operational_flags: true,
        open_billing_configuration: true,
        open_platform_operations: true,
      },
    });

    renderDrawer(
      <AdminAccountManageDrawer isOpen accountId={42} onClose={() => {}} />
    );
    await screen.findByText('Emmenez-moi');

    expect(screen.getByText(/Entreprise de transport/i)).toBeInTheDocument();
    expect(screen.getByText(/Restriction commerciale LIRIE/i)).toBeInTheDocument();
    expect(screen.getAllByText(/Aucune restriction/i).length).toBeGreaterThan(0);
    expect(screen.getByText(/Gouvernance opérationnelle/i)).toBeInTheDocument();
    expect(screen.getByText('12')).toBeInTheDocument();
    expect(screen.getByText('3')).toBeInTheDocument();
    expect(screen.getByText('15')).toBeInTheDocument();
    expect(
      screen.getByRole('heading', { name: 'Services détectés' })
    ).toBeInTheDocument();
    expect(screen.getByText(/Mode : Shadow/i)).toBeInTheDocument();
    expect(screen.queryByText(/Accès commercial/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/sièges consommés/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/licence restante/i)).not.toBeInTheDocument();
    expect(screen.getByText(/aucun quota ni licence/i)).toBeInTheDocument();
    expect(
      screen.getByRole('link', { name: /configuration Billing/i })
    ).toHaveAttribute('href', '/dashboard/admin/pub-1/finance/config');
    expect(
      screen.getByRole('link', { name: /comptes chauffeurs/i })
    ).toHaveAttribute(
      'href',
      '/dashboard/admin/pub-1/partners/users?role=driver&company_id=1'
    );
  });

  it('affiche un message lisible si ownership bloque la preview', async () => {
    previewUserRoleTransition.mockResolvedValue({
      allowed: false,
      blockers: [
        {
          code: 'company_ownership_transition_required',
          message:
            "Transition hors ownership Company requiert l'assistant CP-PR3.",
        },
      ],
      changes: [],
      warnings: [],
    });

    fetchAccountManageContext.mockResolvedValue({
      ...baseCtx,
      account: { ...baseCtx.account, role: 'COMPANY', username: 'owner1' },
      company_profile: {
        company_id: 1,
        name: 'Co',
        contact_email: 'a@b.ch',
        is_approved: true,
        dispatch_enabled: true,
        platform_suspended: false,
        active_drivers_count: 0,
        total_drivers_count: 0,
        inactive_drivers_count: 0,
      },
      commercial_restriction: { company_id: 1, state: 'active' },
      allowed_actions: {
        ...baseCtx.allowed_actions,
        manage_commercial_restriction: true,
        open_billing_configuration: true,
        open_platform_operations: true,
      },
    });

    renderDrawer(
      <AdminAccountManageDrawer isOpen accountId={42} onClose={() => {}} />
    );
    await screen.findByText('Co');

    expect(
      screen.getByText(/assistant ownership CP-PR3/i)
    ).toBeInTheDocument();

    const roleSelect = screen.getByLabelText(/Nouveau rôle/i);
    await userEvent.selectOptions(roleSelect, 'client');

    await userEvent.click(
      screen.getByRole('button', { name: /Prévisualiser et appliquer/i })
    );

    await waitFor(() => {
      expect(screen.getByRole('alert')).toHaveTextContent(/CP-PR3/i);
    });
  });
});
