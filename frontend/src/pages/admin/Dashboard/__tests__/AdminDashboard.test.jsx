import React from 'react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { render, screen } from '@testing-library/react';
import AdminDashboard from '../AdminDashboard';

jest.mock('../../../../services/adminService', () => ({
  fetchAdminDashboardSummary: jest.fn(),
}));

jest.mock('../../../../hooks/useAuthToken', () => ({
  __esModule: true,
  default: () => ({ first_name: 'Drin', username: 'drin' }),
}));

const { fetchAdminDashboardSummary } = require('../../../../services/adminService');

const renderDash = () =>
  render(
    <MemoryRouter initialEntries={['/dashboard/admin/pub-1']}>
      <Routes>
        <Route path="/dashboard/admin/:public_id" element={<AdminDashboard />} />
      </Routes>
    </MemoryRouter>
  );

const baseSummary = {
  generated_at: '2026-08-03T12:32:00.000Z',
  priorities: {
    bookings_pending_action: 0,
    demo_requests_open: 0,
    tenants_suspended: 0,
    platform_alerts_open: 0,
    billing_to_review: 0,
    critical_attention_count: 0,
  },
  kpi_business: {
    bookings_created_7d: 4,
    bookings_completed_7d: 2,
    bookings_canceled_7d: 9,
    bookings_canceled_from_created_7d: 0,
    cancellation_rate_7d: 0,
    platform_invoiced_current_month_chf: 40,
  },
  platform_snippet: {
    overall_status: 'ok',
    open_alerts: 0,
    runbooks_today: 0,
    tenants_in_drift: 0,
    critical_attention_count: 0,
  },
  booking_trends: [],
  recent_activity: [],
};

describe('AdminDashboard — page de décision', () => {
  beforeEach(() => {
    fetchAdminDashboardSummary.mockReset();
  });

  it('affiche l’état sans attention et sans jargon Ops', async () => {
    fetchAdminDashboardSummary.mockResolvedValue(baseSummary);
    renderDash();

    expect(await screen.findByText(/Bonjour Drin/i)).toBeInTheDocument();
    expect(await screen.findByText(/Aucune alerte/i)).toBeInTheDocument();
    expect(screen.getByTestId('admin-dash-attention')).toBeInTheDocument();
    expect(screen.getByText(/Plateforme opérationnelle/i)).toBeInTheDocument();
    expect(screen.queryByTestId('admin-dash-demo-line')).not.toBeInTheDocument();

    const root = screen.getByTestId('admin-dashboard');
    const text = root.textContent || '';
    expect(text).not.toMatch(/Tenants/);
    expect(text).not.toMatch(/\bCR\b/);
    expect(text).not.toMatch(/Runbooks/);
    expect(text).not.toMatch(/Réconciliation/);
  });

  it('affiche le bandeau dégradé et la ligne démo', async () => {
    fetchAdminDashboardSummary.mockResolvedValue({
      ...baseSummary,
      priorities: {
        ...baseSummary.priorities,
        demo_requests_open: 2,
        critical_attention_count: 2,
        platform_alerts_open: 1,
        billing_to_review: 1,
        bookings_pending_action: 3,
      },
      platform_snippet: {
        overall_status: 'degraded',
        open_alerts: 1,
        runbooks_today: 0,
        tenants_in_drift: 1,
        critical_attention_count: 2,
      },
    });
    renderDash();

    expect(await screen.findByText(/Attention requise/i)).toBeInTheDocument();
    expect(screen.queryByText(/Plateforme opérationnelle/i)).not.toBeInTheDocument();
    expect(screen.getByTestId('admin-dash-demo-line')).toHaveTextContent(
      /2 nouvelles demandes de démonstration/i
    );
    expect(screen.getByText(/3 à traiter/i)).toBeInTheDocument();
    expect(screen.getByText(/1 à contrôler/i)).toBeInTheDocument();
  });

  it('formate Mis à jour à en fr-CH depuis generated_at', async () => {
    fetchAdminDashboardSummary.mockResolvedValue(baseSummary);
    renderDash();
    const el = await screen.findByTestId('admin-dash-updated');
    expect(el.textContent).toMatch(/Mis à jour à \d{2}:\d{2}/);
  });
});
