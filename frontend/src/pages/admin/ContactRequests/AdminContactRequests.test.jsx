import React from 'react';
import { MemoryRouter } from 'react-router-dom';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import AdminContactRequests from './AdminContactRequests';
import {
  fetchAdminContactRequests,
  retryAdminContactNotification,
} from '../../../services/adminContactService';

jest.mock('../../../services/adminContactService', () => ({
  fetchAdminContactRequests: jest.fn(),
  retryAdminContactNotification: jest.fn(),
}));

const renderPage = () =>
  render(
    <MemoryRouter initialEntries={['/dashboard/admin/pub-1/partners/contact-requests?status=failed']}>
      <AdminContactRequests />
    </MemoryRouter>
  );

describe('AdminContactRequests', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('affiche une notification interne en échec et permet de la relancer', async () => {
    fetchAdminContactRequests.mockResolvedValue({
      items: [
        {
          id: 7,
          trace_id: 'ct_XFWBW7A4DQAO',
          category: 'institution',
          name: 'Drin JASIQI',
          email: 'drin@example.com',
          organization: 'Clinique Test',
          email_delivery_status: 'failed',
          autoreply_delivery_status: 'sent',
          notification_retry_count: 1,
          notification_last_error: 'hard_bounce',
          created_at: '2026-09-08T11:20:00.000Z',
        },
      ],
      failedCount: 1,
    });
    retryAdminContactNotification.mockResolvedValue({ ok: true });

    renderPage();

    expect(await screen.findByText('ct_XFWBW7A4DQAO')).toBeInTheDocument();
    expect(screen.getByText('Institution / Intégration')).toBeInTheDocument();
    expect(screen.getByText('Échec')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /relancer/i }));
    await waitFor(() => {
      expect(retryAdminContactNotification).toHaveBeenCalledWith(7);
    });
  });
});
