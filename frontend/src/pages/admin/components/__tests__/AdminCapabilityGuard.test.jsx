import React from 'react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor } from '@testing-library/react';
import AdminCapabilityGuard from '../AdminCapabilityGuard';
import { ADMIN_CAP } from '../../capabilities/adminCapabilities';

jest.mock('../../../../hooks/useAdminCapabilities', () => ({
  useAdminCapabilities: jest.fn(),
}));

const { useAdminCapabilities } = require('../../../../hooks/useAdminCapabilities');

function renderGuard(capability) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={['/dashboard/admin/pub-1/advanced/labs/optuna']}>
        <Routes>
          <Route
            path="/dashboard/admin/:public_id/advanced/labs/optuna"
            element={
              <AdminCapabilityGuard capability={capability}>
                <div>Contenu Labs</div>
              </AdminCapabilityGuard>
            }
          />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>
  );
}

describe('AdminCapabilityGuard', () => {
  it('affiche le contenu si la capacité est accordée', () => {
    useAdminCapabilities.mockReturnValue({
      can: () => true,
      isLoading: false,
      enforced: true,
    });
    renderGuard(ADMIN_CAP.LABS_READ);
    expect(screen.getByText('Contenu Labs')).toBeInTheDocument();
  });

  it('refuse l’accès direct si la capacité manque', async () => {
    useAdminCapabilities.mockReturnValue({
      can: () => false,
      isLoading: false,
      enforced: true,
    });
    renderGuard(ADMIN_CAP.LABS_READ);
    await waitFor(() => {
      expect(screen.getByRole('alert')).toHaveTextContent('Accès refusé');
    });
    expect(screen.queryByText('Contenu Labs')).not.toBeInTheDocument();
  });
});
