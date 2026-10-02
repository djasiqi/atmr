import React from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom';
import { invoiceService } from '../../../../../services/invoiceService';
import BillPeriodModal from './BillPeriodModal';

jest.mock('../../../../../hooks/useCompanySocket', () => () => null);

function renderOpen() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>
      <BillPeriodModal open onClose={() => {}} companyId={7} />
    </QueryClientProvider>,
  );
}

describe('BillPeriodModal — sélecteur patient', () => {
  beforeEach(() => {
    jest.spyOn(invoiceService, 'fetchInvoiceCandidates').mockImplementation(
      () => new Promise(() => {}),
    );
    jest.spyOn(invoiceService, 'fetchInstitutions').mockResolvedValue({ institutions: [] });
    jest.spyOn(invoiceService, 'fetchBillablePartners').mockResolvedValue([]);
    jest.spyOn(invoiceService, 'fetchBillingOpportunities').mockResolvedValue({});
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('ouvre tout de suite et ne charge ni institutions ni partenaires', async () => {
    renderOpen();

    expect(screen.getByRole('heading', { name: 'Nouvelle facture' })).toBeInTheDocument();
    const period = screen.getByRole('combobox', { name: 'Période' });
    expect(period).toBeEnabled();
    expect(screen.getByText('Chargement des patients à facturer…')).toBeInTheDocument();

    await waitFor(() => {
      expect(invoiceService.fetchInvoiceCandidates).toHaveBeenCalled();
    });
    expect(invoiceService.fetchInstitutions).not.toHaveBeenCalled();
    expect(invoiceService.fetchBillablePartners).not.toHaveBeenCalled();
    expect(invoiceService.fetchBillingOpportunities).not.toHaveBeenCalled();
  });

  it('une erreur de candidats laisse la période utilisable', async () => {
    invoiceService.fetchInvoiceCandidates.mockRejectedValue(new Error('timeout'));
    renderOpen();

    expect(await screen.findByText('Impossible de charger les patients à facturer. Réessayez.')).toBeInTheDocument();
    expect(screen.getByRole('combobox', { name: 'Période' })).toBeEnabled();
    expect(screen.queryByText('Chargement des patients à facturer…')).not.toBeInTheDocument();
  });

  it('affiche les patients reçus sans fermer la fenêtre', async () => {
    const now = new Date();
    const period = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}`;
    invoiceService.fetchInvoiceCandidates.mockResolvedValue({
      period,
      patients: [
        {
          id: 'client:1|billing_party:2',
          name: 'Jean Dupont',
          billable_count: 1,
          amount: 40,
          client_id: 1,
          billing_party_id: 2,
        },
      ],
    });

    renderOpen();

    const patient = await screen.findByRole('combobox', { name: 'Patient' });
    await waitFor(() => expect(patient).toBeEnabled());
    await userEvent.click(patient);
    expect(await screen.findByRole('option', { name: /Jean Dupont/ })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'Nouvelle facture' })).toBeInTheDocument();
  });

  it('ne charge les institutions que lorsque leur onglet est choisi', async () => {
    renderOpen();
    await userEvent.click(screen.getByRole('radio', { name: /institution/i }));
    await waitFor(() => {
      expect(invoiceService.fetchInstitutions).toHaveBeenCalledWith(7);
    });
    expect(invoiceService.fetchBillablePartners).not.toHaveBeenCalled();
  });
});
