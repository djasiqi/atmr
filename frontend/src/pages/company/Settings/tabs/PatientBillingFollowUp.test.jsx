import { render, screen, waitFor } from '@testing-library/react';
import { fireEvent } from '@testing-library/react';
import apiClient from '../../../../utils/apiClient';
import PatientBillingFollowUp from './PatientBillingFollowUp';

jest.mock('../../../../utils/apiClient', () => ({
  __esModule: true,
  default: { get: jest.fn(), post: jest.fn() },
}));

const dossier = {
  banner: 'PRÉPARATION — FACTURE NON ÉMISE',
  invoice_issuance: false,
  real_signature_collection: false,
  cases: [
    {
      id: 7,
      booking_id: 21,
      company_name: 'Entreprise fictive',
      debtor_label: 'Patient fictif',
      amount_chf: '45.00',
      contractual_basis: 'representative',
      representative_name: 'Proche fictif',
      representative_review: 'pending_review',
      draft_invoice_id: 4,
      history: [{ action: 'representative', to_state: 'administrative' }],
    },
  ],
};

describe('suivi des dossiers patient', () => {
  it('reste absent quand le serveur ferme la rubrique', async () => {
    apiClient.get.mockRejectedValue({ response: { status: 404 } });
    const { container } = render(<PatientBillingFollowUp />);
    await waitFor(() => expect(apiClient.get).toHaveBeenCalled());
    expect(container).toBeEmptyDOMElement();
  });

  it('affiche le dossier et n’envoie pas une acceptation du patient', async () => {
    apiClient.get.mockResolvedValue({ data: dossier });
    apiClient.post.mockResolvedValue({ data: {} });
    render(<PatientBillingFollowUp />);
    expect(await screen.findByText('45.00 CHF')).toBeTruthy();
    expect(screen.getByText('Facture préparée')).toBeTruthy();
    expect(screen.getByText('Pouvoirs : À vérifier')).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: 'Noter le périmètre documenté' }));
    await waitFor(() => expect(apiClient.post).toHaveBeenCalled());
    const body = apiClient.post.mock.calls[0][1];
    expect(body.result).toBe('documented_scope');
    expect(body.powers_verified).toBe(false);
    expect(body.creates_acceptance).toBeUndefined();
  });
});
