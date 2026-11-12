import { render, screen, waitFor } from '@testing-library/react';
import { fireEvent } from '@testing-library/react';
import apiClient from '../../../../utils/apiClient';
import CashRegisterFollowUp from './CashRegisterFollowUp';

jest.mock('../../../../utils/apiClient', () => ({
  __esModule: true,
  default: { get: jest.fn(), post: jest.fn() },
}));

describe('registre de caisse préparé', () => {
  it('reste absent quand le serveur ferme la rubrique', async () => {
    apiClient.get.mockRejectedValue({ response: { status: 404 } });
    render(<CashRegisterFollowUp />);
    expect(await screen.findByText(/pas disponible dans cet environnement/)).toBeTruthy();
    expect(screen.queryByText(/CHF/)).toBeNull();
  });

  it('affiche une réception non activée et distingue la confirmation de remise', async () => {
    apiClient.get.mockResolvedValue({
      data: {
        banner: 'PRÉPARATION — ENCAISSEMENT NON ACTIVÉ',
        production_activation: false,
        cases: [
          {
            id: 3,
            booking_id: 12,
            company_name: 'Entreprise fictive',
            amount_due_chf: '45.00',
            currency: 'CHF',
            state: 'received',
            debtor_label: 'Patient fictif',
            handover_state: 'declared',
            reconciliation_variance_chf: null,
            driver_id: 4,
          },
          {
            id: 4,
            booking_id: 13,
            company_name: 'Entreprise fictive',
            amount_due_chf: '45.00',
            currency: 'CHF',
            state: 'administrative',
            debtor_label: 'Patient fictif',
            handover_state: null,
            reconciliation_variance_chf: null,
            driver_id: 4,
          },
        ],
      },
    });
    apiClient.post.mockResolvedValue({ data: {} });
    render(<CashRegisterFollowUp />);
    expect((await screen.findAllByText('Réglé')).length).toBeGreaterThan(0);
    expect(screen.getAllByText('Vérification nécessaire').length).toBeGreaterThan(0);
    expect(screen.getAllByText('45.00 CHF').length).toBeGreaterThan(0);
    fireEvent.click(screen.getByRole('button', { name: 'Confirmer les fonds reçus' }));
    await waitFor(() =>
      expect(apiClient.post).toHaveBeenCalledWith(
        '/companies/me/cash-register/3/confirm-handover'
      )
    );
    expect(await screen.findByText(/écart de caisse reste un contrôle distinct/)).toBeTruthy();
  });

  it('filtre une exception et ouvre son historique', async () => {
    apiClient.get.mockResolvedValue({
      data: {
        banner: 'PRÉPARATION — ENCAISSEMENT NON ACTIVÉ',
        cases: [
          {
            id: 4,
            booking_id: 13,
            company_name: 'Entreprise fictive',
            amount_due_chf: '45.00',
            currency: 'CHF',
            state: 'administrative',
            debtor_label: 'Patient fictif',
            handover_state: null,
            tendered_chf: '30.00',
            driver_id: 4,
            history: [{ action: 'declare_cash', to_state: 'administrative', actor_role: 'driver' }],
          },
          {
            id: 5,
            booking_id: 14,
            company_name: 'Entreprise fictive',
            amount_due_chf: '45.00',
            currency: 'CHF',
            state: 'received',
            debtor_label: 'Autre',
            handover_state: 'confirmed',
            reconciliation_variance_chf: null,
            driver_id: 8,
          },
        ],
      },
    });
    render(<CashRegisterFollowUp />);
    expect(await screen.findByText(/Espèces détenues 30.00 CHF. Vérification nécessaire./)).toBeTruthy();
    fireEvent.change(screen.getByLabelText('Filtre statut'), { target: { value: 'administrative' } });
    expect(screen.queryByText(/Réservation 14/)).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: 'Historique' }));
    expect(screen.getByText(/Encaissement déclaré/)).toBeTruthy();
  });

  it('consulte les règlements sans confirmation ni export', async () => {
    apiClient.get.mockResolvedValue({
      data: {
        cases: [
          {
            id: 8,
            booking_id: 20,
            company_name: 'Clinique fictive',
            amount_due_chf: '90.00',
            amount_received_chf: '90.00',
            recognized_net_chf: '85.00',
            currency: 'CHF',
            state: 'received',
            debtor_label: 'Patient fictif',
            driver_label: 'Chauffeur A',
            handover_state: 'declared',
            allocated_invoice_payment_id: null,
            invoice_number: 'F-2026-01',
            operations: [{ id: 2, nature: 'correction', net_chf: '-5.00' }],
          },
        ],
      },
    });
    render(<CashRegisterFollowUp focus="reglements" />);
    expect(await screen.findByText(/Réservation 20/)).toBeTruthy();
    expect(screen.getByText(/Montant contractuel : 90.00 CHF/)).toBeTruthy();
    expect(screen.getByText(/Montant encaissé : 90.00 CHF/)).toBeTruthy();
    expect(screen.getByText(/Correction : -5.00 CHF/)).toBeTruthy();
    expect(screen.getByText(/Montant net reconnu : 85.00 CHF/)).toBeTruthy();
    expect(screen.getByText(/Imputation : Facture F-2026-01/)).toBeTruthy();
    expect(screen.queryByRole('button', { name: /Confirmer/ })).toBeNull();
    expect(screen.queryByRole('button', { name: /Exporter/ })).toBeNull();
    expect(screen.queryByRole('button', { name: /paiement|Nouvelle facture|Encaisser/i })).toBeNull();
  });

  it('affiche la position, deux chauffeurs et deux remises partielles', async () => {
    apiClient.get.mockResolvedValue({
      data: {
        position: {
          collected_chf: '90.00',
          corrections_chf: '0.00',
          refunds_chf: '0.00',
          handed_over_chf: '0.00',
          held_chf: '90.00',
          drivers: [
            { driver_id: 1, driver_label: 'Chauffeur A', held_chf: '50.00' },
            { driver_id: 2, driver_label: 'Chauffeur B', held_chf: '40.00' },
          ],
        },
        cases: [
          {
            id: 9,
            booking_id: 21,
            company_name: 'Clinique fictive',
            amount_due_chf: '90.00',
            currency: 'CHF',
            state: 'received',
            debtor_label: 'Patient fictif',
            driver_label: 'Chauffeur A',
            handovers: [
              { id: 11, amount_chf: '40.00', state: 'declared', idempotency_key: 'remise-40' },
              { id: 12, amount_chf: '30.00', state: 'declared', idempotency_key: 'remise-30' },
            ],
          },
        ],
      },
    });
    apiClient.post.mockResolvedValue({ data: {} });
    render(<CashRegisterFollowUp />);
    expect(await screen.findByText(/Espèces encore détenues : 90.00 CHF/)).toBeTruthy();
    expect(screen.getByText(/Chauffeur A : 50.00 CHF encore détenus/)).toBeTruthy();
    expect(screen.getByText(/Chauffeur B : 40.00 CHF encore détenus/)).toBeTruthy();
    expect(screen.getByText(/Remise déclarée : 40.00 CHF/)).toBeTruthy();
    expect(screen.getByText(/Remise déclarée : 30.00 CHF/)).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: 'Confirmer la remise de 40.00 CHF' }));
    await waitFor(() =>
      expect(apiClient.post).toHaveBeenCalledWith(
        '/companies/me/cash-register/9/confirm-handover',
        { idempotency_key: 'remise-40' }
      )
    );
  });

  it('désactive la confirmation sans droit', async () => {
    apiClient.get.mockResolvedValue({
      data: {
        position: { held_chf: '20.00', collected_chf: '20.00', corrections_chf: '0.00', refunds_chf: '0.00', handed_over_chf: '0.00', drivers: [] },
        cases: [
          {
            id: 9,
            booking_id: 21,
            amount_due_chf: '20.00',
            currency: 'CHF',
            state: 'received',
            handover_state: 'declared',
          },
        ],
      },
    });
    render(<CashRegisterFollowUp canManageCash={false} />);
    const button = await screen.findByRole('button', { name: 'Confirmer les fonds reçus' });
    expect(button).toBeDisabled();
    expect(screen.getByText(/réservée au responsable/)).toBeTruthy();
  });

  it('affiche un écart négatif, un écart positif et une régularisation', async () => {
    apiClient.get.mockResolvedValue({
      data: {
        position: { held_chf: '20.00', collected_chf: '90.00', corrections_chf: '0.00', refunds_chf: '0.00', handed_over_chf: '70.00', drivers: [] },
        cases: [
          {
            id: 10,
            booking_id: 22,
            amount_due_chf: '90.00',
            currency: 'CHF',
            state: 'received',
            handovers: [
              { id: 3, amount_chf: '40.00', state: 'confirmed', idempotency_key: 'a' },
              { id: 4, amount_chf: '30.00', state: 'confirmed', idempotency_key: 'b' },
            ],
            reconciliations: [
              { id: 1, handover_id: 3, status: 'constate', variance_chf: '-5.00', open_variance_chf: '-3.00', reason: 'manque' },
              { id: 2, handover_id: 3, status: 'regularisation', variance_chf: '2.00', reason: 'retrouvé' },
              { id: 3, handover_id: 4, status: 'constate', variance_chf: '5.00', open_variance_chf: '5.00', reason: 'surplus' },
            ],
          },
        ],
      },
    });
    render(<CashRegisterFollowUp />);
    expect(await screen.findByText(/Écart à contrôler : -5.00 CHF/)).toBeTruthy();
    expect(screen.getByText(/Écart à contrôler : 5.00 CHF/)).toBeTruthy();
    expect(screen.getByText(/Régularisation : 2.00 CHF/)).toBeTruthy();
    expect(screen.queryByRole('button', { name: /Enregistrer l’écart/ })).toBeNull();
  });

  it('explique un refus sans afficher une réussite', async () => {
    apiClient.get.mockResolvedValue({
      data: {
        cases: [
          {
            id: 3,
            booking_id: 12,
            amount_due_chf: '45.00',
            currency: 'CHF',
            state: 'received',
            handover_state: 'declared',
          },
        ],
      },
    });
    apiClient.post.mockRejectedValue({ response: { status: 403, data: { error_code: 'role_required' } } });
    render(<CashRegisterFollowUp />);
    fireEvent.click(await screen.findByRole('button', { name: 'Confirmer les fonds reçus' }));
    expect(await screen.findByText(/réservée au responsable/)).toBeTruthy();
    expect(screen.queryByText(/Remise confirmée/)).toBeNull();
  });

  it('prépare l’export de caisse et reste vide sans encaissement', async () => {
    const createObjectURL = jest.fn(() => 'blob:caisse');
    const revokeObjectURL = jest.fn();
    global.URL.createObjectURL = createObjectURL;
    global.URL.revokeObjectURL = revokeObjectURL;
    const realCreate = document.createElement.bind(document);
    jest.spyOn(document, 'createElement').mockImplementation((tag) => {
      const node = realCreate(tag);
      if (tag === 'a') node.click = jest.fn();
      return node;
    });
    apiClient.get.mockResolvedValue({
      data: {
        position: {
          collected_chf: '0.00',
          corrections_chf: '0.00',
          refunds_chf: '0.00',
          handed_over_chf: '0.00',
          held_chf: '0.00',
          drivers: [],
        },
        cases: [],
      },
    });
    render(<CashRegisterFollowUp />);
    expect(await screen.findByText(/Aucun encaissement enregistré/)).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: 'Exporter le contrôle' }));
    expect(await screen.findByLabelText('Export de contrôle')).toHaveTextContent('chauffeur;date;transport');
    document.createElement.mockRestore();
  });

  it('annonce le chargement sans inventer de montant', () => {
    apiClient.get.mockReset();
    apiClient.get.mockReturnValue(new Promise(() => {}));
    render(<CashRegisterFollowUp />);
    expect(screen.getByText('Chargement des informations.')).toBeTruthy();
    expect(screen.queryByText(/CHF/)).toBeNull();
    expect(screen.queryByRole('button', { name: /Confirmer|Exporter/ })).toBeNull();
  });

  it('explique un accès refusé sans montant ni action', async () => {
    apiClient.get.mockReset();
    apiClient.post.mockReset();
    apiClient.get.mockRejectedValue({ response: { status: 403 } });
    render(<CashRegisterFollowUp />);
    expect(await screen.findByText(/Accès refusé/)).toBeTruthy();
    expect(screen.queryByText(/CHF/)).toBeNull();
    expect(screen.queryByRole('button', { name: /Confirmer|Exporter/ })).toBeNull();
    expect(apiClient.post).not.toHaveBeenCalled();
  });

  it('refuse une confirmation dont le montant n est pas communiqué', async () => {
    apiClient.get.mockReset();
    apiClient.post.mockReset();
    apiClient.get.mockResolvedValue({
      data: {
        cases: [
          {
            id: 11,
            booking_id: 30,
            state: 'received',
            handovers: [{ id: 8, amount_chf: null, state: 'declared' }],
          },
        ],
      },
    });
    render(<CashRegisterFollowUp />);
    expect(await screen.findByText(/montant de la remise n’est pas communiqué/)).toBeTruthy();
    expect(screen.queryByRole('button', { name: /Confirmer/ })).toBeNull();
    expect(apiClient.post).not.toHaveBeenCalled();
  });

  it('ne crée aucune écriture en consultant la caisse ou les règlements', async () => {
    apiClient.get.mockReset();
    apiClient.post.mockReset();
    apiClient.post.mockImplementation(() => {
      throw new Error('écriture interdite');
    });
    apiClient.get.mockResolvedValue({
      data: { cases: [], position: { held_chf: null, collected_chf: null, drivers: [] } },
    });
    const view = render(<CashRegisterFollowUp />);
    expect(await screen.findByText(/Aucun encaissement enregistré/)).toBeTruthy();
    expect(screen.getByText(/Espèces encore détenues : Non communiqué/)).toBeTruthy();
    expect(apiClient.get).toHaveBeenCalledWith('/companies/me/cash-register');
    expect(apiClient.post).not.toHaveBeenCalled();
    view.rerender(<CashRegisterFollowUp focus="reglements" />);
    expect(screen.getByText(/Aucune saisie ni confirmation de remise/)).toBeTruthy();
    expect(screen.queryByRole('button', { name: /Exporter|Confirmer/ })).toBeNull();
    expect(apiClient.post).not.toHaveBeenCalled();
  });

  it('explique une erreur de chargement sans montant', async () => {
    apiClient.get.mockRejectedValue({ response: { status: 500 } });
    render(<CashRegisterFollowUp focus="reglements" />);
    expect(await screen.findByText(/n’ont pas pu être chargées/)).toBeTruthy();
    expect(screen.queryByText(/CHF/)).toBeNull();
    expect(screen.queryByText(/confirmée/i)).toBeNull();
  });
});
