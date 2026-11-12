import { render, screen } from '@testing-library/react';
import CompanySettlementPreferences from './CompanySettlementPreferences';

jest.mock('../../../../utils/apiClient', () => ({
  __esModule: true,
  default: {
    get: jest.fn(() => Promise.reject(new Error('rubrique inactive'))),
    put: jest.fn(),
  },
}));

describe('préférences de règlement entreprise', () => {
  it('propose trois moyens sans les rendre disponibles et laisse la confirmation décochée', async () => {
    render(<CompanySettlementPreferences companyId={1} />);
    expect(screen.getByLabelText('Paiements et facturation')).toBeTruthy();
    expect(screen.getByText('Tap to Pay')).toBeTruthy();
    expect(screen.getByText('Espèces')).toBeTruthy();
    expect(screen.getByText('Facture directe au patient')).toBeTruthy();
    expect(screen.getAllByText(/Indisponible/).length).toBe(3);
    expect(screen.getAllByText(/Préférence enregistrée : non/).length).toBe(3);
    const notice = screen.getByRole('checkbox', {
      name: /conditions de règlement ont été communiquées au patient/,
    });
    expect(notice.checked).toBe(false);
    expect(
      screen.getByText(/n’est pas une preuve de réception/)
    ).toBeTruthy();
  });
});
