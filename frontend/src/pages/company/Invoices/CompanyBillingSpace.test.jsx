import React from 'react';
import fs from 'fs';
import path from 'path';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import CompanyBillingSpace from './CompanyBillingSpace';

jest.mock('./registry/InvoicesRegistry', () => () => (
  <div>
    <h1>Factures</h1>
    <button type="button">Nouvelle facture</button>
    <button type="button">Situation</button>
    <p>Registre des factures</p>
    <p>Prévisualisation, acomptes et PDF restent dans cette fenêtre.</p>
  </div>
));
jest.mock('../Settings/tabs/CashRegisterFollowUp', () => ({ focus }) => (
  <div>{focus === 'reglements' ? 'Suivi des règlements' : 'Registre de caisse'}</div>
));
jest.mock('../Settings/tabs/SettlementIncidentsFollowUp', () => () => (
  <div>File des signalements de règlement</div>
));

function renderSpace(entry = '/dashboard/company/abc/invoices/clients') {
  return render(
    <MemoryRouter initialEntries={[entry]}>
      <Routes>
        <Route path="/dashboard/company/:public_id/invoices/clients" element={<CompanyBillingSpace />} />
      </Routes>
    </MemoryRouter>
  );
}

describe('espace Facturation', () => {
  it('ouvre les factures existantes et un seul bouton Nouvelle facture', () => {
    renderSpace();
    expect(screen.getByText('Registre des factures')).toBeTruthy();
    expect(screen.getAllByRole('button', { name: 'Nouvelle facture' })).toHaveLength(1);
    expect(screen.getByRole('button', { name: 'Situation' })).toBeTruthy();
    expect(screen.queryByRole('button', { name: 'À préparer' })).toBeNull();
    expect(screen.queryByRole('button', { name: /Générer la facture|Ouvrir Factures pour émettre/i })).toBeNull();
    expect(screen.queryByText('Suivi des règlements')).toBeNull();
  });

  it('ramène l ancien onglet À préparer vers le registre', () => {
    renderSpace('/dashboard/company/abc/invoices/clients?vue=preparer');
    expect(screen.getByText('Registre des factures')).toBeTruthy();
    expect(screen.queryByText('Dossiers patient')).toBeNull();
    expect(screen.queryByRole('button', { name: 'À préparer' })).toBeNull();
  });

  it('retrouve les règlements et la caisse sans bouton d émission', async () => {
    renderSpace();
    await userEvent.click(screen.getByRole('button', { name: 'Règlements' }));
    expect(screen.getByText('Suivi des règlements')).toBeTruthy();
    expect(screen.queryByRole('button', { name: 'Nouvelle facture' })).toBeNull();
    expect(screen.queryByText('Registre des factures')).toBeNull();
    await userEvent.click(screen.getByRole('button', { name: 'Caisse chauffeurs' }));
    expect(screen.getByText('Registre de caisse')).toBeTruthy();
    expect(screen.queryByText('Suivi des règlements')).toBeNull();
    expect(screen.queryByRole('button', { name: /Générer|Nouvelle facture/i })).toBeNull();
    await userEvent.click(screen.getByRole('button', { name: 'Factures' }));
    expect(screen.getByRole('button', { name: 'Nouvelle facture' })).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Situation' })).toBeTruthy();
  });

  it('ouvre la caisse depuis son URL et revient aux factures', async () => {
    renderSpace('/dashboard/company/abc/invoices/clients?vue=caisse');
    expect(screen.getByRole('button', { name: 'Caisse chauffeurs' })).toHaveAttribute('aria-current', 'page');
    expect(screen.getByText('Registre de caisse')).toBeTruthy();
    expect(screen.queryByRole('button', { name: 'Situation' })).toBeNull();
    expect(screen.queryByRole('button', { name: 'Nouvelle facture' })).toBeNull();
    expect(screen.queryByRole('button', { name: 'À préparer' })).toBeNull();
    await userEvent.click(screen.getByRole('button', { name: 'Factures' }));
    expect(screen.getByRole('button', { name: 'Factures' })).toHaveAttribute('aria-current', 'page');
    expect(screen.getByRole('button', { name: 'Situation' })).toBeTruthy();
    expect(screen.getAllByRole('button', { name: 'Nouvelle facture' })).toHaveLength(1);
  });

  it('ouvre les règlements depuis leur URL', () => {
    renderSpace('/dashboard/company/abc/invoices/clients?vue=reglements');
    expect(screen.getByRole('button', { name: 'Règlements' })).toHaveAttribute('aria-current', 'page');
    expect(screen.getByText('Suivi des règlements')).toBeTruthy();
    expect(screen.queryByText('Registre de caisse')).toBeNull();
    expect(screen.queryByRole('button', { name: 'Nouvelle facture' })).toBeNull();
  });

  it('ne présente plus le suivi fictif comme un espace financier', () => {
    const root = path.join(__dirname, '..', '..', '..', '..', '..');
    const screenFile = path.join(
      root,
      'frontend/src/pages/company/Settings/tabs/SettlementSimulationFollowUp.jsx'
    );
    const space = fs.readFileSync(path.join(__dirname, 'CompanyBillingSpace.jsx'), 'utf8');
    const cash = fs.readFileSync(
      path.join(root, 'frontend/src/pages/company/Settings/tabs/CashRegisterFollowUp.jsx'),
      'utf8'
    );
    const route = fs.readFileSync(
      path.join(root, 'backend/routes/company_settlement_simulation.py'),
      'utf8'
    );
    expect(fs.existsSync(screenFile)).toBe(false);
    expect(space).not.toMatch(/SettlementSimulationFollowUp|settlement-simulations|Suivi fictif/);
    expect(cash).not.toMatch(/settlement-simulations/);
    expect(route).not.toMatch(/def delete|session\.delete|DROP TABLE/i);
    renderSpace();
    expect(screen.queryByText(/Suivi fictif/)).toBeNull();
    expect(screen.getByRole('navigation', { name: 'Facturation' }).querySelectorAll('button')).toHaveLength(4);
  });

  it('empile les onglets sur un écran étroit', () => {
    const css = fs.readFileSync(path.join(__dirname, 'CompanyBillingSpace.module.css'), 'utf8');
    expect(css).toMatch(/@media \(max-width: 640px\)/);
    expect(css).toMatch(/grid-template-columns:\s*1fr/);
    renderSpace();
    expect(screen.getByRole('navigation', { name: 'Facturation' }).querySelectorAll('button')).toHaveLength(4);
  });
});
