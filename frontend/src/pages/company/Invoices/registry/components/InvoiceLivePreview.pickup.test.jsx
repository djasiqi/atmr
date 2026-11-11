import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import InvoiceLivePreview from './InvoiceLivePreview';

const partnerLines = [
  {
    id: 1,
    type: 'ride',
    description: 'Eric DEMIERRE — A → B',
    service_date: '2026-09-05',
    line_total: 40,
    amount: 40,
    pickup_label: '08:23',
  },
  {
    id: 2,
    type: 'ride',
    description: 'Michelle BUSSARD — B → A',
    service_date: '2026-09-12',
    line_total: 40,
    amount: 40,
    pickup_label: '11:47',
  },
];

test('affiche les deux heures de prise en charge sans fusionner les jambes', () => {
  render(
    <InvoiceLivePreview
      invoice={{
        billing_strategy: 'partner_monthly',
        line_time_mode: 'pickup',
        invoice_number: 'PARTNER-EM-2026-09-0124',
        subtotal_amount: 80,
        total_amount: 80,
        lines: partnerLines,
      }}
    />,
  );
  expect(screen.getAllByText('Client : DEMIERRE Eric').length).toBeGreaterThan(0);
  expect(screen.getByText('Client : BUSSARD Michelle')).toBeInTheDocument();
  expect(screen.getByText('Prise en charge : 08:23')).toBeInTheDocument();
  expect(screen.getByText('Prise en charge : 11:47')).toBeInTheDocument();
  expect(screen.getByText('Trajet : A → B')).toBeInTheDocument();
  expect(screen.getByText('Trajet : B → A')).toBeInTheDocument();
  expect(screen.queryByText(/Transfert partenaire/)).not.toBeInTheDocument();
  expect(screen.queryByRole('columnheader', { name: 'Prise en charge' })).not.toBeInTheDocument();
  expect(screen.getByText('TOTAL À FACTURER :')).toBeInTheDocument();
});

test('sans option, le tableau institutionnel n’a pas la colonne horaire', () => {
  render(
    <InvoiceLivePreview
      invoice={{
        billing_strategy: 's2_clinic_monthly',
        invoice_number: 'CLIN-1',
        subtotal_amount: 40,
        total_amount: 40,
        lines: [
          {
            id: 1,
            type: 'ride',
            description: 'Trajet : Chemin des Vignes 4 → Clinique',
            service_date: '2026-09-05',
            line_total: 40,
            amount: 40,
            line_meta: { patient_name: 'Claire Bonvin' },
          },
        ],
      }}
    />,
  );
  expect(screen.getByText('Client : BONVIN Claire')).toBeInTheDocument();
  expect(screen.getByText('Trajet : Chemin des Vignes 4 → Clinique')).toBeInTheDocument();
  expect(screen.queryByText(/Prise en charge/)).not.toBeInTheDocument();
  expect(screen.getByText('TOTAL À FACTURER :')).toBeInTheDocument();
  expect(screen.getByText('Description')).toBeInTheDocument();
});

test.each([
  ['patient', 'direct_patient'],
  ['institution', 's2_clinic_monthly'],
  ['partenaire', 'partner_monthly'],
])('%s : la réduction globale reste dans les totaux, pas sous la prestation', (_payer, strategy) => {
  render(
    <InvoiceLivePreview
      invoice={{
        billing_strategy: strategy,
        subtotal_amount: 64.8,
        total_amount: 64.8,
        meta: {
          global_discount: {
            percent: 10,
            amount_ht: 7.2,
            subtotal_before_ht: 72,
          },
        },
        lines: [
          {
            id: 1,
            type: 'ride',
            description: 'Trajet : Chemin des Vignes 4 → Clinique',
            line_total: 64.8,
            line_meta: {
              patient_name: 'Claire Bonvin',
              original_line_total: 72,
            },
          },
        ],
      }}
    />
  );
  expect(screen.getByText('Client : BONVIN Claire')).toBeInTheDocument();
  expect(screen.getByText('Trajet : Chemin des Vignes 4 → Clinique')).toBeInTheDocument();
  expect(screen.getByText('72.00')).toBeInTheDocument();
  expect(screen.queryByText('64.80')).not.toBeInTheDocument();
  expect(screen.queryByText(/Rabais 10 %/)).not.toBeInTheDocument();
  expect(screen.getByText('Réduction globale (10 %)')).toBeInTheDocument();
  expect(screen.getByText('Sous-total HT (avant réduction globale)')).toBeInTheDocument();
  expect(screen.getByText('TOTAL À FACTURER :')).toBeInTheDocument();
});
