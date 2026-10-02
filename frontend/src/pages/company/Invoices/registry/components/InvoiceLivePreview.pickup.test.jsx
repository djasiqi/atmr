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
        lines: partnerLines,
      }}
    />,
  );
  expect(screen.getByText('Prise en charge')).toBeInTheDocument();
  expect(screen.getByText('08:23')).toBeInTheDocument();
  expect(screen.getByText('11:47')).toBeInTheDocument();
  expect(screen.getByText(/A → B/)).toBeInTheDocument();
  expect(screen.getByText(/B → A/)).toBeInTheDocument();
});

test('sans option, le tableau institutionnel n’a pas la colonne horaire', () => {
  render(
    <InvoiceLivePreview
      invoice={{
        billing_strategy: 's2_clinic_monthly',
        invoice_number: 'CLIN-1',
        lines: [
          {
            id: 1,
            type: 'ride',
            description: 'Course clinique',
            service_date: '2026-09-05',
            line_total: 40,
            amount: 40,
          },
        ],
      }}
    />,
  );
  expect(screen.queryByText('Prise en charge')).not.toBeInTheDocument();
  expect(screen.getByText('Description')).toBeInTheDocument();
});
