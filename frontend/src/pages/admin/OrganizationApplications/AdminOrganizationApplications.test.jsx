import React from 'react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { render, screen } from '@testing-library/react';
import AdminOrganizationApplications from './AdminOrganizationApplications';
import AdminOrganizationApplicationDetail from './AdminOrganizationApplicationDetail';
import {
  fetchOrganizationApplication,
  fetchOrganizationApplications,
} from '../../../services/adminOrganizationApplicationService';

jest.mock('../../../services/adminOrganizationApplicationService', () => ({
  fetchOrganizationApplications: jest.fn(),
  fetchOrganizationApplication: jest.fn(),
  decideOrganizationApplication: jest.fn(),
  searchExistingOrganizations: jest.fn(),
}));

const LONG_NAME = 'Établissement médico-social de la Côte et du Jorat — site de Villeneuve';
const LONG_EMAIL = 'bartholomew.fitzgerald@northwind-industries-holdings.example.com';

function renderList() {
  return render(
    <MemoryRouter initialEntries={['/dashboard/admin/pub-1/partners/organization-applications']}>
      <Routes>
        <Route
          path="/dashboard/admin/:public_id/partners/organization-applications"
          element={<AdminOrganizationApplications />}
        />
      </Routes>
    </MemoryRouter>
  );
}

describe('file des inscriptions', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('distingue un dossier incomplet d’un dossier reçu', async () => {
    fetchOrganizationApplications.mockResolvedValue({
      items: [
        {
          public_id: 'incomplet',
          organization_name: 'Transports du Lac',
          kind_label: 'Entreprise',
          applicant_name: 'Nora Meier',
          applicant_email: 'nora@example.test',
          submitted_at: null,
          incomplete: true,
          actionable: false,
          queue_label: 'Incomplet',
          status_label: 'Adresse e-mail à confirmer',
        },
        {
          public_id: 'recu',
          organization_name: LONG_NAME,
          kind_label: 'Institution',
          applicant_name: 'Aleksandra Wiśniewska-Kowalczyk',
          applicant_email: LONG_EMAIL,
          submitted_at: '2026-10-12T12:00:00.000Z',
          incomplete: false,
          actionable: true,
          queue_label: 'À traiter',
          status_label: 'Dossier reçu',
        },
      ],
    });

    renderList();

    expect(await screen.findByText('Dossier incomplet')).toBeInTheDocument();
    expect(screen.getByText(LONG_NAME)).toBeInTheDocument();
    expect(screen.getByText(LONG_EMAIL)).toBeInTheDocument();
    expect(screen.getByText('À traiter')).toBeInTheDocument();
    expect(screen.queryByText('Dossier complet')).not.toBeInTheDocument();
  });

  it('affiche l’état vide', async () => {
    fetchOrganizationApplications.mockResolvedValue({ items: [] });
    renderList();
    expect(await screen.findByText('Aucune demande à afficher')).toBeInTheDocument();
  });

  it('affiche l’erreur de chargement', async () => {
    fetchOrganizationApplications.mockRejectedValue(new Error('réseau'));
    renderList();
    expect(await screen.findByRole('alert')).toHaveTextContent('Impossible de charger les demandes.');
  });
});

describe('détail d’une inscription', () => {
  it('laisse l’approbation inactive et montre un e-mail long', async () => {
    fetchOrganizationApplication.mockResolvedValue({
      public_id: 'recu',
      kind_label: 'Institution',
      organization_name: LONG_NAME,
      incomplete: false,
      actionable: true,
      status: 'received',
      status_label: 'Dossier reçu',
      queue_label: 'À traiter',
      applicant_first_name: 'Aleksandra',
      applicant_last_name: 'Wiśniewska-Kowalczyk',
      applicant_function: 'Direction',
      applicant_email: LONG_EMAIL,
      applicant_phone: '+41790000000',
      street: 'Rue du Lac 4',
      postal_code: '1844',
      city: 'Villeneuve',
      ide_number: null,
      establishment_category: 'ems',
      applicant_review_note: null,
      documents_note: 'Aucun justificatif n’est accepté pour le moment.',
      approval: { enabled: false, reason: 'L’approbation n’est pas disponible.' },
      sla: {
        notice: 'Votre demande sera examinée dans un délai maximal de 24 heures administratives cumulées.',
        calendar: 'pending',
        due_at: null,
        suspensions: [],
      },
      checks: {
        identity: { confirmed: null },
        organization: { confirmed: null },
      },
      events: [],
    });

    render(
      <MemoryRouter initialEntries={['/dashboard/admin/pub-1/partners/organization-applications/recu']}>
        <Routes>
          <Route
            path="/dashboard/admin/:public_id/partners/organization-applications/:applicationId"
            element={<AdminOrganizationApplicationDetail />}
          />
        </Routes>
      </MemoryRouter>
    );

    expect(await screen.findByRole('button', { name: 'Approuver et activer' })).toBeDisabled();
    expect(screen.getByText(LONG_EMAIL)).toBeInTheDocument();
    expect(screen.queryByText(/échéance/i)).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Prendre en charge' })).toBeEnabled();
  });
});
