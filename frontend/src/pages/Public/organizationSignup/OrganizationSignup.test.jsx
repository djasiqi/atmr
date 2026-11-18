import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import ApplicationStatusCard from './ApplicationStatusCard';
import OrganizationSignupChoice from './OrganizationSignupChoice';
import OrganizationSignupForm from './OrganizationSignupForm';
import { WORST_CASE_APPLICATION } from './worstCaseApplication';
import { SLA_NOTICE, submissionErrorMessage, NETWORK_INTERRUPTION_MESSAGE } from '../../../services/organizationApplicationService';

describe('préinscription organisation', () => {
  it('propose le particulier, les demandes professionnelles et l’invitation chauffeur', () => {
    render(
      <MemoryRouter>
        <OrganizationSignupChoice />
      </MemoryRouter>
    );
    expect(screen.getByRole('link', { name: /Particulier/ })).toHaveAttribute(
      'href',
      '/login?mode=signup'
    );
    expect(screen.getByRole('link', { name: /Entreprise de transport/ })).toHaveAttribute(
      'href',
      '/inscription/organisation/entreprise'
    );
    expect(screen.getByRole('link', { name: /Institution/ })).toBeInTheDocument();
    expect(
      screen.getByText('Votre entreprise de transport doit vous inviter à rejoindre LIRIE.')
    ).toBeInTheDocument();
    expect(screen.queryByRole('link', { name: /Chauffeur/ })).not.toBeInTheDocument();
    expect(screen.queryByText(/Google/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Apple/)).not.toBeInTheDocument();
    expect(screen.getByText(SLA_NOTICE)).toBeInTheDocument();
  });

  it('limite les champs du formulaire entreprise', () => {
    render(
      <MemoryRouter>
        <OrganizationSignupForm kind="company" />
      </MemoryRouter>
    );
    expect(screen.getByRole('textbox', { name: 'Raison sociale' })).toHaveAttribute('maxLength', '200');
    expect(screen.getByRole('textbox', { name: 'E-mail professionnel' })).toHaveAttribute('maxLength', '255');
    expect(document.querySelector('[name="role"]')).toBeNull();
  });

  it('affiche un dossier long sans date d’échéance', () => {
    render(<ApplicationStatusCard application={WORST_CASE_APPLICATION} />);
    expect(screen.getByText(WORST_CASE_APPLICATION.organization_name)).toBeInTheDocument();
    expect(screen.getByText(WORST_CASE_APPLICATION.applicant_email)).toBeInTheDocument();
    expect(screen.getByText(SLA_NOTICE)).toBeInTheDocument();
    expect(screen.queryByText(/échéance/i)).not.toBeInTheDocument();
    expect(screen.getByText(/Aucun accès ne vous a été ouvert/)).toBeInTheDocument();
    expect(screen.queryByRole('button')).not.toBeInTheDocument();
  });

  it('conserve une explication française quand le réseau coupe l’envoi', () => {
    expect(submissionErrorMessage(new Error('Network Error'))).toBe(NETWORK_INTERRUPTION_MESSAGE);
    expect(NETWORK_INTERRUPTION_MESSAGE).toMatch(/restent dans le formulaire/);
    expect(NETWORK_INTERRUPTION_MESSAGE).toMatch(/peut-être déjà reçu/);
    expect(NETWORK_INTERRUPTION_MESSAGE).not.toMatch(/dernier lien reçu/);
    expect(NETWORK_INTERRUPTION_MESSAGE).not.toMatch(/perdu/);
    expect(submissionErrorMessage({ response: { status: 400, data: { message: 'Indiquez la rue.' } } }))
      .toBe('Indiquez la rue.');
  });
});
