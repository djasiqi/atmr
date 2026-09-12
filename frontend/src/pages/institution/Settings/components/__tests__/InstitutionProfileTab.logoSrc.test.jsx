import { render } from '@testing-library/react';
import InstitutionProfileTab from '../InstitutionProfileTab';
import { resolveLogoUrl } from '../../../../../utils/resolveLogoUrl';

const mockMe = {
  name: 'Clinique',
  institution_type: 'clinic',
  address: 'Rue 1',
  contact_email: 'a@b.c',
  contact_phone: '',
  notes: '',
  institution_role: 'institution_admin',
  logo_url: null,
};

jest.mock('../../../../../hooks/useInstitutionData', () => ({
  useInstitutionMe: () => ({ data: mockMe, isLoading: false }),
  useUpdateInstitution: () => ({ mutateAsync: jest.fn() }),
  institutionQueryKeys: { me: () => ['institution', 'me'] },
}));

jest.mock('@tanstack/react-query', () => ({
  useQueryClient: () => ({ invalidateQueries: jest.fn() }),
}));

jest.mock('../../../../../services/institutionService', () => ({
  uploadInstitutionLogo: jest.fn(),
  deleteInstitutionLogo: jest.fn(),
}));

jest.mock('../../../../../utils/institutionPermissions', () => ({
  isAdmin: () => true,
}));

jest.mock('../../../../../components/common/AddressAutocomplete', () => () => null);
jest.mock('../ChipSelect', () => () => null);
jest.mock('sonner', () => ({ toast: { success: jest.fn(), error: jest.fn() } }));

describe('InstitutionProfileTab logo src', () => {
  beforeEach(() => {
    mockMe.logo_url = null;
  });

  it('refuse javascript: dans le preview résolu', () => {
    mockMe.logo_url = `javascript${':'}alert(1)`;
    const { queryByAltText } = render(<InstitutionProfileTab />);
    expect(queryByAltText("Logo de l'institution")).toBeNull();
  });

  it('refuse data:text/html dans le preview résolu', () => {
    mockMe.logo_url = 'data:text/html,<script>alert(1)</script>';
    const { queryByAltText } = render(<InstitutionProfileTab />);
    expect(queryByAltText("Logo de l'institution")).toBeNull();
  });

  it('affiche un logo https valide', () => {
    mockMe.logo_url = 'https://cdn.example.com/inst.png';
    const { getByAltText } = render(<InstitutionProfileTab />);
    expect(getByAltText("Logo de l'institution")).toHaveAttribute(
      'src',
      resolveLogoUrl('https://cdn.example.com/inst.png')
    );
  });
});
