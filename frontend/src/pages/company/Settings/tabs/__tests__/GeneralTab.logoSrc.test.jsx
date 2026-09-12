import { render } from '@testing-library/react';
import GeneralTab from '../GeneralTab';
import { resolveLogoUrl } from '../../../../../utils/resolveLogoUrl';

jest.mock('../../../../../components/common/AddressAutocomplete', () => () => null);

const baseProps = {
  company: { logo_url: '/uploads/company_logos/logo.png' },
  isEditing: false,
  form: {},
  fieldErrors: {},
  handleChange: jest.fn(),
  handleAddressSelect: jest.fn(),
  handleDomicileAddressSelect: jest.fn(),
  onClickPickFile: jest.fn(),
  onPickFile: jest.fn(),
  logoUrlEditOpen: false,
  setLogoUrlEditOpen: jest.fn(),
  logoUrlInput: '',
  setLogoUrlInput: jest.fn(),
  onSaveLogoUrl: jest.fn(),
  onRemoveLogo: jest.fn(),
  logoBusy: false,
};

describe('GeneralTab logo src', () => {
  it('affiche un logo https valide', () => {
    const src = resolveLogoUrl('https://cdn.example.com/logo.png');
    const { getByAltText } = render(<GeneralTab {...baseProps} logoPreview={src} />);
    expect(getByAltText("Logo de l'entreprise")).toHaveAttribute('src', src);
  });

  it('n’affiche pas d’img si javascript: est résolu', () => {
    const { queryByAltText } = render(
      <GeneralTab {...baseProps} logoPreview={resolveLogoUrl(`javascript${':'}alert(1)`)} />
    );
    expect(queryByAltText("Logo de l'entreprise")).toBeNull();
  });

  it('n’affiche pas d’img si data:text/html est résolu', () => {
    const { queryByAltText } = render(
      <GeneralTab
        {...baseProps}
        logoPreview={resolveLogoUrl('data:text/html,<script>alert(1)</script>')}
      />
    );
    expect(queryByAltText("Logo de l'entreprise")).toBeNull();
  });

  it('affiche un blob de preview locale', () => {
    const src = resolveLogoUrl('blob:http://localhost/preview', { allowPreview: true });
    const { getByAltText } = render(<GeneralTab {...baseProps} logoPreview={src} />);
    expect(getByAltText("Logo de l'entreprise")).toHaveAttribute('src', src);
  });
});
