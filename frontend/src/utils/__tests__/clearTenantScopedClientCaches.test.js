jest.mock('../../App', () => ({
  queryClient: {
    clear: jest.fn(),
    getQueryData: jest.fn(),
    setQueryData: jest.fn(),
  },
}));

jest.mock('../../components/common/AddressAutocomplete', () => ({
  clearAddressAutocompleteCache: jest.fn(),
}));

const { clearTenantScopedClientCaches } = require('../clearTenantScopedClientCaches');
const { queryClient } = require('../../App');
const { clearAddressAutocompleteCache } = require('../../components/common/AddressAutocomplete');

describe('clearTenantScopedClientCaches', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('vide React Query et le cache autocomplete (A→logout→B)', () => {
    queryClient.setQueryData(['company', 'A', 'dashboard'], { secret: 'from-A' });
    clearTenantScopedClientCaches();
    expect(queryClient.clear).toHaveBeenCalledTimes(1);
    expect(clearAddressAutocompleteCache).toHaveBeenCalledTimes(1);
  });

  it('ne laisse pas de donnée résiduelle après clear simulé A→B', () => {
    const store = new Map();
    queryClient.setQueryData.mockImplementation((key, value) => {
      store.set(JSON.stringify(key), value);
    });
    queryClient.getQueryData.mockImplementation((key) => store.get(JSON.stringify(key)));
    queryClient.clear.mockImplementation(() => {
      store.clear();
    });

    queryClient.setQueryData(['company', 1, 'clients'], [{ id: 1, name: 'Client A' }]);
    expect(queryClient.getQueryData(['company', 1, 'clients'])).toEqual([
      { id: 1, name: 'Client A' },
    ]);

    clearTenantScopedClientCaches();

    expect(queryClient.getQueryData(['company', 1, 'clients'])).toBeUndefined();
  });
});
