import { queryKeys } from '../queryKeys';

describe('queryKeys', () => {
  it('builds stable institution request keys with params', () => {
    const key = queryKeys.institutionRequests({
      status: 'SENT',
      external_reference: 'EXT-123',
      patient_id: 42,
      date_from: '2026-04-01',
      date_to: '2026-04-30',
      page: 3,
      per_page: 50,
    });

    expect(key).toEqual([
      'institution',
      'requests',
      'SENT',
      'EXT-123',
      '42',
      '2026-04-01',
      '2026-04-30',
      '3',
      '50',
    ]);
  });

  it('normalizes patients search key', () => {
    expect(queryKeys.institutionPatients('  JoHN  ')).toEqual(['institution', 'patients', 'john']);
  });
});
