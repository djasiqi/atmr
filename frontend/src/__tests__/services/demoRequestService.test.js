import {
  DEMO_ORGANIZATION_TYPES,
  DEMO_USE_CASES,
  submitDemoRequest,
} from '../../services/demoRequestService';
import apiClient from '../../utils/apiClient';

jest.mock('../../utils/apiClient');

describe('demoRequestService', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('expose les listes de qualification', () => {
    expect(DEMO_ORGANIZATION_TYPES).toContain('transport_company');
    expect(DEMO_USE_CASES).toContain('planning_dispatch');
  });

  it('soumet une demande de demonstration', async () => {
    apiClient.post.mockResolvedValue({ data: { ok: true } });
    const payload = {
      name: 'Marie Curie',
      email: 'marie@example.com',
      organization: 'Clinique Test',
      organization_type: 'clinic',
      use_case: 'planning_dispatch',
      integration_required: 'yes',
      timing: 'immediate',
      preferred_slot: 'this_week',
      preferred_period: 'morning',
      privacy_consent: true,
    };

    const result = await submitDemoRequest(payload, { publicRequest: true });

    expect(apiClient.post).toHaveBeenCalledWith('/demo-requests', payload, {
      baseURL: '/api/app',
      _targetEnv: 'app',
      skipCsrf: true,
      skipAuthRedirect: true,
      skipEnvRouting: true,
    });
    expect(result).toEqual({ ok: true });
  });
});
