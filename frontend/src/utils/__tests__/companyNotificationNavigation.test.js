import { resolveCompanyNotificationLink } from '../companyNotificationNavigation';

describe('resolveCompanyNotificationLink', () => {
  const baseArgs = {
    dashboardRoot: '/dashboard',
    companyPublicId: 'emmenex-moi',
  };

  it('dirige vers le dispatch avec panneau ouvert pour institution_change_request', () => {
    const link = resolveCompanyNotificationLink({
      ...baseArgs,
      notif: {
        event_type: 'institution_change_request',
        metadata: { booking_id: 31004, change_request_id: 42 },
      },
    });
    expect(link).toBe(
      '/dashboard/company/emmenex-moi/dispatch?booking=31004&focus=change_request',
    );
  });

  it('dirige vers le dispatch si booking_id présent (request_updated acceptée)', () => {
    const link = resolveCompanyNotificationLink({
      ...baseArgs,
      notif: {
        event_type: 'request_updated',
        metadata: { booking_id: 99, request_id: 1 },
      },
    });
    expect(link).toBe('/dashboard/company/emmenex-moi/dispatch?booking=99');
  });

  it('dirige vers le dashboard institution pour new_request sans booking', () => {
    const link = resolveCompanyNotificationLink({
      ...baseArgs,
      notif: {
        event_type: 'new_request',
        metadata: { request_id: 12, offer_id: 7 },
      },
    });
    expect(link).toBe(
      '/dashboard/company/emmenex-moi?tab=institution&request=12&offer=7',
    );
  });
});
