import { canRelaunchInstitutionRequest } from '../institutionRequestDispatch';

describe('canRelaunchInstitutionRequest', () => {
  it('autorise la relance quand seules des offres PENDING expirées existent', () => {
    const req = {
      status: 'SENT',
      dispatch: {
        can_relaunch: false,
        has_pending_offers: false,
        has_only_expired_pending: true,
      },
    };
    expect(canRelaunchInstitutionRequest(req)).toBe(true);
  });

  it('refuse la relance si une offre active est encore en attente', () => {
    const req = {
      status: 'SENT',
      dispatch: {
        can_relaunch: false,
        has_pending_offers: true,
        has_only_expired_pending: false,
      },
    };
    expect(canRelaunchInstitutionRequest(req)).toBe(false);
  });
});
