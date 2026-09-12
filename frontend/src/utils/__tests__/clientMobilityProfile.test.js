import {
  getMobilityProfileForUser,
  saveMobilityProfileForEmail,
} from '../clientMobilityProfile';

const STORAGE_KEY = 'lirie_client_mobility_profile_v1';

describe('clientMobilityProfile', () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it('persiste les flags mobilité sans contact d’urgence ni notes', () => {
    saveMobilityProfileForEmail('client@example.com', {
      needsWheelchair: true,
      needsDoorToDoorAssistance: true,
      assistanceLevel: 'high',
      emergencyContact: '+41 79 000 00 00',
      notes: 'PII à ne pas stocker',
    });

    const raw = JSON.parse(localStorage.getItem(STORAGE_KEY));
    const persisted = raw.byEmail['client@example.com'];
    expect(persisted.needsWheelchair).toBe(true);
    expect(persisted.assistanceLevel).toBe('high');
    expect(persisted.emergencyContact).toBeUndefined();
    expect(persisted.notes).toBeUndefined();
    expect(JSON.stringify(raw)).not.toMatch(/79 000/);
    expect(JSON.stringify(raw)).not.toMatch(/PII/);

    const read = getMobilityProfileForUser({ email: 'client@example.com' });
    expect(read.emergencyContact).toBe('');
    expect(read.notes).toBe('');
    expect(read.needsWheelchair).toBe(true);
  });

  it('purge un profil legacy déjà stocké avec PII', () => {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        byEmail: {
          'old@example.com': {
            needsWheelchair: false,
            emergencyContact: 'Alice +41000',
            notes: 'secret médical',
          },
        },
        byPublicId: {},
        last: { emergencyContact: 'Alice +41000', notes: 'secret médical' },
      })
    );

    const profile = getMobilityProfileForUser({ email: 'old@example.com' });
    expect(profile.emergencyContact).toBe('');
    expect(profile.notes).toBe('');

    const raw = localStorage.getItem(STORAGE_KEY);
    expect(raw).not.toMatch(/Alice/);
    expect(raw).not.toMatch(/secret médical/);
  });
});
