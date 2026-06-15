import {
  combineMissionDateTime,
  derivePickupTimeConfirmed,
  applyDepartureToPayload,
  normalizeMissionDate,
  isInstantInPast,
  isInstantBeforeLead,
  extractHHMM,
} from '../missionScheduleForm';

describe('missionScheduleForm', () => {
  describe('normalizeMissionDate', () => {
    it('accepte ISO YYYY-MM-DD', () => {
      expect(normalizeMissionDate('2026-06-13')).toBe('2026-06-13');
    });

    it('convertit DD.MM.YYYY', () => {
      expect(normalizeMissionDate('13.06.2026')).toBe('2026-06-13');
    });
  });

  describe('combineMissionDateTime', () => {
    it('combine date mission et heure HH:MM (ISO naïf Genève)', () => {
      const iso = combineMissionDateTime('2026-06-13', '09:00');
      expect(iso).toBe('2026-06-13T09:00:00');
    });

    it('retourne null si date absente (scénario C — date non commitée)', () => {
      expect(combineMissionDateTime('', '09:00')).toBeNull();
      expect(combineMissionDateTime(null, '09:00')).toBeNull();
    });

    it('retourne null si heure absente', () => {
      expect(combineMissionDateTime('2026-06-13', '')).toBeNull();
    });
  });

  describe('derivePickupTimeConfirmed', () => {
    it('true si heure présente', () => {
      expect(derivePickupTimeConfirmed('09:00')).toBe(true);
    });

    it('false si heure vide', () => {
      expect(derivePickupTimeConfirmed('')).toBe(false);
    });
  });

  describe('applyDepartureToPayload', () => {
    it('pose scheduled_time quand départ confirmé', () => {
      const payload = {};
      const ok = applyDepartureToPayload(payload, {
        missionDate: '2026-06-13',
        pickupTime: '09:00',
      });
      expect(ok).toBe(true);
      expect(payload.pickup_time_confirmed).toBe(true);
      expect(payload.scheduled_time).toBe('2026-06-13T09:00:00');
      expect(payload.scheduled_time_type).toBe('departure');
    });

    it('ne pose pas pickup_time_confirmed sans heure', () => {
      const payload = {};
      const ok = applyDepartureToPayload(payload, {
        missionDate: '2026-06-13',
        pickupTime: '',
      });
      expect(ok).toBe(true);
      expect(payload.pickup_time_confirmed).toBe(false);
      expect(payload.scheduled_time).toBeUndefined();
    });

    it('échoue si heure présente mais ISO impossible (date manquante)', () => {
      const payload = { pickup_time_confirmed: true };
      const ok = applyDepartureToPayload(payload, {
        missionDate: '',
        pickupTime: '09:00',
      });
      expect(ok).toBe(false);
      expect(payload.pickup_time_confirmed).toBe(true);
      expect(payload.scheduled_time).toBeUndefined();
    });
  });

  describe('isInstantInPast', () => {
    it('true pour un instant nettement passé', () => {
      const past = new Date(Date.now() - 60 * 60 * 1000).toISOString();
      expect(isInstantInPast(past)).toBe(true);
    });

    it('false pour maintenant (tolérance départ)', () => {
      const now = new Date().toISOString();
      expect(isInstantInPast(now)).toBe(false);
    });

    it('false pour le futur', () => {
      const future = new Date(Date.now() + 60 * 60 * 1000).toISOString();
      expect(isInstantInPast(future)).toBe(false);
    });

    it('false si valeur absente ou invalide', () => {
      expect(isInstantInPast(null)).toBe(false);
      expect(isInstantInPast('not-a-date')).toBe(false);
    });
  });

  describe('isInstantBeforeLead', () => {
    it('true si rendez-vous à moins d\'1h', () => {
      const soon = new Date(Date.now() + 30 * 60 * 1000).toISOString();
      expect(isInstantBeforeLead(soon)).toBe(true);
    });

    it('true si rendez-vous dans le passé', () => {
      const past = new Date(Date.now() - 30 * 60 * 1000).toISOString();
      expect(isInstantBeforeLead(past)).toBe(true);
    });

    it('false si rendez-vous à plus d\'1h', () => {
      const later = new Date(Date.now() + 90 * 60 * 1000).toISOString();
      expect(isInstantBeforeLead(later)).toBe(false);
    });
  });

  describe('extractHHMM', () => {
    it('garde une valeur HH:MM', () => {
      expect(extractHHMM('14:30')).toBe('14:30');
    });

    it('extrait l\'heure d\'un ISO', () => {
      expect(extractHHMM('2026-06-13T14:05:00')).toBe('14:05');
    });

    it('retourne vide si valeur absente', () => {
      expect(extractHHMM('')).toBe('');
      expect(extractHHMM(null)).toBe('');
    });
  });
});
