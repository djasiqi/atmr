import {
  normalizeStatus,
  isCompletedStatus,
  isCanceledStatus,
  getStatusTab,
} from './reservationStatusUtils';

describe('reservationStatusUtils', () => {
  describe('normalizeStatus', () => {
    it('retourne lowercase et trim', () => {
      expect(normalizeStatus('COMPLETED')).toBe('completed');
      expect(normalizeStatus('  Return_Completed  ')).toBe('return_completed');
    });
    it('gère null/undefined', () => {
      expect(normalizeStatus(null)).toBe('');
      expect(normalizeStatus(undefined)).toBe('');
    });
  });

  describe('isCompletedStatus', () => {
    it('inclut completed et return_completed', () => {
      expect(isCompletedStatus('completed')).toBe(true);
      expect(isCompletedStatus('return_completed')).toBe(true);
      expect(isCompletedStatus('COMPLETED')).toBe(true);
      expect(isCompletedStatus('RETURN_COMPLETED')).toBe(true);
    });
    it('inclut return completed (espace)', () => {
      expect(isCompletedStatus('return completed')).toBe(true);
    });
    it('exclut pending et canceled', () => {
      expect(isCompletedStatus('pending')).toBe(false);
      expect(isCompletedStatus('canceled')).toBe(false);
    });
  });

  describe('isCanceledStatus', () => {
    it('inclut canceled et cancelled', () => {
      expect(isCanceledStatus('canceled')).toBe(true);
      expect(isCanceledStatus('cancelled')).toBe(true);
    });
  });

  describe('getStatusTab', () => {
    it('completed et return_completed => completed', () => {
      expect(getStatusTab('completed')).toBe('completed');
      expect(getStatusTab('return_completed')).toBe('completed');
    });
    it('pending => pending', () => {
      expect(getStatusTab('pending')).toBe('pending');
    });
    it('accepted/assigned/in_progress => in_progress', () => {
      expect(getStatusTab('accepted')).toBe('in_progress');
      expect(getStatusTab('assigned')).toBe('in_progress');
      expect(getStatusTab('in_progress')).toBe('in_progress');
    });
    it('canceled => canceled', () => {
      expect(getStatusTab('canceled')).toBe('canceled');
    });
  });
});
