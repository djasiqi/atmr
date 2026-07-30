/**
 * Tests gate séquence temps réel Lot 3.
 */
const {
  setSnapshotCursor,
  evaluateRealtimeSequence,
  resetCompanySequenceState,
  getLastAppliedSeq,
} = require('../companyRealtimeSequenceGate');

describe('companyRealtimeSequenceGate', () => {
  beforeEach(() => {
    resetCompanySequenceState(1);
  });

  it('refuse les event_seq ≤ snapshot_cursor', () => {
    setSnapshotCursor(1, 10);
    expect(evaluateRealtimeSequence(1, 10).accept).toBe(false);
    expect(evaluateRealtimeSequence(1, 5).accept).toBe(false);
    expect(evaluateRealtimeSequence(1, 11).accept).toBe(true);
    expect(getLastAppliedSeq(1)).toBe(11);
  });

  it('détecte un trou de séquence', () => {
    setSnapshotCursor(1, 1);
    evaluateRealtimeSequence(1, 2);
    const gap = evaluateRealtimeSequence(1, 5);
    expect(gap.accept).toBe(true);
    expect(gap.gapDetected).toBe(true);
  });

  it('accepte les événements sans event_seq (legacy)', () => {
    setSnapshotCursor(1, 100);
    expect(evaluateRealtimeSequence(1, null).accept).toBe(true);
    expect(evaluateRealtimeSequence(1, 0).accept).toBe(true);
  });
});
