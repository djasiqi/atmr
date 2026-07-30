/**
 * Tests gate séquence temps réel (inspect / commit).
 */
const {
  setSnapshotCursor,
  inspectSequence,
  commitAppliedSequence,
  evaluateRealtimeSequence,
  resetCompanySequenceState,
  getLastAppliedSeq,
  setSubscribedCursor,
  bufferRealtimeEvent,
  drainContiguousBuffer,
  clearResyncAfterBootstrapSuccess,
  isResyncRequired,
  isRealtimeDegraded,
} = require('../companyRealtimeSequenceGate');

describe('companyRealtimeSequenceGate', () => {
  beforeEach(() => {
    resetCompanySequenceState(1);
  });

  it('inspect refuse les event_seq ≤ snapshot_cursor sans commit', () => {
    setSnapshotCursor(1, 10);
    expect(inspectSequence(1, 10).accept).toBe(false);
    expect(inspectSequence(1, 5).accept).toBe(false);
    const ok = inspectSequence(1, 11);
    expect(ok.accept).toBe(true);
    expect(getLastAppliedSeq(1)).toBe(10);
    commitAppliedSequence(1, 11);
    expect(getLastAppliedSeq(1)).toBe(11);
  });

  it('détecte un trou sans avancer le curseur', () => {
    setSnapshotCursor(1, 1);
    commitAppliedSequence(1, 2);
    const gap = inspectSequence(1, 5);
    expect(gap.accept).toBe(true);
    expect(gap.gapDetected).toBe(true);
    expect(isResyncRequired(1)).toBe(true);
    expect(getLastAppliedSeq(1)).toBe(2);
  });

  it('rejette les événements sans event_seq valide', () => {
    setSnapshotCursor(1, 100);
    expect(inspectSequence(1, null).accept).toBe(false);
    expect(inspectSequence(1, 0).accept).toBe(false);
  });

  it('mode dégradé si snapshot_cursor null', () => {
    setSnapshotCursor(1, null);
    expect(isRealtimeDegraded(1)).toBe(true);
    expect(inspectSequence(1, 1).accept).toBe(false);
  });

  it('flush contig nécessite tous les seq jusqu’à subscribed_cursor', () => {
    setSnapshotCursor(1, 2);
    setSubscribedCursor(1, 4);
    bufferRealtimeEvent(1, 3, { a: 1 });
    expect(drainContiguousBuffer(1).complete).toBe(false);
    bufferRealtimeEvent(1, 4, { a: 2 });
    const drained = drainContiguousBuffer(1);
    expect(drained.complete).toBe(true);
    expect(drained.events).toHaveLength(2);
  });

  it('clearResyncAfterBootstrapSuccess remet l’état', () => {
    setSnapshotCursor(1, 1);
    inspectSequence(1, 5);
    expect(isResyncRequired(1)).toBe(true);
    clearResyncAfterBootstrapSuccess(1);
    expect(isResyncRequired(1)).toBe(false);
  });

  it('evaluateRealtimeSequence (compat) commit seulement sans trou', () => {
    setSnapshotCursor(1, 1);
    expect(evaluateRealtimeSequence(1, 2).accept).toBe(true);
    expect(getLastAppliedSeq(1)).toBe(2);
  });
});
