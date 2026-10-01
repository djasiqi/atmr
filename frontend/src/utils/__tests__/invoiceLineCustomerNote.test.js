import {
  CANONICAL_CLIENT_VISIBLE_FIELD,
  collectCustomerVisibleNotes,
  collectCustomerVisibleNotesForPreview,
  normalizeCustomerVisibleNote,
} from '../invoiceLineCustomerNote';

describe('invoiceLineCustomerNote', () => {
  it('utilise adjustment_note comme champ canonique', () => {
    expect(CANONICAL_CLIENT_VISIBLE_FIELD).toBe('adjustment_note');
  });

  it('ignore vide et espaces', () => {
    expect(normalizeCustomerVisibleNote(null)).toBeNull();
    expect(normalizeCustomerVisibleNote('')).toBeNull();
    expect(normalizeCustomerVisibleNote('   ')).toBeNull();
  });

  it('conserve les notes distinctes dans l’ordre, sans doublon exact', () => {
    expect(
      collectCustomerVisibleNotes([
        { adjustment_note: 'Note A' },
        { adjustment_note: 'Note A' },
        { adjustment_note: 'Note B' },
        { notes_medical: 'INTERNE' },
      ]),
    ).toEqual(['Note A', 'Note B']);
  });

  it('affiche la note du retour sur la primaire (paire)', () => {
    const primary = {
      id: 1,
      reservation_id: 10,
      adjustment_note: '',
      line_meta: { round_trip_merge_partner_reservation_id: 11 },
    };
    const retour = {
      id: 2,
      reservation_id: 11,
      adjustment_note: 'Note uniquement retour',
    };
    expect(collectCustomerVisibleNotesForPreview(primary, [primary, retour])).toEqual([
      'Note uniquement retour',
    ]);
  });

  it('conserve une note sur ligne à 0.00 CHF', () => {
    expect(
      collectCustomerVisibleNotes([
        { adjustment_note: 'Annuler - Reservation non justifiable', line_total: 0 },
      ]),
    ).toEqual(['Annuler - Reservation non justifiable']);
  });

  it('ne lit pas les champs internes', () => {
    expect(
      collectCustomerVisibleNotes([
        {
          notes_medical: 'Allergie',
          internal_note: 'ops',
          invoice_note: 'interne',
          line_note: 'interne',
        },
      ]),
    ).toEqual([]);
  });
});
