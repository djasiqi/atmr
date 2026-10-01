import {
  ROUND_TRIP_LINE_STRUCTURE,
  attachedRoundTripBookingIds,
  canShowRoundTripLegExcludeActions,
  getRoundTripAuditLegs,
  invoiceLineClientArTag,
  invoiceLineHasBothRoundTripLegs,
  invoiceLineRepresentsFullRoundTrip,
  isAnyRoundTripLine,
  isSingleMergedRoundTripLine,
  lineEditorContextArTag,
  roundTripLineStructure,
  transportDescriptionsAreStrictReverse,
} from '../invoiceLineRoundTrip';

/** Cas réel (#5557) : réservation créée en A/R, une seule jambe facturée, drapeaux API posés. */
const FLAG_ONLY_SINGLE_LEG_LINE = {
  id: 5557,
  reservation_id: 5557,
  line_total: 20,
  line_meta: {
    billing_unit: 'round_trip',
    transport_type: 'A/R',
    is_round_trip_leg: true,
    primary_booking_id: 5557,
    booking_ids: [5557],
  },
};

/** Ligne fusionnée S1/S2 : aller + retour rattachés à la même ligne. */
const MERGED_BOTH_LEGS_LINE = {
  id: 5558,
  reservation_id: 101,
  line_total: 40,
  line_meta: {
    billing_unit: 'round_trip',
    transport_type: 'A/R',
    primary_booking_id: 101,
    booking_ids: [101, 202],
    round_trip_secondary_reservation_ids: [202],
  },
};

/** Paire deux lignes (enrichissement API) : aller + retour sur deux lignes distinctes. */
const PAIR_PRIMARY_LINE = {
  id: 1,
  reservation_id: 301,
  description: 'Trajet Domicile → Hôpital',
  line_meta: { round_trip_merge_partner_reservation_id: 302, is_round_trip_leg: true },
};
const PAIR_RETURN_LINE = {
  id: 2,
  reservation_id: 302,
  description: 'Trajet Hôpital → Domicile',
  line_meta: {
    preview_hide_merged_round_trip: true,
    round_trip_merge_primary_reservation_id: 301,
  },
};

describe('information (badge A/R) vs structure (droits)', () => {
  it('le badge A/R reste informatif sur une ligne mono-réservation…', () => {
    expect(isSingleMergedRoundTripLine(FLAG_ONLY_SINGLE_LEG_LINE)).toBe(true);
    expect(isAnyRoundTripLine(FLAG_ONLY_SINGLE_LEG_LINE)).toBe(true);
    expect(lineEditorContextArTag(FLAG_ONLY_SINGLE_LEG_LINE)).toBe('A/R');
  });

  it('…mais ne donne aucun droit : structure SINGLE, pas de découpage', () => {
    expect(invoiceLineHasBothRoundTripLegs(FLAG_ONLY_SINGLE_LEG_LINE)).toBe(false);
    expect(roundTripLineStructure(FLAG_ONLY_SINGLE_LEG_LINE)).toBe(
      ROUND_TRIP_LINE_STRUCTURE.SINGLE
    );
    expect(canShowRoundTripLegExcludeActions(FLAG_ONLY_SINGLE_LEG_LINE)).toBe(false);
    expect(getRoundTripAuditLegs(FLAG_ONLY_SINGLE_LEG_LINE)).toBeNull();
  });

  it('le prix ne joue aucun rôle : 40 CHF sans seconde réservation reste SINGLE', () => {
    const pricey = {
      ...FLAG_ONLY_SINGLE_LEG_LINE,
      line_total: 40,
      line_meta: { ...FLAG_ONLY_SINGLE_LEG_LINE.line_meta, booking_ids: [5557] },
    };
    expect(roundTripLineStructure(pricey)).toBe(ROUND_TRIP_LINE_STRUCTURE.SINGLE);
    expect(canShowRoundTripLegExcludeActions(pricey)).toBe(false);
  });
});

describe('attachedRoundTripBookingIds / invoiceLineHasBothRoundTripLegs', () => {
  it('lit booking_ids (≥ 2 entrées distinctes)', () => {
    expect(attachedRoundTripBookingIds(MERGED_BOTH_LEGS_LINE)).toEqual([101, 202]);
    expect(invoiceLineHasBothRoundTripLegs(MERGED_BOTH_LEGS_LINE)).toBe(true);
  });

  it('retombe sur reservation_id + réservation(s) secondaire(s) (méta héritée)', () => {
    expect(
      attachedRoundTripBookingIds({
        reservation_id: 7,
        line_meta: { round_trip_secondary_reservation_id: 8 },
      })
    ).toEqual([7, 8]);
    expect(
      attachedRoundTripBookingIds({
        reservation_id: 7,
        line_meta: { round_trip_secondary_reservation_ids: [8, 9] },
      })
    ).toEqual([7, 8, 9]);
  });

  it('exige reservation_id pour la résolution par réservations secondaires (comme le backend)', () => {
    expect(
      invoiceLineHasBothRoundTripLegs({
        reservation_id: null,
        line_meta: { round_trip_secondary_reservation_id: 8 },
      })
    ).toBe(false);
  });

  it('ignore les doublons et les identifiants invalides', () => {
    expect(
      invoiceLineHasBothRoundTripLegs({
        reservation_id: 7,
        line_meta: { billing_unit: 'round_trip', booking_ids: [7, 7, null, 'x'] },
      })
    ).toBe(false);
    expect(
      invoiceLineHasBothRoundTripLegs({
        reservation_id: 7,
        line_meta: { round_trip_secondary_reservation_ids: [7] },
      })
    ).toBe(false);
  });

  it('ne compte pas le partenaire d’une paire deux lignes comme rattaché à la ligne', () => {
    expect(invoiceLineHasBothRoundTripLegs(PAIR_PRIMARY_LINE)).toBe(false);
  });

  it('est faux après exclusion d’une jambe en aperçu période', () => {
    expect(
      invoiceLineHasBothRoundTripLegs({
        ...MERGED_BOTH_LEGS_LINE,
        line_meta: { ...MERGED_BOTH_LEGS_LINE.line_meta, period_preview_single_leg: 'outbound' },
      })
    ).toBe(false);
  });
});

describe('roundTripLineStructure', () => {
  it('MERGED_BOTH_LEGS pour une ligne fusionnée à deux réservations', () => {
    expect(roundTripLineStructure(MERGED_BOTH_LEGS_LINE)).toBe(
      ROUND_TRIP_LINE_STRUCTURE.MERGED_BOTH_LEGS
    );
  });

  it('PAIR_PRIMARY / PAIR_RETURN pour une paire deux lignes présente', () => {
    const all = [PAIR_PRIMARY_LINE, PAIR_RETURN_LINE];
    expect(roundTripLineStructure(PAIR_PRIMARY_LINE, all)).toBe(
      ROUND_TRIP_LINE_STRUCTURE.PAIR_PRIMARY
    );
    expect(roundTripLineStructure(PAIR_RETURN_LINE, all)).toBe(
      ROUND_TRIP_LINE_STRUCTURE.PAIR_RETURN
    );
    // Sans la liste des lignes : on fait confiance à la méta de paire.
    expect(roundTripLineStructure(PAIR_PRIMARY_LINE)).toBe(ROUND_TRIP_LINE_STRUCTURE.PAIR_PRIMARY);
  });

  it('SINGLE quand la méta de paire pointe vers une ligne absente de la facture', () => {
    expect(roundTripLineStructure(PAIR_PRIMARY_LINE, [PAIR_PRIMARY_LINE])).toBe(
      ROUND_TRIP_LINE_STRUCTURE.SINGLE
    );
    expect(roundTripLineStructure(PAIR_RETURN_LINE, [PAIR_RETURN_LINE])).toBe(
      ROUND_TRIP_LINE_STRUCTURE.SINGLE
    );
  });

  it('SINGLE pour une ligne sans méta', () => {
    expect(roundTripLineStructure({ reservation_id: 1 })).toBe(ROUND_TRIP_LINE_STRUCTURE.SINGLE);
  });
});

describe('canShowRoundTripLegExcludeActions', () => {
  it('affiche « sans retour / sans aller » pour un A/R fusionné à deux réservations', () => {
    expect(canShowRoundTripLegExcludeActions(MERGED_BOTH_LEGS_LINE)).toBe(true);
  });

  it('affiche les actions sur l’aller d’une paire deux lignes, pas sur le retour', () => {
    const all = [PAIR_PRIMARY_LINE, PAIR_RETURN_LINE];
    expect(canShowRoundTripLegExcludeActions(PAIR_PRIMARY_LINE, all)).toBe(true);
    expect(canShowRoundTripLegExcludeActions(PAIR_RETURN_LINE, all)).toBe(false);
  });

  it('masque les actions si le partenaire de paire n’est plus sur la facture', () => {
    expect(canShowRoundTripLegExcludeActions(PAIR_PRIMARY_LINE, [PAIR_PRIMARY_LINE])).toBe(false);
  });

  it('masque les actions sans booking_ids ni réservation secondaire', () => {
    expect(
      canShowRoundTripLegExcludeActions({
        reservation_id: 7,
        line_meta: { billing_unit: 'round_trip', transport_type: 'A/R' },
      })
    ).toBe(false);
  });

  it('accepte round_trip_secondary_reservation_id seul (méta héritée)', () => {
    expect(
      canShowRoundTripLegExcludeActions({
        reservation_id: 7,
        line_meta: { round_trip_secondary_reservation_id: 8 },
      })
    ).toBe(true);
  });

  it('masque les actions après exclusion d’une jambe en aperçu période', () => {
    expect(
      canShowRoundTripLegExcludeActions({
        reservation_id: 7,
        line_meta: {
          billing_unit: 'round_trip',
          booking_ids: [7, 8],
          period_preview_single_leg: 'outbound',
        },
      })
    ).toBe(false);
  });
});

describe('getRoundTripAuditLegs', () => {
  it('expose les deux booking_id d’un A/R regroupé', () => {
    const legs = getRoundTripAuditLegs({
      reservation_id: 101,
      line_meta: {
        billing_unit: 'round_trip',
        primary_booking_id: 101,
        booking_ids: [101, 202],
        round_trip_merge_partner_reservation_id: 202,
        round_trip_primary_amount_ht: 40,
        round_trip_partner_amount_ht: 40,
      },
    });
    expect(legs).not.toBeNull();
    expect(legs.segmentsCount).toBe(2);
    expect(legs.outbound.bookingId).toBe(101);
    expect(legs.inbound.bookingId).toBe(202);
    expect(legs.outbound.amountHt).toBe(40);
    expect(legs.inbound.amountHt).toBe(40);
  });

  it('reconnaît un aller-retour miroir et refuse une chaîne', () => {
    expect(
      transportDescriptionsAreStrictReverse(
        'Trajet Domicile → Hôpital',
        'Trajet Hôpital → Domicile'
      )
    ).toBe(true);
    expect(
      transportDescriptionsAreStrictReverse(
        'Trajet Hôpitaux Universitaires de Genève (HUG) → Clinique de Joli-Mont',
        'Trajet Clinique de Joli-Mont → Avenue Ernest-Pictet 9'
      )
    ).toBe(false);
  });

  it('ne fusionne pas une ligne simple', () => {
    expect(
      getRoundTripAuditLegs({
        reservation_id: 7,
        line_meta: { billing_unit: 'single', booking_ids: [7] },
      })
    ).toBeNull();
  });
});

/** EM-2026-09-0065 : 02.09 aller simple (flag historique) + 03.09 A/R fusionné. */
const EM_0209_SINGLE = {
  id: 65,
  reservation_id: 2002,
  line_total: 45,
  description: 'Trajet Chem. du Val-de-Travers 12, Versoix → HUG',
  line_meta: {
    billing_unit: 'round_trip',
    transport_type: 'A/R',
    is_round_trip_leg: true,
    booking_ids: [2002],
    service_date: '2026-09-02',
  },
};
const EM_0309_MERGED = {
  id: 66,
  reservation_id: 2003,
  line_total: 90,
  description:
    'Trajet Chem. du Val-de-Travers 12, Versoix → Place de la Diversité 3, Meyrin',
  line_meta: {
    billing_unit: 'round_trip',
    transport_type: 'A/R',
    booking_ids: [2003, 2004],
    round_trip_secondary_reservation_ids: [2004],
    service_date: '2026-09-03',
  },
};

describe('tag client [A/R] (parité HTML / PDF)', () => {
  it('aller simple réel : aucun A/R', () => {
    const line = { reservation_id: 1, line_total: 45, line_meta: { booking_ids: [1] } };
    expect(invoiceLineClientArTag(line)).toBeNull();
    expect(invoiceLineRepresentsFullRoundTrip(line)).toBe(false);
  });

  it('aller-retour fusionné : A/R', () => {
    expect(invoiceLineClientArTag(MERGED_BOTH_LEGS_LINE)).toBe('A/R');
    expect(invoiceLineRepresentsFullRoundTrip(MERGED_BOTH_LEGS_LINE)).toBe(true);
  });

  it('is_round_trip historique + une seule jambe : pas de A/R client (badge éditeur inchangé)', () => {
    expect(lineEditorContextArTag(FLAG_ONLY_SINGLE_LEG_LINE)).toBe('A/R');
    expect(invoiceLineClientArTag(FLAG_ONLY_SINGLE_LEG_LINE)).toBeNull();
  });

  it('deux réservations rattachées : A/R', () => {
    expect(invoiceLineRepresentsFullRoundTrip(MERGED_BOTH_LEGS_LINE)).toBe(true);
  });

  it('paire deux lignes : A/R sur la primaire seulement', () => {
    const all = [PAIR_PRIMARY_LINE, PAIR_RETURN_LINE];
    expect(invoiceLineClientArTag(PAIR_PRIMARY_LINE, all)).toBe('A/R');
    expect(invoiceLineClientArTag(PAIR_RETURN_LINE, all)).toBeNull();
  });

  it('partenaire absent : retombe en SINGLE, pas de A/R', () => {
    expect(invoiceLineClientArTag(PAIR_PRIMARY_LINE, [PAIR_PRIMARY_LINE])).toBeNull();
  });

  it('montant 45 avec deux jambes : la structure gagne', () => {
    const cheapMerged = { ...MERGED_BOTH_LEGS_LINE, line_total: 45 };
    expect(invoiceLineClientArTag(cheapMerged)).toBe('A/R');
  });

  it('montant 90 avec une seule jambe : la structure gagne', () => {
    const priceySingle = { ...FLAG_ONLY_SINGLE_LEG_LINE, line_total: 90 };
    expect(invoiceLineClientArTag(priceySingle)).toBeNull();
  });

  it('EM-2026-09-0065 : 02.09 sans A/R, 03.09 avec A/R', () => {
    const all = [EM_0209_SINGLE, EM_0309_MERGED];
    expect(invoiceLineClientArTag(EM_0209_SINGLE, all)).toBeNull();
    expect(invoiceLineClientArTag(EM_0309_MERGED, all)).toBe('A/R');
  });

  it('le champ API booléen est honoré s’il est présent', () => {
    expect(
      invoiceLineRepresentsFullRoundTrip({
        ...FLAG_ONLY_SINGLE_LEG_LINE,
        invoice_line_represents_full_round_trip: true,
      })
    ).toBe(true);
    expect(
      invoiceLineRepresentsFullRoundTrip({
        ...MERGED_BOTH_LEGS_LINE,
        invoice_line_represents_full_round_trip: false,
      })
    ).toBe(false);
  });
});
