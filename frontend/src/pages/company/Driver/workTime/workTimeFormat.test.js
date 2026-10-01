import {
  activityLines,
  aggregateDisplayedDays,
  contractualCopy,
  dayMetrics,
  driverNarrative,
  entryDuration,
  entryNeedsReview,
  reviewItemsForMode,
  reviewReason,
  fleetSentence,
  narrativeSource,
  adjustmentFormDefaults,
  formatClock,
  formatDateTimeLocal,
  formatMinutes,
  formatRouteKm,
  qualityBadge,
  routeProposalCopy,
  segmentsLabel,
  totalMinutes,
  workedHints,
} from './workTimeFormat';

describe('formatMinutes', () => {
  it('affiche des heures et minutes', () => {
    expect(formatMinutes(0)).toBe('0h00');
    expect(formatMinutes(40)).toBe('0h40');
    expect(formatMinutes(80)).toBe('1h20');
    expect(formatMinutes(null)).toBe('—');
  });

  it('affiche l’heure suisse proposée pour une validation', () => {
    expect(formatClock('2026-09-28T07:40:00Z')).toBe('09:40');
    expect(formatClock('2026-09-28T08:00:00Z')).toBe('10:00');
  });
});

describe('rectification d’horaire', () => {
  it('préremplit la date de la course quand l’arrivée manque', () => {
    expect(formatDateTimeLocal('2026-09-29T21:15:00Z')).toBe('2026-09-29T23:15');
    expect(
      adjustmentFormDefaults({
        date: '2026-09-29',
        effective_arrived_at: null,
        effective_completed_at: '2026-09-29T21:15:00Z',
      })
    ).toEqual({
      arrived: '2026-09-29T23:15',
      completed: '2026-09-29T23:15',
      arrivalRecorded: false,
    });
  });

  it('conserve l’arrivée déjà enregistrée', () => {
    expect(
      adjustmentFormDefaults({
        date: '2026-09-30',
        effective_arrived_at: '2026-09-29T22:13:00Z',
        effective_completed_at: '2026-09-29T22:31:00Z',
      })
    ).toEqual({
      arrived: '2026-09-30T00:13',
      completed: '2026-09-30T00:31',
      arrivalRecorded: true,
    });
  });
});

describe('proposition de trajet', () => {
  const entry = {
    routing_status: 'ok',
    route_minutes: 12,
    margin_minutes: 5,
    proposed_worked_minutes: 17,
    route_distance_m: 4236,
  };

  it('sépare le temps de trajet estimé des minutes offertes', () => {
    expect(formatRouteKm(4236)).toBe('4,2 km');
    const copy = routeProposalCopy(entry);
    expect(copy.proposedLabel).toBe('17 min proposées');
    expect(copy.reference).toBe('Temps de trajet estimé : 12 min');
    expect(copy.trace).toBe('Voiture · 4,2 km');
    expect(copy.margin).toBe('Offertes au chauffeur : +5 min');
    expect(copy.facts).toEqual([
      ['Temps de trajet estimé', '12 min'],
      ['Distance routière', '4,2 km'],
      ['Minutes offertes par course', '+5 min'],
      ['Temps proposé', '17 min'],
    ]);
  });

  it('n’invente pas de libellé si la route OSRM est indisponible', () => {
    expect(routeProposalCopy({ routing_status: 'unavailable', route_minutes: 6 })).toBeNull();
  });
});

describe('affichage transports', () => {
  it('distingue le total réel du forfait et la règle de clôture', () => {
    const source = {
      real_transport_minutes: 40,
      flat_transport_minutes: 60,
      real_added_minutes: 20,
      flat_added_minutes: 0,
    };
    expect(totalMinutes(source, 'real')).toBe(60);
    expect(totalMinutes(source, 'flat')).toBe(60);
    expect(contractualCopy({
      versions: [{ mode: 'flat_per_trip', transport_flat_minutes: 30 }],
    })).toBe('Règle contractuelle : Forfait par transport · 30 min');
    expect(contractualCopy({ versions: [{ mode: 'flat_per_trip' }, { mode: 'real_time' }] }))
      .toBe('Règle contractuelle : plusieurs versions sur cette période');
    expect(entryDuration({
      kind: 'manual',
      real_minutes: 20,
      flat_status: 'requires_review',
      flat_minutes: null,
    }, 'real', false)).toBe(20);
    expect(entryDuration({
      kind: 'manual',
      real_minutes: 20,
      flat_status: 'requires_review',
      flat_minutes: null,
    }, 'flat', false)).toBeNull();
  });
});

describe('invariant phrase et journées', () => {
  const verified = (minutes) => ({
    kind: 'transport',
    real_minutes: minutes,
    worked_minutes: minutes,
    flat_status: 'calculated',
    flat_minutes: 30,
    work_time_status: 'verified',
  });

  it('reprend la somme exacte des journées affichées', () => {
    const days = [
      { date: '2026-09-28', entries: [verified(18)] },
      {
        date: '2026-09-30',
        entries: [
          verified(18),
          {
            kind: 'manual',
            is_manual: true,
            real_minutes: 60,
            worked_minutes: 60,
            flat_status: 'calculated',
            flat_minutes: 30,
          },
        ],
      },
    ];
    const aggregate = aggregateDisplayedDays(days, 'real', false);
    const rows = days.map((day) => dayMetrics(day, 'real', false));
    expect(aggregate.transport_count).toBe(rows.reduce((sum, row) => sum + row.transportCount, 0));
    expect(aggregate.transport_minutes).toBe(rows.reduce((sum, row) => sum + row.transportMinutes, 0));
    expect(aggregate.added_minutes).toBe(rows.reduce((sum, row) => sum + row.addedMinutes, 0));
    expect(aggregate.transport_minutes).toBe(36);
    const text = driverNarrative('Léa', narrativeSource(aggregate, 'real'), 'real');
    expect(text).toContain('2 transports');
    expect(text).toContain('0h36');
    expect(text).toContain('1h00');
    expect(text).not.toContain('0h14');
  });

  it('ne cite pas une durée partielle quand une journée est à vérifier', () => {
    const days = [
      { entries: [verified(18)] },
      {
        entries: [{
          kind: 'transport',
          real_minutes: null,
          work_time_status: 'pending_validation',
          flat_status: 'calculated',
          flat_minutes: 30,
        }],
      },
    ];
    const aggregate = aggregateDisplayedDays(days, 'real', false);
    expect(aggregate.transport_count).toBe(2);
    expect(aggregate.transport_pending).toBe(true);
    const text = driverNarrative('Léa', narrativeSource(aggregate, 'real'), 'real');
    expect(text).toContain('2 transports');
    expect(text).toContain('reste à vérifier');
    expect(text).not.toContain('0h18');
    expect(text).not.toContain('0h00');
  });

  it('aligne le compteur, l’état et une durée inconnue', () => {
    const unresolved = {
      kind: 'transport',
      real_minutes: 14,
      worked_minutes: 14,
      compensated_minutes: 0,
      review_modes: ['real', 'flat'],
      flat_status: 'calculated',
      flat_minutes: 30,
      work_time_status: 'validated_estimate',
    };
    const manual = {
      kind: 'manual',
      is_manual: true,
      real_minutes: 30,
      worked_minutes: 30,
      compensated_minutes: 30,
      flat_status: 'calculated',
      flat_minutes: 30,
    };
    const days = [
      { date: '2026-09-28', entries: [unresolved] },
      { date: '2026-09-30', entries: [manual] },
    ];
    const open = dayMetrics(days[0], 'real', true);
    expect(open.transportCount).toBe(1);
    expect(open.statusLabel).toBe('À vérifier');
    expect(open.transportLabel).toBe('À vérifier');
    expect(open.transportLabel).not.toBe('0h00');
    expect(entryDuration(unresolved, 'real', true)).toBeNull();
    const aggregate = aggregateDisplayedDays(days, 'real', true);
    expect(aggregate.review_count).toBe(1);
    expect(aggregate.transport_pending).toBe(true);
    expect(aggregate.added_minutes).toBe(30);
    const text = driverNarrative('Emmenez MOI', narrativeSource(aggregate, 'real'), 'real');
    expect(text).toContain('1 transport');
    expect(text).toContain('Le temps de ce transport reste à vérifier');
    expect(text).toContain('0h30');
    expect(text).not.toContain('0h00');
    expect(text).not.toContain('représentant');

    const resolved = { ...unresolved, review_modes: [], compensated_minutes: 14 };
    expect(entryNeedsReview(resolved, 'real')).toBe(false);
    expect(dayMetrics({ entries: [resolved] }, 'real', true).statusLabel).toBe('Validé');
    expect(aggregateDisplayedDays([{ entries: [resolved] }, days[1]], 'real', true).review_count).toBe(0);
  });

  it('n’affiche jamais 0h00 quand la durée n’existe pas', () => {
    const entry = {
      kind: 'transport',
      real_minutes: null,
      worked_minutes: null,
      compensated_minutes: 0,
      work_time_status: 'incomplete',
    };
    const metrics = dayMetrics({ entries: [entry] }, 'real', false);
    expect(metrics.transportCount).toBe(1);
    expect(metrics.statusLabel).toBe('À vérifier');
    expect(metrics.transportLabel).toBe('À vérifier');
    expect(String(metrics.transportLabel)).not.toBe('0h00');
    expect(String(metrics.totalLabel)).not.toBe('0h00');
  });

  it('la file contient exactement les éléments du compteur', () => {
    const items = Array.from({ length: 5 }, (_, index) => ({
      driver_id: index + 1,
      modes: ['real', 'flat'],
      reasons: ['journey_split_across_drivers'],
      date: '2026-09-28',
      pickup_label: 'HUG',
      dropoff_label: 'Pictet',
    }));
    expect(reviewItemsForMode(items, 'real')).toHaveLength(5);
    expect(reviewItemsForMode(items, 'flat')).toHaveLength(5);
    expect(reviewReason(items[0])).toBe('Trajet réparti entre plusieurs chauffeurs');
    expect(reviewItemsForMode([{ modes: ['flat'] }], 'real')).toHaveLength(0);
  });
});

describe('phrases', () => {
  const source = {
    transport_count: 2,
    real_transport_minutes: 14,
    real_added_minutes: 30,
    flat_transport_minutes: 60,
    flat_added_minutes: 0,
  };

  it('résume la flotte et le chauffeur', () => {
    expect(fleetSentence(source, 'real')).toBe('2 transports · 0h14 transport · +0h30 ajouté = 0h44');
    expect(driverNarrative('Léa', source, 'real')).toBe(
      'Durant cette période, Léa a effectué 2 transports représentant 0h14 de temps de transport, auxquels s’ajoutent 0h30 de temps supplémentaire.',
    );
    expect(fleetSentence(source, 'flat')).toBe('2 transports · 1h00 transport = 1h00');
  });
});

describe('libellés', () => {
  it('distingue segments et trajets forfaitaires', () => {
    const label = segmentsLabel({
      completed_segments_count: 3,
      incomplete_segments_count: 1,
      compensated_journeys_count: 1,
      pending_journeys_count: 1,
    });
    expect(label.segments).toBe('3 segments réalisés');
    expect(label.journeys).toBe('1 trajet forfaitaire');
    expect(qualityBadge('estimated_historical')).toBe('≈ Estimé');
  });

  it('résume l’activité sans afficher les zéros secondaires', () => {
    expect(activityLines({
      compensated_journeys_count: 1,
      completed_segments_count: 3,
    }).main).toBe('1 trajet · 3 segments');
    expect(activityLines({
      incomplete_segments_count: 6,
    })).toEqual({ main: '0 trajet', extra: ['6 incomplets'] });
    expect(workedHints({
      worked_verified_minutes: 40,
      worked_estimated_minutes: 60,
      manual_minutes: 0,
      worked_adjusted_minutes: 0,
    })).toEqual(['dont ≈ 1h00 estimées']);
  });
});
