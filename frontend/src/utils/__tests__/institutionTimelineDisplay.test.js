import {
  buildOperationalTimeline,
  consolidateTimelineApiEvents,
  formatRouteJourneyEvent,
  getTimelineDisplayEvent,
  isSameTimelineMinute,
} from '../institutionTimelineDisplay';

describe('institutionTimelineDisplay', () => {
  const baseTime = '2026-06-15T17:17:00+02:00';

  const buildContext = (apiEvents, request = null) => ({
    request,
    apiEvents,
    hasRequestConverted: apiEvents.some((e) => e.event_type === 'request_converted'),
    offerAccepted: apiEvents.find((e) => e.event_type === 'offer_accepted') || null,
    externalCarrierAssigned: apiEvents.find((e) => e.event_type === 'external_carrier_assigned') || null,
    hasRouteJourney: false,
  });

  it('isSameTimelineMinute compare la minute calendaire (pas la seconde)', () => {
    expect(isSameTimelineMinute('2026-06-15T17:03:02+02:00', '2026-06-15T17:03:58+02:00')).toBe(true);
    expect(isSameTimelineMinute('2026-06-15T17:03:00+02:00', '2026-06-15T17:04:00+02:00')).toBe(false);
  });

  it('masque request_created si envoi même minute (17:03:02 / 17:03:58)', () => {
    const events = [
      { id: 1, event_type: 'request_created', created_at: '2026-06-15T17:03:02+02:00', label: 'Demande créée' },
      { id: 2, event_type: 'request_sent', created_at: '2026-06-15T17:03:58+02:00', label: 'Demande envoyée' },
    ];
    const out = buildOperationalTimeline({ apiEvents: events });
    expect(out).toHaveLength(1);
    expect(out[0].event).toBe('Demande envoyée');
  });

  it('affiche création et envoi si minutes différentes', () => {
    const events = [
      { id: 1, event_type: 'request_created', created_at: '2026-06-15T17:03:00+02:00', label: 'Demande créée' },
      { id: 2, event_type: 'request_sent', created_at: '2026-06-15T17:04:00+02:00', label: 'Demande envoyée' },
    ];
    const out = buildOperationalTimeline({ apiEvents: events });
    expect(out.map((e) => e.event).sort()).toEqual(['Demande créée', 'Demande envoyée']);
  });

  it('affiche l\'auteur de l\'envoi via request_sent.actor_name', () => {
    const events = [
      { id: 1, event_type: 'request_sent', created_at: '2026-06-15T21:06:00+02:00', payload: { actor_name: 'Drin Jasiqi' } },
      { id: 2, event_type: 'offer_sent', created_at: '2026-06-15T21:06:05+02:00', payload: { company_name: 'A' } },
    ];
    const out = buildOperationalTimeline({ apiEvents: events });
    expect(out).toHaveLength(1);
    expect(out[0].event).toBe('Demande envoyée — Drin Jasiqi');
  });

  it('agrège plusieurs offer_sent en une Demande envoyée sans sous-titre', () => {
    const events = [
      { id: 1, event_type: 'offer_sent', created_at: '2026-06-15T17:03:00+02:00', payload: { company_name: 'A' } },
      { id: 2, event_type: 'offer_sent', created_at: '2026-06-15T17:03:10+02:00', payload: { company_name: 'B' } },
      { id: 3, event_type: 'offer_sent', created_at: '2026-06-15T17:03:20+02:00', payload: { company_name: 'C' } },
    ];
    const out = buildOperationalTimeline({ apiEvents: events });
    expect(out).toHaveLength(1);
    expect(out[0].event).toBe('Demande envoyée');
  });

  it('fusionne offer_accepted dans request_converted avec nom dynamique', () => {
    const events = [
      { id: 1, event_type: 'offer_accepted', created_at: baseTime, payload: { company_name: 'Transport X' } },
      { id: 2, event_type: 'request_converted', created_at: baseTime, payload: {} },
    ];
    const out = buildOperationalTimeline({
      apiEvents: events,
      request: { accepted_by_company: { name: 'Transport X' } },
    });
    expect(out).toHaveLength(1);
    expect(out[0].event).toBe('Réservation confirmée — Transport X');
  });

  it('masque booking_created si request_converted présent', () => {
    const events = [
      { id: 1, event_type: 'request_converted', created_at: baseTime, payload: { company_name: 'Transport X' } },
      { id: 2, event_type: 'booking_created', created_at: baseTime, label: 'Course créée' },
    ];
    const out = buildOperationalTimeline({ apiEvents: events });
    expect(out).toHaveLength(1);
    expect(out[0].event).toBe('Réservation confirmée — Transport X');
  });

  it('masque field_updated si parcours modifié à la même minute', () => {
    const t = '2026-06-15T17:04:00+02:00';
    const events = [
      {
        id: 1,
        event_type: 'field_updated',
        created_at: t,
        payload: { changed_fields: ['multi_stop'] },
      },
      {
        id: 2,
        event_type: 'route_legs_reorganized',
        created_at: t,
        payload: { after_legs: [{}, {}] },
      },
    ];
    const out = buildOperationalTimeline({ apiEvents: events });
    expect(out).toHaveLength(1);
    expect(out[0].event).toBe('Parcours modifié');
  });

  it('affiche le nom de l\'acteur pour parcours/horaire modifié', () => {
    const events = [
      {
        id: 1,
        event_type: 'route_legs_reorganized',
        created_at: '2026-06-15T17:04:00+02:00',
        payload: { after_legs: [{}, {}], actor_name: 'Drin Jasiqi' },
      },
      {
        id: 2,
        event_type: 'field_updated',
        created_at: '2026-06-15T18:00:00+02:00',
        payload: { changed_fields: ['scheduled_time'], actor_name: 'Drin Jasiqi' },
      },
    ];
    const out = buildOperationalTimeline({ apiEvents: events });
    expect(out.map((e) => e.event)).toEqual([
      'Horaire modifié — Drin Jasiqi',
      'Parcours modifié — Drin Jasiqi',
    ]);
  });

  it('affiche Horaire modifié pour field_updated horaires seuls', () => {
    const events = [
      {
        id: 1,
        event_type: 'field_updated',
        created_at: '2026-06-15T12:00:00+02:00',
        payload: { changed_fields: ['scheduled_time'] },
      },
    ];
    const out = buildOperationalTimeline({ apiEvents: events });
    expect(out[0].event).toBe('Horaire modifié');
  });

  it('masque patient_boarded si route_journey présent', () => {
    const events = [
      { id: 1, event_type: 'patient_boarded', created_at: '2026-06-15T18:00:00+02:00' },
    ];
    const out = buildOperationalTimeline({
      apiEvents: events,
      bookingSummary: {
        route_journey: [
          {
            type: 'pickup',
            date: '2026-06-15T18:00:00+02:00',
            is_return: false,
            leg_count: 1,
          },
        ],
        is_round_trip: true,
      },
      request: { is_round_trip: true },
    });
    expect(out.some((e) => e.event === 'Patient pris en charge — Aller')).toBe(true);
    expect(out.filter((e) => e.event.includes('Patient pris en charge'))).toHaveLength(1);
    expect(out.some((e) => e.event === 'Patient pris en charge' && !e.event.includes('—'))).toBe(false);
  });

  it('affiche chauffeur secondary et masque driver_reassigned', () => {
    const events = [
      { id: 1, event_type: 'driver_assigned', created_at: '2026-06-15T17:20:00+02:00', payload: { driver_name: 'Khalid ALAOUI' } },
      { id: 2, event_type: 'driver_reassigned', created_at: '2026-06-15T17:25:00+02:00', payload: { driver_name: 'Ahmed X' } },
    ];
    const out = buildOperationalTimeline({ apiEvents: events });
    expect(out.map((e) => e.event)).toEqual(['Chauffeur assigné — Khalid ALAOUI']);
  });

  it('conserve ordre chronologique si chauffeur assigné après prise en charge', () => {
    const events = [
      { id: 1, event_type: 'driver_assigned', created_at: '2026-06-15T18:05:00+02:00', payload: { driver_name: 'Khalid' } },
    ];
    const out = buildOperationalTimeline({
      apiEvents: events,
      bookingSummary: {
        route_journey: [
          {
            type: 'pickup',
            date: '2026-06-15T18:00:00+02:00',
            is_return: false,
            leg_count: 1,
          },
        ],
        is_round_trip: true,
      },
      request: { is_round_trip: true },
    });
    expect(out[0].event).toBe('Chauffeur assigné — Khalid');
    expect(out[1].event).toBe('Patient pris en charge — Aller');
  });

  it('projette transporteur externe avec nom propagé à la clôture', () => {
    const events = [
      {
        id: 1,
        event_type: 'external_carrier_assigned',
        created_at: '2026-06-15T08:15:00+02:00',
        payload: { carrier_name: 'Taxi Dupont' },
      },
      {
        id: 2,
        event_type: 'external_mission_completed',
        created_at: '2026-06-15T11:30:00+02:00',
        payload: {},
      },
    ];
    const out = buildOperationalTimeline({ apiEvents: events });
    expect(out.map((e) => e.event)).toEqual([
      'Mission terminée — Taxi Dupont',
      'Transporteur externe affecté — Taxi Dupont',
    ]);
  });

  it('formatRouteJourneyEvent — aller-retour', () => {
    expect(formatRouteJourneyEvent({ type: 'pickup', is_return: false, leg_count: 1 }, { is_round_trip: true }))
      .toEqual({ label: 'Patient pris en charge — Aller', type: 'pickup' });
    expect(formatRouteJourneyEvent({ type: 'dropoff', is_return: false, leg_count: 1 }, { is_round_trip: true }))
      .toEqual({ label: 'Patient déposé — Destination', type: 'dropoff' });
    expect(formatRouteJourneyEvent({ type: 'pickup', is_return: true, leg_count: 1 }, { is_round_trip: true }))
      .toEqual({ label: 'Patient repris en charge — Retour', type: 'pickup' });
    expect(formatRouteJourneyEvent({ type: 'dropoff', is_return: true, leg_count: 1 }, { is_round_trip: true }))
      .toEqual({ label: 'Retour terminé — Institution', type: 'dropoff' });
  });

  it('formatRouteJourneyEvent — multi-stop', () => {
    expect(formatRouteJourneyEvent({
      type: 'dropoff', leg_index: 2, is_final_leg: true, leg_count: 2, is_return: false,
    }, {}))
      .toEqual({ label: 'Transport terminé — Destination finale', type: 'dropoff' });
    expect(formatRouteJourneyEvent({
      type: 'pickup', leg_index: 1, leg_count: 2, is_return: false,
    }, {}))
      .toEqual({ label: 'Prise en charge — Étape 1', type: 'pickup' });
  });

  it('enrichit annulation depuis booking_summary', () => {
    const events = [
      { id: 1, event_type: 'cancelled', created_at: '2026-06-15T19:00:00+02:00', payload: {} },
    ];
    const out = buildOperationalTimeline({
      apiEvents: events,
      bookingSummary: { cancellation_display_label: 'Patient indisponible' },
    });
    expect(out[0].event).toBe('Transport annulé — Patient indisponible');
    expect(out[0].type).toBe('cancel');
  });

  it('affiche le motif et l\'auteur de l\'annulation', () => {
    const events = [
      {
        id: 1,
        event_type: 'cancelled',
        created_at: '2026-06-15T19:00:00+02:00',
        payload: {
          cancellation_display_label: 'Patient indisponible',
          actor_name: 'Drin Jasiqi',
        },
      },
    ];
    const out = buildOperationalTimeline({ apiEvents: events });
    expect(out[0].event).toBe('Transport annulé — Patient indisponible — Drin Jasiqi');
    expect(out[0].type).toBe('cancel');
  });

  it('n\'écrase pas le libellé d\'annulation enrichi via booking_summary', () => {
    const events = [
      {
        id: 1,
        event_type: 'cancelled',
        created_at: '2026-06-15T19:00:00+02:00',
        payload: {
          cancellation_display_label: 'Patient indisponible',
          actor_name: 'Drin Jasiqi',
        },
      },
    ];
    const out = buildOperationalTimeline({
      apiEvents: events,
      bookingSummary: { cancellation_display_label: 'Patient indisponible' },
    });
    expect(out[0].event).toBe('Transport annulé — Patient indisponible — Drin Jasiqi');
  });

  it('ajoute Transport annulé en repli si demande annulée sans événement timeline', () => {
    const events = [
      { id: 1, event_type: 'request_sent', created_at: '2026-06-15T17:29:00+02:00', payload: {} },
    ];
    const out = buildOperationalTimeline({
      apiEvents: events,
      request: {
        status: 'cancelled',
        cancelled_at: '2026-06-15T18:00:00+02:00',
        cancel_reason: 'Plus besoin',
      },
    });
    expect(out[0].event).toBe('Transport annulé — Plus besoin');
    expect(out[0].type).toBe('cancel');
    expect(out.some((e) => e.event === 'Demande envoyée')).toBe(true);
  });

  it('consolidateTimelineApiEvents délègue à buildOperationalTimeline', () => {
    const events = [
      { id: 1, event_type: 'request_converted', created_at: baseTime, payload: { company_name: 'X' } },
    ];
    expect(consolidateTimelineApiEvents(events)[0].event).toBe('Réservation confirmée — X');
  });

  it('getTimelineDisplayEvent retourne null pour types techniques masqués', () => {
    const ctx = buildContext([]);
    const send = null;
    expect(getTimelineDisplayEvent({ event_type: 'status_changed', created_at: baseTime }, ctx, send)).toBeNull();
    expect(getTimelineDisplayEvent({ event_type: 'driver_reassigned', created_at: baseTime, payload: { driver_name: 'X' } }, ctx, send)).toBeNull();
  });

  it('affiche demande + acceptation de modification (aligné transporteur)', () => {
    const events = [
      { id: 1, event_type: 'request_sent', created_at: '2026-07-21T22:23:00+02:00', payload: { actor_name: 'Drin Jasiqi' } },
      {
        id: 2,
        event_type: 'request_converted',
        created_at: '2026-07-21T22:24:00+02:00',
        payload: { company_name: 'Emmenez Moi' },
      },
      {
        id: 3,
        event_type: 'change_confirmation_requested',
        created_at: '2026-07-21T22:28:00+02:00',
        payload: {
          actor_name: 'Drin Jasiqi',
          changed_fields: { scheduled_time: true },
          proposed_patch: { scheduled_time: '2026-07-22T11:15:00' },
        },
      },
      {
        id: 4,
        event_type: 'change_accepted_by_company',
        created_at: '2026-07-21T22:30:00+02:00',
        payload: { action_type: 'CHANGE_TIME' },
      },
      {
        id: 5,
        event_type: 'field_updated',
        created_at: '2026-07-21T22:30:10+02:00',
        payload: { changed_fields: { scheduled_time: true } },
      },
    ];
    const out = buildOperationalTimeline({ apiEvents: events });
    expect(out.map((e) => e.event)).toEqual([
      'Modification acceptée par le transporteur',
      'Modification institution — Drin Jasiqi',
      'Réservation confirmée — Emmenez Moi',
      'Demande envoyée — Drin Jasiqi',
    ]);
  });

  it('affiche demande d’annulation institution + confirmation transporteur', () => {
    const events = [
      {
        id: 1,
        event_type: 'change_confirmation_requested',
        created_at: '2026-07-21T20:40:00+02:00',
        payload: {
          actor_name: 'Drin Jasiqi',
          proposed_patch: { _cancellation: true },
        },
      },
      {
        id: 2,
        event_type: 'change_accepted_by_company',
        created_at: '2026-07-21T22:17:00+02:00',
        payload: { action_type: 'CANCELLATION' },
      },
    ];
    const out = buildOperationalTimeline({ apiEvents: events });
    expect(out.map((e) => e.event)).toEqual([
      'Annulation confirmée par le transporteur',
      'Demande d’annulation institution — Drin Jasiqi',
    ]);
  });

  it('ne garde qu’une seule demande d’annulation institution en cas de doublons', () => {
    const events = [
      {
        id: 1,
        event_type: 'change_confirmation_requested',
        created_at: '2026-07-29T10:11:00+02:00',
        payload: {
          actor_name: 'User #91596',
          proposed_patch: { _cancellation: true },
        },
      },
      {
        id: 2,
        event_type: 'change_confirmation_requested',
        created_at: '2026-07-29T10:12:00+02:00',
        payload: {
          actor_name: 'User #91596',
          proposed_patch: { _cancellation: true },
        },
      },
    ];
    const out = buildOperationalTimeline({ apiEvents: events });
    const cancelRequests = out.filter((e) => (
      String(e.event || '').startsWith('Demande d’annulation institution')
    ));
    expect(cancelRequests).toHaveLength(1);
    expect(cancelRequests[0].event).toBe(
      'Demande d’annulation institution — User #91596',
    );
  });
});
