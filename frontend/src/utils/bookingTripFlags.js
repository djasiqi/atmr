/**
 * Dérive la liste de badges parcours depuis trip_flags API ou fallback legacy.
 */

export function resolveTripFlagsFromBooking(booking, routeGroupSizes = {}) {
  const flags = booking?.trip_flags;
  if (flags) {
    if (
      process.env.NODE_ENV !== 'production'
      && booking
      && booking.display_model !== 'booking'
    ) {
      // eslint-disable-next-line no-console
      console.warn(
        '[canonical-display] trip_flags sans display_model booking — fallback legacy',
        { id: booking?.id },
      );
    }
    return {
      roundTrip: Boolean(flags.round_trip),
      returnLeg: Boolean(flags.return_leg),
      multiStop: Boolean(flags.multi_stop),
      legNumber: flags.leg_number ?? null,
      legCount: flags.leg_count ?? null,
      transferred: Boolean(flags.transferred),
      changeRequestPending: Boolean(flags.change_request_pending),
    };
  }

  const gid = booking?.route_group_id;
  const legCount = gid ? (routeGroupSizes[gid] || 1) : 1;
  const isReturn = Boolean(booking?.is_return);
  const isRoundTripOutbound = !isReturn && (Boolean(booking?.is_round_trip) || Boolean(booking?.has_return));

  return {
    roundTrip: isRoundTripOutbound,
    returnLeg: isReturn,
    multiStop: Boolean(gid && legCount > 1),
    legNumber: booking?.route_sequence_number ?? null,
    legCount: gid && legCount > 1 ? legCount : null,
    transferred: Boolean(booking?.is_transferred),
    changeRequestPending: ['pending', 'escalation_required', 'expired'].includes(
      String(booking?.active_change_request?.status || '').toLowerCase()
    ),
  };
}

/** @returns {Array<{ key: string, label: string, title?: string, variant?: string }>} */
export function buildTripBadgeDescriptors(flags) {
  const badges = [];
  if (flags.roundTrip) {
    badges.push({
      key: 'round_trip',
      label: 'Aller-retour',
      title: 'Demande aller-retour : cette ligne est l’aller ; une course retour est liée.',
      variant: 'roundTrip',
    });
  }
  if (flags.returnLeg) {
    badges.push({
      key: 'return_leg',
      label: 'Retour',
      title: 'Course retour liée à l’aller (même dossier client).',
      variant: 'returnLeg',
    });
  }
  if (flags.multiStop && flags.legNumber && flags.legCount) {
    badges.push({
      key: 'multi_stop',
      label: `Trajet ${flags.legNumber}/${flags.legCount}`,
      title: 'Trajet d’un parcours multi-destinations.',
      variant: 'routeLeg',
    });
  }
  if (flags.transferred) {
    badges.push({
      key: 'transferred',
      label: 'Transférée',
      variant: 'transfer',
    });
  }
  return badges;
}
