/** Modèle d'affichage du parcours, aligné sur route_steps. */

export function createRouteKey() {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID();
  }
  return `step-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

export function createPickupPoint() {
  return {
    key: createRouteKey(),
    role: 'pickup',
    location: '',
    lat: null,
    lon: null,
    accessNotes: '',
  };
}

export function createDestinationPoint(missionDate = '') {
  return {
    key: createRouteKey(),
    role: 'destination',
    location: '',
    lat: null,
    lon: null,
    arrivalDate: missionDate || '',
    arrivalTime: '',
    departureDate: missionDate || '',
    departureTime: '',
    destinationKind: 'medical',
    establishment: '',
    establishmentId: null,
    service: '',
    serviceId: null,
    doctor: '',
    accessNotes: '',
  };
}

export function createInitialRoute(missionDate = '') {
  return [createPickupPoint(), createDestinationPoint(missionDate)];
}

function asPickup(point) {
  return {
    key: point.key || createRouteKey(),
    role: 'pickup',
    location: point.location || '',
    lat: point.lat ?? null,
    lon: point.lon ?? null,
    accessNotes: point.accessNotes || '',
  };
}

function asDestination(point, missionDate = '') {
  return {
    key: point.key || createRouteKey(),
    role: 'destination',
    location: point.location || '',
    lat: point.lat ?? null,
    lon: point.lon ?? null,
    arrivalDate: point.arrivalDate || missionDate || '',
    arrivalTime: point.arrivalTime || '',
    departureDate: point.departureDate || missionDate || '',
    departureTime: point.departureTime || '',
    destinationKind: point.destinationKind === 'medical' ? 'medical' : 'other',
    establishment: point.establishment || '',
    establishmentId: point.establishmentId ?? null,
    service: point.service || '',
    serviceId: point.serviceId ?? null,
    doctor: point.doctor || '',
    accessNotes: point.accessNotes || '',
  };
}

/** L'index décide du rôle : 0 = départ, le reste = destinations. Le retour n'est pas dans la liste. */
export function reorderRoutePoints(points, from, to) {
  if (
    !Array.isArray(points) ||
    from === to ||
    from < 0 ||
    to < 0 ||
    from >= points.length ||
    to >= points.length
  ) {
    return points;
  }
  const next = [...points];
  const [moved] = next.splice(from, 1);
  next.splice(to, 0, moved);
  return next.map((point, index) => (index === 0 ? asPickup(point) : asDestination(point)));
}

export function segmentLabels(destinationCount, isRoundTrip) {
  const names = ['Départ'];
  const count = Math.max(0, Number(destinationCount) || 0);
  for (let index = 1; index <= count; index += 1) {
    names.push(`Destination ${index}`);
  }
  if (isRoundTrip) names.push('Retour');
  const labels = [];
  for (let index = 0; index < names.length - 1; index += 1) {
    labels.push(`${names[index]} → ${names[index + 1]}`);
  }
  return labels;
}
