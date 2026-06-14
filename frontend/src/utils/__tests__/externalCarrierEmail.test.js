import {
  buildCarrierEmail,
  buildCarrierMailto,
  formatMissionDate,
  getRequesterContact,
  getRequestNumber,
  shouldDisplayReference,
} from '../externalCarrierEmail';

const baseRequest = {
  id: 42,
  external_reference: 'HUG-2026-551',
  mission_date: '2026-06-14',
  scheduled_time: '2026-06-14T09:00:00+02:00',
  pickup_time_confirmed: true,
  patient: { first_name: 'Khalid', last_name: 'ALAOUI', dob: '1945-03-28' },
  contact_on_site: {
    requester_name: 'Marc Mouchet',
    requester_phone: '076 000 00 00',
    requester_service: 'Admissions',
  },
  legs: [
    {
      sequence_index: 0,
      pickup_location: 'Chemin des Courbes 9, 1247, Anières',
      dropoff_location: 'HUG, Rue Gabrielle-Perret-Gentil 4, 1205, Genève',
      scheduled_time: '2026-06-14T10:00:00+02:00',
      time_confirmed: true,
    },
  ],
  external_carrier: { reason: 'Dépannage' },
};

describe('formatMissionDate', () => {
  it('formate YYYY-MM-DD sans new Date', () => {
    expect(formatMissionDate('2026-06-14')).toBe('14.06.2026');
    expect(formatMissionDate('2026-06-14T00:00:00Z')).toBe('14.06.2026');
  });

  it('retourne une chaîne vide pour une valeur invalide', () => {
    expect(formatMissionDate(null)).toBe('');
    expect(formatMissionDate('invalid')).toBe('');
  });
});

describe('getRequesterContact', () => {
  it('retourne des chaînes vides si contact_on_site est null', () => {
    expect(getRequesterContact({ contact_on_site: null })).toEqual({
      requester_name: '',
      requester_phone: '',
      requester_service: '',
    });
  });

  it('retourne des chaînes vides si contact_on_site est {}', () => {
    expect(getRequesterContact({ contact_on_site: {} })).toEqual({
      requester_name: '',
      requester_phone: '',
      requester_service: '',
    });
  });

  it('lit les champs requester', () => {
    expect(getRequesterContact(baseRequest)).toEqual({
      requester_name: 'Marc Mouchet',
      requester_phone: '076 000 00 00',
      requester_service: 'Admissions',
    });
  });
});

describe('getRequestNumber / shouldDisplayReference', () => {
  it('privilégie external_reference', () => {
    expect(getRequestNumber(baseRequest)).toBe('HUG-2026-551');
  });

  it('retombe sur id si pas de external_reference', () => {
    expect(getRequestNumber({ id: 99 })).toBe(99);
  });

  it('n\'affiche pas Référence si déjà utilisée comme N° de demande', () => {
    expect(shouldDisplayReference({ id: 42, external_reference: 'HUG-2026-551' })).toBe(false);
    expect(shouldDisplayReference({ id: 42, external_reference: 'DPI-2024-999' })).toBe(false);
    expect(shouldDisplayReference({ id: 42 })).toBe(false);
  });
});

describe('buildCarrierEmail', () => {
  it('produit un corps concis : patient (dob), trajet à puces, contact, signature', () => {
    const { subject, body } = buildCarrierEmail(baseRequest, {
      institutionName: 'Clinique Les Hauts d\'Anières',
      institutionPhone: '022 000 00 00',
    });
    expect(subject).toBe(
      'Bon de transport #HUG-2026-551 — Khalid ALAOUI — Clinique Les Hauts d\'Anières — 14.06.2026',
    );
    expect(body).toContain('Veuillez trouver ci-joint le bon de transport concernant :');
    expect(body).toContain('Patient : Khalid ALAOUI (28.03.1945)');
    expect(body).toContain('Trajet :');
    expect(body).toContain('• Départ : Chemin des Courbes 9, 1247, Anières (départ à 09:00)');
    expect(body).toContain('• Destination : HUG, Rue Gabrielle-Perret-Gentil 4, 1205, Genève (rendez-vous à 10:00)');
    expect(body).toContain('Contact : 076 000 00 00');
    expect(body).toContain('Cordialement,');
    expect(body).toContain('Clinique Les Hauts d\'Anières');
  });

  it('reste minimal : pas de sections, type de trajet, LIRIE ni date de mission dans le corps', () => {
    const { body } = buildCarrierEmail(baseRequest, {
      institutionName: 'Clinique X',
      institutionPhone: '022 000 00 00',
    });
    expect(body).not.toContain('INFORMATIONS GÉNÉRALES');
    expect(body).not.toContain('CONTACT');
    expect(body).not.toContain('Type de trajet');
    expect(body).not.toContain('N° de demande');
    expect(body).not.toContain('Date de mission');
    expect(body).not.toContain('Mission transmise via la plateforme LIRIE');
    expect(body).not.toContain('téléchargé sur votre poste');
  });

  it('met la date de mission dans l\'objet sans décalage horaire', () => {
    const { subject } = buildCarrierEmail({
      mission_date: '2026-06-14',
      scheduled_time: '2026-06-14T00:00:00Z',
    });
    expect(subject).toContain('14.06.2026');
  });

  it('affiche la date de naissance entre parenthèses après le patient', () => {
    const { body } = buildCarrierEmail({
      id: 7,
      patient: { first_name: 'Eliane', last_name: 'STOFER', dob: '1945-03-28' },
    });
    expect(body).toContain('Patient : Eliane STOFER (28.03.1945)');
  });

  it('omet la date de naissance si absente', () => {
    const { body } = buildCarrierEmail({
      id: 7,
      patient: { first_name: 'Eliane', last_name: 'STOFER' },
    });
    expect(body).toContain('Patient : Eliane STOFER');
    expect(body).not.toContain('(');
  });

  it('retombe sur institutionPhone si pas de requester_phone', () => {
    const { body } = buildCarrierEmail(
      { id: 1, contact_on_site: null },
      { institutionPhone: '022 111 22 33' },
    );
    expect(body).toContain('Contact : 022 111 22 33');
  });

  it('liste le retour comme puce quand return_to_institution est vrai', () => {
    const { body } = buildCarrierEmail({
      ...baseRequest,
      return_to_institution: true,
      legs: [
        { sequence_index: 0, pickup_location: 'A', dropoff_location: 'B' },
        { sequence_index: 1, pickup_location: 'B', dropoff_location: 'A' },
      ],
    });
    expect(body).toContain('• Retour : A');
  });

  it('liste les étapes multi-stop comme puces', () => {
    const { body } = buildCarrierEmail({
      ...baseRequest,
      legs: [
        { sequence_index: 0, pickup_location: 'A', dropoff_location: 'B' },
        { sequence_index: 1, pickup_location: 'B', dropoff_location: 'C' },
      ],
    });
    expect(body).toContain('• Étape 1 : B');
    expect(body).toContain('• Étape 2 : C');
  });
});

describe('buildCarrierMailto', () => {
  it('encode le destinataire, l\'objet et le corps', () => {
    const href = buildCarrierMailto('ops@transporteur.ch', baseRequest, {
      institutionName: 'Clinique X',
    });
    expect(href.startsWith('mailto:ops%40transporteur.ch?')).toBe(true);
    expect(href).toContain('subject=');
    expect(href).toContain('body=');
    expect(href).toContain(encodeURIComponent('Bon de transport #HUG-2026-551'));
  });
});
