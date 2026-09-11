import {
  findMissingMedicalDestinationDetails,
  hasMissingMedicalDestinationDetails,
  hasServiceOrDoctor,
  inferDestinationType,
  isMedicalDestinationType,
  suggestDestinationTypeFromPlace,
} from '../institutionDestinationDetails';

describe('institutionDestinationDetails', () => {
  describe('hasServiceOrDoctor', () => {
    it('accepte service seul, médecin seul, ou les deux', () => {
      expect(hasServiceOrDoctor('Radiologie', '')).toBe(true);
      expect(hasServiceOrDoctor('  ', 'Dr Martin')).toBe(true);
      expect(hasServiceOrDoctor('Cardio', 'Dr Martin')).toBe(true);
    });

    it('refuse les deux vides', () => {
      expect(hasServiceOrDoctor('', '')).toBe(false);
      expect(hasServiceOrDoctor('   ', null)).toBe(false);
    });
  });

  describe('isMedicalDestinationType', () => {
    it('n’est vrai que pour medical explicite', () => {
      expect(isMedicalDestinationType('medical')).toBe(true);
      expect(isMedicalDestinationType('other')).toBe(false);
      expect(isMedicalDestinationType('')).toBe(false);
      expect(isMedicalDestinationType(undefined)).toBe(false);
    });
  });

  describe('suggestDestinationTypeFromPlace', () => {
    it('suggère medical uniquement via les types POI, pas le libellé', () => {
      expect(suggestDestinationTypeFromPlace({
        types: ['hospital'],
        name: 'HUG',
      })).toBe('medical');
      expect(suggestDestinationTypeFromPlace({
        types: ['restaurant'],
        name: 'Restaurant Les Armures',
      })).toBe('other');
      expect(suggestDestinationTypeFromPlace({
        name: 'Restaurant Les Armures',
      })).toBeNull();
    });
  });

  describe('inferDestinationType', () => {
    it('conserve un type explicite', () => {
      expect(inferDestinationType({ destinationType: 'other' })).toBe('other');
      expect(inferDestinationType({ destinationType: 'medical' })).toBe('medical');
    });

    it('ne transforme pas un type omis en médical sans service/médecin', () => {
      expect(inferDestinationType({})).toBe('other');
    });

    it('rétrocompat : service déjà saisi ⇒ médical', () => {
      expect(inferDestinationType({ service: 'Radiologie' })).toBe('medical');
    });
  });

  describe('findMissingMedicalDestinationDetails', () => {
    it('n’exige rien si le type est omis (legacy / other)', () => {
      const found = findMissingMedicalDestinationDetails({
        dropoffService: '',
        dropoffDoctor: '',
      });
      expect(found.principal).toBe(false);
    });

    it('n’exige rien pour other / restaurant / hôtel', () => {
      expect(findMissingMedicalDestinationDetails({
        destinationType: 'other',
        dropoffService: '',
        dropoffDoctor: '',
      }).principal).toBe(false);
    });

    it('exige service ou médecin uniquement si medical', () => {
      expect(findMissingMedicalDestinationDetails({
        destinationType: 'medical',
        dropoffService: '',
        dropoffDoctor: '',
      }).principal).toBe(true);
    });

    it('accepte service seul sur une destination medical', () => {
      const found = findMissingMedicalDestinationDetails({
        destinationType: 'medical',
        dropoffService: 'Radiologie',
        dropoffDoctor: '',
      });
      expect(found.principal).toBe(false);
      expect(hasMissingMedicalDestinationDetails({
        destinationType: 'medical',
        dropoffService: 'Radiologie',
      })).toBe(false);
    });

    it('n’exige rien pour un retour domicile', () => {
      const found = findMissingMedicalDestinationDetails({
        destinationType: 'domicile',
        dropoffService: '',
        dropoffDoctor: '',
      });
      expect(found.principal).toBe(false);
    });

    it('signale uniquement les étapes supplémentaires médicales incomplètes', () => {
      const found = findMissingMedicalDestinationDetails({
        destinationType: 'other',
        extraStops: [
          { dropoff_location: 'Restaurant', destination_type: 'other' },
          { dropoff_location: 'HUG', destination_type: 'medical' },
          { dropoff_location: 'Cabinet', destination_type: 'medical', dropoff_doctor: 'Dr X' },
        ],
      });
      expect(found.principal).toBe(false);
      expect(found.extraStopIndexes).toEqual([1]);
    });

    it('ignore les étapes sans adresse', () => {
      const found = findMissingMedicalDestinationDetails({
        destinationType: 'domicile',
        extraStops: [{ dropoff_location: '  ', destination_type: 'medical' }],
      });
      expect(found.extraStopIndexes).toEqual([]);
    });

    it('n’applique pas la règle aux livraisons matériel', () => {
      const found = findMissingMedicalDestinationDetails({
        missionType: 'material_delivery',
        destinationType: 'medical',
        extraStops: [{ dropoff_location: 'HUG', destination_type: 'medical' }],
      });
      expect(found.principal).toBe(false);
      expect(found.extraStopIndexes).toEqual([]);
    });
  });
});
