import {
  buildInitialDestinations,
  extractAddressFromPlace,
  filterTripTypesForInstitution,
  routingDropoffDetails,
} from '../institutionRouteForm';

describe('institutionRouteForm', () => {
  describe('buildInitialDestinations', () => {
    it('initialise les destinations depuis les legs avec heures RDV', () => {
      const request = {
        return_to_institution: false,
        legs: [
          {
            sequence_index: 0,
            dropoff_location: 'HUG, Genève',
            dropoff_establishment: 'HUG',
            scheduled_time: '2026-06-15T20:00:00',
            time_confirmed: true,
            booking_id: 31006,
          },
        ],
      };
      const bs = {
        pickup_location: 'Anières',
        scheduled_time: '2026-06-15T19:00:00',
        medical_facility: 'HUG',
      };

      const dests = buildInitialDestinations(request, bs);
      expect(dests).toHaveLength(1);
      expect(dests[0].address).toBe('HUG, Genève');
      expect(dests[0].scheduled_time).toBe('2026-06-15T20:00:00');
      expect(dests[0].time_confirmed).toBe(true);
      expect(dests[0].booking_id).toBe(31006);
    });

    it('exclut le leg retour A/R', () => {
      const request = {
        return_to_institution: true,
        legs: [
          {
            sequence_index: 0,
            dropoff_location: 'HUG',
            scheduled_time: '2026-06-15T20:00:00',
            time_confirmed: true,
          },
          {
            sequence_index: 1,
            dropoff_location: 'Domicile',
            scheduled_time: '2026-06-15T22:00:00',
            time_confirmed: true,
          },
        ],
      };

      const dests = buildInitialDestinations(request);
      expect(dests).toHaveLength(1);
      expect(dests[0].address).toBe('HUG');
    });

    it('retombe sur booking_summary sans legs', () => {
      const request = { dropoff_location: 'Clinique' };
      const bs = {
        dropoff_location: 'HUG',
        medical_facility: 'HUG',
        hospital_service: 'Radio',
        doctor_name: 'Dr. Martin',
      };

      const dests = buildInitialDestinations(request, bs);
      expect(dests[0].address).toBe('HUG');
      expect(dests[0].establishment).toBe('HUG');
      expect(dests[0].service).toBe('Radio');
      expect(dests[0].doctor).toBe('Dr. Martin');
    });
  });

  describe('routingDropoffDetails', () => {
    it('lit les détails routing billing', () => {
      const request = {
        billing_details: {
          routing: {
            dropoff_establishment: 'Clinique',
            dropoff_service: 'Urgences',
            dropoff_doctor: 'Dr. X',
          },
        },
      };
      expect(routingDropoffDetails(request)).toEqual({
        establishment: 'Clinique',
        service: 'Urgences',
        doctor: 'Dr. X',
      });
    });
  });

  describe('extractAddressFromPlace', () => {
    it('préfère label puis address', () => {
      expect(extractAddressFromPlace({ label: 'Rue A 1' })).toBe('Rue A 1');
      expect(extractAddressFromPlace({ address: 'Rue B 2' })).toBe('Rue B 2');
    });
  });

  describe('filterTripTypesForInstitution', () => {
    const tripTypes = [
      { value: 'inst_to_dest', label: 'Institution → Dest.' },
      { value: 'dom_to_dest', label: 'Domicile → Dest.' },
      { value: 'return_home', label: 'Retour domicile' },
    ];

    it('masque Domicile → Dest. pour une clinique', () => {
      const filtered = filterTripTypesForInstitution(tripTypes, 'clinic');
      expect(filtered.map((t) => t.value)).toEqual(['inst_to_dest', 'return_home']);
    });

    it('conserve Domicile → Dest. pour IMAD et curatelle', () => {
      expect(filterTripTypesForInstitution(tripTypes, 'imad')).toHaveLength(3);
      expect(filterTripTypesForInstitution(tripTypes, 'curatelle')).toHaveLength(3);
    });
  });
});
