import { MEDICAL_DESTINATION_OR_ERROR } from '../institutionDestinationDetails';
import {
  APPOINTMENT_LEAD_REQUIRED,
  CONFIRMED_TIME_REQUIRED,
  DROPOFF_ADDRESS_REQUIRED,
  EXTRA_STOP_ADDRESS_REQUIRED,
  MISSION_DATE_REQUIRED,
  PATIENT_REQUIRED_FOR_DOMICILE,
  PICKUP_ADDRESS_REQUIRED,
  REQUIRED_FIELDS_TOAST,
  collectInstitutionRequestFormErrors,
  fieldErrorsMap,
  formErrorId,
  scrollToFirstFormError,
} from '../institutionRequestFormErrors';

const base = {
  mission_type: 'patient_transport',
  pickup_type: 'institution',
  dropoff_type: 'other',
  destination_type: 'other',
  pickup_location: 'Chemin des Courbes 9, 1247, Anières',
  dropoff_location: 'Restaurant Les Armures, Genève',
  pickup_time: '',
  dropoff_time: '',
  intermediate_stops: [],
};

describe('institutionRequestFormErrors', () => {
  it('signale la date de mission vide en premier', () => {
    const errors = collectInstitutionRequestFormErrors({
      formData: { ...base, mission_date: '' },
    });
    expect(errors[0]).toEqual({
      key: 'mission_date',
      message: MISSION_DATE_REQUIRED,
      fieldId: 'mission_date',
    });
    expect(formErrorId('mission_date')).toBe('mission_date-error');
    expect(REQUIRED_FIELDS_TOAST).toMatch(/champs obligatoires/);
  });

  it('collecte date + destination médicale ensemble', () => {
    const errors = collectInstitutionRequestFormErrors({
      formData: {
        ...base,
        mission_date: '',
        destination_type: 'medical',
        dropoff_service: '',
        dropoff_doctor: '',
      },
    });
    const keys = errors.map((e) => e.key);
    expect(keys).toContain('mission_date');
    expect(keys).toContain('medical_principal');
    expect(fieldErrorsMap(errors).medical_principal).toBe(MEDICAL_DESTINATION_OR_ERROR);
  });

  it('n’exige pas Service/Médecin pour Autre lieu', () => {
    const errors = collectInstitutionRequestFormErrors({
      formData: {
        ...base,
        mission_date: '2026-12-01',
        destination_type: 'other',
      },
    });
    expect(errors.map((e) => e.key)).not.toContain('medical_principal');
  });

  it('signale départ et arrivée manquants', () => {
    const errors = collectInstitutionRequestFormErrors({
      formData: {
        ...base,
        mission_date: '2026-12-01',
        pickup_location: '',
        pickup_type: 'other',
        dropoff_location: '',
      },
      institutionAddress: '',
    });
    expect(errors.map((e) => e.key)).toEqual(
      expect.arrayContaining(['pickup_location', 'dropoff_location']),
    );
    expect(fieldErrorsMap(errors).pickup_location).toBe(PICKUP_ADDRESS_REQUIRED);
    expect(fieldErrorsMap(errors).dropoff_location).toBe(DROPOFF_ADDRESS_REQUIRED);
  });

  it('signale l’étape médicale incomplète en multi-stop', () => {
    const errors = collectInstitutionRequestFormErrors({
      formData: {
        ...base,
        mission_date: '2026-12-01',
        destination_type: 'other',
        intermediate_stops: [
          {
            dropoff_location: 'HUG',
            destination_type: 'medical',
            dropoff_service: '',
            dropoff_doctor: '',
          },
        ],
      },
    });
    expect(errors.some((e) => e.key === 'medical_extra_0')).toBe(true);
    expect(errors.find((e) => e.key === 'medical_extra_0').fieldId).toBe('stop_service_0');
  });

  it('exige une heure confirmée en envoi LIRIE', () => {
    const errors = collectInstitutionRequestFormErrors({
      formData: { ...base, mission_date: '2026-12-01' },
      isLirieSendMode: true,
    });
    const confirmed = errors.find((e) => e.key === 'confirmed_time');
    expect(confirmed.message).toBe(CONFIRMED_TIME_REQUIRED);
    expect(confirmed.fieldId).toBe('confirmed_time');
  });

  it('expose le message de délai RDV', () => {
    expect(APPOINTMENT_LEAD_REQUIRED).toMatch(/rendez-vous/);
  });

  it('signale l’adresse manquante d’une étape multi-stop commencée', () => {
    const errors = collectInstitutionRequestFormErrors({
      formData: {
        ...base,
        mission_date: '2026-12-01',
        intermediate_stops: [
          {
            dropoff_location: '',
            destination_type: 'medical',
            dropoff_service: '',
            dropoff_doctor: '',
          },
        ],
      },
    });
    expect(errors.some((e) => e.key === 'extra_stop_location_0')).toBe(true);
    expect(fieldErrorsMap(errors).extra_stop_location_0).toBe(EXTRA_STOP_ADDRESS_REQUIRED);
  });

  it('scroll vers le premier champ en erreur', () => {
    const raf = jest.spyOn(window, 'requestAnimationFrame').mockImplementation((cb) => {
      cb();
      return 1;
    });
    const focus = jest.fn();
    const scrollIntoView = jest.fn();
    const getElementById = jest.spyOn(document, 'getElementById').mockReturnValue({
      scrollIntoView,
      focus,
    });
    scrollToFirstFormError([{ key: 'mission_date', message: MISSION_DATE_REQUIRED, fieldId: 'mission_date' }]);
    expect(getElementById).toHaveBeenCalledWith('mission_date');
    expect(scrollIntoView).toHaveBeenCalled();
    expect(focus).toHaveBeenCalled();
    getElementById.mockRestore();
    raf.mockRestore();
  });

  it('exige un patient pour un trajet domicile', () => {
    const errors = collectInstitutionRequestFormErrors({
      formData: {
        ...base,
        mission_date: '2026-12-01',
        pickup_type: 'domicile',
        pickup_location: '',
        patient_id: '',
      },
    });
    expect(errors[0]).toEqual({
      key: 'patient_id',
      message: PATIENT_REQUIRED_FOR_DOMICILE,
      fieldId: 'patient-select',
    });
  });
});
