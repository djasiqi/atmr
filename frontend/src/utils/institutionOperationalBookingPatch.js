/**
 * Payload PATCH booking institution (édition opérationnelle du panneau détail).
 * Source : état édité (destinations / heures flushées), pas l'objet request d'origine.
 */

import { DESTINATION_TYPE_OTHER } from './institutionDestinationDetails';
import { combineMissionDateTime, extractHHMM } from './missionScheduleForm';

export const applyFlushedDestinationTimes = (destinations, missionDate, flushedTimes) =>
  (destinations || []).map((dest, index) => {
    const hhmm = extractHHMM(flushedTimes?.[index]) || extractHHMM(dest.scheduled_time);
    return {
      ...dest,
      scheduled_time: combineMissionDateTime(missionDate, hhmm) || '',
      time_confirmed: Boolean(hhmm),
    };
  });

export const buildOperationalBookingPatch = ({
  editVersion,
  accessForm = {},
  pickupLocation,
  pickupTime,
  missionDate,
  destinations,
  returnToInstitution,
  returnTime,
}) => {
  const cleanedDestinations = (destinations || [])
    .map((d) => ({ ...d, address: (d.address || '').trim() }))
    .filter((d) => d.address);

  const firstDest = cleanedDestinations[0];
  if (!firstDest) {
    return null;
  }

  const pickupIso = combineMissionDateTime(missionDate, pickupTime);
  const legAppointments = cleanedDestinations.map((dest, index) => ({
    index,
    scheduled_time: dest.scheduled_time || null,
  }));

  const payload = {
    version: Number(editVersion) || 1,
    reason: (accessForm.reason || '').trim() || undefined,
    customer_name: (accessForm.customer_name || '').trim() || undefined,
    pickup_location: (pickupLocation || '').trim(),
    dropoff_location: firstDest.address,
    scheduled_time: pickupIso || undefined,
    medical_facility: (firstDest.establishment || '').trim() || null,
    hospital_service: (firstDest.service || '').trim() || null,
    doctor_name: (firstDest.doctor || '').trim() || null,
    destination_type: firstDest.destinationType || DESTINATION_TYPE_OTHER,
    pickup_floor: (accessForm.pickup_floor || '').trim() || null,
    pickup_door_code: (accessForm.pickup_door_code || '').trim() || null,
    dropoff_floor: (accessForm.dropoff_floor || '').trim() || null,
    dropoff_door_code: (accessForm.dropoff_door_code || '').trim() || null,
    pickup_access_notes: (accessForm.pickup_access_notes || '').trim() || null,
    dropoff_access_notes: (accessForm.dropoff_access_notes || '').trim() || null,
    notes_medical: (accessForm.notes_medical || '').trim() || null,
    wheelchair_need: Boolean(accessForm.wheelchair_need),
    wheelchair_client_has: Boolean(accessForm.wheelchair_client_has),
    delivery_description: (accessForm.delivery_description || '').trim() || null,
    leg_appointments: legAppointments,
  };

  if (firstDest.scheduled_time) {
    payload.appointment_time = firstDest.scheduled_time;
  }

  if (returnToInstitution) {
    const returnIso = combineMissionDateTime(missionDate, returnTime);
    payload.return_appointment_time = returnIso || null;
  }

  return payload;
};
