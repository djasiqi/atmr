import React, { useMemo, useState } from 'react';
import { FaPlus, FaTrash, FaRoute } from 'react-icons/fa';
import { toast } from 'sonner';
import AddressAutocomplete from '../../../components/common/AddressAutocomplete';
import { useUpdateRequest } from '../../../hooks/useInstitutionData';
import InlineDatePicker from '../../../components/ui/InlineDatePicker';
import RouteStepTimeField from '../../../components/institution/RouteStepTimeField';
import ConfirmRequestEditModal from './ConfirmRequestEditModal';
import { combineMissionDateTime, extractHHMM } from '../../../utils/missionScheduleForm';
import { extractWallClockDate } from '../../../utils/missionTimeDisplay';
import MedicalDestinationDetails from '../../../components/institution/MedicalDestinationDetails';
import {
  DESTINATION_TYPE_MEDICAL,
  DESTINATION_TYPE_OTHER,
  suggestDestinationTypeFromPlace,
} from '../../../utils/institutionDestinationDetails';
import {
  REQUIRED_FIELDS_TOAST,
  collectInstitutionRequestFormErrors,
  fieldErrorsMap,
  formErrorId,
  scrollToFirstFormError,
} from '../../../utils/institutionRequestFormErrors';
import {
  buildInitialDestinations,
  buildInitialReturnBilling,
  extractAddressFromPlace,
  extractPlaceDetails,
  mapDestinationsToIntermediateStops,
} from '../../../utils/institutionRouteForm';
import s from './RequestDetailPanel.module.css';

const parseDate = (iso) => {
  if (!iso) return '';
  if (/^\d{4}-\d{2}-\d{2}$/.test(String(iso))) return String(iso);
  return extractWallClockDate(iso) || String(iso).slice(0, 10);
};

const parseTime = (iso) => extractHHMM(iso);

const InstitutionRequestEdit = ({ request, onCancel, onSaved }) => {
  const wasMultiStop = Boolean(request?.multi_stop);
  const needsCarrierAck = ['SENT', 'ACCEPTED'].includes(request?.status);
  const [showConfirmModal, setShowConfirmModal] = useState(false);
  const [pendingPayload, setPendingPayload] = useState(null);
  const [returnToInstitution, setReturnToInstitution] = useState(
    Boolean(request?.return_to_institution),
  );
  const returnBilling = useMemo(() => buildInitialReturnBilling(request), [request]);

  const [pickupLocation, setPickupLocation] = useState(request.pickup_location || '');
  const [destinations, setDestinations] = useState(() => buildInitialDestinations(request));
  const [missionDate, setMissionDate] = useState(
    parseDate(request.mission_date || request.scheduled_time),
  );
  const [pickupTime, setPickupTime] = useState(parseTime(request.scheduled_time));
  const [pickupTimeConfirmed, setPickupTimeConfirmed] = useState(
    Boolean(request.pickup_time_confirmed),
  );
  const [returnTime, setReturnTime] = useState(parseTime(request.return_time));
  const [returnTimeConfirmed, setReturnTimeConfirmed] = useState(
    Boolean(request.return_time_confirmed),
  );
  const [notes, setNotes] = useState(request.notes || '');
  const initialMobility = useMemo(() => request.mobility || {}, [request.mobility]);
  const [mobility, setMobility] = useState(() => ({
    wheelchair: Boolean(request.requires_wheelchair || initialMobility.wheelchair),
    vehicle_wheelchair: Boolean(initialMobility.vehicle_wheelchair),
    needs_assistance: Boolean(request.requires_assistance || initialMobility.needs_assistance),
  }));
  const [assistanceType, setAssistanceType] = useState(initialMobility.assistance_type || '');
  const [fieldErrors, setFieldErrors] = useState({});

  const clearFieldErrors = (...keys) => {
    setFieldErrors((prev) => {
      if (!keys.some((k) => prev[k])) return prev;
      const next = { ...prev };
      keys.forEach((k) => { delete next[k]; });
      return next;
    });
  };

  const remapEditErrorIds = (errors) => (errors || []).map((err) => {
    if (err.key === 'mission_date') return { ...err, fieldId: 'edit-mission-date' };
    if (err.key === 'pickup_location') return { ...err, fieldId: 'edit-pickup-location' };
    if (err.key === 'pickup_time') return { ...err, fieldId: 'edit-pickup-time' };
    if (err.key === 'dropoff_location') return { ...err, fieldId: 'edit-dest-address-0' };
    if (err.key === 'medical_principal') return { ...err, fieldId: 'edit-dest-service-0' };
    const loc = /^extra_stop_location_(\d+)$/.exec(err.key);
    if (loc) return { ...err, fieldId: `edit-dest-address-${Number(loc[1]) + 1}` };
    const time = /^extra_stop_time_(\d+)$/.exec(err.key);
    if (time) return { ...err, fieldId: `edit-dest-time-${Number(time[1]) + 1}` };
    const med = /^medical_extra_(\d+)$/.exec(err.key);
    if (med) return { ...err, fieldId: `edit-dest-service-${Number(med[1]) + 1}` };
    return err;
  });

  const toggleMobility = (key) => {
    setMobility((prev) => {
      const next = { ...prev, [key]: !prev[key] };
      if (key === 'wheelchair' && next.wheelchair) next.vehicle_wheelchair = false;
      if (key === 'vehicle_wheelchair' && next.vehicle_wheelchair) next.wheelchair = false;
      return next;
    });
  };

  const updateMutation = useUpdateRequest();

  const setDestinationField = (index, field, value) => {
    setDestinations((prev) =>
      prev.map((d, i) => (i === index ? { ...d, [field]: value } : d)),
    );
    if (field === 'service' || field === 'doctor' || field === 'destinationType') {
      clearFieldErrors(index === 0 ? 'medical_principal' : `medical_extra_${index - 1}`);
    }
    if (field === 'address') {
      clearFieldErrors(index === 0 ? 'dropoff_location' : `extra_stop_location_${index - 1}`);
    }
  };

  const setDestinationTime = (index, timeHHMM) => {
    const iso = combineMissionDateTime(missionDate, timeHHMM);
    setDestinationField(index, 'scheduled_time', iso || '');
  };

  const addDestination = () => {
    setDestinations((prev) => [
      ...prev,
      {
        address: '',
        establishment: '',
        service: '',
        doctor: '',
        destinationType: DESTINATION_TYPE_MEDICAL,
        scheduled_time: '',
        time_confirmed: false,
        use_custom_billing: false,
        destination_billing_override: 'patient',
      },
    ]);
  };

  const removeDestination = (index) => {
    setDestinations((prev) => prev.filter((_, i) => i !== index));
  };

  const setDestinationFromSelection = (index, item) => {
    const address = extractAddressFromPlace(item);
    const { establishment, doctor } = extractPlaceDetails(item);
    const suggested = suggestDestinationTypeFromPlace(item)
      || (doctor ? 'medical' : null);
    setDestinations((prev) =>
      prev.map((dest, i) => (
        i === index
          ? {
            ...dest,
            address,
            establishment,
            doctor,
            ...(suggested ? { destinationType: suggested } : {}),
          }
          : dest
      )),
    );
    clearFieldErrors(
      index === 0 ? 'dropoff_location' : `extra_stop_location_${index - 1}`,
      index === 0 ? 'medical_principal' : `medical_extra_${index - 1}`,
    );
  };

  const buildPayload = () => {
    const extraStops = destinations.slice(1).map((d) => ({
      dropoff_location: d.address,
      destination_type: d.destinationType,
      dropoff_service: d.service,
      dropoff_doctor: d.doctor,
      dropoff_establishment: d.establishment,
      scheduled_time: d.scheduled_time,
    }));
    const validationErrors = remapEditErrorIds(collectInstitutionRequestFormErrors({
      formData: {
        mission_type: request.mission_type || 'patient_transport',
        mission_date: missionDate,
        pickup_location: pickupLocation,
        pickup_type: 'other',
        dropoff_location: destinations[0]?.address,
        dropoff_type: 'other',
        destination_type: destinations[0]?.destinationType,
        dropoff_service: destinations[0]?.service,
        dropoff_doctor: destinations[0]?.doctor,
        pickup_time: pickupTime,
        return_to_institution: returnToInstitution,
        return_time: returnTime,
        intermediate_stops: extraStops,
      },
    }));
    if (validationErrors.length > 0) {
      setFieldErrors(fieldErrorsMap(validationErrors));
      toast.error(REQUIRED_FIELDS_TOAST);
      scrollToFirstFormError(validationErrors);
      return null;
    }
    setFieldErrors({});

    const cleanedDestinations = destinations
      .map((d) => ({ ...d, address: (d.address || '').trim() }))
      .filter((d) => d.address);

    const payload = {
      mission_date: missionDate,
      pickup_location: pickupLocation.trim(),
      pickup_time_confirmed: pickupTimeConfirmed,
      notes: notes || null,
    };

    const isMultiRoute = returnToInstitution
      || cleanedDestinations.length > 1
      || wasMultiStop;

    if (isMultiRoute) {
      payload.multi_stop = true;
      payload.return_to_institution = returnToInstitution;
      payload.is_round_trip = false;
      payload.intermediate_stops = mapDestinationsToIntermediateStops(cleanedDestinations);

      const pickupIso = combineMissionDateTime(missionDate, pickupTime);
      if (pickupIso) {
        payload.scheduled_time = pickupIso;
        payload.scheduled_time_type = 'departure';
      }

      if (returnToInstitution) {
        const retIso = combineMissionDateTime(missionDate, returnTime);
        if (retIso) payload.return_scheduled_time = retIso;
        payload.return_time_confirmed = returnTimeConfirmed;
        payload.return_stop = {
          use_custom_billing: Boolean(returnBilling.use_custom_billing),
          destination_billing_override: returnBilling.use_custom_billing
            ? (returnBilling.destination_billing_override || 'patient')
            : null,
        };
      }
    } else {
      const dest = cleanedDestinations[0];
      payload.dropoff_location = dest.address;
      payload.dropoff_type = 'other';
      payload.destination_type = dest.destinationType || DESTINATION_TYPE_OTHER;
      if (dest.establishment) payload.dropoff_establishment = dest.establishment;
      payload.dropoff_service = dest.service?.trim() || null;
      payload.dropoff_doctor = dest.doctor?.trim() || null;

      const pickupIso = combineMissionDateTime(missionDate, pickupTime);
      if (pickupIso) {
        payload.scheduled_time = pickupIso;
        payload.scheduled_time_type = 'departure';
      } else if (dest.scheduled_time) {
        payload.scheduled_time = dest.scheduled_time;
        payload.scheduled_time_type = 'arrival';
        payload.pickup_time_confirmed = false;
        payload.appointment_time_confirmed = Boolean(dest.time_confirmed);
      }
    }

    payload.mobility = {
      ...initialMobility,
      wheelchair: mobility.wheelchair,
      vehicle_wheelchair: mobility.vehicle_wheelchair,
      needs_assistance: mobility.needs_assistance,
      assistance_type: mobility.needs_assistance ? assistanceType.trim() : '',
      walking: !mobility.wheelchair && !mobility.vehicle_wheelchair && !initialMobility.stretcher,
    };

    return payload;
  };

  const submitPayload = (payload, { withCarrierAck = false } = {}) => {
    const data = withCarrierAck
      ? { ...payload, acknowledge_carrier_impact: true }
      : payload;

    updateMutation.mutate(
      { requestId: request.id, data },
      {
        onSuccess: () => {
          setShowConfirmModal(false);
          setPendingPayload(null);
          onSaved?.({ carrierNotified: withCarrierAck });
        },
        onError: (err) => {
          const dataErr = err?.response?.data;
          if (dataErr?.code === 'carrier_ack_required') {
            setPendingPayload(payload);
            setShowConfirmModal(true);
            return;
          }
          toast.error(dataErr?.error || 'Erreur lors de la modification.');
        },
      },
    );
  };

  const handleSave = () => {
    const payload = buildPayload();
    if (!payload) return;

    if (needsCarrierAck) {
      setPendingPayload(payload);
      setShowConfirmModal(true);
      return;
    }

    submitPayload(payload);
  };

  const handleConfirmSave = () => {
    if (!pendingPayload) return;
    submitPayload(pendingPayload, { withCarrierAck: true });
  };

  const showReturnRow = returnToInstitution;

  return (
    <div className={s.section}>
      <div className={s.sectionHeader}>
        <div className={`${s.sectionIcon} ${s.sectionIconBrand}`}><FaRoute /></div>
        <h3 className={s.sectionTitle}>Modifier le trajet</h3>
      </div>
      <div className={s.routeEdit}>
        <div className={s.routeEditRow}>
          <div className={s.routeMarker}>
            <span className={`${s.routeDot} ${s.routeDotStart}`} />
            <span className={s.routeConnector} />
          </div>
          <div className={s.routeEditBody}>
            <div className={s.routeStopLabel}>Départ</div>
            <div className={s.routeEditAddressRow}>
              <div className={s.editInputWrap}>
                <AddressAutocomplete
                  name="pickup_location"
                  inputId="edit-pickup-location"
                  value={pickupLocation}
                  onChange={(e) => {
                    setPickupLocation(e?.target?.value ?? e ?? '');
                    clearFieldErrors('pickup_location');
                  }}
                  onSelect={(item) => {
                    setPickupLocation(extractAddressFromPlace(item));
                    clearFieldErrors('pickup_location');
                  }}
                  placeholder="Adresse de départ"
                  aria-invalid={Boolean(fieldErrors.pickup_location) || undefined}
                  aria-describedby={fieldErrors.pickup_location ? formErrorId('edit-pickup-location') : undefined}
                />
              </div>
              <RouteStepTimeField
                inputId="edit-pickup-time"
                timeValue={pickupTime}
                timeConfirmed={pickupTimeConfirmed}
                onTimeChange={setPickupTime}
                onConfirmedChange={setPickupTimeConfirmed}
              />
            </div>
            {fieldErrors.pickup_location && (
              <p id={formErrorId('edit-pickup-location')} className={s.fieldError} role="alert">
                {fieldErrors.pickup_location}
              </p>
            )}
          </div>
        </div>

        {destinations.map((dest, index) => (
          <div className={s.routeEditRow} key={`dest-${index}`}>
            <div className={s.routeMarker}>
              <span className={`${s.routeDot} ${s.routeDotMid}`} />
              <span className={s.routeConnector} />
            </div>
            <div className={s.routeEditBody}>
              <div className={s.routeEditHeader}>
                <span className={s.routeStopLabel}>Destination {index + 1}</span>
                {destinations.length > 1 && (
                  <button
                    type="button"
                    className={s.routeEditRemove}
                    onClick={() => removeDestination(index)}
                    aria-label={`Supprimer la destination ${index + 1}`}
                  >
                    <FaTrash size={11} />
                  </button>
                )}
              </div>
              <div className={s.routeEditAddressRow}>
                <div className={s.editInputWrap}>
                  <AddressAutocomplete
                    name={`edit_destination_${index}`}
                    inputId={`edit-dest-address-${index}`}
                    value={dest.address}
                    onChange={(e) => setDestinationField(
                      index,
                      'address',
                      e?.target?.value ?? e ?? '',
                    )}
                    onSelect={(item) => setDestinationFromSelection(index, item)}
                    placeholder="Adresse de destination"
                    aria-invalid={Boolean(
                      index === 0
                        ? fieldErrors.dropoff_location
                        : fieldErrors[`extra_stop_location_${index - 1}`],
                    ) || undefined}
                    aria-describedby={
                      (index === 0 && fieldErrors.dropoff_location && formErrorId(`edit-dest-address-${index}`))
                      || (index > 0 && fieldErrors[`extra_stop_location_${index - 1}`]
                        && formErrorId(`edit-dest-address-${index}`))
                      || undefined
                    }
                  />
                </div>
                <RouteStepTimeField
                  inputId={`edit-dest-time-${index}`}
                  timeValue={parseTime(dest.scheduled_time)}
                  timeConfirmed={Boolean(dest.time_confirmed)}
                  onTimeChange={(v) => setDestinationTime(index, v)}
                  onConfirmedChange={(v) => setDestinationField(index, 'time_confirmed', v)}
                />
              </div>
              {(index === 0 ? fieldErrors.dropoff_location : fieldErrors[`extra_stop_location_${index - 1}`]) && (
                <p id={formErrorId(`edit-dest-address-${index}`)} className={s.fieldError} role="alert">
                  {index === 0 ? fieldErrors.dropoff_location : fieldErrors[`extra_stop_location_${index - 1}`]}
                </p>
              )}
              <div className={s.routeEditDetails}>
                <MedicalDestinationDetails
                  compact
                  establishmentId={`edit-dest-establishment-${index}`}
                  serviceId={`edit-dest-service-${index}`}
                  doctorId={`edit-dest-doctor-${index}`}
                  establishment={dest.establishment}
                  service={dest.service}
                  doctor={dest.doctor}
                  destinationType={dest.destinationType || DESTINATION_TYPE_OTHER}
                  onDestinationTypeChange={(value) => setDestinationField(index, 'destinationType', value)}
                  onEstablishmentChange={(e) => setDestinationField(index, 'establishment', e.target.value)}
                  onServiceChange={(e) => setDestinationField(index, 'service', e.target.value)}
                  onDoctorChange={(e) => setDestinationField(index, 'doctor', e.target.value)}
                  showError={Boolean(
                    index === 0
                      ? fieldErrors.medical_principal
                      : fieldErrors[`medical_extra_${index - 1}`],
                  )}
                />
              </div>
            </div>
          </div>
        ))}

        <div className={s.routeEditActions}>
          <button type="button" className={s.routeEditAdd} onClick={addDestination}>
            <FaPlus size={10} /> Ajouter une destination
          </button>
          <button
            type="button"
            className={`${s.routeEditReturnBtn} ${returnToInstitution ? s.routeEditReturnBtnActive : ''}`}
            aria-pressed={returnToInstitution}
            title="Aller / retour : ajoute le retour au départ en fin de parcours"
            onClick={() => setReturnToInstitution((prev) => !prev)}
          >
            ⇄ A/R
          </button>
        </div>

        {showReturnRow && (
          <div className={s.routeEditRow}>
            <div className={s.routeMarker}>
              <span className={`${s.routeDot} ${s.routeDotEnd}`} />
            </div>
            <div className={s.routeEditBody}>
              <div className={s.routeStopLabel}>Retour</div>
              <div className={s.routeEditAddressRow}>
                <div className={s.routeStopAddress}>{pickupLocation || '—'}</div>
                <RouteStepTimeField
                  inputId="edit-return-time"
                  timeValue={returnTime}
                  timeConfirmed={returnTimeConfirmed}
                  onTimeChange={setReturnTime}
                  onConfirmedChange={setReturnTimeConfirmed}
                />
              </div>
            </div>
          </div>
        )}
      </div>

      <div className={s.editRow}>
        <div className={s.editField}>
          <label htmlFor="edit-mission-date" className={s.editLabel}>
            Date de mission
          </label>
          <InlineDatePicker
            inputId="edit-mission-date"
            value={missionDate}
            onChange={(v) => {
              setMissionDate(v);
              clearFieldErrors('mission_date');
            }}
            placeholder="Date"
            invalid={Boolean(fieldErrors.mission_date)}
            describedBy={fieldErrors.mission_date ? formErrorId('edit-mission-date') : undefined}
          />
          {fieldErrors.mission_date && (
            <p id={formErrorId('edit-mission-date')} className={s.fieldError} role="alert">
              {fieldErrors.mission_date}
            </p>
          )}
        </div>
      </div>

      <label className={s.editLabel}>
        Notes
        <textarea
          className={s.editTextarea}
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
          rows={2}
        />
      </label>

      <div className={s.editField}>
        <span className={s.editLabel}>Besoins spécifiques</span>
        <div className={s.needsRow}>
          <button
            type="button"
            aria-pressed={mobility.wheelchair}
            className={`${s.needsChip} ${mobility.wheelchair ? s.needsChipActive : ''}`}
            onClick={() => toggleMobility('wheelchair')}
          >
            ♿ Fauteuil
          </button>
          <button
            type="button"
            aria-pressed={mobility.vehicle_wheelchair}
            className={`${s.needsChip} ${mobility.vehicle_wheelchair ? s.needsChipActive : ''}`}
            onClick={() => toggleMobility('vehicle_wheelchair')}
          >
            🏥 Prendre chaise
          </button>
          <button
            type="button"
            aria-pressed={mobility.needs_assistance}
            className={`${s.needsChip} ${mobility.needs_assistance ? s.needsChipActive : ''}`}
            onClick={() => toggleMobility('needs_assistance')}
          >
            🤝 Assistance
          </button>
        </div>
        {mobility.needs_assistance && (
          <input
            className={s.editInput}
            style={{ marginTop: 6 }}
            value={assistanceType}
            onChange={(e) => setAssistanceType(e.target.value)}
            placeholder="Préciser le type d'assistance (ex: aide au transfert, accompagnement…)"
          />
        )}
      </div>

      <div className={s.editActions}>
        <button
          type="button"
          className={`${s.actionBtn} ${s.btnSecondary}`}
          onClick={onCancel}
        >
          Annuler
        </button>
        <button
          type="button"
          className={`${s.actionBtn} ${s.btnPrimary}`}
          onClick={handleSave}
          disabled={updateMutation.isPending}
        >
          {updateMutation.isPending ? '...' : 'Enregistrer'}
        </button>
      </div>

      {showConfirmModal && (
        <ConfirmRequestEditModal
          requestStatus={request.status}
          onClose={() => {
            if (updateMutation.isPending) return;
            setShowConfirmModal(false);
            setPendingPayload(null);
          }}
          onConfirm={handleConfirmSave}
          loading={updateMutation.isPending}
        />
      )}
    </div>
  );
};

export default InstitutionRequestEdit;
