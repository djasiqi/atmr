import React, { useMemo, useState } from 'react';
import { FaPlus, FaRoute, FaTrash, FaWheelchair } from 'react-icons/fa';
import { FiCheck, FiFileText, FiHome, FiMapPin, FiUser, FiX } from 'react-icons/fi';
import AddressAutocomplete from '../../../components/common/AddressAutocomplete';
import RouteStepTimeField from '../../../components/institution/RouteStepTimeField';
import InlineDatePicker from '../../../components/ui/InlineDatePicker';
import {
  buildInitialDestinations,
  extractAddressFromPlace,
  extractPlaceDetails,
} from '../../../utils/institutionRouteForm';
import {
  combineMissionDateTime,
  extractHHMM,
  isInstantInPast,
} from '../../../utils/missionScheduleForm';
import { extractWallClockDate } from '../../../utils/missionTimeDisplay';
import s from './RequestDetailPanel.module.css';

const parseDate = (iso) => {
  if (!iso) return '';
  if (/^\d{4}-\d{2}-\d{2}$/.test(String(iso))) return String(iso);
  return extractWallClockDate(iso) || String(iso).slice(0, 10);
};

const parseTime = (iso) => extractHHMM(iso);

const textOrEmpty = (value) => (value == null ? '' : String(value));

const InstitutionOperationalEdit = ({
  request,
  bookingId,
  editVersion,
  isEnRoute,
  onCancel,
  onSaved,
  patchMutation,
}) => {
  const bs = useMemo(() => request.booking_summary || {}, [request.booking_summary]);
  const patientName = request.patient
    ? `${request.patient.first_name || ''} ${request.patient.last_name || ''}`.trim()
    : bs.customer_name || 'Patient';
  const isReturnTrip = Boolean(bs.is_return || request.is_return);
  const isConverted = request.status === 'CONVERTED';

  const [pickupLocation, setPickupLocation] = useState(
    textOrEmpty(bs.pickup_location || request.pickup_location),
  );
  const [destinations, setDestinations] = useState(() => buildInitialDestinations(request, bs));
  const [missionDate, setMissionDate] = useState(
    parseDate(bs.scheduled_time || request.mission_date || request.scheduled_time),
  );
  const [pickupTime, setPickupTime] = useState(parseTime(bs.scheduled_time));
  const [pickupTimeConfirmed, setPickupTimeConfirmed] = useState(bs.time_confirmed !== false);
  const [returnToInstitution, setReturnToInstitution] = useState(
    Boolean(request.return_to_institution),
  );
  const [returnTime, setReturnTime] = useState(parseTime(request.return_time));
  const [returnTimeConfirmed, setReturnTimeConfirmed] = useState(
    Boolean(request.return_time_confirmed),
  );

  const initialAccess = useMemo(() => ({
    customer_name: textOrEmpty(bs.customer_name || patientName),
    pickup_floor: textOrEmpty(bs.pickup_floor),
    pickup_door_code: textOrEmpty(bs.pickup_door_code),
    dropoff_floor: textOrEmpty(bs.dropoff_floor),
    dropoff_door_code: textOrEmpty(bs.dropoff_door_code),
    pickup_access_notes: textOrEmpty(bs.pickup_access_notes),
    dropoff_access_notes: textOrEmpty(bs.dropoff_access_notes),
    notes_medical: textOrEmpty(bs.notes_medical || request.notes),
    wheelchair_need: Boolean(bs.wheelchair_need || request.requires_wheelchair),
    wheelchair_client_has: Boolean(bs.wheelchair_client_has),
    delivery_description: textOrEmpty(bs.delivery_description || request.delivery_description),
    reason: '',
  }), [bs, patientName, request]);

  const [accessForm, setAccessForm] = useState(initialAccess);
  const [saveError, setSaveError] = useState(null);

  const isDateTimeInPast = useMemo(() => {
    if (bs.time_confirmed === false) return false;
    const pickupIso = combineMissionDateTime(missionDate, pickupTime);
    return isInstantInPast(pickupIso);
  }, [missionDate, pickupTime, bs.time_confirmed]);

  const handleAccessChange = (field, value) => {
    setAccessForm((prev) => ({ ...prev, [field]: value }));
  };

  const setDestinationField = (index, field, value) => {
    setDestinations((prev) =>
      prev.map((d, i) => (i === index ? { ...d, [field]: value } : d)),
    );
  };

  const setDestinationTime = (index, timeHHMM) => {
    const iso = combineMissionDateTime(missionDate, timeHHMM);
    setDestinationField(index, 'scheduled_time', iso || '');
    setDestinationField(index, 'time_confirmed', Boolean(timeHHMM?.trim()));
  };

  const addDestination = () => {
    if (isConverted) return;
    setDestinations((prev) => [
      ...prev,
      {
        address: '',
        establishment: '',
        service: '',
        doctor: '',
        scheduled_time: '',
        time_confirmed: false,
        booking_id: null,
        leg_index: null,
      },
    ]);
  };

  const removeDestination = (index) => {
    if (destinations.length <= 1) return;
    setDestinations((prev) => prev.filter((_, i) => i !== index));
  };

  const setDestinationFromSelection = (index, item) => {
    const address = extractAddressFromPlace(item);
    const { establishment, doctor } = extractPlaceDetails(item);
    setDestinations((prev) =>
      prev.map((dest, i) => (
        i === index
          ? { ...dest, address, establishment, doctor }
          : dest
      )),
    );
  };

  const handleSave = () => {
    if (!bookingId) {
      setSaveError('Booking introuvable pour cette demande.');
      return;
    }

    const cleanedDestinations = destinations
      .map((d) => ({ ...d, address: (d.address || '').trim() }))
      .filter((d) => d.address);

    if (!pickupLocation.trim() || cleanedDestinations.length === 0) {
      setSaveError('Le départ et au moins une destination sont obligatoires.');
      return;
    }
    if (!missionDate) {
      setSaveError('La date de mission est obligatoire.');
      return;
    }
    if (isDateTimeInPast) {
      setSaveError("La date et l'heure de départ sont dans le passé.");
      return;
    }
    if (isEnRoute && accessForm.reason.trim().length < 10) {
      setSaveError('Motif obligatoire (10 caractères min.) pour une modification en route.');
      return;
    }

    const firstDest = cleanedDestinations[0];
    const pickupIso = combineMissionDateTime(missionDate, pickupTime);

    const legAppointments = cleanedDestinations.map((dest, index) => ({
      index,
      scheduled_time: dest.scheduled_time || null,
    }));

    const payload = {
      version: Number(editVersion) || 1,
      reason: accessForm.reason.trim() || undefined,
      customer_name: accessForm.customer_name.trim() || undefined,
      pickup_location: pickupLocation.trim(),
      dropoff_location: firstDest.address,
      scheduled_time: pickupIso || undefined,
      medical_facility: firstDest.establishment.trim() || null,
      hospital_service: firstDest.service.trim() || null,
      doctor_name: firstDest.doctor.trim() || null,
      pickup_floor: accessForm.pickup_floor.trim() || null,
      pickup_door_code: accessForm.pickup_door_code.trim() || null,
      dropoff_floor: accessForm.dropoff_floor.trim() || null,
      dropoff_door_code: accessForm.dropoff_door_code.trim() || null,
      pickup_access_notes: accessForm.pickup_access_notes.trim() || null,
      dropoff_access_notes: accessForm.dropoff_access_notes.trim() || null,
      notes_medical: accessForm.notes_medical.trim() || null,
      wheelchair_need: Boolean(accessForm.wheelchair_need),
      wheelchair_client_has: Boolean(accessForm.wheelchair_client_has),
      delivery_description: accessForm.delivery_description.trim() || null,
      leg_appointments: legAppointments,
    };

    if (firstDest.scheduled_time) {
      payload.appointment_time = firstDest.scheduled_time;
    }

    if (returnToInstitution) {
      const returnIso = combineMissionDateTime(missionDate, returnTime);
      payload.return_appointment_time = returnIso || null;
    }

    if (isEnRoute) {
      const msg = 'Cette modification alertera le transporteur et le chauffeur. Confirmer ?';
      if (!window.confirm(msg)) return;
    }

    setSaveError(null);
    patchMutation.mutate(
      { bookingId, data: payload, requestId: request.id },
      {
        onSuccess: () => onSaved?.(),
        onError: (err) => {
          const data = err?.response?.data;
          const fieldErrors = data?.details?.errors;
          const firstFieldMsg = fieldErrors && typeof fieldErrors === 'object'
            ? Object.values(fieldErrors).flat().find(Boolean)
            : null;
          const message = firstFieldMsg
            || data?.message
            || data?.error
            || 'Erreur lors de la sauvegarde.';
          if (err?.response?.status === 409) {
            setSaveError(data?.error || 'Conflit de version, rechargez le détail.');
          } else {
            setSaveError(message);
          }
        },
      },
    );
  };

  const homeAccessNotesField = isReturnTrip ? 'dropoff_access_notes' : 'pickup_access_notes';
  const hospitalAccessNotesField = isReturnTrip ? 'pickup_access_notes' : 'dropoff_access_notes';

  const fieldId = (suffix) => `institution-booking-${bookingId || 'new'}-${suffix}`;

  const homeBlock = (
    <div className={s.editGroup} key="home-access">
      <div className={s.editGroupTitle}>
        <FiMapPin className={s.editGroupIcon} size={12} />
        {isReturnTrip ? 'Accès arrivée' : 'Accès départ'}
      </div>
      <div className={s.editRow}>
        <input
          id={fieldId('pickup-floor')}
          name="pickup_floor"
          autoComplete="off"
          aria-label={isReturnTrip ? 'Étage à l\u2019arrivée' : 'Étage au départ'}
          className={s.editInput}
          value={accessForm.pickup_floor}
          onChange={(e) => handleAccessChange('pickup_floor', e.target.value)}
          placeholder="Étage"
        />
        <input
          id={fieldId('pickup-door-code')}
          name="pickup_door_code"
          autoComplete="off"
          aria-label={isReturnTrip ? 'Code porte à l\u2019arrivée' : 'Code porte au départ'}
          className={s.editInput}
          value={accessForm.pickup_door_code}
          onChange={(e) => handleAccessChange('pickup_door_code', e.target.value)}
          placeholder="Code porte"
        />
      </div>
      <input
        id={fieldId(`access-${homeAccessNotesField}`)}
        name={homeAccessNotesField}
        autoComplete="off"
        aria-label={isReturnTrip ? 'Consignes d\u2019arrivée' : 'Consignes de départ'}
        className={s.editInput}
        value={accessForm[homeAccessNotesField]}
        onChange={(e) => handleAccessChange(homeAccessNotesField, e.target.value)}
        placeholder={isReturnTrip ? 'Consignes arrivée' : 'Consignes départ'}
        style={{ marginTop: 6 }}
      />
    </div>
  );

  const hospitalBlock = (
    <div className={s.editGroup} key="hospital-access">
      <div className={s.editGroupTitle}>
        <FiHome className={s.editGroupIcon} size={12} />
        {isReturnTrip ? 'Accès départ' : 'Accès arrivée'}
      </div>
      <div className={s.editRow}>
        <input
          id={fieldId('dropoff-floor')}
          name="dropoff_floor"
          autoComplete="off"
          aria-label={isReturnTrip ? 'Étage au départ' : 'Étage à l\u2019arrivée'}
          className={s.editInput}
          value={accessForm.dropoff_floor}
          onChange={(e) => handleAccessChange('dropoff_floor', e.target.value)}
          placeholder="Étage"
        />
        <input
          id={fieldId('dropoff-door-code')}
          name="dropoff_door_code"
          autoComplete="off"
          aria-label={isReturnTrip ? 'Code porte au départ' : 'Code porte à l\u2019arrivée'}
          className={s.editInput}
          value={accessForm.dropoff_door_code}
          onChange={(e) => handleAccessChange('dropoff_door_code', e.target.value)}
          placeholder="Code porte"
        />
      </div>
      <input
        id={fieldId(`access-${hospitalAccessNotesField}`)}
        name={hospitalAccessNotesField}
        autoComplete="off"
        aria-label={isReturnTrip ? 'Consignes de départ' : 'Consignes d\u2019arrivée'}
        className={s.editInput}
        value={accessForm[hospitalAccessNotesField]}
        onChange={(e) => handleAccessChange(hospitalAccessNotesField, e.target.value)}
        placeholder={isReturnTrip ? 'Consignes départ' : 'Consignes arrivée'}
        style={{ marginTop: 8 }}
      />
    </div>
  );

  const showReturnRow = returnToInstitution;

  return (
    <div className={s.editForm}>
      <div className={s.editContext}>
        <span className={s.editContextLabel}>{patientName}</span>
        <span className={s.editContextSep} />
        <span className={s.editContextMeta}>#{bookingId}</span>
      </div>

      {isEnRoute && (
        <div className={s.enRouteWarning}>
          Course en route : toute modification déclenche une alerte au transporteur et au chauffeur.
        </div>
      )}

      <div className={s.section}>
        <div className={s.sectionHeader}>
          <div className={`${s.sectionIcon} ${s.sectionIconBrand}`}>
            <FaRoute />
          </div>
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
                    inputId={fieldId('pickup-location')}
                    value={pickupLocation}
                    onChange={(e) => setPickupLocation(e?.target?.value ?? e ?? '')}
                    onSelect={(item) => setPickupLocation(extractAddressFromPlace(item))}
                    placeholder="Adresse de départ"
                  />
                </div>
                <RouteStepTimeField
                  inputId={fieldId('pickup-time')}
                  timeValue={pickupTime}
                  timeConfirmed={pickupTimeConfirmed}
                  onTimeChange={setPickupTime}
                  onConfirmedChange={setPickupTimeConfirmed}
                  label="Heure de départ"
                />
              </div>
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
                  <span className={s.routeStopLabel}>
                    {destinations.length === 1 ? 'Destination' : `Destination ${index + 1}`}
                  </span>
                  {destinations.length > 1 && !isConverted && (
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
                      inputId={fieldId(`dest-address-${index}`)}
                      value={dest.address}
                      onChange={(e) => setDestinationField(
                        index,
                        'address',
                        e?.target?.value ?? e ?? '',
                      )}
                      onSelect={(item) => setDestinationFromSelection(index, item)}
                      placeholder="Adresse de destination"
                    />
                  </div>
                  <RouteStepTimeField
                    inputId={fieldId(`dest-time-${index}`)}
                    timeValue={parseTime(dest.scheduled_time)}
                    timeConfirmed={Boolean(dest.time_confirmed)}
                    onTimeChange={(v) => setDestinationTime(index, v)}
                    onConfirmedChange={(v) => setDestinationField(index, 'time_confirmed', v)}
                    label="Heure du rendez-vous"
                  />
                </div>
                <div className={s.routeEditDetails}>
                  <input
                    className={s.editInput}
                    value={dest.establishment}
                    onChange={(e) => setDestinationField(index, 'establishment', e.target.value)}
                    placeholder="Établissement / Lieu"
                  />
                  <input
                    className={s.editInput}
                    value={dest.service}
                    onChange={(e) => setDestinationField(index, 'service', e.target.value)}
                    placeholder="Service"
                  />
                  <input
                    className={s.editInput}
                    value={dest.doctor}
                    onChange={(e) => setDestinationField(index, 'doctor', e.target.value)}
                    placeholder="Médecin"
                  />
                </div>
              </div>
            </div>
          ))}

          {!isConverted && (
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
          )}

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
                    inputId={fieldId('return-time')}
                    timeValue={returnTime}
                    timeConfirmed={returnTimeConfirmed}
                    onTimeChange={setReturnTime}
                    onConfirmedChange={setReturnTimeConfirmed}
                    label="Heure de retour"
                  />
                </div>
              </div>
            </div>
          )}
        </div>

        <div className={s.editRow}>
          <div className={s.editField}>
            <label
              htmlFor={fieldId('mission-date')}
              className={`${s.editLabel} ${isDateTimeInPast ? s.fieldErrorLabel : ''}`}
            >
              Date de mission
            </label>
            <InlineDatePicker
              inputId={fieldId('mission-date')}
              value={missionDate}
              onChange={(v) => setMissionDate(v)}
              placeholder="Date"
            />
          </div>
        </div>
        {isDateTimeInPast && (
          <div className={s.fieldErrorHint}>La date et l&apos;heure de départ sont dans le passé</div>
        )}
      </div>

      <div className={s.editDivider} />
      {isReturnTrip ? hospitalBlock : homeBlock}
      <div className={s.editDivider} />
      {isReturnTrip ? homeBlock : hospitalBlock}
      <div className={s.editDivider} />

      <div className={s.editGroup}>
        <div className={s.editGroupTitle}>
          <FiUser className={s.editGroupIcon} size={12} />
          Patient
        </div>
        <input
          id={fieldId('customer-name')}
          name="customer_name"
          autoComplete="name"
          aria-label="Nom du patient"
          className={s.editInput}
          value={accessForm.customer_name}
          onChange={(e) => handleAccessChange('customer_name', e.target.value)}
          placeholder="Nom du patient"
        />
        <textarea
          id={fieldId('notes-medical')}
          name="notes_medical"
          aria-label="Notes médicales"
          className={s.editTextarea}
          value={accessForm.notes_medical}
          onChange={(e) => handleAccessChange('notes_medical', e.target.value)}
          placeholder="Pathologie, difficultés, mobilité…"
          rows={2}
          style={{ marginTop: 8 }}
        />
        <div className={s.toggleGroup}>
          <button
            type="button"
            className={`${s.toggleBtn} ${accessForm.wheelchair_need ? s.toggleBtnActive : ''}`}
            aria-pressed={accessForm.wheelchair_need}
            onClick={() => handleAccessChange('wheelchair_need', !accessForm.wheelchair_need)}
          >
            <span className={s.toggleIcon} aria-hidden="true">
              <FaWheelchair size={14} />
            </span>
            <span className={s.toggleLabel}>Fauteuil roulant requis</span>
            <span className={s.toggleState} aria-hidden="true">
              {accessForm.wheelchair_need ? <FiCheck size={14} /> : <FiX size={14} />}
            </span>
          </button>
          <button
            type="button"
            className={`${s.toggleBtn} ${accessForm.wheelchair_client_has ? s.toggleBtnActive : ''}`}
            aria-pressed={accessForm.wheelchair_client_has}
            onClick={() => handleAccessChange('wheelchair_client_has', !accessForm.wheelchair_client_has)}
          >
            <span className={s.toggleIcon} aria-hidden="true">
              <FaWheelchair size={14} />
            </span>
            <span className={s.toggleLabel}>Le patient dispose de son fauteuil</span>
            <span className={s.toggleState} aria-hidden="true">
              {accessForm.wheelchair_client_has ? <FiCheck size={14} /> : <FiX size={14} />}
            </span>
          </button>
        </div>
      </div>

      {isEnRoute && (
        <>
          <div className={s.editDivider} />
          <div className={s.editGroup}>
            <div className={s.editGroupTitle}>
              <FiFileText className={s.editGroupIcon} size={12} />
              Motif
            </div>
            <textarea
              id={fieldId('reason')}
              name="reason"
              aria-label="Motif de modification en route"
              className={s.editTextarea}
              value={accessForm.reason}
              onChange={(e) => handleAccessChange('reason', e.target.value)}
              rows={2}
              placeholder="Motif obligatoire (min. 10 caractères)"
            />
          </div>
        </>
      )}

      {saveError && <div className={s.saveErrorBanner}>{saveError}</div>}

      <div className={s.editFooter}>
        <button type="button" className={s.editCancelBtn} onClick={onCancel} disabled={patchMutation.isPending}>
          Annuler
        </button>
        <button type="button" className={s.editSaveBtn} onClick={handleSave} disabled={patchMutation.isPending}>
          {patchMutation.isPending ? 'Enregistrement…' : 'Enregistrer'}
        </button>
      </div>
    </div>
  );
};

export default InstitutionOperationalEdit;
