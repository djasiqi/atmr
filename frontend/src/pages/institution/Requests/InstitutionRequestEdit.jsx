import React, { useMemo, useState } from 'react';
import { FaPlus, FaTrash, FaRoute } from 'react-icons/fa';
import AddressAutocomplete from '../../../components/common/AddressAutocomplete';
import { useUpdateRequest } from '../../../hooks/useInstitutionData';
import InlineDatePicker from '../../../components/ui/InlineDatePicker';
import RouteStepTimeField from '../../../components/institution/RouteStepTimeField';
import ConfirmRequestEditModal from './ConfirmRequestEditModal';
import { combineMissionDateTime, extractHHMM } from '../../../utils/missionScheduleForm';
import { extractWallClockDate } from '../../../utils/missionTimeDisplay';
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
    setDestinations((prev) =>
      prev.map((dest, i) => (
        i === index
          ? {
            ...dest,
            address,
            establishment,
            doctor,
          }
          : dest
      )),
    );
  };

  const buildPayload = () => {
    const cleanedDestinations = destinations
      .map((d) => ({ ...d, address: (d.address || '').trim() }))
      .filter((d) => d.address);

    if (!pickupLocation.trim() || cleanedDestinations.length === 0) {
      window.alert('Le départ et au moins une destination sont obligatoires.');
      return null;
    }
    if (!missionDate) {
      window.alert('La date de mission est obligatoire.');
      return null;
    }

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
      if (dest.establishment) payload.dropoff_establishment = dest.establishment;
      if (dest.service) payload.dropoff_service = dest.service;
      if (dest.doctor) payload.dropoff_doctor = dest.doctor;

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
          window.alert(dataErr?.error || 'Erreur lors de la modification.');
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
                  onChange={(e) => setPickupLocation(e?.target?.value ?? e ?? '')}
                  onSelect={(item) => setPickupLocation(extractAddressFromPlace(item))}
                  placeholder="Adresse de départ"
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
            onChange={(v) => setMissionDate(v)}
            placeholder="Date"
          />
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
