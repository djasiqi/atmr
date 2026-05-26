import React, { useMemo, useState } from 'react';
import { FiCheck, FiClock, FiFileText, FiHome, FiMapPin, FiUser, FiX } from 'react-icons/fi';
import { FaWheelchair } from 'react-icons/fa';
import AddressAutocomplete from '../../../components/common/AddressAutocomplete';
import InlineDatePicker from '../../../components/ui/InlineDatePicker';
import InlineTimePicker from '../../../components/ui/InlineTimePicker';
import s from './RequestDetailPanel.module.css';

const pad2 = (n) => String(n).padStart(2, '0');

const parseDate = (value) => {
  if (!value) return '';
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return String(value).slice(0, 10);
  return `${d.getFullYear()}-${pad2(d.getMonth() + 1)}-${pad2(d.getDate())}`;
};

const parseTime = (value) => {
  if (!value) return '';
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return '';
  return `${pad2(d.getHours())}:${pad2(d.getMinutes())}`;
};

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

  const initialForm = useMemo(() => {
    const scheduled = bs.scheduled_time || request.scheduled_time;
    return {
      pickup_location: textOrEmpty(bs.pickup_location || request.pickup_location),
      dropoff_location: textOrEmpty(bs.dropoff_location || request.dropoff_location),
      scheduled_date: parseDate(scheduled),
      scheduled_time: parseTime(scheduled),
      customer_name: textOrEmpty(bs.customer_name || patientName),
      medical_facility: textOrEmpty(bs.medical_facility),
      hospital_service: textOrEmpty(bs.hospital_service),
      doctor_name: textOrEmpty(bs.doctor_name),
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
    };
  }, [bs, patientName, request]);

  const [form, setForm] = useState(initialForm);
  const [saveError, setSaveError] = useState(null);

  const isDateTimeInPast = useMemo(() => {
    if (!form.scheduled_date || !form.scheduled_time || form.scheduled_time === '00:00') return false;
    const combined = new Date(`${form.scheduled_date}T${form.scheduled_time}:00`);
    return !Number.isNaN(combined.getTime()) && combined < new Date();
  }, [form.scheduled_date, form.scheduled_time]);

  const handleChange = (field, value) => {
    setForm((prev) => ({ ...prev, [field]: value }));
  };

  const handleSave = () => {
    if (!bookingId) {
      setSaveError('Booking introuvable pour cette demande.');
      return;
    }
    if (isDateTimeInPast) {
      setSaveError("La date et l'heure de la course sont dans le passé.");
      return;
    }
    if (isEnRoute && form.reason.trim().length < 10) {
      setSaveError('Motif obligatoire (10 caractères min.) pour une modification en route.');
      return;
    }

    const payload = {
      version: editVersion,
      reason: form.reason.trim() || undefined,
      customer_name: form.customer_name.trim() || undefined,
      pickup_location: form.pickup_location.trim(),
      dropoff_location: form.dropoff_location.trim(),
      scheduled_time: form.scheduled_date && form.scheduled_time
        ? `${form.scheduled_date}T${form.scheduled_time}:00`
        : undefined,
      medical_facility: form.medical_facility.trim() || null,
      hospital_service: form.hospital_service.trim() || null,
      doctor_name: form.doctor_name.trim() || null,
      pickup_floor: form.pickup_floor.trim() || null,
      pickup_door_code: form.pickup_door_code.trim() || null,
      dropoff_floor: form.dropoff_floor.trim() || null,
      dropoff_door_code: form.dropoff_door_code.trim() || null,
      pickup_access_notes: form.pickup_access_notes.trim() || null,
      dropoff_access_notes: form.dropoff_access_notes.trim() || null,
      notes_medical: form.notes_medical.trim() || null,
      wheelchair_need: Boolean(form.wheelchair_need),
      wheelchair_client_has: Boolean(form.wheelchair_client_has),
      delivery_description: form.delivery_description.trim() || null,
    };

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
          if (err?.response?.status === 409) {
            setSaveError(data?.error || 'Conflit de version, rechargez le détail.');
          } else {
            setSaveError(data?.error || 'Erreur lors de la sauvegarde.');
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
          value={form.pickup_floor}
          onChange={(e) => handleChange('pickup_floor', e.target.value)}
          placeholder="Étage"
        />
        <input
          id={fieldId('pickup-door-code')}
          name="pickup_door_code"
          autoComplete="off"
          aria-label={isReturnTrip ? 'Code porte à l\u2019arrivée' : 'Code porte au départ'}
          className={s.editInput}
          value={form.pickup_door_code}
          onChange={(e) => handleChange('pickup_door_code', e.target.value)}
          placeholder="Code porte"
        />
      </div>
      <input
        id={fieldId(`access-${homeAccessNotesField}`)}
        name={homeAccessNotesField}
        autoComplete="off"
        aria-label={isReturnTrip ? 'Consignes d\u2019arrivée' : 'Consignes de départ'}
        className={s.editInput}
        value={form[homeAccessNotesField]}
        onChange={(e) => handleChange(homeAccessNotesField, e.target.value)}
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
      <input
        id={fieldId('medical-facility')}
        name="medical_facility"
        autoComplete="organization"
        aria-label="Établissement / lieu"
        className={s.editInput}
        value={form.medical_facility}
        onChange={(e) => handleChange('medical_facility', e.target.value)}
        placeholder="Établissement / lieu"
      />
      <div className={s.editRow} style={{ marginTop: 8 }}>
        <input
          id={fieldId('hospital-service')}
          name="hospital_service"
          autoComplete="off"
          aria-label="Service"
          className={s.editInput}
          value={form.hospital_service}
          onChange={(e) => handleChange('hospital_service', e.target.value)}
          placeholder="Service"
        />
        <input
          id={fieldId('doctor-name')}
          name="doctor_name"
          autoComplete="off"
          aria-label="Médecin référent"
          className={s.editInput}
          value={form.doctor_name}
          onChange={(e) => handleChange('doctor_name', e.target.value)}
          placeholder="Médecin"
        />
      </div>
      <input
        id={fieldId(`access-${hospitalAccessNotesField}`)}
        name={hospitalAccessNotesField}
        autoComplete="off"
        aria-label={isReturnTrip ? 'Consignes de départ' : 'Consignes d\u2019arrivée'}
        className={s.editInput}
        value={form[hospitalAccessNotesField]}
        onChange={(e) => handleChange(hospitalAccessNotesField, e.target.value)}
        placeholder={isReturnTrip ? 'Consignes départ' : 'Consignes arrivée'}
        style={{ marginTop: 8 }}
      />
    </div>
  );

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

      <div className={s.editGroup}>
        <div className={s.editGroupTitle}>
          <FiMapPin className={s.editGroupIcon} size={12} />
          Itinéraire
        </div>
        <div className={s.editRoute}>
          <div className={s.editRouteTrack}>
            <span className={s.editRouteDotA} />
            <span className={s.editRouteLine} />
            <span className={s.editRouteDotB} />
          </div>
          <div className={s.editRouteFields}>
            <div className={s.editInputWrap}>
              <AddressAutocomplete
                name="pickup_location"
                inputId={fieldId('pickup-location')}
                value={form.pickup_location}
                onChange={(e) => handleChange('pickup_location', e?.target?.value ?? e)}
                placeholder="Départ"
              />
            </div>
            <div className={s.editInputWrap}>
              <AddressAutocomplete
                name="dropoff_location"
                inputId={fieldId('dropoff-location')}
                value={form.dropoff_location}
                onChange={(e) => handleChange('dropoff_location', e?.target?.value ?? e)}
                placeholder="Destination"
              />
            </div>
          </div>
        </div>
      </div>

      <div className={s.editDivider} />
      {isReturnTrip ? hospitalBlock : homeBlock}
      <div className={s.editDivider} />
      {isReturnTrip ? homeBlock : hospitalBlock}
      <div className={s.editDivider} />

      <div className={s.editGroup}>
        <div className={s.editGroupTitle}>
          <FiClock className={s.editGroupIcon} size={12} />
          Horaire
        </div>
        <div className={s.editRow}>
          <div className={s.editField}>
            <label
              htmlFor={fieldId('scheduled-date')}
              className={`${s.editLabel} ${isDateTimeInPast ? s.fieldErrorLabel : ''}`}
            >
              Date
            </label>
            <InlineDatePicker
              inputId={fieldId('scheduled-date')}
              value={form.scheduled_date}
              onChange={(v) => handleChange('scheduled_date', v)}
              placeholder="Date"
            />
          </div>
          <div className={s.editField}>
            <label
              htmlFor={fieldId('scheduled-time')}
              className={`${s.editLabel} ${isDateTimeInPast ? s.fieldErrorLabel : ''}`}
            >
              Heure
            </label>
            <InlineTimePicker
              inputId={fieldId('scheduled-time')}
              value={form.scheduled_time}
              onChange={(v) => handleChange('scheduled_time', v)}
              className={isDateTimeInPast ? s.fieldErrorInput : ''}
            />
          </div>
        </div>
        {isDateTimeInPast && <div className={s.fieldErrorHint}>La date et l'heure sont dans le passé</div>}
      </div>

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
          value={form.customer_name}
          onChange={(e) => handleChange('customer_name', e.target.value)}
          placeholder="Nom du patient"
        />
        <textarea
          id={fieldId('notes-medical')}
          name="notes_medical"
          aria-label="Notes médicales"
          className={s.editTextarea}
          value={form.notes_medical}
          onChange={(e) => handleChange('notes_medical', e.target.value)}
          placeholder="Pathologie, difficultés, mobilité…"
          rows={2}
          style={{ marginTop: 8 }}
        />
        <div className={s.toggleGroup}>
          <button
            type="button"
            className={`${s.toggleBtn} ${form.wheelchair_need ? s.toggleBtnActive : ''}`}
            aria-pressed={form.wheelchair_need}
            onClick={() => handleChange('wheelchair_need', !form.wheelchair_need)}
          >
            <span className={s.toggleIcon} aria-hidden="true">
              <FaWheelchair size={14} />
            </span>
            <span className={s.toggleLabel}>Fauteuil roulant requis</span>
            <span className={s.toggleState} aria-hidden="true">
              {form.wheelchair_need ? <FiCheck size={14} /> : <FiX size={14} />}
            </span>
          </button>
          <button
            type="button"
            className={`${s.toggleBtn} ${form.wheelchair_client_has ? s.toggleBtnActive : ''}`}
            aria-pressed={form.wheelchair_client_has}
            onClick={() => handleChange('wheelchair_client_has', !form.wheelchair_client_has)}
          >
            <span className={s.toggleIcon} aria-hidden="true">
              <FaWheelchair size={14} />
            </span>
            <span className={s.toggleLabel}>Le patient dispose de son fauteuil</span>
            <span className={s.toggleState} aria-hidden="true">
              {form.wheelchair_client_has ? <FiCheck size={14} /> : <FiX size={14} />}
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
              value={form.reason}
              onChange={(e) => handleChange('reason', e.target.value)}
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
