import React, { useState } from 'react';
import { useUpdateRequest } from '../../../hooks/useInstitutionData';
import InlineDatePicker from '../../../components/ui/InlineDatePicker';
import InlineTimePicker from '../../../components/ui/InlineTimePicker';
import s from './RequestDetailPanel.module.css';

const pad2 = (n) => String(n).padStart(2, '0');

const parseDate = (iso) => {
  if (!iso) return '';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return String(iso).slice(0, 10);
  return `${d.getFullYear()}-${pad2(d.getMonth() + 1)}-${pad2(d.getDate())}`;
};

const parseTime = (iso) => {
  if (!iso) return '';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return '';
  return `${pad2(d.getHours())}:${pad2(d.getMinutes())}`;
};

const InstitutionRequestEdit = ({ request, onCancel, onSaved }) => {
  const [pickupLocation, setPickupLocation] = useState(request.pickup_location || '');
  const [dropoffLocation, setDropoffLocation] = useState(request.dropoff_location || '');
  const [scheduledDate, setScheduledDate] = useState(parseDate(request.scheduled_time));
  const [scheduledTime, setScheduledTime] = useState(parseTime(request.scheduled_time));
  const [notes, setNotes] = useState(request.notes || '');
  const [requiresWheelchair, setRequiresWheelchair] = useState(
    Boolean(request.requires_wheelchair),
  );

  const updateMutation = useUpdateRequest();

  const handleSave = () => {
    if (!pickupLocation.trim() || !dropoffLocation.trim()) {
      window.alert('Adresses départ et arrivée obligatoires.');
      return;
    }
    const payload = {
      pickup_location: pickupLocation.trim(),
      dropoff_location: dropoffLocation.trim(),
      notes: notes || null,
    };
    if (scheduledDate && scheduledTime) {
      const combined = new Date(`${scheduledDate}T${scheduledTime}:00`);
      if (!Number.isNaN(combined.getTime())) {
        payload.scheduled_time = combined.toISOString();
      }
    }
    if (typeof requiresWheelchair === 'boolean') {
      payload.mobility = requiresWheelchair ? 'wheelchair' : 'walking';
    }

    updateMutation.mutate(
      { requestId: request.id, data: payload },
      {
        onSuccess: () => onSaved?.(),
        onError: (err) => {
          const data = err?.response?.data;
          window.alert(data?.error || 'Erreur lors de la modification.');
        },
      },
    );
  };

  return (
    <div className={s.section}>
      <label className={s.editLabel}>
        Départ
        <input
          className={s.editInput}
          value={pickupLocation}
          onChange={(e) => setPickupLocation(e.target.value)}
        />
      </label>
      <label className={s.editLabel}>
        Arrivée
        <input
          className={s.editInput}
          value={dropoffLocation}
          onChange={(e) => setDropoffLocation(e.target.value)}
        />
      </label>
      <div className={s.editRow}>
        <div className={s.editField}>
          <label htmlFor="edit-scheduled-date" className={s.editLabel}>
            Date
          </label>
          <InlineDatePicker
            inputId="edit-scheduled-date"
            value={scheduledDate}
            onChange={(v) => setScheduledDate(v)}
            placeholder="Date"
          />
        </div>
        <div className={s.editField}>
          <label htmlFor="edit-scheduled-time" className={s.editLabel}>
            Heure
          </label>
          <InlineTimePicker
            inputId="edit-scheduled-time"
            value={scheduledTime}
            onChange={(v) => setScheduledTime(v)}
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
      <label className={s.editCheckbox}>
        <input
          type="checkbox"
          checked={requiresWheelchair}
          onChange={(e) => setRequiresWheelchair(e.target.checked)}
        />
        Fauteuil roulant requis
      </label>
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
    </div>
  );
};

export default InstitutionRequestEdit;
