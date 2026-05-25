import React, { useState } from 'react';
import { useUpdateRequest } from '../../../hooks/useInstitutionData';
import s from './RequestDetailPanel.module.css';

const toLocalDateTimeInputValue = (iso) => {
  if (!iso) return '';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return '';
  const pad = (n) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
};

const InstitutionRequestEdit = ({ request, onCancel, onSaved }) => {
  const [pickupLocation, setPickupLocation] = useState(request.pickup_location || '');
  const [dropoffLocation, setDropoffLocation] = useState(request.dropoff_location || '');
  const [scheduledTime, setScheduledTime] = useState(
    toLocalDateTimeInputValue(request.scheduled_time),
  );
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
    if (scheduledTime) {
      payload.scheduled_time = new Date(scheduledTime).toISOString();
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
      <label className={s.editLabel}>
        Horaire
        <input
          type="datetime-local"
          className={s.editInput}
          value={scheduledTime}
          onChange={(e) => setScheduledTime(e.target.value)}
        />
      </label>
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
