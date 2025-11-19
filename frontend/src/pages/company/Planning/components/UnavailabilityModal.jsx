import React, { useEffect, useState } from 'react';

export default function UnavailabilityModal({ open, initial, onClose, onSave }) {
  const [form, setForm] = useState({
    driver_id: '',
    start_local: '',
    end_local: '',
    reason: 'other',
    note: '',
  });

  useEffect(() => {
    if (open) {
      setForm({
        driver_id: initial?.driver_id || '',
        start_local: initial?.start_local || '',
        end_local: initial?.end_local || '',
        reason: initial?.reason || 'other',
        note: initial?.note || '',
      });
    }
  }, [open, initial]);

  if (!open) return null;

  const save = () => onSave(form);

  return (
    <div style={{ position: 'fixed', inset: 0, background: '#0006' }} onClick={onClose}>
      <div
        style={{
          background: '#fff',
          padding: 16,
          maxWidth: 420,
          margin: '10vh auto',
          borderRadius: 8,
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <h3>Indisponibilité</h3>
        <label>
          Driver ID
          <input
            value={form.driver_id}
            onChange={(e) => setForm({ ...form, driver_id: e.target.value })}
          />
        </label>
        <label>
          Début
          <input
            value={form.start_local}
            onChange={(e) => setForm({ ...form, start_local: e.target.value })}
          />
        </label>
        <label>
          Fin
          <input
            value={form.end_local}
            onChange={(e) => setForm({ ...form, end_local: e.target.value })}
          />
        </label>
        <label>
          Raison
          <select
            value={form.reason}
            onChange={(e) => setForm({ ...form, reason: e.target.value })}
          >
            <option value="other">other</option>
            <option value="vacation">vacation</option>
            <option value="sick">sick</option>
            <option value="training">training</option>
          </select>
        </label>
        <label>
          Note
          <input value={form.note} onChange={(e) => setForm({ ...form, note: e.target.value })} />
        </label>
        <div style={{ marginTop: 12, display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
          <button onClick={onClose}>Annuler</button>
          <button onClick={save}>Enregistrer</button>
        </div>
      </div>
    </div>
  );
}
