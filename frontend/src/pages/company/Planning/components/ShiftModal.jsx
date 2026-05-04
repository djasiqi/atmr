import React, { useEffect, useState } from 'react';
import styles from './Modal.module.css';

export default function ShiftModal({ open, initial, onClose, onSave }) {
  const [form, setForm] = useState({
    driver_id: '',
    start_local: '',
    end_local: '',
    type: 'regular',
    status: 'planned',
  });

  useEffect(() => {
    if (open) {
      setForm({
        driver_id: initial?.driver_id || '',
        start_local: initial?.start_local || '',
        end_local: initial?.end_local || '',
        type: initial?.type || 'regular',
        status: initial?.status || 'planned',
      });
    }
  }, [open, initial]);

  if (!open) return null;

  const save = () => onSave(form);

  return (
    <div className={styles.modalOverlay} onClick={onClose}>
      <div className={styles.modalContent} onClick={(e) => e.stopPropagation()}>
        <h3>Shift</h3>
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
          Type
          <select value={form.type} onChange={(e) => setForm({ ...form, type: e.target.value })}>
            <option value="regular">regular</option>
            <option value="night">night</option>
            <option value="overtime">overtime</option>
          </select>
        </label>
        <label>
          Statut
          <select
            value={form.status}
            onChange={(e) => setForm({ ...form, status: e.target.value })}
          >
            <option value="planned">planned</option>
            <option value="confirmed">confirmed</option>
            <option value="done">done</option>
          </select>
        </label>
        <div className={styles.modalActions}>
          <button onClick={onClose}>Annuler</button>
          <button onClick={save}>Enregistrer</button>
        </div>
      </div>
    </div>
  );
}
