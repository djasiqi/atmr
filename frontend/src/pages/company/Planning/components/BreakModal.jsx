import React, { useEffect, useState } from 'react';
import styles from './Modal.module.css';

export default function BreakModal({ open, initial, onClose, onSave }) {
  const [form, setForm] = useState({
    shift_id: '',
    start_local: '',
    end_local: '',
    type: 'mandatory',
  });

  useEffect(() => {
    if (open) {
      setForm({
        shift_id: initial?.shift_id || '',
        start_local: initial?.start_local || '',
        end_local: initial?.end_local || '',
        type: initial?.type || 'mandatory',
      });
    }
  }, [open, initial]);

  if (!open) return null;

  const save = () => onSave(form);

  return (
    <div className={styles.modalOverlay} onClick={onClose}>
      <div className={styles.modalContent} onClick={(e) => e.stopPropagation()}>
        <h3>Pause</h3>
        <label>
          Shift ID
          <input
            value={form.shift_id}
            onChange={(e) => setForm({ ...form, shift_id: e.target.value })}
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
            <option value="mandatory">mandatory</option>
            <option value="optional">optional</option>
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
