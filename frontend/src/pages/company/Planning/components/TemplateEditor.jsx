import React, { useState } from 'react';

export default function TemplateEditor({ open, onClose, onSave }) {
  const [rows, setRows] = useState([]);
  if (!open) return null;

  const addRow = () =>
    setRows((r) => [
      ...r,
      { driver_id: '', weekday: 1, start_time: '08:00:00', end_time: '17:00:00' },
    ]);

  const save = () => onSave(rows);

  return (
    <div style={{ position: 'fixed', inset: 0, background: '#0006' }} onClick={onClose}>
      <div
        style={{
          background: '#fff',
          padding: 16,
          maxWidth: 640,
          margin: '8vh auto',
          borderRadius: 8,
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <h3>Modèle hebdomadaire</h3>
        <button onClick={addRow}>+ Ligne</button>
        <ul>
          {rows.map((r, i) => (
            <li key={i} style={{ marginTop: 8 }}>
              <input
                placeholder="driver_id"
                value={r.driver_id}
                onChange={(e) => {
                  const c = [...rows];
                  c[i].driver_id = e.target.value;
                  setRows(c);
                }}
              />
              <input
                placeholder="weekday"
                value={r.weekday}
                onChange={(e) => {
                  const c = [...rows];
                  c[i].weekday = Number(e.target.value);
                  setRows(c);
                }}
              />
              <input
                placeholder="start_time"
                value={r.start_time}
                onChange={(e) => {
                  const c = [...rows];
                  c[i].start_time = e.target.value;
                  setRows(c);
                }}
              />
              <input
                placeholder="end_time"
                value={r.end_time}
                onChange={(e) => {
                  const c = [...rows];
                  c[i].end_time = e.target.value;
                  setRows(c);
                }}
              />
            </li>
          ))}
        </ul>
        <div style={{ marginTop: 12, display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
          <button onClick={onClose}>Fermer</button>
          <button onClick={save}>Enregistrer</button>
        </div>
      </div>
    </div>
  );
}
