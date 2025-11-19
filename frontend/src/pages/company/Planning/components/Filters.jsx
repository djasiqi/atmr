import React from 'react';

export default function Filters({ value, onChange }) {
  const v = value || {};
  return (
    <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
      <input
        placeholder="Recherche"
        value={v.q || ''}
        onChange={(e) => onChange({ ...v, q: e.target.value })}
      />
      <select value={v.status || ''} onChange={(e) => onChange({ ...v, status: e.target.value })}>
        <option value="">Tous statuts</option>
        <option value="planned">Planned</option>
        <option value="confirmed">Confirmed</option>
        <option value="done">Done</option>
      </select>
    </div>
  );
}
