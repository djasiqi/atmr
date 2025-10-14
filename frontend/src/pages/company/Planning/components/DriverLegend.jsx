import React from "react";

export default function DriverLegend({ drivers = [] }) {
  return (
    <div style={{ padding: 8, border: "1px solid #eee", borderRadius: 6 }}>
      <div style={{ fontWeight: 600, marginBottom: 6 }}>Chauffeurs</div>
      <ul
        style={{ maxHeight: 200, overflow: "auto", margin: 0, paddingLeft: 16 }}
      >
        {drivers.map((d) => (
          <li key={d.id}>{d.full_name || d.username || `Driver #${d.id}`}</li>
        ))}
      </ul>
    </div>
  );
}
