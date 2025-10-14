import React from "react";
import usePlanningKPIs from "../hooks/usePlanningKPIs";

export default function KPIs({ items = [], mode = "day" }) {
  const k = usePlanningKPIs(items, mode);

  return (
    <div style={{ display: "flex", gap: 12 }}>
      <Badge label="Total" value={k.total} />
      <Badge label="Planned" value={k.planned} />
      <Badge label="Confirmed" value={k.confirmed} />
      <Badge label="Done" value={k.done} />
    </div>
  );
}

function Badge({ label, value }) {
  return (
    <div style={{ padding: 8, background: "#f3f4f6", borderRadius: 6, minWidth: 80, textAlign: "center" }}>
      <div style={{ fontSize: 12, color: "#6b7280" }}>{label}</div>
      <div style={{ fontWeight: 700 }}>{value}</div>
    </div>
  );
}


