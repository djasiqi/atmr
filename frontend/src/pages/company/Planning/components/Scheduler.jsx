import React, { useMemo, useRef, useState } from "react";

// Very lightweight timeline grid with basic drag/resize interactions (MVP adapter)
export default function Scheduler({
  items = [],
  onCreate,
  onUpdate,
  onDelete,
  overlays = {},
}) {
  const [drag, setDrag] = useState(null); // {id, startMs, endMs, mode:"move"|"resize-start"|"resize-end"}
  const containerRef = useRef(null);

  const dayStart = useMemo(() => {
    const d = new Date();
    d.setHours(6, 0, 0, 0);
    return d.getTime();
  }, []);
  const dayEnd = useMemo(() => {
    const d = new Date();
    d.setHours(22, 0, 0, 0);
    return d.getTime();
  }, []);
  const spanMs = dayEnd - dayStart;

  const statusColor = (s) => {
    const st = String(s.status || "planned").toLowerCase();
    if (st === "active") return "#0f766e";
    if (st === "completed") return "#2563eb";
    if (st === "cancelled") return "#b91c1c";
    return "#9ca3af"; // planned
  };

  const toX = (ms) =>
    `${Math.max(0, Math.min(1, (ms - dayStart) / spanMs)) * 100}%`;

  const onMouseDown = (e, item, edge) => {
    e.stopPropagation();
    setDrag({
      id: item?.id || null,
      startMs: new Date(item.start_local).getTime(),
      endMs: new Date(item.end_local).getTime(),
      mode: edge,
    });
  };

  const onMouseMove = (e) => {
    if (!drag) return;
    const step = 15 * 60000; // 15min grid
    const delta =
      (e.movementX / (containerRef.current?.clientWidth || 1)) * spanMs;
    if (drag.mode === "move") {
      setDrag((d) => ({
        ...d,
        startMs: d.startMs + delta,
        endMs: d.endMs + delta,
      }));
    } else if (drag.mode === "resize-start") {
      setDrag((d) => ({
        ...d,
        startMs: Math.min(d.endMs - step, d.startMs + delta),
      }));
    } else if (drag.mode === "resize-end") {
      setDrag((d) => ({
        ...d,
        endMs: Math.max(d.startMs + step, d.endMs + delta),
      }));
    }
  };

  const onMouseUp = () => {
    if (!drag) return;
    const round = (ms) => Math.round(ms / (15 * 60000)) * (15 * 60000);
    const payload = {
      start_local: new Date(round(drag.startMs)).toISOString(),
      end_local: new Date(round(drag.endMs)).toISOString(),
    };
    if (drag.id) onUpdate?.(drag.id, payload);
    setDrag(null);
  };

  const rendered = items.map((s) => {
    const startMs = new Date(s.start_local).getTime();
    const endMs = new Date(s.end_local).getTime();
    const left = toX(startMs);
    const right = toX(endMs);
    const width = `calc(${right} - ${left})`;
    return (
      <div
        key={s.id}
        style={{
          position: "relative",
          height: 28,
          marginBottom: 6,
          background: "#f9fafb",
          borderRadius: 6,
        }}
      >
        <div
          title={`${s.type || "Shift"} · ${s.status}`}
          style={{
            position: "absolute",
            left,
            width,
            top: 4,
            bottom: 4,
            background: statusColor(s),
            color: "#fff",
            borderRadius: 6,
            display: "flex",
            alignItems: "center",
            padding: "0 8px",
            cursor: "move",
          }}
          onMouseDown={(e) => onMouseDown(e, s, "move")}
        >
          <div
            style={{
              fontSize: 12,
              fontWeight: 600,
              whiteSpace: "nowrap",
              overflow: "hidden",
              textOverflow: "ellipsis",
            }}
          >
            {s.type || "Shift"} · {s.vehicle_plate || "—"}
          </div>
          <div
            onMouseDown={(e) => onMouseDown(e, s, "resize-start")}
            style={{
              position: "absolute",
              left: 0,
              top: 0,
              bottom: 0,
              width: 6,
              cursor: "ew-resize",
            }}
          />
          <div
            onMouseDown={(e) => onMouseDown(e, s, "resize-end")}
            style={{
              position: "absolute",
              right: 0,
              top: 0,
              bottom: 0,
              width: 6,
              cursor: "ew-resize",
            }}
          />
        </div>
      </div>
    );
  });

  return (
    <div
      ref={containerRef}
      onMouseMove={onMouseMove}
      onMouseUp={onMouseUp}
      style={{
        background: "#fff",
        border: "1px solid #e5e7eb",
        borderRadius: 8,
        padding: 12,
      }}
    >
      <div style={{ marginBottom: 8, fontWeight: 700 }}>Timeline (MVP)</div>
      <div style={{ position: "relative" }}>{rendered}</div>
      <div style={{ marginTop: 8 }}>
        <button
          onClick={() =>
            onCreate?.({
              driver_id: items[0]?.driver_id || 1,
              start_local: new Date().toISOString(),
              end_local: new Date(Date.now() + 2 * 3600 * 1000).toISOString(),
              type: "regular",
              status: "planned",
            })
          }
        >
          + Créer shift 2h
        </button>
      </div>
    </div>
  );
}
