// src/pages/Dashboard/ReservationChart.jsx
import React from "react";
import {
  ResponsiveContainer,
  LineChart,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  Line,
} from "recharts";

const ReservationChart = ({ reservations }) => {
  // On compte par jour "Genève" fourni par l'API, et on garde une clé UTC pour trier
  const dataMap = {};
  reservations.forEach((r) => {
    const label = r.date_formatted || new Date(r.scheduled_time).toLocaleDateString("fr-FR");
    const keyUtc = (r.scheduled_time || "").slice(0, 10); // "YYYY-MM-DD" (UTC)
    if (!dataMap[label]) dataMap[label] = { count: 0, keyUtc };
    dataMap[label].count += 1;
  });

  const chartData = Object.entries(dataMap)
    .map(([date, { count, keyUtc }]) => ({ date, count, keyUtc }))
    .sort((a, b) => (a.keyUtc < b.keyUtc ? -1 : a.keyUtc > b.keyUtc ? 1 : 0));

  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={chartData}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="date" />
        <YAxis allowDecimals={false} />
        <Tooltip />
        <Line type="monotone" dataKey="count" stroke="#00796b" />
      </LineChart>
    </ResponsiveContainer>
  );
};

export default ReservationChart;
