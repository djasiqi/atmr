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

const ReservationChart = ({ reservations = [] }) => {
  // On compte par jour "Genève" fourni par l'API, et on garde une clé UTC pour trier
  const dataMap = {};
  
  // Ensure reservations is an array
  const reservationArray = Array.isArray(reservations) ? reservations : [];
  
  // Generate some default data if empty
  if (reservationArray.length === 0) {
    const today = new Date();
    const yesterday = new Date(today);
    yesterday.setDate(yesterday.getDate() - 1);
    const twoDaysAgo = new Date(today);
    twoDaysAgo.setDate(twoDaysAgo.getDate() - 2);
    
    // Format dates
    const formatDate = (date) => {
      return date.toLocaleDateString("fr-FR");
    };
    
    // Add sample data
    dataMap[formatDate(twoDaysAgo)] = { count: 0, keyUtc: twoDaysAgo.toISOString().slice(0, 10) };
    dataMap[formatDate(yesterday)] = { count: 0, keyUtc: yesterday.toISOString().slice(0, 10) };
    dataMap[formatDate(today)] = { count: 0, keyUtc: today.toISOString().slice(0, 10) };
  } else {
    // Process actual reservation data
    reservationArray.forEach((r) => {
      try {
        const label = r.date_formatted || new Date(r.scheduled_time || r.pickup_time || r.date_time || new Date()).toLocaleDateString("fr-FR");
        const keyUtc = ((r.scheduled_time || r.pickup_time || r.date_time || "").slice(0, 10)) || new Date().toISOString().slice(0, 10); // "YYYY-MM-DD" (UTC)
        if (!dataMap[label]) dataMap[label] = { count: 0, keyUtc };
        dataMap[label].count += 1;
      } catch (e) {
        console.error("Error processing reservation for chart:", e);
      }
    });
  }

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
