// DriverYearCalendar.jsx
import React from "react";
import { Calendar } from "react-yearly-calendar";

function DriverYearCalendar({ absences }) {
  // absences = [{ start_date: '2025-07-10', end_date: '2025-07-20', ... }, ...]

  // Convertir ce tableau en un ensemble de “highlighted days”.
  // Par exemple, on peut colorer les jours d'absence en rouge.

  // Ex: on construit un objet { '2025-07-10': 'vacation', '2025-07-11': 'vacation', ... }
  // Selon la doc de react-yearly-calendar, tu peux passer “customClasses” ou un “disabledDates”.

  const customDatesStyles = [];
  absences.forEach((abs) => {
    let current = new Date(abs.start_date);
    const end = new Date(abs.end_date);
    while (current <= end) {
      const iso = current.toISOString().split("T")[0];
      customDatesStyles.push({
        date: iso,
        className: "vacationDay", // tu définiras la classe CSS
      });
      current.setDate(current.getDate() + 1);
    }
  });

  return (
    <div>
      <Calendar year={2025} customDatesStyles={customDatesStyles} />
    </div>
  );
}

export default DriverYearCalendar;
