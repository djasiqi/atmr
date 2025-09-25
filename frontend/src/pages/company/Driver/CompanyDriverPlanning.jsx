// CompanyDriverPlanning.jsx
import React, { useEffect, useState } from "react";
import DriverYearCalendar from "./DriverYearCalendar";
// ou un composant Gantt, etc.
import { fetchDriverVacations, fetchCompanyDriver } from "../../../services/driverPlanningService";
// import ou define d’autres services
// import styles si besoin

function CompanyDriverPlanning() {
  const [driver, setDriver] = useState([]);
  const [selectedDriver, setSelectedDriver] = useState(null);
  const [vacations, setVacations] = useState([]);

  useEffect(() => {
    // Charger la liste de chauffeurs
    fetchCompanyDriver().then((res) => setDriver(res));
  }, []);

  useEffect(() => {
    if (selectedDriver) {
      // Charger les vacances du chauffeur
      fetchDriverVacations(selectedDriver.id).then((res) => setVacations(res));
    }
  }, [selectedDriver]);

  function handleDriverChange(event) {
    const driverId = event.target.value;
    const found = driver.find((d) => d.id === Number(driverId));
    setSelectedDriver(found);
  }

  return (
    <div>
      <h1>Planning des chauffeurs</h1>

      <label>Chauffeur : </label>
      <select onChange={handleDriverChange}>
        <option value="">Sélectionner un chauffeur</option>
        {driver.map((drv) => (
          <option key={drv.id} value={drv.id}>
            {drv.username}
          </option>
        ))}
      </select>

      {selectedDriver && (
        <div>
          <h2>Calendrier de {selectedDriver.username}</h2>
          <DriverYearCalendar vacations={vacations} />
        </div>
      )}
    </div>
  );
}

export default CompanyDriverPlanning;
