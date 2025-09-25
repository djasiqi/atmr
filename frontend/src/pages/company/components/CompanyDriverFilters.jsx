// src/pages/company/Driver/components/CompanyDriverFilters.jsx
import React from "react";
import styles from "./CompanyDriverFilters.module.css"; // Créez ce fichier CSS

const CompanyDriverFilters = ({
  searchTerm,
  setSearchTerm,
  statusFilter,
  setStatusFilter,
}) => {
  return (
    <div className={styles.filters}>
      <input
        type="text"
        placeholder="Rechercher un chauffeur..."
        value={searchTerm}
        onChange={(e) => setSearchTerm(e.target.value)}
        className={styles.searchInput}
      />
      <select
        value={statusFilter}
        onChange={(e) => setStatusFilter(e.target.value)}
        className={styles.statusFilter}
      >
        <option value="all">Tous</option>
        <option value="available">Disponible</option>
        <option value="on_trip">En course</option>
        <option value="inactive">Indisponible</option>
      </select>
    </div>
  );
};

export default CompanyDriverFilters;
