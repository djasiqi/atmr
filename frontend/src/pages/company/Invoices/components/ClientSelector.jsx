import React, { useEffect, useState } from "react";
import { fetchCompanyClients } from "../../../../services/companyService";
import styles from "./ClientSelector.module.css";
export default function ClientSelector({ onSelectClient }) {
  const [clients, setClients] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadClients() {
      try {
        setLoading(true);
        const data = await fetchCompanyClients();
        // ⚠️ Sécuriser ici :
        setClients(Array.isArray(data) ? data : []);
      } finally {
        setLoading(false);
      }
    }
    loadClients();
  }, []);

  return (
    <div className={styles.selectorWrap}>
      <label className={styles.label} htmlFor="clientSelector">Choisir un client :</label>
      <select
        id="clientSelector"
        className={styles.select}
        disabled={loading}
        onChange={(e) => onSelectClient(e.target.value || null)}
        defaultValue=""
      >
        <option value="">-- Sélectionner --</option>
        {(clients || []).map((client) => (
          <option key={client.id} value={client.id}>
            {client.first_name} {client.last_name} {client.email ? `(${client.email})` : ""}
          </option>
        ))}
      </select>
      {loading && <span className={styles.loading}>Chargement clients...</span>}
    </div>
  );
}

