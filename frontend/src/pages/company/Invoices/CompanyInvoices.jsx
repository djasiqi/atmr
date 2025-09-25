import React, { useEffect, useState } from "react";
import styles from "./CompanyInvoices.module.css";
import CompanyHeader from "../../../components/layout/Header/CompanyHeader";
import CompanySidebar from "../../../components/layout/Sidebar/CompanySidebar/CompanySidebar";
import { fetchCompanyInvoices } from "../../../services/companyService";

const CompanyInvoices = () => {
  const [invoices, setInvoices] = useState([]);   // ← Toujours tableau
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const loadInvoices = async () => {
      try {
        setLoading(true);
        const data = await fetchCompanyInvoices();
        setInvoices(Array.isArray(data?.invoices) ? data.invoices : []); // ← SAFE
      } catch (err) {
        console.error("Erreur lors du chargement des factures :", err);
        setError("Erreur lors du chargement des factures.");
      } finally {
        setLoading(false);
      }
    };
    loadInvoices();
  }, []);

  return (
    <div className={styles.companyContainer}>
      <CompanyHeader />
      <div className={styles.dashboard}>
        <CompanySidebar />
        <main className={styles.content}>
          <h1>Factures</h1>
          {error && <div className={styles.error}>{error}</div>}
          {loading ? (
            <p>Chargement...</p>
          ) : invoices.length === 0 ? (
            <p>Aucune facture pour le moment.</p>
          ) : (
            <table className={styles.table}>
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Date</th>
                  <th>Montant</th>
                  <th>Statut</th>
                </tr>
              </thead>
              <tbody>
                {invoices.map((invoice) => (
                  <tr key={invoice.id}>
                    <td>{invoice.id}</td>
                    <td>
                      {invoice.date
                        ? new Date(invoice.date).toLocaleDateString("fr-FR")
                        : "-"}
                    </td>
                    <td>{invoice.amount} CHF</td>
                    <td>{invoice.status}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </main>
      </div>
    </div>
  );
};

export default CompanyInvoices;
