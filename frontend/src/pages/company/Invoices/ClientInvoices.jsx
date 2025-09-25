import React, { useEffect, useState } from "react";
import styles from "./ClientInvoices.module.css";
import CompanyHeader from "../../../components/layout/Header/CompanyHeader";
import CompanySidebar from "../../../components/layout/Sidebar/CompanySidebar/CompanySidebar";
import { fetchClientReservations, fetchCompanyClients } from "../../../services/companyService";
import ClientSelector from "./components/ClientSelector";
import { FaFilePdf } from "react-icons/fa";
import { mergeInvoiceAndQRBill } from "../../../utils/mergePDFs";

// Pour limiter les appels fetchCompanyClients, on garde une copie locale
const clientsCache = {};

const ClientInvoices = () => {
  const [selectedClientId, setSelectedClientId] = useState(null);
  const [reservations, setReservations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isExporting, setIsExporting] = useState(false);
  const [selectedMonth, setSelectedMonth] = useState("");
  const [clientInfo, setClientInfo] = useState(null);

  // Charge la liste des réservations du client sélectionné
  useEffect(() => {
    if (!selectedClientId) {
      setReservations([]);
      setClientInfo(null);
      return;
    }
    const loadData = async () => {
      try {
        setLoading(true);
        setError(null);
        // fetch réservations
        const data = await fetchClientReservations(selectedClientId);
        setReservations(Array.isArray(data?.reservations) ? data.reservations : []);
        // fetch info client (depuis cache ou API)
        let client = clientsCache[selectedClientId];
        if (!client) {
          const clients = await fetchCompanyClients();
          client = clients.find((c) => String(c.id) === String(selectedClientId));
          if (client) clientsCache[selectedClientId] = client;
        }
        setClientInfo(client || null);
      } catch (err) {
        setError(err?.error || "Erreur lors du chargement des données.");
        setReservations([]);
        setClientInfo(null);
      } finally {
        setLoading(false);
      }
    };
    loadData();
  }, [selectedClientId]);

  // Handler d'export PDF
  const handleExportClientPDF = async () => {
    if (!selectedMonth) {
      alert("Veuillez sélectionner un mois avant d'exporter.");
      return;
    }
    if (!reservations || reservations.length === 0) {
      alert("Aucune réservation à exporter.");
      return;
    }
    if (!clientInfo) {
      alert("Impossible de retrouver les informations du client.");
      return;
    }
    setIsExporting(true);
    try {
      // Filtrer les réservations du mois choisi
      const monthBookings = reservations.filter(
        (booking) =>
          new Date(booking.scheduled_time).getMonth() + 1 ===
          parseInt(selectedMonth, 10)
      );
      if (monthBookings.length === 0) {
        alert("Aucune réservation trouvée pour ce mois.");
        setIsExporting(false);
        return;
      }

      await mergeInvoiceAndQRBill({
        totalPrice: monthBookings.reduce((acc, curr) => acc + (curr.amount || 0), 0),
        client: {
          firstName: clientInfo.first_name || clientInfo.username || "Non spécifié",
          lastName: clientInfo.last_name || clientInfo.username || "Non spécifié",
          address: clientInfo.address || "",
          zipCode: clientInfo.zip_code || "",
          city: clientInfo.city || "",
        },
        bookings: monthBookings,
        // company: { ... } // à ajouter si tu veux un logo/nom société sur le PDF
      });

      alert("Facture générée avec succès !");
    } catch (error) {
      console.error("Erreur lors de l'exportation du PDF :", error);
      alert("Une erreur est survenue lors de l'exportation.");
    }
    setIsExporting(false);
  };

  // Formatage pour affichage
  const formattedReservations = reservations.map((reservation) => ({
    ...reservation,
    dateFormatted: reservation.scheduled_time
      ? new Date(reservation.scheduled_time).toLocaleDateString("fr-FR")
      : "-",
    trajet: `${reservation.pickup_location || "?"} → ${reservation.dropoff_location || "?"}`,
    amountDisplay: reservation.amount != null ? `${reservation.amount} CHF` : "Non défini",
  }));

  return (
    <div className={styles.companyContainer}>
      <CompanyHeader />
      <div className={styles.dashboard}>
        <CompanySidebar />
        <main className={styles.content}>
          {/* Sélecteur de client */}
          <ClientSelector onSelectClient={setSelectedClientId} />

          {error && <div className={styles.error}>{error}</div>}
          {loading && <p>Chargement...</p>}

          {/* Sélecteur de mois */}
          {!loading && reservations.length > 0 && (
            <div style={{ marginBottom: 12 }}>
              <select
                value={selectedMonth}
                onChange={(e) => setSelectedMonth(e.target.value)}
                className={styles.monthSelect}
              >
                <option value="">📅 Sélectionner un mois</option>
                {[...Array(12)].map((_, i) => (
                  <option key={i + 1} value={i + 1}>
                    {new Date(2025, i).toLocaleString("fr-FR", { month: "long" })}
                  </option>
                ))}
              </select>
            </div>
          )}

          {/* Bouton export PDF */}
          {!loading && reservations.length > 0 && (
            <div style={{ margin: "20px 0" }}>
              <button
                className={styles.exportBtn}
                onClick={handleExportClientPDF}
                disabled={isExporting || !selectedMonth}
              >
                <FaFilePdf /> {isExporting ? "Exportation..." : "Exporter la facture PDF"}
              </button>
            </div>
          )}

          {/* Tableau des réservations */}
          {!loading && reservations.length > 0 && (
            <table className={styles.table}>
              <thead>
                <tr>
                  <th>ID Réservation</th>
                  <th>Date</th>
                  <th>Trajet</th>
                  <th>Montant</th>
                </tr>
              </thead>
              <tbody>
                {formattedReservations.map((reservation) => (
                  <tr key={reservation.id}>
                    <td>{reservation.id}</td>
                    <td>{reservation.dateFormatted}</td>
                    <td>{reservation.trajet}</td>
                    <td>{reservation.amountDisplay}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}

          {!loading && selectedClientId && reservations.length === 0 && (
            <p>Aucune réservation pour ce client.</p>
          )}
        </main>
      </div>
    </div>
  );
};

export default ClientInvoices;
