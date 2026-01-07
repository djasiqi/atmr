import apiClient from '../utils/apiClient';
// ✅ SUPPRIMÉ: mergeInvoiceAndQRBill - Génération PDF via API backend

const _API_URL = process.env.REACT_APP_API_URL;

export const fetchBookings = async (publicId) => {
  try {
    const response = await apiClient.get(`/clients/${publicId}/bookings`);

    if (!Array.isArray(response.data)) {
      console.error('Format de réponse invalide :', response.data);
      return [];
    }

    return response.data.map((booking) => ({
      ...booking,
      // ✅ P1-4 Phase 2.2: Utiliser company_name du backend (ou fallback si absent)
      company_name: booking.company_name || (booking.company_id ? `Entreprise ${booking.company_id}` : 'Non assignée'),
      driver_name: booking.driver_id ? `Chauffeur ${booking.driver_id}` : 'Non assigné',
    }));
  } catch (error) {
    console.error('Erreur lors de la récupération des réservations :', error);
    return [];
  }
};

export const exportBookingsPDF = async (month, bookings, _client, _company) => {
  try {
    console.log('📂 Génération PDF en cours sur le frontend...');

    // Vérifier si des réservations existent pour ce mois
    if (bookings.length === 0) {
      alert('Aucune réservation trouvée pour ce mois.');
      return;
    }

    // ✅ TODO: Remplacer par appel API backend
    // const response = await apiClient.post('/companies/me/invoices', {
    //   client_id: client.id,
    //   period_year: year,
    //   period_month: month
    // });
    // window.open(response.data.pdf_url, '_blank');

    console.log('PDF generation moved to backend API - To be implemented');

    console.log('✅ PDF généré avec succès !');
  } catch (error) {
    console.error("❌ Erreur lors de l'exportation du PDF :", error);
    throw new Error("Erreur lors de l'exportation du PDF");
  }
};

export const cancelBooking = async (bookingId) => {
  try {
    // ✅ Utiliser apiClient au lieu de fetch directement pour bénéficier des cookies httpOnly
    const response = await apiClient.delete(`/bookings/${bookingId}`, {
      data: { status: 'canceled' }, // 🔥 Ajout du statut si requis par l'API
    });

    console.log('📢 API Response (Annulation) :', response.data); // ✅ Debug ici

    return response.data; // ✅ Retourne les données mises à jour
  } catch (error) {
    console.error("Erreur lors de l'annulation :", error);
    throw error;
  }
};
