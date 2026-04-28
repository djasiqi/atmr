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

    return response.data.map((booking) => {
      const fromApi =
        (booking.driver_name && String(booking.driver_name).trim()) ||
        (booking.driver?.full_name && String(booking.driver.full_name).trim()) ||
        '';
      return {
        ...booking,
        // ✅ P1-4 Phase 2.2: Utiliser company_name du backend (ou fallback si absent)
        company_name:
          booking.company_name ||
          (booking.company_id ? `Entreprise ${booking.company_id}` : 'Non assignée'),
        // Ne pas écraser driver_name : le backend envoie le vrai nom. L’ancien placeholder
        // « Chauffeur {id} » faisait échouer hasDriverAssigned() côté portail client.
        ...(fromApi ? { driver_name: fromApi } : {}),
      };
    });
  } catch (error) {
    console.error('Erreur lors de la récupération des réservations :', error);
    throw error;
  }
};

export const exportBookingsPDF = async (month, bookings, _client, _company) => {
  if (!Array.isArray(bookings) || bookings.length === 0) {
    const emptyError = new Error('Aucune donnée exportable pour la période sélectionnée.');
    emptyError.code = 'empty_period';
    throw emptyError;
  }
  const clientPublicId = _client?.public_id || _client?.id || null;
  const payload = {
    client_public_id: clientPublicId,
    period: month,
  };
  if (month === 'custom') {
    const dates = bookings
      .map((item) => Date.parse(item?.scheduled_time))
      .filter((value) => Number.isFinite(value))
      .sort((a, b) => a - b);
    if (dates.length === 0) {
      const customError = new Error('Période personnalisée invalide.');
      customError.code = 'custom_period_invalid';
      throw customError;
    }
    payload.from = new Date(dates[0]).toISOString();
    payload.to = new Date(dates[dates.length - 1]).toISOString();
  }

  const response = await apiClient.post('/bookings/clients/me/bookings/export-pdf', payload, {
    responseType: 'blob',
    headers: {
      Accept: 'application/pdf, application/json',
    },
  });

  const contentType = String(response?.headers?.['content-type'] || '').toLowerCase();
  if (contentType.includes('application/json')) {
    const asText = await response.data.text();
    const parsed = JSON.parse(asText || '{}');
    if (parsed?.pdf_url) {
      return {
        pdfUrl: parsed.pdf_url,
        periodLabel: parsed.period_label || null,
        totalAmount: parsed.total_amount ?? null,
        rowsCount: parsed.rows_count ?? null,
      };
    }
    throw new Error(parsed?.message || "Impossible d'exporter le PDF.");
  }

  if (!contentType.includes('application/pdf')) {
    throw new Error("Format d'export PDF invalide.");
  }

  const blob = new Blob([response.data], { type: 'application/pdf' });
  const href = window.URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  const periodLabel = String(month || 'periode');
  anchor.href = href;
  anchor.download = `courses_${periodLabel}.pdf`;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  window.URL.revokeObjectURL(href);

  return {
    periodLabel: response?.headers?.['x-period-label'] || null,
    totalAmount: response?.headers?.['x-total-amount'] || null,
    rowsCount: response?.headers?.['x-rows-count'] || null,
  };
};

/** Mini-chat booking (portail client ↔ entreprise), même API que l’espace entreprise. */
export const fetchBookingMessagesForClient = async (bookingId, { limit = 30, beforeId } = {}) => {
  const params = new URLSearchParams({ limit: String(limit) });
  if (beforeId) params.set('before_id', String(beforeId));
  const { data } = await apiClient.get(`/bookings/${bookingId}/messages?${params}`);
  return data;
};

export const sendBookingMessageAsClient = async (bookingId, content) => {
  const { data } = await apiClient.post(`/bookings/${bookingId}/messages`, { content });
  return data;
};

export const cancelBooking = async (bookingId) => {
  try {
    // ✅ Utiliser apiClient au lieu de fetch directement pour bénéficier des cookies httpOnly
    const response = await apiClient.delete(`/bookings/${bookingId}`, {
      data: { status: 'canceled' }, // 🔥 Ajout du statut si requis par l'API
    });

    if (process.env.NODE_ENV === 'development') {
      console.log('📢 API Response (Annulation) :', response.data);
    }

    return response.data; // ✅ Retourne les données mises à jour
  } catch (error) {
    console.error("Erreur lors de l'annulation :", error);
    throw error;
  }
};
