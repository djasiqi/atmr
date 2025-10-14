// frontend/src/services/analyticsService.js
/**
 * Service pour les appels API Analytics
 */
import apiClient from "../utils/apiClient";

/**
 * Récupère les analytics pour le dashboard
 * @param {string} period - Période ('7d', '30d', '90d', '1y')
 * @param {string} startDate - Date de début (YYYY-MM-DD) optionnelle
 * @param {string} endDate - Date de fin (YYYY-MM-DD) optionnelle
 * @returns {Promise} Analytics data
 */
export const fetchDashboardAnalytics = async ({
  period = "30d",
  startDate = null,
  endDate = null,
} = {}) => {
  try {
    const params = { period };
    
    if (startDate && endDate) {
      params.start_date = startDate;
      params.end_date = endDate;
    }

    const { data } = await apiClient.get("/analytics/dashboard", { params });
    
    if (data.success) {
      return data.data;
    } else {
      throw new Error(data.error || "Erreur lors du chargement des analytics");
    }
  } catch (error) {
    console.error("fetchDashboardAnalytics failed:", error?.response?.data || error);
    throw error;
  }
};

/**
 * Récupère les insights intelligents
 * @param {number} lookbackDays - Nombre de jours à analyser
 * @returns {Promise} Insights data
 */
export const fetchInsights = async (lookbackDays = 30) => {
  try {
    const { data } = await apiClient.get("/analytics/insights", {
      params: { lookback_days: lookbackDays },
    });
    
    if (data.success) {
      return data.data;
    } else {
      throw new Error(data.error || "Erreur lors du chargement des insights");
    }
  } catch (error) {
    console.error("fetchInsights failed:", error?.response?.data || error);
    throw error;
  }
};

/**
 * Récupère le résumé hebdomadaire
 * @param {string} weekStart - Date de début de semaine (YYYY-MM-DD) optionnelle
 * @returns {Promise} Weekly summary data
 */
export const fetchWeeklySummary = async (weekStart = null) => {
  try {
    const params = weekStart ? { week_start: weekStart } : {};
    const { data } = await apiClient.get("/analytics/weekly-summary", { params });
    
    if (data.success) {
      return data.data;
    } else {
      throw new Error(data.error || "Erreur lors du chargement du résumé hebdomadaire");
    }
  } catch (error) {
    console.error("fetchWeeklySummary failed:", error?.response?.data || error);
    throw error;
  }
};

/**
 * Exporte les analytics en CSV ou JSON
 * @param {string} startDate - Date de début (YYYY-MM-DD)
 * @param {string} endDate - Date de fin (YYYY-MM-DD)
 * @param {string} format - Format d'export ('csv' ou 'json')
 * @returns {Promise} Export data
 */
export const exportAnalytics = async (startDate, endDate, format = "csv") => {
  try {
    if (!startDate || !endDate) {
      throw new Error("Les dates de début et de fin sont requises");
    }

    const response = await apiClient.get("/analytics/export", {
      params: {
        start_date: startDate,
        end_date: endDate,
        format,
      },
      responseType: format === "csv" ? "blob" : "json",
    });

    return response.data;
  } catch (error) {
    console.error("exportAnalytics failed:", error?.response?.data || error);
    throw error;
  }
};

/**
 * Télécharge un fichier CSV
 * @param {Blob} blob - Données blob du CSV
 * @param {string} filename - Nom du fichier
 */
export const downloadCsvFile = (blob, filename) => {
  const url = window.URL.createObjectURL(new Blob([blob]));
  const link = document.createElement("a");
  link.href = url;
  link.setAttribute("download", filename);
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.URL.revokeObjectURL(url);
};

/**
 * Ouvre les données JSON dans un nouvel onglet
 * @param {Object} jsonData - Données JSON
 */
export const openJsonInNewTab = (jsonData) => {
  const jsonStr = JSON.stringify(jsonData, null, 2);
  const blob = new Blob([jsonStr], { type: "application/json" });
  const url = window.URL.createObjectURL(blob);
  window.open(url, "_blank");
  window.URL.revokeObjectURL(url);
};

