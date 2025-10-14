// frontend/src/pages/company/Analytics/AnalyticsDashboard.jsx
/**
 * 📊 DASHBOARD ANALYTICS
 *
 * Affiche les métriques de performance du système de dispatch :
 * - KPIs clés (courses, ponctualité, retards, qualité)
 * - Graphiques de tendances (recharts)
 * - Insights intelligents
 * - Export de données
 */

import React, { useState, useEffect, useCallback } from "react";
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Area,
  AreaChart,
} from "recharts";
import CompanyHeader from "../../../components/layout/Header/CompanyHeader";
import CompanySidebar from "../../../components/layout/Sidebar/CompanySidebar/CompanySidebar";
import styles from "./AnalyticsDashboard.module.css";
import {
  fetchDashboardAnalytics,
  exportAnalytics,
  downloadCsvFile,
  openJsonInNewTab,
} from "../../../services/analyticsService";

const AnalyticsDashboard = () => {
  // États
  const [period, setPeriod] = useState("7d"); // Changé à 7d pour inclure le 15 octobre
  const [loading, setLoading] = useState(false);
  const [analytics, setAnalytics] = useState(null);
  const [error, setError] = useState(null);

  // Récupération des analytics
  const fetchAnalytics = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const data = await fetchDashboardAnalytics({ period });
      setAnalytics(data);
    } catch (err) {
      setError(err.message || "Impossible de charger les analytics");
      console.error("Failed to fetch analytics:", err);
    } finally {
      setLoading(false);
    }
  }, [period]);

  useEffect(() => {
    fetchAnalytics();
  }, [fetchAnalytics]);

  // Rendu du loading analytics
  if (loading) {
    return (
      <div className={styles.companyContainer}>
        <CompanyHeader />
        <div className={styles.dashboardLayout}>
          <CompanySidebar />
          <div className={styles.mainContent}>
            <div className={styles.loadingContainer}>
              <div className={styles.spinner}></div>
              <p>Chargement de l'entreprise...</p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Rendu du loading analytics
  if (loading) {
    return (
      <div className={styles.companyContainer}>
        <CompanyHeader />
        <div className={styles.dashboardLayout}>
          <CompanySidebar />
          <div className={styles.mainContent}>
            <div className={styles.loadingContainer}>
              <div className={styles.spinner}></div>
              <p>Chargement des analytics...</p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Rendu de l'erreur
  if (error) {
    return (
      <div className={styles.companyContainer}>
        <CompanyHeader />
        <div className={styles.dashboardLayout}>
          <CompanySidebar />
          <div className={styles.mainContent}>
            <div className={styles.errorContainer}>
              <h2>❌ Erreur</h2>
              <p>{error}</p>
              <button onClick={fetchAnalytics} className={styles.retryButton}>
                Réessayer
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Données vides
  if (!analytics || !analytics.trends || analytics.trends.length === 0) {
    return (
      <div className={styles.companyContainer}>
        <CompanyHeader />
        <div className={styles.dashboardLayout}>
          <CompanySidebar />
          <div className={styles.mainContent}>
            <div className={styles.emptyState}>
              <h2>📊 Analytics</h2>
              <p>Aucune donnée disponible pour le moment.</p>
              <p className={styles.emptyHint}>
                Lancez des dispatches pour commencer à collecter des métriques.
              </p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  const { summary, trends, insights } = analytics;

  return (
    <div className={styles.companyContainer}>
      <CompanyHeader />

      <div className={styles.dashboardLayout}>
        <CompanySidebar />

        <div className={styles.mainContent}>
          <div className={styles.analytics}>
            {/* Header avec sélecteur de période */}
            <header className={styles.analyticsHeader}>
              <div className={styles.headerLeft}>
                <h1>📊 Analytics & Performance</h1>
                <p className={styles.subtitle}>
                  Analyse de la performance du système de dispatch
                </p>
              </div>

              <div className={styles.periodSelector}>
                <button
                  className={
                    period === "7d" ? styles.periodActive : styles.periodButton
                  }
                  onClick={() => setPeriod("7d")}
                >
                  7 jours
                </button>
                <button
                  className={
                    period === "30d" ? styles.periodActive : styles.periodButton
                  }
                  onClick={() => setPeriod("30d")}
                >
                  30 jours
                </button>
                <button
                  className={
                    period === "90d" ? styles.periodActive : styles.periodButton
                  }
                  onClick={() => setPeriod("90d")}
                >
                  90 jours
                </button>
              </div>
            </header>

            {/* KPI Cards */}
            <div className={styles.kpiGrid}>
              <div className={styles.kpiCard}>
                <div className={styles.kpiIcon}>📦</div>
                <div className={styles.kpiContent}>
                  <h3 className={styles.kpiLabel}>Total Courses</h3>
                  <p className={styles.kpiValue}>
                    {summary.total_bookings || 0}
                  </p>
                </div>
              </div>

              <div className={styles.kpiCard}>
                <div className={styles.kpiIcon}>✅</div>
                <div className={styles.kpiContent}>
                  <h3 className={styles.kpiLabel}>Taux à l'heure</h3>
                  <p className={styles.kpiValue}>
                    {summary.avg_on_time_rate
                      ? summary.avg_on_time_rate.toFixed(1)
                      : "0.0"}
                    %
                  </p>
                </div>
              </div>

              <div className={styles.kpiCard}>
                <div className={styles.kpiIcon}>⏱️</div>
                <div className={styles.kpiContent}>
                  <h3 className={styles.kpiLabel}>Retard moyen</h3>
                  <p className={styles.kpiValue}>
                    {summary.avg_delay_minutes
                      ? summary.avg_delay_minutes.toFixed(1)
                      : "0.0"}{" "}
                    min
                  </p>
                </div>
              </div>

              <div className={styles.kpiCard}>
                <div className={styles.kpiIcon}>⭐</div>
                <div className={styles.kpiContent}>
                  <h3 className={styles.kpiLabel}>Score Qualité</h3>
                  <p className={styles.kpiValue}>
                    {summary.avg_quality_score
                      ? summary.avg_quality_score.toFixed(0)
                      : "0"}
                    /100
                  </p>
                </div>
              </div>
            </div>

            {/* Insights */}
            {insights && insights.length > 0 && (
              <div className={styles.insightsSection}>
                <h2 className={styles.sectionTitle}>
                  💡 Insights & Recommandations
                </h2>
                <div className={styles.insightsList}>
                  {insights.map((insight, index) => (
                    <div
                      key={index}
                      className={`${styles.insightCard} ${
                        styles[
                          `insight${
                            insight.priority.charAt(0).toUpperCase() +
                            insight.priority.slice(1)
                          }`
                        ]
                      }`}
                    >
                      <div className={styles.insightHeader}>
                        <span className={styles.insightIcon}>
                          {insight.type === "success"
                            ? "✅"
                            : insight.type === "warning"
                            ? "⚠️"
                            : "ℹ️"}
                        </span>
                        <span className={styles.insightTitle}>
                          {insight.title}
                        </span>
                      </div>
                      <p className={styles.insightMessage}>{insight.message}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Graphiques */}
            <div className={styles.chartsGrid}>
              {/* Graphique 1: Volume de Courses */}
              <div className={styles.chartCard}>
                <h3 className={styles.chartTitle}>📦 Volume de Courses</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={trends}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                    <XAxis
                      dataKey="date"
                      tick={{ fontSize: 12, fill: "#6b7280" }}
                      tickFormatter={(value) => {
                        const d = new Date(value);
                        return `${d.getDate()}/${d.getMonth() + 1}`;
                      }}
                    />
                    <YAxis tick={{ fontSize: 12, fill: "#6b7280" }} />
                    <Tooltip
                      contentStyle={{
                        background: "#ffffff",
                        border: "1px solid #e5e7eb",
                        borderRadius: "8px",
                      }}
                    />
                    <Legend />
                    <Bar
                      dataKey="bookings"
                      fill="#0f766e"
                      name="Courses"
                      radius={[8, 8, 0, 0]}
                    />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* Graphique 2: Taux de Ponctualité */}
              <div className={styles.chartCard}>
                <h3 className={styles.chartTitle}>✅ Taux de Ponctualité</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <AreaChart data={trends}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                    <XAxis
                      dataKey="date"
                      tick={{ fontSize: 12, fill: "#6b7280" }}
                      tickFormatter={(value) => {
                        const d = new Date(value);
                        return `${d.getDate()}/${d.getMonth() + 1}`;
                      }}
                    />
                    <YAxis
                      tick={{ fontSize: 12, fill: "#6b7280" }}
                      domain={[0, 100]}
                    />
                    <Tooltip
                      contentStyle={{
                        background: "#ffffff",
                        border: "1px solid #e5e7eb",
                        borderRadius: "8px",
                      }}
                      formatter={(value) => `${value.toFixed(1)}%`}
                    />
                    <Legend />
                    <Area
                      type="monotone"
                      dataKey="on_time_rate"
                      stroke="#10b981"
                      fill="#10b981"
                      fillOpacity={0.6}
                      name="% à l'heure"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>

              {/* Graphique 3: Évolution des Retards */}
              <div className={styles.chartCard}>
                <h3 className={styles.chartTitle}>⏱️ Évolution des Retards</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={trends}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                    <XAxis
                      dataKey="date"
                      tick={{ fontSize: 12, fill: "#6b7280" }}
                      tickFormatter={(value) => {
                        const d = new Date(value);
                        return `${d.getDate()}/${d.getMonth() + 1}`;
                      }}
                    />
                    <YAxis tick={{ fontSize: 12, fill: "#6b7280" }} />
                    <Tooltip
                      contentStyle={{
                        background: "#ffffff",
                        border: "1px solid #e5e7eb",
                        borderRadius: "8px",
                      }}
                      formatter={(value) => `${value.toFixed(1)} min`}
                    />
                    <Legend />
                    <Line
                      type="monotone"
                      dataKey="avg_delay"
                      stroke="#ef4444"
                      strokeWidth={2}
                      dot={{ fill: "#ef4444", r: 4 }}
                      activeDot={{ r: 6 }}
                      name="Retard moyen (min)"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              {/* Graphique 4: Score de Qualité */}
              <div className={styles.chartCard}>
                <h3 className={styles.chartTitle}>⭐ Score de Qualité</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <AreaChart data={trends}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                    <XAxis
                      dataKey="date"
                      tick={{ fontSize: 12, fill: "#6b7280" }}
                      tickFormatter={(value) => {
                        const d = new Date(value);
                        return `${d.getDate()}/${d.getMonth() + 1}`;
                      }}
                    />
                    <YAxis
                      tick={{ fontSize: 12, fill: "#6b7280" }}
                      domain={[0, 100]}
                    />
                    <Tooltip
                      contentStyle={{
                        background: "#ffffff",
                        border: "1px solid #e5e7eb",
                        borderRadius: "8px",
                      }}
                      formatter={(value) => `${value.toFixed(0)}/100`}
                    />
                    <Legend />
                    <Area
                      type="monotone"
                      dataKey="quality_score"
                      stroke="#8b5cf6"
                      fill="#8b5cf6"
                      fillOpacity={0.6}
                      name="Score (/100)"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Bouton Export */}
            <div className={styles.actionsBar}>
              <button
                className={styles.exportButton}
                onClick={() => handleExport("csv")}
              >
                📥 Exporter en CSV
              </button>
              <button
                className={styles.exportButton}
                onClick={() => handleExport("json")}
              >
                📄 Exporter en JSON
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  // Handler pour l'export
  async function handleExport(format) {
    try {
      const today = new Date().toISOString().split("T")[0];
      const startDate = new Date();
      startDate.setDate(
        startDate.getDate() - (period === "7d" ? 7 : period === "90d" ? 90 : 30)
      );
      const start = startDate.toISOString().split("T")[0];

      const data = await exportAnalytics(start, today, format);

      if (format === "csv") {
        downloadCsvFile(data, `analytics_${start}_${today}.csv`);
      } else {
        openJsonInNewTab(data);
      }
    } catch (err) {
      console.error("Erreur lors de l'export:", err);
      setError(err.message || "Impossible d'exporter les données");
    }
  }
};

export default AnalyticsDashboard;
