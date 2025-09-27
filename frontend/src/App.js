import React, { useEffect } from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import apiClient from "./utils/apiClient";

import DefaultLayout from "./store/layouts/DefaultLayout";
import Home from "./pages/Home/Home";
import SignUp from "./pages/Auth/Signup";
import Login from "./pages/Auth/Login";
import DashboardRedirect from "./pages/Auth/DashboardRedirect";
import ForgotPassword from "./pages/Auth/ForgotPassword";
import ResetPassword from "./pages/Auth/ResetPassword";
import ProtectedRoute from "./utils/ProtectedRoute";
import AdminDashboard from "./pages/admin/Dashboard/AdminDashboard";
import AdminUsers from "./pages/admin/Users/AdminUsers";
import ClientDashboard from "./pages/client/Dashboard/ClientDashboard";
import AccountUser from "./pages/client/Account/AccountUser";
import ReservationsPage from "./pages/client/Reservations/ReservationsPage";
import DriverDashboard from "./pages/driver/Dashboard/DriverDashboard";
import DriverSchedulePage from "./pages/driver/DriverSchedulePage";
import DriverMapPage from "./pages/driver/Map/DriverMapPage";
import DriverHistoryPage from "./pages/driver/History/DriverHistoryPage";
import DriverSettingsPage from "./pages/driver/Settings/DriverSettingsPage";
import CompanyDashboard from "./pages/company/Dashboard/CompanyDashboard";
import CompanyReservations from "./pages/company/Reservations/CompanyReservations";
import CompanyDriver from "./pages/company/Driver/CompanyDriver";
import CompanyDriverPlanning from "./pages/company/Driver/CompanyDriverPlanning";
import CompanyInvoices from "./pages/company/Invoices/CompanyInvoices";
import ClientInvoices from "./pages/company/Invoices/ClientInvoices";
import CompanySettings from "./pages/company/Settings/CompanySettings";
import AIOptimizedRoute from "./pages/ai/AIOptimizedRoute";
import Dashboard from "./pages/Home/Dashboard";
import Unauthorized from "./pages/Error/Unauthorized";
import NotFound from "./pages/Error/NotFound";

// ──────────────────────────────────────────────────────────
// Query Client (déclaré hors composant pour éviter recréation)
const queryClient = new QueryClient();

// Keep-alive user activity
let lastActivity = Date.now();
function resetActivityTimer() {
  lastActivity = Date.now();
}

// Rafraîchissement automatique du token toutes les 5 min si actif
function setupTokenAutoRefresh() {
  // Écoute activité
  window.addEventListener("mousemove", resetActivityTimer);
  window.addEventListener("keydown", resetActivityTimer);
  window.addEventListener("touchstart", resetActivityTimer);

  const id = setInterval(async () => {
    const now = Date.now();
    const token = localStorage.getItem("authToken");
    if (token && now - lastActivity < 55 * 60 * 1000) {
      try {
        const { data } = await apiClient.post(
          "/auth/refresh-token",
          {},
          { headers: { Authorization: `Bearer ${token}` } }
        );
        localStorage.setItem("authToken", data.access_token);
      } catch (e) {
        console.error("Erreur rafraîchissement token", e);
      }
    }
  }, 5 * 60 * 1000);

  // cleanup
  return () => {
    clearInterval(id);
    window.removeEventListener("mousemove", resetActivityTimer);
    window.removeEventListener("keydown", resetActivityTimer);
    window.removeEventListener("touchstart", resetActivityTimer);
  };
}
// ──────────────────────────────────────────────────────────

const App = () => {
  // refresh via refreshToken (toutes les 5 min)
  useEffect(() => {
    const cleanup = setupTokenAutoRefresh();

    const refreshAuthToken = async () => {
      const refreshToken = localStorage.getItem("refreshToken");
      if (!refreshToken) return;
      try {
        const { data } = await apiClient.post(
          "/auth/refresh-token",
          {},
          { headers: { Authorization: `Bearer ${refreshToken}` } }
        );
        localStorage.setItem("authToken", data.access_token);
      } catch {
        localStorage.removeItem("authToken");
        localStorage.removeItem("refreshToken");
      }
    };

    const iv = setInterval(refreshAuthToken, 5 * 60 * 1000);
    return () => {
      cleanup();
      clearInterval(iv);
    };
  }, []);

  return (
    <QueryClientProvider client={queryClient}>
      <Router>
        <Routes>
          <Route path="/" element={<DefaultLayout><Home /></DefaultLayout>} />
          <Route path="/signup" element={<DefaultLayout><SignUp /></DefaultLayout>} />
          <Route path="/login" element={<DefaultLayout><Login /></DefaultLayout>} />
          <Route path="/dashboard" element={<ProtectedRoute><DashboardRedirect /></ProtectedRoute>} />
          <Route path="/forgot-password" element={<DefaultLayout><ForgotPassword /></DefaultLayout>} />
          <Route path="/reset-password/:token" element={<DefaultLayout><ResetPassword /></DefaultLayout>} />
          <Route path="/force-reset-password/:userId" element={<DefaultLayout><ResetPassword resetMode="userId" /></DefaultLayout>} />
          <Route path="/dashboard" element={<ProtectedRoute><Dashboard /></ProtectedRoute>} />

          <Route path="/dashboard/admin/:public_id" element={
            <ProtectedRoute allowedRoles={["admin"]}><AdminDashboard /></ProtectedRoute>
          }/>
          <Route path="/dashboard/admin/:public_id/users" element={
            <ProtectedRoute allowedRoles={["admin"]}><AdminUsers /></ProtectedRoute>
          }/>

          <Route path="/dashboard/client/:id" element={
            <ProtectedRoute allowedRoles={["client"]}><ClientDashboard /></ProtectedRoute>
          }/>
          <Route path="/dashboard/account/:public_id" element={
            <ProtectedRoute allowedRoles={["client"]}><AccountUser /></ProtectedRoute>
          }/>
          <Route path="/reservations/:public_id" element={
            <ProtectedRoute allowedRoles={["client"]}><ReservationsPage /></ProtectedRoute>
          }/>

          <Route path="/dashboard/driver/:public_id" element={
            <ProtectedRoute allowedRoles={["driver"]}><DriverDashboard /></ProtectedRoute>
          }/>
          <Route path="/driver/schedule" element={<ProtectedRoute allowedRoles={["driver"]}><DriverSchedulePage /></ProtectedRoute>} />
          <Route path="/driver/map" element={<ProtectedRoute allowedRoles={["driver"]}><DriverMapPage /></ProtectedRoute>} />
          <Route path="/driver/history" element={<ProtectedRoute allowedRoles={["driver"]}><DriverHistoryPage /></ProtectedRoute>} />
          <Route path="/driver/settings" element={<ProtectedRoute allowedRoles={["driver"]}><DriverSettingsPage /></ProtectedRoute>} />

          <Route path="/dashboard/company/:public_id" element={
            <ProtectedRoute allowedRoles={["company"]}><CompanyDashboard /></ProtectedRoute>
          }/>
          <Route path="/dashboard/company/:public_id/reservations" element={
            <ProtectedRoute allowedRoles={["company"]}><CompanyReservations /></ProtectedRoute>
          }/>
          <Route path="/dashboard/company/:public_id/drivers" element={
            <ProtectedRoute allowedRoles={["company"]}><CompanyDriver /></ProtectedRoute>
          }/>
          <Route path="/dashboard/company/:public_id/driver/planning" element={
            <ProtectedRoute allowedRoles={["company"]}><CompanyDriverPlanning /></ProtectedRoute>
          }/>
          <Route path="/dashboard/company/:public_id/invoices" element={
            <ProtectedRoute allowedRoles={["company"]}><CompanyInvoices /></ProtectedRoute>
          }/>
          <Route path="/dashboard/company/:public_id/invoices/clients" element={
            <ProtectedRoute allowedRoles={["company"]}><ClientInvoices /></ProtectedRoute>
          }/>
          <Route path="/dashboard/company/:public_id/settings" element={
            <ProtectedRoute allowedRoles={["company"]}><CompanySettings /></ProtectedRoute>
          }/>

          <Route path="/ai-route" element={<AIOptimizedRoute />} />
          <Route path="/unauthorized" element={<Unauthorized />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </Router>
    </QueryClientProvider>
  );
};

export default App;
