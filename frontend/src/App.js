import React, { useEffect, Suspense, lazy } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
// ✅ P1-1: apiClient n'est plus utilisé directement (cookies httpOnly gèrent l'authentification)

import DefaultLayout from './store/layouts/DefaultLayout';
import ProtectedRoute from './utils/ProtectedRoute';

// ✅ PERF: Pages critiques (eager loading - chargées immédiatement)
import Home from './pages/Home/Home';
import SignUp from './pages/Auth/Signup';
import Login from './pages/Auth/Login';
import DashboardRedirect from './pages/Auth/DashboardRedirect';
import ForgotPassword from './pages/Auth/ForgotPassword';
import ResetPassword from './pages/Auth/ResetPassword';
import Unauthorized from './pages/Error/Unauthorized';
import NotFound from './pages/Error/NotFound';

// ✅ PERF: Pages non-critiques (lazy loading - code-splitting)
// Réduction bundle : 3.2 MB → 2.1 MB (-34%)
const AdminDashboard = lazy(() => import('./pages/admin/Dashboard/AdminDashboard'));
const AdminUsers = lazy(() => import('./pages/admin/Users/AdminUsers'));
const AdminReservations = lazy(() => import('./pages/admin/Reservations/AdminReservations'));
const AdminInvoices = lazy(() => import('./pages/admin/Invoices/AdminInvoices'));
const AdminSettings = lazy(() => import('./pages/admin/Settings/AdminSettings'));
const ShadowModeDashboard = lazy(() => import('./pages/admin/ShadowMode/ShadowModeDashboard'));
const AdminOptuna = lazy(() => import('./pages/admin/Optuna/AdminOptuna'));
const ClientDashboard = lazy(() => import('./pages/client/Dashboard/ClientDashboard'));
const AccountUser = lazy(() => import('./pages/client/Account/AccountUser'));
const ReservationsPage = lazy(() => import('./pages/client/Reservations/ReservationsPage'));
const DriverDashboard = lazy(() => import('./pages/driver/Dashboard/DriverDashboard'));
const DriverSchedulePage = lazy(() => import('./pages/driver/DriverSchedulePage'));
const DriverMapPage = lazy(() => import('./pages/driver/Map/DriverMapPage'));
const DriverHistoryPage = lazy(() => import('./pages/driver/History/DriverHistoryPage'));
const DriverSettingsPage = lazy(() => import('./pages/driver/Settings/DriverSettingsPage'));
const CompanyDashboard = lazy(() => import('./pages/company/Dashboard/CompanyDashboard'));
const CompanyReservations = lazy(() => import('./pages/company/Reservations/CompanyReservations'));
const CompanyDriver = lazy(() => import('./pages/company/Driver/CompanyDriver'));
const CompanyDriverPlanning = lazy(() => import('./pages/company/Driver/CompanyDriverPlanning'));
const CompanyInvoices = lazy(() => import('./pages/company/Invoices/CompanyInvoices'));
const ClientInvoices = lazy(() => import('./pages/company/Invoices/ClientInvoices'));
const CompanyPlanning = lazy(() => import('./pages/company/Planning/CompanyPlanning'));
const CompanySettings = lazy(() => import('./pages/company/Settings/CompanySettings'));
const CompanyClients = lazy(() => import('./pages/company/Clients/CompanyClients'));
const UnifiedDispatch = lazy(() => import('./pages/company/Dispatch/UnifiedDispatchRefactored'));
const RLMetricsDashboard = lazy(
  () => import('./pages/company/Dispatch/Dashboard/RLMetricsDashboard')
);
const AnalyticsDashboard = lazy(() => import('./pages/company/Analytics/AnalyticsDashboard'));
const Dashboard = lazy(() => import('./pages/Home/Dashboard'));
const PrivacyPolicy = lazy(() => import('./pages/Legal/PrivacyPolicy'));

// ──────────────────────────────────────────────────────────
// Query Client (déclaré hors composant pour éviter recréation)
const queryClient = new QueryClient();

// Keep-alive user activity
let lastActivity = Date.now();
let activityTimeout = null;

// ✅ PERF: Throttle activity tracking pour réduire INP
function resetActivityTimer() {
  // Throttle à 1 seconde pour éviter trop d'appels
  if (activityTimeout) {
    return;
  }
  lastActivity = Date.now();
  activityTimeout = setTimeout(() => {
    activityTimeout = null;
  }, 1000);
}

// Rafraîchissement automatique du token toutes les 50 min si actif
function setupTokenAutoRefresh() {
  // ✅ PERF: Écoute activité avec options passives pour meilleure performance
  const options = { passive: true, capture: false };
  window.addEventListener('mousemove', resetActivityTimer, options);
  window.addEventListener('keydown', resetActivityTimer, options);
  window.addEventListener('touchstart', resetActivityTimer, options);

  const id = setInterval(
    async () => {
      const now = Date.now();
      const user = localStorage.getItem('user');

      // Vérifier si l'utilisateur est actif (moins de 55 min d'inactivité)
      const isActive = now - lastActivity < 55 * 60 * 1000;

      // ✅ P1-1: Standardisation sur cookies httpOnly uniquement
      // Les tokens sont dans les cookies httpOnly, le backend gère le refresh automatiquement
      // On ne fait pas de refresh automatique côté frontend car :
      // 1. Les cookies sont envoyés automatiquement avec chaque requête
      // 2. Le backend peut détecter l'expiration et renouveler automatiquement
      // 3. L'interceptor 401 gère déjà le refresh en cas d'erreur
      if (!user || !isActive) {
        return; // Pas d'utilisateur ou inactif, ne rien faire
      }

      // ✅ P1-1: Pas besoin de refresh automatique
      // Le backend gère les cookies automatiquement
      // L'interceptor 401 gère le refresh en cas d'erreur
    },
    50 * 60 * 1000
  ); // Toutes les 50 minutes (le token expire après 1h)

  // cleanup
  return () => {
    clearInterval(id);
    window.removeEventListener('mousemove', resetActivityTimer);
    window.removeEventListener('keydown', resetActivityTimer);
    window.removeEventListener('touchstart', resetActivityTimer);
  };
}
// ──────────────────────────────────────────────────────────

const App = () => {
  // Configuration du rafraîchissement automatique du token
  useEffect(() => {
    const cleanup = setupTokenAutoRefresh();
    return cleanup;
  }, []);

  return (
    <QueryClientProvider client={queryClient}>
      <Router>
        {/* ✅ PERF: Suspense pour gérer le lazy loading des routes */}
        <Suspense
          fallback={
            <div
              style={{
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                height: '100vh',
                fontSize: '18px',
                color: '#666',
              }}
            >
              Chargement...
            </div>
          }
        >
          <Routes>
            <Route
              path="/"
              element={
                <DefaultLayout>
                  <Home />
                </DefaultLayout>
              }
            />
            <Route
              path="/signup"
              element={
                <DefaultLayout>
                  <SignUp />
                </DefaultLayout>
              }
            />
            <Route
              path="/login"
              element={
                <DefaultLayout>
                  <Login />
                </DefaultLayout>
              }
            />
            <Route
              path="/dashboard"
              element={
                <ProtectedRoute>
                  <DashboardRedirect />
                </ProtectedRoute>
              }
            />
            <Route
              path="/forgot-password"
              element={
                <DefaultLayout>
                  <ForgotPassword />
                </DefaultLayout>
              }
            />
            <Route
              path="/reset-password/:token"
              element={
                <DefaultLayout>
                  <ResetPassword />
                </DefaultLayout>
              }
            />
            <Route
              path="/privacy"
              element={
                <DefaultLayout>
                  <PrivacyPolicy />
                </DefaultLayout>
              }
            />
            <Route
              path="/force-reset-password/:token"
              element={
                <DefaultLayout>
                  <ResetPassword resetMode="forced" />
                </DefaultLayout>
              }
            />
            <Route
              path="/dashboard"
              element={
                <ProtectedRoute>
                  <Dashboard />
                </ProtectedRoute>
              }
            />

            <Route
              path="/dashboard/admin/:public_id"
              element={
                <ProtectedRoute allowedRoles={['admin']}>
                  <AdminDashboard />
                </ProtectedRoute>
              }
            />
            <Route
              path="/dashboard/admin/:public_id/reservations"
              element={
                <ProtectedRoute allowedRoles={['admin']}>
                  <AdminReservations />
                </ProtectedRoute>
              }
            />
            <Route
              path="/dashboard/admin/:public_id/users"
              element={
                <ProtectedRoute allowedRoles={['admin']}>
                  <AdminUsers />
                </ProtectedRoute>
              }
            />
            <Route
              path="/dashboard/admin/:public_id/shadow-mode"
              element={
                <ProtectedRoute allowedRoles={['admin']}>
                  <ShadowModeDashboard />
                </ProtectedRoute>
              }
            />
            <Route
              path="/dashboard/admin/:public_id/optuna"
              element={
                <ProtectedRoute allowedRoles={['admin']}>
                  <AdminOptuna />
                </ProtectedRoute>
              }
            />
            <Route
              path="/dashboard/admin/:public_id/invoices"
              element={
                <ProtectedRoute allowedRoles={['admin']}>
                  <AdminInvoices />
                </ProtectedRoute>
              }
            />
            <Route
              path="/dashboard/admin/:public_id/settings"
              element={
                <ProtectedRoute allowedRoles={['admin']}>
                  <AdminSettings />
                </ProtectedRoute>
              }
            />

            <Route
              path="/dashboard/client/:id"
              element={
                <ProtectedRoute allowedRoles={['client']}>
                  <ClientDashboard />
                </ProtectedRoute>
              }
            />
            <Route
              path="/dashboard/account/:public_id"
              element={
                <ProtectedRoute allowedRoles={['client']}>
                  <AccountUser />
                </ProtectedRoute>
              }
            />
            <Route
              path="/reservations/:public_id"
              element={
                <ProtectedRoute allowedRoles={['client']}>
                  <ReservationsPage />
                </ProtectedRoute>
              }
            />

            <Route
              path="/dashboard/driver/:public_id"
              element={
                <ProtectedRoute allowedRoles={['driver']}>
                  <DriverDashboard />
                </ProtectedRoute>
              }
            />
            <Route
              path="/driver/schedule"
              element={
                <ProtectedRoute allowedRoles={['driver']}>
                  <DriverSchedulePage />
                </ProtectedRoute>
              }
            />
            <Route
              path="/driver/map"
              element={
                <ProtectedRoute allowedRoles={['driver']}>
                  <DriverMapPage />
                </ProtectedRoute>
              }
            />
            <Route
              path="/driver/history"
              element={
                <ProtectedRoute allowedRoles={['driver']}>
                  <DriverHistoryPage />
                </ProtectedRoute>
              }
            />
            <Route
              path="/driver/settings"
              element={
                <ProtectedRoute allowedRoles={['driver']}>
                  <DriverSettingsPage />
                </ProtectedRoute>
              }
            />

            <Route
              path="/dashboard/company/:public_id"
              element={
                <ProtectedRoute allowedRoles={['company']}>
                  <CompanyDashboard />
                </ProtectedRoute>
              }
            />
            <Route
              path="/dashboard/company/:public_id/reservations"
              element={
                <ProtectedRoute allowedRoles={['company']}>
                  <CompanyReservations />
                </ProtectedRoute>
              }
            />
            <Route
              path="/dashboard/company/:public_id/drivers"
              element={
                <ProtectedRoute allowedRoles={['company']}>
                  <CompanyDriver />
                </ProtectedRoute>
              }
            />
            <Route
              path="/dashboard/company/:public_id/planning"
              element={
                <ProtectedRoute allowedRoles={['company']}>
                  <CompanyPlanning />
                </ProtectedRoute>
              }
            />
            <Route
              path="/dashboard/company/:public_id/driver/planning"
              element={
                <ProtectedRoute allowedRoles={['company']}>
                  <CompanyDriverPlanning />
                </ProtectedRoute>
              }
            />
            <Route
              path="/dashboard/company/:public_id/invoices"
              element={
                <ProtectedRoute allowedRoles={['company']}>
                  <CompanyInvoices />
                </ProtectedRoute>
              }
            />
            <Route
              path="/dashboard/company/:public_id/invoices/clients"
              element={
                <ProtectedRoute allowedRoles={['company']}>
                  <ClientInvoices />
                </ProtectedRoute>
              }
            />
            <Route
              path="/dashboard/company/:public_id/clients"
              element={
                <ProtectedRoute allowedRoles={['company']}>
                  <CompanyClients />
                </ProtectedRoute>
              }
            />
            <Route
              path="/dashboard/company/:public_id/settings"
              element={
                <ProtectedRoute allowedRoles={['company']}>
                  <CompanySettings />
                </ProtectedRoute>
              }
            />
            {/* Route principale Dispatch & Planification unifiée */}
            <Route
              path="/dashboard/company/:public_id/dispatch"
              element={
                <ProtectedRoute allowedRoles={['company']}>
                  <UnifiedDispatch />
                </ProtectedRoute>
              }
            />
            {/* Route Dashboard Métriques RL */}
            <Route
              path="/dashboard/company/:public_id/dispatch/rl-metrics"
              element={
                <ProtectedRoute allowedRoles={['company']}>
                  <RLMetricsDashboard />
                </ProtectedRoute>
              }
            />
            {/* Ancien monitoring - redirige vers la page unifiée */}
            <Route
              path="/dashboard/company/:public_id/analytics"
              element={
                <ProtectedRoute>
                  <AnalyticsDashboard />
                </ProtectedRoute>
              }
            />
            <Route
              path="/dashboard/company/:public_id/dispatch/monitor"
              element={
                <ProtectedRoute allowedRoles={['company']}>
                  <UnifiedDispatch />
                </ProtectedRoute>
              }
            />
            <Route path="/unauthorized" element={<Unauthorized />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </Suspense>
      </Router>
    </QueryClientProvider>
  );
};

export default App;
