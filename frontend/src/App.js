import React, { useEffect, Suspense, lazy } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import apiClient from './utils/apiClient';

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
const RLMetricsDashboard = lazy(() =>
  import('./pages/company/Dispatch/Dashboard/RLMetricsDashboard')
);
const AnalyticsDashboard = lazy(() => import('./pages/company/Analytics/AnalyticsDashboard'));
const Dashboard = lazy(() => import('./pages/Home/Dashboard'));

// ──────────────────────────────────────────────────────────
// Query Client (déclaré hors composant pour éviter recréation)
const queryClient = new QueryClient();

// Keep-alive user activity
let lastActivity = Date.now();
function resetActivityTimer() {
  lastActivity = Date.now();
}

// Rafraîchissement automatique du token toutes les 50 min si actif
function setupTokenAutoRefresh() {
  // Écoute activité
  window.addEventListener('mousemove', resetActivityTimer);
  window.addEventListener('keydown', resetActivityTimer);
  window.addEventListener('touchstart', resetActivityTimer);

  const id = setInterval(async () => {
    const now = Date.now();
    const refreshToken = localStorage.getItem('refreshToken');
    const authToken = localStorage.getItem('authToken');

    // Vérifier si l'utilisateur est actif (moins de 55 min d'inactivité)
    const isActive = now - lastActivity < 55 * 60 * 1000;

    if (!refreshToken || !authToken || !isActive) {
      return; // Ne rien faire si pas de tokens ou utilisateur inactif
    }

    try {
      // Token refresh en cours
      const { data } = await apiClient.post(
        '/auth/refresh-token',
        {},
        {
          headers: {
            Authorization: `Bearer ${refreshToken}`,
          },
          // Ignorer l'intercepteur d'erreur pour éviter les redirections automatiques
          skipAuthRedirect: true,
        }
      );

      if (data.access_token) {
        localStorage.setItem('authToken', data.access_token);
      }
    } catch (e) {
      console.error('❌ Erreur rafraîchissement token:');
      console.error('  - Status:', e.response?.status);
      console.error('  - Data:', e.response?.data);
      console.error('  - Headers:', e.response?.headers);

      // Ne supprimer les tokens que si le refresh token est vraiment invalide (401, 422)
      if (e.response?.status === 401 || e.response?.status === 422) {
        console.warn('⚠️ Refresh token invalide, nettoyage des tokens et redirection...');
        localStorage.removeItem('authToken');
        localStorage.removeItem('refreshToken');
        localStorage.removeItem('user');
        localStorage.removeItem('public_id');
        // Rediriger vers la page de connexion seulement après un délai
        setTimeout(() => {
          window.location.href = '/login';
        }, 1000);
      }
    }
  }, 50 * 60 * 1000); // Toutes les 50 minutes (le token expire après 1h)

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
              path="/force-reset-password/:userId"
              element={
                <DefaultLayout>
                  <ResetPassword resetMode="userId" />
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
