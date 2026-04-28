import React, { useEffect, Suspense, lazy } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate, useLocation, useParams } from 'react-router-dom';
import CompanyEnterpriseLayout from './components/layout/CompanyEnterpriseLayout';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
// ✅ P1-1: apiClient n'est plus utilisé directement (cookies httpOnly gèrent l'authentification)

import DefaultLayout from './store/layouts/DefaultLayout';
import ProtectedRoute from './utils/ProtectedRoute';
import PlatformSegmentGuard from './pages/admin/PlatformOps/PlatformSegmentGuard';
import GoogleMapsProvider from './components/common/GoogleMapsProvider';
import PwaOfflineBanner from './components/common/PwaOfflineBanner';
import { Toaster } from 'sonner';
import Home from './pages/Home/Home';
import { hasActiveSession } from './utils/webAuthSession';
import BookNewRedirect from './pages/Auth/BookNewRedirect';

// ✅ PERF: Pages critiques (eager loading - chargées immédiatement)
import Login from './pages/Auth/Login';
import AppNamespaceRedirect from './pages/Auth/AppNamespaceRedirect';
import DashboardRedirect from './pages/Auth/DashboardRedirect';
import ForgotPassword from './pages/Auth/ForgotPassword';
import ResetPassword from './pages/Auth/ResetPassword';
import SignupActivation from './pages/Auth/SignupActivation';
import Unauthorized from './pages/Error/Unauthorized';
import NotFound from './pages/Error/NotFound';

// ✅ PERF: Pages non-critiques (lazy loading - code-splitting)
// Réduction bundle : 3.2 MB → 2.1 MB (-34%)
const AdminDashboard = lazy(() => import('./pages/admin/Dashboard/AdminDashboard'));
const AdminUsers = lazy(() => import('./pages/admin/Users/AdminUsers'));
const AdminReservations = lazy(() => import('./pages/admin/Reservations/AdminReservations'));
const AdminBookingDetail = lazy(() => import('./pages/admin/Reservations/AdminBookingDetail'));
const AdminInvoices = lazy(() => import('./pages/admin/Invoices/AdminInvoices'));
const AdminPlatformBilling = lazy(() => import('./pages/admin/PlatformBilling/AdminPlatformBilling'));
const AdminBillingHub = lazy(() => import('./pages/admin/Billing/AdminBillingHub'));
const AdminBillingTransportConfig = lazy(() =>
  import('./pages/admin/Billing/AdminBillingTransportConfig')
);
const AdminPilotageCompanyDetail = lazy(() =>
  import('./pages/admin/Invoices/AdminPilotageCompanyDetail')
);

/** Ancienne URL détail pilotage → hub Facturation. */
function RedirectToBillingPilotageCompany() {
  const { public_id, companyId } = useParams();
  return (
    <Navigate
      to={`/dashboard/admin/${public_id}/billing/pilotage/companies/${companyId}`}
      replace
    />
  );
}

function ScrollToTopOnNavigation() {
  const location = useLocation();

  useEffect(() => {
    window.scrollTo({ top: 0, left: 0, behavior: 'auto' });
  }, [location.pathname, location.search]);

  return null;
}

function LegacySignupRedirect() {
  const location = useLocation();
  const params = new URLSearchParams(location.search || '');
  params.set('mode', 'signup');
  return <Navigate to={`/login?${params.toString()}`} replace />;
}

const MAP_ROUTE_PATTERNS = [
  /^\/dashboard\/client\/.+/,
  /^\/dashboard\/company\/[^/]+\/.+/,
  /^\/demo\/dashboard\/company\/[^/]+\/.+/,
  /^\/driver\/map(?:\/|$)/,
];

function GoogleMapsRouteScope({ children }) {
  const location = useLocation();
  // Tout le périmètre « dashboard entreprise » (index + sous-routes) a besoin du contexte
  // Google Maps pour DriverLiveMap, cartes paramètres, etc. Avant : seul l’index avec
  // ?live_map=1 ou les sous-chemins du type .../company/:id/xxx étaient couverts ;
  // l’index seul n’avait pas de GoogleMapsProvider → isLoaded restait false (défaut
  // du context) et la carte restait en « Chargement de la carte... » au retour sur la page.
  const isCompanyEnterpriseArea = /^\/(?:demo\/)?dashboard\/company\/[^/]+/.test(
    location.pathname
  );
  const needsGoogleMaps =
    isCompanyEnterpriseArea ||
    MAP_ROUTE_PATTERNS.some((pattern) => pattern.test(location.pathname));
  if (!needsGoogleMaps) {
    return children;
  }
  return <GoogleMapsProvider>{children}</GoogleMapsProvider>;
}
const AdminSettings = lazy(() => import('./pages/admin/Settings/AdminSettings'));
const AdminDemoRequests = lazy(() => import('./pages/admin/DemoRequests/AdminDemoRequests'));
const AdminLayout = lazy(() => import('./pages/admin/AdminLayout'));
const PlatformLayout = lazy(() => import('./pages/admin/PlatformOps/PlatformLayout'));
const PlatformOverviewPage = lazy(() => import('./pages/admin/PlatformOps/PlatformOverviewPage'));
const PlatformTenantsPage = lazy(() => import('./pages/admin/PlatformOps/PlatformTenantsPage'));
const PlatformRunbooksPage = lazy(() => import('./pages/admin/PlatformOps/PlatformRunbooksPage'));
const PlatformAuditPage = lazy(() => import('./pages/admin/PlatformOps/PlatformAuditPage'));
const PlatformRuntimePage = lazy(() => import('./pages/admin/PlatformOps/PlatformRuntimePage'));
const PlatformReconciliationPage = lazy(() => import('./pages/admin/PlatformOps/PlatformReconciliationPage'));
const PlatformInvestigationPage = lazy(() => import('./pages/admin/PlatformOps/PlatformInvestigationPage'));
const ShadowModeDashboard = lazy(() => import('./pages/admin/ShadowMode/ShadowModeDashboard'));
const AdminOptuna = lazy(() => import('./pages/admin/Optuna/AdminOptuna'));
const ClientDashboard = lazy(() => import('./pages/client/Dashboard/ClientDashboard'));
const AccountUser = lazy(() => import('./pages/client/Account/AccountUser'));
const ReservationsPage = lazy(() => import('./pages/client/Reservations/ReservationsPage'));
const ClientSaferpayPaymentReturn = lazy(() =>
  import('./pages/client/Payment/ClientSaferpayPaymentReturn')
);
const ClientSaferpayCheckoutStart = lazy(() =>
  import('./pages/client/Payment/ClientSaferpayCheckoutStart')
);
const GuestSaferpayAppReturn = lazy(() => import('./pages/Public/GuestSaferpayAppReturn'));
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
const PrivacyPolicy = lazy(() => import('./pages/Legal/PrivacyPolicy'));
const TermsOfService = lazy(() => import('./pages/Legal/TermsOfService'));
const LegalNotice = lazy(() => import('./pages/Legal/LegalNotice'));
const DeplacezVousPage = lazy(() => import('./pages/Public/DeplacezVousPage'));
const ConduirePage = lazy(() => import('./pages/Public/ConduirePage'));
const ProfessionnelPage = lazy(() => import('./pages/Public/ProfessionnelPage'));
const AProposPage = lazy(() => import('./pages/Public/AProposPage'));
const AidePage = lazy(() => import('./pages/Public/AidePage'));
const Contact = lazy(() => import('./pages/Legal/Contact'));
const DemoRequest = lazy(() => import('./pages/Legal/DemoRequest'));
const ContactSupport = lazy(() => import('./pages/Legal/ContactSupport'));
const ContactInstitution = lazy(() => import('./pages/Legal/ContactInstitution'));
const ContactTransport = lazy(() => import('./pages/Legal/ContactTransport'));
const ContactDemo = lazy(() => import('./pages/Legal/ContactDemo'));
const ContactBilling = lazy(() => import('./pages/Legal/ContactBilling'));
const ContactFamily = lazy(() => import('./pages/Legal/ContactFamily'));
const DemoHome = lazy(() => import('./pages/demo/DemoHome'));
const DemoAccessConsume = lazy(() => import('./pages/demo/DemoAccessConsume'));

/** Anciens liens e-mail / favoris `/client/payment/worldline/*` → Saferpay. */
function LegacyWorldlinePaymentUrlRedirect({ targetBase }) {
  const { search } = useLocation();
  return <Navigate replace to={`${targetBase}${search}`} />;
}

// ✅ ÉTAPE 6: Pages Institution (lazy loading)
const AcceptInvite = lazy(() => import('./pages/Auth/AcceptInvite'));
const InstitutionLayout = lazy(() => import('./pages/institution/Layout/InstitutionLayout'));
const InstitutionDashboard = lazy(() => import('./pages/institution/Dashboard/InstitutionDashboard'));
const InstitutionRequests = lazy(() => import('./pages/institution/Requests/InstitutionRequests'));
const InstitutionRequestCreate = lazy(() => import('./pages/institution/Requests/InstitutionRequestCreate'));
const InstitutionRequestDetail = lazy(() => import('./pages/institution/Requests/InstitutionRequestDetail'));
const InstitutionPatients = lazy(() => import('./pages/institution/Patients/InstitutionPatients'));
const InstitutionSettings = lazy(() => import('./pages/institution/Settings/InstitutionSettings'));

// ──────────────────────────────────────────────────────────
// Query Client (déclaré hors composant pour éviter recréation)
// Exporté pour permettre le nettoyage du cache au logout
export const queryClient = new QueryClient();

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
      const hasSession = hasActiveSession();

      // Vérifier si l'utilisateur est actif (moins de 55 min d'inactivité)
      const isActive = now - lastActivity < 55 * 60 * 1000;

      // ✅ P1-1: Standardisation sur cookies httpOnly uniquement
      // Les tokens sont dans les cookies httpOnly, le backend gère le refresh automatiquement
      // On ne fait pas de refresh automatique côté frontend car :
      // 1. Les cookies sont envoyés automatiquement avec chaque requête
      // 2. Le backend peut détecter l'expiration et renouveler automatiquement
      // 3. L'interceptor 401 gère déjà le refresh en cas d'erreur
      if (!hasSession || !isActive) {
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

  useEffect(() => {
    let cancelled = false;
    let idleId = null;
    let timerId = null;

    const loadSharedComponentsCss = () => {
      if (cancelled) return;
      import('./styles/components.css').catch(() => {});
    };

    // Déférer les styles utilitaires volumineux non critiques au premier écran.
    if (typeof window.requestIdleCallback === 'function') {
      idleId = window.requestIdleCallback(loadSharedComponentsCss, { timeout: 1800 });
    } else {
      timerId = window.setTimeout(loadSharedComponentsCss, 800);
    }

    return () => {
      cancelled = true;
      if (idleId != null && typeof window.cancelIdleCallback === 'function') {
        window.cancelIdleCallback(idleId);
      }
      if (timerId != null) {
        window.clearTimeout(timerId);
      }
    };
  }, []);

  return (
    <QueryClientProvider client={queryClient}>
      <Router
        future={{
          v7_startTransition: true,
          v7_relativeSplatPath: true,
        }}
      >
        <ScrollToTopOnNavigation />
        <Toaster position="top-right" richColors closeButton />
        <PwaOfflineBanner />
        {/* ✅ PERF: Suspense pour gérer le lazy loading des routes */}
        <GoogleMapsRouteScope>
          <Suspense
            fallback={
              <div
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  justifyContent: 'center',
                  alignItems: 'center',
                  height: '100vh',
                  gap: '12px',
                  padding: '24px',
                }}
              >
                <div
                  style={{
                    width: 'min(760px, 100%)',
                    height: '22px',
                    borderRadius: '8px',
                    background: 'linear-gradient(90deg, #eef2f7 0%, #f8fafc 50%, #eef2f7 100%)',
                  }}
                />
                <div
                  style={{
                    width: 'min(760px, 100%)',
                    height: '180px',
                    borderRadius: '12px',
                    background: 'linear-gradient(90deg, #eef2f7 0%, #f8fafc 50%, #eef2f7 100%)',
                  }}
                />
                <div style={{ fontSize: '15px', color: '#64748b' }}>Chargement de la page…</div>
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
              path="/deplacez-vous"
              element={
                <DefaultLayout>
                  <DeplacezVousPage />
                </DefaultLayout>
              }
            />
            <Route
              path="/conduire"
              element={
                <DefaultLayout>
                  <ConduirePage />
                </DefaultLayout>
              }
            />
            <Route
              path="/professionnel"
              element={
                <DefaultLayout>
                  <ProfessionnelPage />
                </DefaultLayout>
              }
            />
            <Route
              path="/a-propos"
              element={
                <DefaultLayout>
                  <AProposPage />
                </DefaultLayout>
              }
            />
            <Route
              path="/aide"
              element={
                <DefaultLayout>
                  <AidePage />
                </DefaultLayout>
              }
            />
            <Route
              path="/signup"
              element={
                <LegacySignupRedirect />
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
              path="/guest/payment/saferpay/return"
              element={
                <Suspense fallback={null}>
                  <GuestSaferpayAppReturn />
                </Suspense>
              }
            />
            <Route path="/app/*" element={<AppNamespaceRedirect />} />
            <Route
              path="/book/new"
              element={
                <ProtectedRoute>
                  <BookNewRedirect />
                </ProtectedRoute>
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
              path="/demo/dashboard"
              element={
                <ProtectedRoute>
                  <DashboardRedirect forceDemoNamespace />
                </ProtectedRoute>
              }
            />
            <Route
              path="/demo/home"
              element={
                <ProtectedRoute>
                  <DefaultLayout>
                    <DemoHome />
                  </DefaultLayout>
                </ProtectedRoute>
              }
            />
            <Route
              path="/demo-access/consume"
              element={
                <DefaultLayout>
                  <DemoAccessConsume />
                </DefaultLayout>
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
              path="/activate-account"
              element={
                <DefaultLayout>
                  <SignupActivation />
                </DefaultLayout>
              }
            />
            <Route
              path="/invite/:token"
              element={<AcceptInvite />}
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
              path="/conditions"
              element={
                <DefaultLayout>
                  <TermsOfService />
                </DefaultLayout>
              }
            />
            <Route
              path="/mentions-legales"
              element={
                <DefaultLayout>
                  <LegalNotice />
                </DefaultLayout>
              }
            />
            <Route
              path="/contact"
              element={
                <DefaultLayout>
                  <Contact />
                </DefaultLayout>
              }
            />
            <Route
              path="/contact/support"
              element={
                <DefaultLayout>
                  <ContactSupport />
                </DefaultLayout>
              }
            />
            <Route
              path="/contact/institution"
              element={
                <DefaultLayout>
                  <ContactInstitution />
                </DefaultLayout>
              }
            />
            <Route
              path="/contact/transport"
              element={
                <DefaultLayout>
                  <ContactTransport />
                </DefaultLayout>
              }
            />
            <Route
              path="/contact/demo"
              element={
                <DefaultLayout>
                  <ContactDemo />
                </DefaultLayout>
              }
            />
            <Route
              path="/contact/billing"
              element={
                <DefaultLayout>
                  <ContactBilling />
                </DefaultLayout>
              }
            />
            <Route
              path="/contact/family"
              element={
                <DefaultLayout>
                  <ContactFamily />
                </DefaultLayout>
              }
            />
            <Route
              path="/demo-request"
              element={
                <DefaultLayout>
                  <DemoRequest />
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
              path="/dashboard/admin/:public_id"
              element={
                <ProtectedRoute allowedRoles={['admin']}>
                  <AdminLayout />
                </ProtectedRoute>
              }
            >
              <Route index element={<AdminDashboard />} />
              <Route path="reservations/:bookingId" element={<AdminBookingDetail />} />
              <Route path="reservations" element={<AdminReservations />} />
              <Route path="users" element={<AdminUsers />} />
              <Route path="shadow-mode" element={<ShadowModeDashboard />} />
              <Route path="optuna" element={<AdminOptuna />} />
              <Route path="billing" element={<AdminBillingHub />}>
                <Route index element={<Navigate to="pilotage" replace />} />
                <Route path="pilotage" element={<AdminInvoices />} />
                <Route
                  path="pilotage/companies/:companyId"
                  element={<AdminPilotageCompanyDetail />}
                />
                <Route path="releves" element={<AdminPlatformBilling />} />
                <Route path="config" element={<AdminBillingTransportConfig />} />
              </Route>
              <Route path="invoices" element={<Navigate to="../billing/pilotage" replace />} />
              <Route
                path="invoices/pilotage/companies/:companyId"
                element={<RedirectToBillingPilotageCompany />}
              />
              <Route path="platform-billing" element={<Navigate to="../billing/releves" replace />} />
              <Route path="settings" element={<AdminSettings />} />
              <Route path="demo-requests" element={<AdminDemoRequests />} />
              <Route path="platform-ops" element={<PlatformLayout />}>
                <Route index element={<Navigate to="overview" replace />} />
                <Route
                  path="overview"
                  element={
                    <PlatformSegmentGuard segment="overview">
                      <PlatformOverviewPage />
                    </PlatformSegmentGuard>
                  }
                />
                <Route
                  path="tenants"
                  element={
                    <PlatformSegmentGuard segment="tenants">
                      <PlatformTenantsPage />
                    </PlatformSegmentGuard>
                  }
                />
                <Route
                  path="runbooks"
                  element={
                    <PlatformSegmentGuard segment="runbooks">
                      <PlatformRunbooksPage />
                    </PlatformSegmentGuard>
                  }
                />
                <Route
                  path="audit"
                  element={
                    <PlatformSegmentGuard segment="audit">
                      <PlatformAuditPage />
                    </PlatformSegmentGuard>
                  }
                />
                <Route
                  path="runtime"
                  element={
                    <PlatformSegmentGuard segment="runtime">
                      <PlatformRuntimePage />
                    </PlatformSegmentGuard>
                  }
                />
                <Route
                  path="reconciliation"
                  element={
                    <PlatformSegmentGuard segment="reconciliation">
                      <PlatformReconciliationPage />
                    </PlatformSegmentGuard>
                  }
                />
                <Route
                  path="investigation"
                  element={
                    <PlatformSegmentGuard segment="investigation">
                      <PlatformInvestigationPage />
                    </PlatformSegmentGuard>
                  }
                />
              </Route>
            </Route>

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
              path="/client/payment/saferpay/return"
              element={
                <ProtectedRoute allowedRoles={['client']}>
                  <Suspense fallback={null}>
                    <ClientSaferpayPaymentReturn />
                  </Suspense>
                </ProtectedRoute>
              }
            />
            <Route
              path="/client/payment/saferpay/start"
              element={
                <ProtectedRoute allowedRoles={['client']}>
                  <Suspense fallback={null}>
                    <ClientSaferpayCheckoutStart />
                  </Suspense>
                </ProtectedRoute>
              }
            />
            <Route
              path="/client/payment/worldline/return"
              element={
                <LegacyWorldlinePaymentUrlRedirect targetBase="/client/payment/saferpay/return" />
              }
            />
            <Route
              path="/client/payment/worldline/start"
              element={
                <LegacyWorldlinePaymentUrlRedirect targetBase="/client/payment/saferpay/start" />
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
                  <CompanyEnterpriseLayout />
                </ProtectedRoute>
              }
            >
              <Route index element={<CompanyDashboard />} />
              <Route path="reservations" element={<CompanyReservations />} />
              <Route path="drivers" element={<CompanyDriver />} />
              <Route path="planning" element={<CompanyPlanning />} />
              <Route path="driver/planning" element={<CompanyDriverPlanning />} />
              <Route path="invoices" element={<CompanyInvoices />} />
              <Route path="invoices/clients" element={<ClientInvoices />} />
              <Route path="clients" element={<CompanyClients />} />
              <Route path="settings" element={<CompanySettings />} />
              <Route path="dispatch/rl-metrics" element={<RLMetricsDashboard />} />
              <Route path="dispatch/monitor" element={<UnifiedDispatch />} />
              <Route path="dispatch" element={<UnifiedDispatch />} />
              <Route path="analytics" element={<AnalyticsDashboard />} />
            </Route>
            <Route
              path="/demo/dashboard/company/:public_id"
              element={
                <ProtectedRoute allowedRoles={['company']}>
                  <CompanyEnterpriseLayout />
                </ProtectedRoute>
              }
            >
              <Route index element={<CompanyDashboard />} />
              <Route path="reservations" element={<CompanyReservations />} />
              <Route path="drivers" element={<CompanyDriver />} />
              <Route path="planning" element={<CompanyPlanning />} />
              <Route path="driver/planning" element={<CompanyDriverPlanning />} />
              <Route path="invoices" element={<CompanyInvoices />} />
              <Route path="invoices/clients" element={<ClientInvoices />} />
              <Route path="clients" element={<CompanyClients />} />
              <Route path="settings" element={<CompanySettings />} />
              <Route path="dispatch/rl-metrics" element={<RLMetricsDashboard />} />
              <Route path="dispatch/monitor" element={<UnifiedDispatch />} />
              <Route path="dispatch" element={<UnifiedDispatch />} />
              <Route path="analytics" element={<AnalyticsDashboard />} />
            </Route>

            {/* ✅ ÉTAPE 6: Routes Institution (layout commun avec sidebar) */}
            <Route
              path="/dashboard/institution/:public_id"
              element={
                <ProtectedRoute allowedRoles={['institution']}>
                  <InstitutionLayout />
                </ProtectedRoute>
              }
            >
              <Route index element={<InstitutionDashboard />} />
              <Route path="requests" element={<InstitutionRequests />} />
              <Route path="requests/new" element={<InstitutionRequestCreate />} />
              <Route path="requests/:requestId" element={<InstitutionRequestDetail />} />
              <Route path="patients" element={<InstitutionPatients />} />
              <Route path="settings" element={<InstitutionSettings />} />
            </Route>
            <Route
              path="/demo/dashboard/institution/:public_id"
              element={
                <ProtectedRoute allowedRoles={['institution']}>
                  <InstitutionLayout />
                </ProtectedRoute>
              }
            >
              <Route index element={<InstitutionDashboard />} />
              <Route path="requests" element={<InstitutionRequests />} />
              <Route path="requests/new" element={<InstitutionRequestCreate />} />
              <Route path="requests/:requestId" element={<InstitutionRequestDetail />} />
              <Route path="patients" element={<InstitutionPatients />} />
              <Route path="settings" element={<InstitutionSettings />} />
            </Route>

            <Route path="/unauthorized" element={<Unauthorized />} />
            <Route path="*" element={<NotFound />} />
            </Routes>
          </Suspense>
        </GoogleMapsRouteScope>
      </Router>
    </QueryClientProvider>
  );
};

export default App;
