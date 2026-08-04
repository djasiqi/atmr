import React, { useEffect, Suspense, lazy } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate, useLocation, useParams } from 'react-router-dom';
import CompanyEnterpriseLayout from './components/layout/CompanyEnterpriseLayout';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { wrapInvalidateQueries } from './utils/companyDashboardPerfInstrumentation';
import { installCompanyDashboardApiTiming } from './utils/companyDashboardApiTiming';
import { startCompanyDashboardWebVitals } from './utils/companyDashboardWebPerf';
// ✅ P1-1: apiClient n'est plus utilisé directement (cookies httpOnly gèrent l'authentification)

import DefaultLayout from './store/layouts/DefaultLayout';
import ProtectedRoute from './utils/ProtectedRoute';
import PlatformSegmentGuard from './pages/admin/PlatformOps/PlatformSegmentGuard';
import AdminCapabilityGuard from './pages/admin/components/AdminCapabilityGuard';
import { ADMIN_CAP } from './pages/admin/capabilities/adminCapabilities';
import {
  RedirectLegacyReservations,
  RedirectLegacyReservationDetail,
  RedirectLegacyUsers,
  RedirectLegacyDemoRequests,
  RedirectLegacyBilling,
  RedirectLegacyBillingReleves,
  RedirectLegacyBillingConfig,
  RedirectLegacySettings,
  RedirectLegacyShadowMode,
  RedirectLegacyOptuna,
  RedirectLegacyPlatformOpsIndex,
  RedirectLegacyPlatformOpsSegment,
  RedirectToAdminFinance,
  RedirectPartnersToOrganizations,
} from './pages/admin/routing/adminLegacyRedirects';
import { adminPaths } from './pages/admin/routing/adminRoutePaths';
import GoogleMapsProvider from './components/common/GoogleMapsProvider';
import PwaOfflineBanner from './components/common/PwaOfflineBanner';
import { Toaster } from 'sonner';
import Home from './pages/Home/Home';
import BookNewRedirect from './pages/Auth/BookNewRedirect';
import { recordUserActivity } from './utils/userActivityTracker';
import {
  isRecoverableAuthError,
  isFreshTokenRequiredError,
  isSessionExpiredError,
  isRateLimitError,
} from './utils/queryAuthError';
import { SessionBootstrapProvider } from './contexts/SessionBootstrapContext';
import AuthNavigationBridge from './components/auth/AuthNavigationBridge';
import RouteSeoManager from './components/seo/RouteSeoManager';

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
const AdminOrganizations = lazy(() => import('./pages/admin/Organizations/AdminOrganizations'));
const AdminOrganizationDetail = lazy(
  () => import('./pages/admin/Organizations/AdminOrganizationDetail')
);
const AdminReservations = lazy(() => import('./pages/admin/Reservations/AdminReservations'));
const AdminBookingDetail = lazy(() => import('./pages/admin/Reservations/AdminBookingDetail'));
const AdminBillingOverview = lazy(() => import('./pages/admin/Billing/AdminBillingOverview'));
const AdminPlatformBilling = lazy(() => import('./pages/admin/PlatformBilling/AdminPlatformBilling'));
const AdminPlatformInvoicesRegistry = lazy(() =>
  import('./pages/admin/PlatformBilling/registry/AdminPlatformInvoicesRegistry')
);
const AdminBillingHub = lazy(() => import('./pages/admin/Billing/AdminBillingHub'));
const AdminBillingTransportConfig = lazy(() =>
  import('./pages/admin/Billing/AdminBillingTransportConfig')
);

/** Anciennes URLs facturation / pilotage → hub Finance. */
function RedirectToAdminBilling() {
  return <RedirectToAdminFinance />;
}

function RedirectLegacyPlatformBilling() {
  const { public_id } = useParams();
  const location = useLocation();
  return (
    <Navigate
      to={{
        pathname: adminPaths.financeReleves(public_id),
        search: location.search,
        hash: location.hash,
      }}
      replace
      state={location.state}
    />
  );
}

function ScrollToTopOnNavigation() {
  const location = useLocation();

  useEffect(() => {
    window.scrollTo({ top: 0, left: 0, behavior: 'auto' });
  }, [location.pathname, location.search]);

  useEffect(() => {
    recordUserActivity();
  }, [location.pathname, location.search]);

  return null;
}

function LegacySignupRedirect() {
  const location = useLocation();
  const params = new URLSearchParams(location.search || '');
  params.set('mode', 'signup');
  return <Navigate to={`/login?${params.toString()}`} replace />;
}

/** Routes entreprise qui rendent réellement une carte Google (pas factures/planning/dispatch). */
const COMPANY_MAP_ROUTE_PATTERNS = [
  /^\/(?:demo\/)?dashboard\/company\/[^/]+\/?$/,
  /^\/(?:demo\/)?dashboard\/company\/[^/]+\/(reservations|drivers|settings)(?:\/|$)/,
];

const MAP_ROUTE_PATTERNS = [
  /^\/dashboard\/client\/.+/,
  /^\/driver\/map(?:\/|$)/,
  ...COMPANY_MAP_ROUTE_PATTERNS,
];

function GoogleMapsRouteScope({ children }) {
  const location = useLocation();
  const pathname = location.pathname;
  const needsGoogleMaps = MAP_ROUTE_PATTERNS.some((pattern) => pattern.test(pathname));
  const isCompanyMapRoute = COMPANY_MAP_ROUTE_PATTERNS.some((pattern) => pattern.test(pathname));
  if (!needsGoogleMaps) {
    return children;
  }
  return <GoogleMapsProvider autoLoad={!isCompanyMapRoute}>{children}</GoogleMapsProvider>;
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
const InstitutionPatients = lazy(() => import('./pages/institution/Patients/InstitutionPatients'));
const InstitutionSettings = lazy(() => import('./pages/institution/Settings/InstitutionSettings'));

// ──────────────────────────────────────────────────────────
// Query Client (déclaré hors composant pour éviter recréation)
// Exporté pour permettre le nettoyage du cache au logout
export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry(failureCount, error) {
        if (
          isRecoverableAuthError(error) ||
          isFreshTokenRequiredError(error) ||
          isSessionExpiredError(error) ||
          isRateLimitError(error)
        ) {
          return false;
        }
        return failureCount < 2;
      },
      meta: {
        suppressAuthError: false,
      },
    },
    mutations: {
      retry(failureCount, error) {
        if (
          isRecoverableAuthError(error) ||
          isFreshTokenRequiredError(error) ||
          isSessionExpiredError(error) ||
          isRateLimitError(error)
        ) {
          return false;
        }
        return failureCount < 1;
      },
    },
  },
});
wrapInvalidateQueries(queryClient);
if (typeof window !== 'undefined') {
  installCompanyDashboardApiTiming();
  startCompanyDashboardWebVitals();
}

// ──────────────────────────────────────────────────────────

const App = () => {
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
      <SessionBootstrapProvider>
        <Router
          future={{
            v7_startTransition: true,
            v7_relativeSplatPath: true,
          }}
        >
        <AuthNavigationBridge />
        <RouteSeoManager />
        <ScrollToTopOnNavigation />
        <Toaster
          position="top-right"
          richColors
          closeButton
          containerAriaLabel="Notifications"
          closeButtonAriaLabel="Fermer la notification"
        />
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
              path="/force-reset-password"
              element={
                <DefaultLayout hideAuthEntry>
                  <ResetPassword resetMode="forced" />
                </DefaultLayout>
              }
            />
            <Route
              path="/force-reset-password/:token"
              element={
                <DefaultLayout hideAuthEntry>
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

              {/* Architecture cible — 6 workspaces */}
              <Route path="operations" element={<Navigate to="bookings" replace />} />
              <Route path="operations/bookings/:bookingId" element={<AdminBookingDetail />} />
              <Route path="operations/bookings" element={<AdminReservations />} />

              <Route path="partners" element={<RedirectPartnersToOrganizations />} />
              <Route path="partners/organizations" element={<AdminOrganizations />} />
              <Route
                path="partners/organizations/:publicId"
                element={<AdminOrganizationDetail />}
              />
              <Route path="partners/users" element={<AdminUsers />} />
              <Route path="partners/demo-requests" element={<AdminDemoRequests />} />

              <Route path="finance" element={<AdminBillingHub />}>
                <Route index element={<Navigate to="factures" replace />} />
                <Route path="releves" element={<Navigate to="../factures" replace />} />
                <Route path="factures" element={<AdminPlatformInvoicesRegistry />} />
                <Route path="config" element={<AdminBillingTransportConfig />} />
              </Route>

              <Route path="configuration" element={<AdminSettings />} />

              <Route path="advanced" element={<Navigate to="platform/overview" replace />} />
              <Route path="advanced/platform" element={<PlatformLayout />}>
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
              <Route
                path="advanced/labs/shadow-mode"
                element={
                  <AdminCapabilityGuard capability={ADMIN_CAP.LABS_READ}>
                    <ShadowModeDashboard />
                  </AdminCapabilityGuard>
                }
              />
              <Route
                path="advanced/labs/optuna"
                element={
                  <AdminCapabilityGuard capability={ADMIN_CAP.LABS_READ}>
                    <AdminOptuna />
                  </AdminCapabilityGuard>
                }
              />

              {/* Redirections legacy (conservent search/hash/state) */}
              <Route path="reservations/:bookingId" element={<RedirectLegacyReservationDetail />} />
              <Route path="reservations" element={<RedirectLegacyReservations />} />
              <Route path="users" element={<RedirectLegacyUsers />} />
              <Route path="demo-requests" element={<RedirectLegacyDemoRequests />} />
              <Route path="billing" element={<RedirectLegacyBilling />} />
              <Route path="billing/releves" element={<RedirectLegacyBillingReleves />} />
              <Route path="billing/config" element={<RedirectLegacyBillingConfig />} />
              <Route path="billing/pilotage" element={<RedirectToAdminBilling />} />
              <Route path="billing/pilotage/companies/:companyId" element={<RedirectToAdminBilling />} />
              <Route path="settings" element={<RedirectLegacySettings />} />
              <Route path="shadow-mode" element={<RedirectLegacyShadowMode />} />
              <Route path="optuna" element={<RedirectLegacyOptuna />} />
              <Route path="invoices" element={<RedirectToAdminBilling />} />
              <Route
                path="invoices/pilotage/companies/:companyId"
                element={<RedirectToAdminBilling />}
              />
              <Route path="platform-billing" element={<RedirectLegacyPlatformBilling />} />
              <Route path="platform-ops" element={<RedirectLegacyPlatformOpsIndex />} />
              <Route path="platform-ops/:segment" element={<RedirectLegacyPlatformOpsSegment />} />
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
              <Route path="requests/:requestId" element={<InstitutionRequests />} />
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
              <Route path="requests/:requestId" element={<InstitutionRequests />} />
              <Route path="patients" element={<InstitutionPatients />} />
              <Route path="settings" element={<InstitutionSettings />} />
            </Route>

            <Route path="/unauthorized" element={<Unauthorized />} />
            <Route path="*" element={<NotFound />} />
            </Routes>
          </Suspense>
        </GoogleMapsRouteScope>
        </Router>
      </SessionBootstrapProvider>
    </QueryClientProvider>
  );
};

export default App;
