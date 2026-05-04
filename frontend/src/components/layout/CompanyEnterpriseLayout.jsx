import React, { Suspense, useEffect } from 'react';
import { Outlet } from 'react-router-dom';
import CompanyShellProvider from '../../providers/CompanyShellProvider';
import { schedulePrefetchGoogleMaps } from '../../utils/googleMapsLoader';

/** Fallback léger pour les routes entreprise lazy — évite le plein écran App.js pendant le téléchargement du chunk. */
function CompanyOutletFallback() {
  return (
    <div
      style={{
        padding: '32px 24px',
        maxWidth: 960,
        margin: '0 auto',
        minHeight: '320px',
      }}
    >
      <div
        style={{
          width: '100%',
          height: '18px',
          borderRadius: '8px',
          marginBottom: '14px',
          background: 'linear-gradient(90deg, #eef2f7 0%, #f8fafc 50%, #eef2f7 100%)',
        }}
      />
      <div
        style={{
          width: '100%',
          height: '140px',
          borderRadius: '12px',
          background: 'linear-gradient(90deg, #eef2f7 0%, #f8fafc 50%, #eef2f7 100%)',
        }}
      />
      <div style={{ fontSize: '15px', color: '#64748b', marginTop: '16px' }}>
        Chargement du module…
      </div>
    </div>
  );
}

/** Précharge les chunks les plus utilisés après paint (cache webpack → navigation quasi instantanée). */
function schedulePrefetchCompanyChunks() {
  const loaders = [
    () => import('../../pages/company/Dashboard/CompanyDashboard'),
    () => import('../../pages/company/Invoices/ClientInvoices'),
    () => import('../../pages/company/Invoices/CompanyInvoices'),
    () => import('../../pages/company/Reservations/CompanyReservations'),
  ];
  const run = () => {
    loaders.forEach((fn) => {
      void fn().catch(() => {});
    });
  };
  if (typeof window.requestIdleCallback === 'function') {
    const id = window.requestIdleCallback(run, { timeout: 1200 });
    return () => window.cancelIdleCallback(id);
  }
  const t = window.setTimeout(run, 350);
  return () => window.clearTimeout(t);
}

/**
 * Layout routes entreprise : précharge le profil via {@link CompanyShellProvider} une fois par arborescence.
 * Précharge le SDK Google Maps (idle + repli) pour limiter l’attente sur la vue carte / dispatch.
 * Suspense interne : le fallback plein écran global (App.js) ne s’affiche plus lors des navigations lazy sous cette arborescence.
 */
export default function CompanyEnterpriseLayout() {
  useEffect(() => {
    return schedulePrefetchGoogleMaps();
  }, []);

  useEffect(() => {
    return schedulePrefetchCompanyChunks();
  }, []);

  return (
    <CompanyShellProvider>
      <Suspense fallback={<CompanyOutletFallback />}>
        <Outlet />
      </Suspense>
    </CompanyShellProvider>
  );
}
