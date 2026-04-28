import React, { useEffect } from 'react';
import { Outlet } from 'react-router-dom';
import CompanyShellProvider from '../../providers/CompanyShellProvider';
import { schedulePrefetchGoogleMaps } from '../../utils/googleMapsLoader';

/**
 * Layout routes entreprise : précharge le profil via {@link CompanyShellProvider} une fois par arborescence.
 * Précharge le SDK Google Maps (idle + repli) pour limiter l’attente sur la vue carte / dispatch.
 */
export default function CompanyEnterpriseLayout() {
  useEffect(() => {
    return schedulePrefetchGoogleMaps();
  }, []);

  return (
    <CompanyShellProvider>
      <Outlet />
    </CompanyShellProvider>
  );
}
