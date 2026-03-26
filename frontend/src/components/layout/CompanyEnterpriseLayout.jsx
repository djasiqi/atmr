import React from 'react';
import { Outlet } from 'react-router-dom';
import CompanyShellProvider from '../../providers/CompanyShellProvider';

/**
 * Layout routes entreprise : précharge le profil via {@link CompanyShellProvider} une fois par arborescence.
 */
export default function CompanyEnterpriseLayout() {
  return (
    <CompanyShellProvider>
      <Outlet />
    </CompanyShellProvider>
  );
}
