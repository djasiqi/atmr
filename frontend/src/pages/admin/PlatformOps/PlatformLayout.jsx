import React from 'react';
import { Outlet } from 'react-router-dom';
import adminShell from '../adminShell.module.css';

/**
 * Sous-layout plateforme : Outlet uniquement.
 * Sous-nav et filtrage : AdminWorkspaceNav + PlatformSegmentGuard sur les routes.
 */
const PlatformLayout = () => (
  <main className={adminShell.content}>
    <Outlet />
  </main>
);

export default PlatformLayout;
