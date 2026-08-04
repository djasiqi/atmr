import React from 'react';
import { Outlet } from 'react-router-dom';
import shell from '../adminShell.module.css';

/**
 * Hub Finance — Outlet uniquement (titre porté par la page enfant).
 */
const AdminBillingHub = () => (
  <main className={shell.content}>
    <Outlet />
  </main>
);

export default AdminBillingHub;
