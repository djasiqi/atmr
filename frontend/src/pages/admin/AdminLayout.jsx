import React from 'react';
import { Outlet } from 'react-router-dom';
import HeaderDashboard from '../../components/layout/Header/HeaderDashboard';
import AdminSidebar from '../../components/layout/Sidebar/AdminSidebar/AdminSidebar';
import styles from './Dashboard/AdminDashboard.module.css';

/**
 * Cadre commun admin : header + sidebar + pages filles (Outlet).
 * Les routes sont définies sous /dashboard/admin/:public_id/* dans App.js.
 */
const AdminLayout = () => (
  <div className={styles.adminContainer}>
    <HeaderDashboard />
    <div className={styles.dashboard}>
      <AdminSidebar />
      <Outlet />
    </div>
  </div>
);

export default AdminLayout;
