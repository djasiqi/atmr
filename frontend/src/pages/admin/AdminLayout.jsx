import React from 'react';
import { Outlet, useLocation, useParams } from 'react-router-dom';
import AdminSidebar from '../../components/layout/Sidebar/AdminSidebar/AdminSidebar';
import AdminTopbar from './shell/AdminTopbar';
import AdminWorkspaceNav from './shell/AdminWorkspaceNav';
import shellStyles from './shell/AdminShell.module.css';
import './shell/adminTokens.css';

/**
 * Cadre commun admin : sidebar + topbar + sous-nav workspace + pages filles.
 * Routes sous /dashboard/admin/:public_id/* dans App.js.
 */
const AdminLayout = () => {
  const { public_id: publicId } = useParams();
  const location = useLocation();

  return (
    <div className={`adminShell ${shellStyles.adminShell}`}>
      <AdminSidebar />
      <div className={shellStyles.adminMain}>
        <AdminTopbar publicId={publicId} pathname={location.pathname} />
        <AdminWorkspaceNav />
        <div className={shellStyles.adminMainBody}>
          <Outlet />
        </div>
      </div>
    </div>
  );
};

export default AdminLayout;
