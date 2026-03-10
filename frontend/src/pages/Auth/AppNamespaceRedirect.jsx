import React from 'react';
import { Navigate, useLocation } from 'react-router-dom';

const AppNamespaceRedirect = () => {
  const location = useLocation();
  const nextPath = location.pathname.replace(/^\/app/, '') || '/dashboard';
  const authEnv = (localStorage.getItem('lirie_auth_env') || '').toLowerCase();
  const scopedPath =
    authEnv === 'demo' && nextPath.startsWith('/dashboard') ? `/demo${nextPath}` : nextPath;
  const nextUrl = `${scopedPath}${location.search || ''}`;
  return <Navigate to={nextUrl} replace />;
};

export default AppNamespaceRedirect;
