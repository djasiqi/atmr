import React from 'react';
import { Navigate, useLocation } from 'react-router-dom';
import { getAuthEnv } from '../../utils/webAuthSession';

const AppNamespaceRedirect = () => {
  const location = useLocation();
  const nextPath = location.pathname.replace(/^\/app/, '') || '/dashboard';
  const authEnv = getAuthEnv();
  const scopedPath =
    authEnv === 'demo' && nextPath.startsWith('/dashboard') ? `/demo${nextPath}` : nextPath;
  const nextUrl = `${scopedPath}${location.search || ''}`;
  return <Navigate to={nextUrl} replace />;
};

export default AppNamespaceRedirect;
