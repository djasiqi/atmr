import React from 'react';
import { Navigate, useLocation } from 'react-router-dom';
import { jwtDecode } from 'jwt-decode';

// Clés localStorage : snake_case (nouveau) + fallback camelCase pendant migration.
const STORAGE_KEYS = {
  company: {
    token: 'company_access_token',
    refresh: 'company_refresh_token',
    tokenLegacy: 'company_authToken',
    refreshLegacy: 'company_refreshToken',
    user: 'company_user',
    publicId: 'company_public_id',
  },
  driver: {
    token: 'driver_access_token',
    refresh: 'driver_refresh_token',
    tokenLegacy: 'driver_authToken',
    refreshLegacy: 'driver_refreshToken',
    user: 'driver_user',
    publicId: 'driver_public_id',
  },
  // ✅ ÉTAPE 6: Clés pour les utilisateurs institution
  institution: {
    token: 'institution_access_token',
    refresh: 'institution_refresh_token',
    tokenLegacy: null,
    refreshLegacy: null,
    user: 'institution_user',
    publicId: 'institution_public_id',
  },
  legacy: { token: 'authToken', refresh: 'refreshToken', tokenLegacy: null, refreshLegacy: null, user: 'user', publicId: 'public_id' },
};

const getToken = (keys) =>
  localStorage.getItem(keys.token) || (keys.tokenLegacy ? localStorage.getItem(keys.tokenLegacy) : null);

const getCurrentAuthEnv = () =>
  localStorage.getItem('lirie_auth_env') === 'demo' ? 'demo' : 'app';

const getStorageKeys = (allowedRoles) => {
  if (!Array.isArray(allowedRoles) || allowedRoles.length === 0) return STORAGE_KEYS.legacy;
  const roles = allowedRoles.map((r) => String(r).toLowerCase());
  if (roles.includes('company') || roles.includes('admin')) return STORAGE_KEYS.company;
  if (roles.includes('driver')) return STORAGE_KEYS.driver;
  // ✅ ÉTAPE 6: Support rôle institution
  if (roles.includes('institution')) return STORAGE_KEYS.institution;
  return STORAGE_KEYS.legacy;
};

const normalizeRole = (rawRole) => {
  const role = String(rawRole || '').trim().toLowerCase();
  if (!role) return '';
  if (role.startsWith('institution')) return 'institution';
  if (role.startsWith('company') || role.startsWith('transport_company')) return 'company';
  return role;
};

const clearSession = (keys) => {
  try {
    localStorage.removeItem(keys.token);
    localStorage.removeItem(keys.refresh);
    if (keys.tokenLegacy) localStorage.removeItem(keys.tokenLegacy);
    if (keys.refreshLegacy) localStorage.removeItem(keys.refreshLegacy);
    localStorage.removeItem(keys.user);
    localStorage.removeItem(keys.publicId);
  } catch (_) {}
};

const ProtectedRoute = ({ allowedRoles, children }) => {
  const location = useLocation();
  const keys = getStorageKeys(allowedRoles);
  const env = getCurrentAuthEnv();
  const isDemoDashboardPath =
    location.pathname === '/dashboard' ||
    location.pathname.startsWith('/dashboard/company/') ||
    location.pathname.startsWith('/dashboard/institution/');
  if (env === 'demo' && isDemoDashboardPath) {
    return (
      <Navigate
        to={`/demo${location.pathname}${location.search || ''}`}
        replace
      />
    );
  }
  const envToken = localStorage.getItem(`${env}_access_token`) || localStorage.getItem('authToken');
  const token = getToken(keys) || envToken;
  const rawUser =
    localStorage.getItem(keys.user) ||
    localStorage.getItem(`${env}_user`) ||
    localStorage.getItem('user');
  const user = rawUser ? JSON.parse(rawUser) : null;

  if (!token && !user) {
    return <Navigate to="/login" replace state={{ from: location }} />;
  }

  let role = null;
  if (token) {
    try {
      const payload = jwtDecode(token);
      const now = Math.floor(Date.now() / 1000);
      if (typeof payload.exp === 'number' && payload.exp <= now) {
        clearSession(keys);
        return <Navigate to="/login" replace state={{ from: location }} />;
      }
      role = normalizeRole(payload?.role ?? user?.role ?? '');
    } catch {
      // Fallback robuste: si le token scoped est absent/invalide mais le user est présent,
      // on continue avec le rôle stocké (utile pour certains flux démo).
      role = normalizeRole(user?.role ?? '');
      if (!role) {
        clearSession(keys);
        return <Navigate to="/login" replace state={{ from: location }} />;
      }
    }
  } else {
    role = normalizeRole(user?.role ?? '');
  }

  if (Array.isArray(allowedRoles) && allowedRoles.length > 0) {
    const allowed = allowedRoles.map((r) => normalizeRole(r));
    if (!allowed.includes(role)) {
      return <Navigate to="/unauthorized" replace />;
    }
  }

  return children;
};

export default ProtectedRoute;
