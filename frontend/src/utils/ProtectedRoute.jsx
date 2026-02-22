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

const getStorageKeys = (allowedRoles) => {
  if (!Array.isArray(allowedRoles) || allowedRoles.length === 0) return STORAGE_KEYS.legacy;
  const roles = allowedRoles.map((r) => String(r).toLowerCase());
  if (roles.includes('company') || roles.includes('admin')) return STORAGE_KEYS.company;
  if (roles.includes('driver')) return STORAGE_KEYS.driver;
  // ✅ ÉTAPE 6: Support rôle institution
  if (roles.includes('institution')) return STORAGE_KEYS.institution;
  return STORAGE_KEYS.legacy;
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
  const token = getToken(keys);
  const rawUser = localStorage.getItem(keys.user);
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
      role = String(payload?.role ?? user?.role ?? '').toLowerCase();
    } catch {
      clearSession(keys);
      return <Navigate to="/login" replace state={{ from: location }} />;
    }
  } else {
    role = String(user?.role ?? '').toLowerCase();
  }

  if (Array.isArray(allowedRoles) && allowedRoles.length > 0) {
    const allowed = allowedRoles.map((r) => String(r).toLowerCase());
    if (!allowed.includes(role)) {
      return <Navigate to="/unauthorized" replace />;
    }
  }

  return children;
};

export default ProtectedRoute;
