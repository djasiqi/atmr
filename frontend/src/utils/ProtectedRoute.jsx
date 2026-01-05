import React from 'react';
import { Navigate, useLocation } from 'react-router-dom';
import { jwtDecode } from 'jwt-decode';

const ProtectedRoute = ({ allowedRoles, children }) => {
  const location = useLocation();
  const token = localStorage.getItem('authToken');
  const rawUser = localStorage.getItem('user');
  const user = rawUser ? JSON.parse(rawUser) : null;

  // ✅ Vérifier l'authentification : soit token dans localStorage (mobile), soit infos utilisateur (web avec cookies)
  // Si on utilise des cookies httpOnly, le token n'est pas dans localStorage, mais les infos utilisateur sont stockées
  if (!token && !user) {
    return <Navigate to="/login" replace state={{ from: location }} />;
  }

  // ✅ Si on a un token dans localStorage (mode mobile), vérifier son expiration
  // Si on utilise des cookies (pas de token), on compte sur le backend pour vérifier l'authentification
  let role = null;
  if (token) {
    try {
      const payload = jwtDecode(token);
      const now = Math.floor(Date.now() / 1000);
      if (typeof payload.exp === 'number' && payload.exp <= now) {
        localStorage.removeItem('authToken');
        localStorage.removeItem('refreshToken');
        localStorage.removeItem('user');
        return <Navigate to="/login" replace state={{ from: location }} />;
      }
      role = String(payload?.role ?? user?.role ?? '').toLowerCase();
    } catch {
      // Token invalide, supprimer et rediriger
      localStorage.removeItem('authToken');
      localStorage.removeItem('refreshToken');
      localStorage.removeItem('user');
      return <Navigate to="/login" replace state={{ from: location }} />;
    }
  } else {
    // ✅ Mode cookies httpOnly : utiliser les infos utilisateur stockées
    // Le backend vérifiera l'authentification via les cookies
    role = String(user?.role ?? '').toLowerCase();
  }

  // Si des rôles sont exigés, comparer en lowercase
  if (Array.isArray(allowedRoles) && allowedRoles.length > 0) {
    const allowed = allowedRoles.map((r) => String(r).toLowerCase());
    if (!allowed.includes(role)) {
      return <Navigate to="/unauthorized" replace />;
    }
  }

  return children;
};

export default ProtectedRoute;
