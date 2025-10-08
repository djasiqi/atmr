import React from "react";
import { Navigate, useLocation } from "react-router-dom";
import { jwtDecode } from "jwt-decode";

const ProtectedRoute = ({ allowedRoles, children }) => {
  const location = useLocation();
  const token = localStorage.getItem("authToken");
  const rawUser = localStorage.getItem("user");
  const user = rawUser ? JSON.parse(rawUser) : null;

  // Pas de token → login
  if (!token) {
    return <Navigate to="/login" replace state={{ from: location }} />;
  }

  // Vérif expiration & rôle depuis le token (source de vérité)
  let role = null;
  try {
    const payload = jwtDecode(token);
    const now = Math.floor(Date.now() / 1000);
    if (typeof payload.exp === "number" && payload.exp <= now) {
      localStorage.removeItem("authToken");
      localStorage.removeItem("refreshToken");
      return <Navigate to="/login" replace state={{ from: location }} />;
    }
    role = String(payload?.role ?? user?.role ?? "").toLowerCase();
  } catch {
    return <Navigate to="/login" replace state={{ from: location }} />;
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
