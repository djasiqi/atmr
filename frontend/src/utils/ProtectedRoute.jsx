import React from 'react';
import { Navigate, useLocation } from 'react-router-dom';
import { getAuthEnv, getEnvUser } from './webAuthSession';
import { useSessionBootstrap } from '../contexts/SessionBootstrapContext';

// Clés localStorage : snake_case (nouveau) + fallback camelCase pendant migration.
const STORAGE_KEYS = {
  company: {
    user: 'company_user',
    publicId: 'company_public_id',
  },
  driver: {
    user: 'driver_user',
    publicId: 'driver_public_id',
  },
  institution: {
    user: 'institution_user',
    publicId: 'institution_public_id',
  },
  legacy: { user: 'user', publicId: 'public_id' },
};

const getStorageKeys = (allowedRoles) => {
  if (!Array.isArray(allowedRoles) || allowedRoles.length === 0) return STORAGE_KEYS.legacy;
  const roles = allowedRoles.map((r) => String(r).toLowerCase());
  if (roles.includes('company') || roles.includes('admin')) return STORAGE_KEYS.company;
  if (roles.includes('driver')) return STORAGE_KEYS.driver;
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

// Resout la destination d'onboarding. Pour l'instant seul le changement
// de mot de passe est cable. Etendre ici pour CGU / profil / MFA.
export const resolveOnboardingRedirect = (u, pathname) => {
  if (u?.force_password_change && !pathname.startsWith('/force-reset-password')) {
    return '/force-reset-password';
  }
  return null;
};

const readJsonUser = (storageKey) => {
  try {
    const raw = localStorage.getItem(storageKey);
    return raw ? JSON.parse(raw) : null;
  } catch (_) {
    return null;
  }
};

const ProtectedRoute = ({ allowedRoles, children }) => {
  const location = useLocation();
  const keys = getStorageKeys(allowedRoles);
  const env = getAuthEnv();
  const { status, user: bootstrapUser } = useSessionBootstrap();

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

  if (status === 'loading' || status === 'idle') {
    return (
      <div
        style={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          minHeight: '40vh',
        }}
        aria-live="polite"
      >
        Vérification de la session…
      </div>
    );
  }

  const storageScopedUser = readJsonUser(keys.user);
  const envUser = getEnvUser(env);
  // Priorité bootstrap, puis storage scopé (company_user…), puis env.
  // Si le bootstrap est encore l'ancienne session (race juste après login),
  // on préfère le storage scopé quand son rôle matche la route.
  let user = bootstrapUser || storageScopedUser || envUser;

  if (status === 'anonymous') {
    // Login vient d'écrire la session en localStorage avant que le bootstrap
    // bascule : ne pas renvoyer au login si une session scopée est déjà là.
    if (storageScopedUser || envUser) {
      user = storageScopedUser || envUser;
    } else {
      return <Navigate to="/login" replace state={{ from: location }} />;
    }
  }

  if (status === 'error' && !user) {
    return <Navigate to="/login" replace state={{ from: location }} />;
  }

  const mustOnboard = user?.must_complete_onboarding ?? user?.force_password_change;
  const onboardingDestination = resolveOnboardingRedirect(user, location.pathname);
  if (mustOnboard && !onboardingDestination) {
    console.warn(
      'must_complete_onboarding=true mais aucune destination configuree',
      { reasons: user?.onboarding_reasons }
    );
  }
  if (mustOnboard && onboardingDestination) {
    return (
      <Navigate
        to={onboardingDestination}
        replace
        state={{ from: location }}
      />
    );
  }

  let role = normalizeRole(user?.role ?? '');

  if (Array.isArray(allowedRoles) && allowedRoles.length > 0) {
    const allowed = allowedRoles.map((r) => normalizeRole(r));
    if (!allowed.includes(role)) {
      // Race login : bootstrap encore sur l'ancien rôle (ex. admin) alors que
      // writeAuthSession a déjà posé company_user / app_user corrects.
      const fallbackCandidates = [storageScopedUser, envUser].filter(Boolean);
      const matching = fallbackCandidates.find((candidate) =>
        allowed.includes(normalizeRole(candidate?.role ?? ''))
      );
      if (matching) {
        user = matching;
        role = normalizeRole(matching.role ?? '');
      } else {
        return <Navigate to="/unauthorized" replace />;
      }
    }
  }

  return children;
};

export default ProtectedRoute;
