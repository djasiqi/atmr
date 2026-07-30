import React, { useState, useEffect, useMemo, useRef } from 'react';
import { useNavigate, useLocation, Link, useSearchParams } from 'react-router-dom';

import apiClient, { cleanLocalSession } from '../../utils/apiClient';
import { jwtDecode } from 'jwt-decode';
import { queryClient } from '../../App';
import { buildSafeAppPath, pathFromNextQueryParam } from '../../utils/safeReturnPath';
import { hasActiveSession, normalizeAuthRole, writeAuthSession } from '../../utils/webAuthSession';
import {
  beginLoginSession,
  clearExplicitLogoutMarker,
  endLoginSession,
  hasRecentExplicitLogout,
} from '../../utils/sessionLogoutState';
import { linkMobilityProfileToUser, saveMobilityProfileForEmail } from '../../utils/clientMobilityProfile';
import {
  getPendingActivationByEmail,
  removePendingActivationByEmail,
  setPendingActivationSession,
} from '../../utils/activationSessionStore';
import useAuthToken from '../../hooks/useAuthToken';
import AddressAutocomplete from '../../components/common/AddressAutocomplete';
import styles from './Login.module.css';
import institutionStyles from '../institution/Requests/InstitutionRequestForm.module.css';
import { getApiErrorMessage } from '../../utils/apiErrorMessage';

const REMEMBER_KEY = 'lirie_remember_me';
const SIGNUP_DISABLED =
  process.env.REACT_APP_SIGNUP_DISABLED === 'true' || process.env.REACT_APP_SIGNUP_DISABLED === '1';
const CIVILITY_OPTIONS = [
  { value: '', label: 'Civilité' },
  { value: 'Mme', label: 'Mme' },
  { value: 'M.', label: 'M.' },
  { value: 'Autre', label: 'Autre' },
];

const rolePathSegment = (role) => {
  const normalized = normalizeAuthRole(role);
  if (normalized === 'admin') return 'admin';
  if (normalized === 'company') return 'company';
  if (normalized === 'institution') return 'institution';
  if (normalized === 'driver') return 'driver';
  if (normalized === 'client') return 'client';
  return normalized;
};

const isReturnPathAllowedForRole = (path, role) => {
  if (!path) return false;
  const segment = rolePathSegment(role);
  if (!segment) return false;

  const normalizedPath = String(path);
  const dashboardMatch = normalizedPath.match(
    /^\/(?:app\/)?(?:demo\/)?dashboard\/([^/?#]+)(?:[/?#]|$)/
  );
  if (dashboardMatch) {
    return normalizeAuthRole(dashboardMatch[1]) === segment;
  }

  if (segment === 'client' && normalizedPath.startsWith('/client/')) {
    return true;
  }

  return !/^\/(?:app\/)?(?:demo\/)?dashboard(?:[/?#]|$)/.test(normalizedPath);
};

const firstAllowedReturnPath = (role, ...paths) =>
  paths.find((path) => isReturnPathAllowedForRole(path, role)) || null;

const EyeIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
    <circle cx="12" cy="12" r="3" />
  </svg>
);

const EyeOffIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24" />
    <line x1="1" y1="1" x2="23" y2="23" />
  </svg>
);

const Login = () => {
  const location = useLocation();
  const [searchParams] = useSearchParams();
  const authUser = useAuthToken();
  const modeParam = String(searchParams.get('mode') || '').toLowerCase();
  const authMode = modeParam === 'signup' ? 'signup' : 'login';
  const isSignupMode = authMode === 'signup';
  const justActivated = location.state?.activated === true;
  const nextFromQuery = pathFromNextQueryParam(searchParams.get('next') || '');
  const fromProtectedRoute = location.state?.from;
  const safeReturnFromState =
    fromProtectedRoute?.pathname != null
      ? buildSafeAppPath(fromProtectedRoute.pathname, fromProtectedRoute.search || '')
      : null;

  const [loginFormData, setLoginFormData] = useState({ email: '', password: '' });
  const [signupFormData, setSignupFormData] = useState({
    civility: '',
    firstName: '',
    lastName: '',
    username: '',
    email: '',
    password: '',
    confirmPassword: '',
    phone: '',
    address: '',
    needsWheelchair: false,
    needsElectricWheelchair: false,
    needsWalkingAid: false,
    needsDoorToDoorAssistance: false,
    assistanceLevel: '',
    emergencyContact: '',
    mobilityNotes: '',
  });
  const [showLoginPassword, setShowLoginPassword] = useState(false);
  const [showSignupPassword, setShowSignupPassword] = useState(false);
  const [isCivilityOpen, setIsCivilityOpen] = useState(false);
  const [rememberMe, setRememberMe] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');
  const [successMessage, setSuccessMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const navigate = useNavigate();
  const civilityDropdownRef = useRef(null);
  const hasSafeNext = Boolean(nextFromQuery || safeReturnFromState);
  const loginSearch = useMemo(() => {
    const params = new URLSearchParams();
    params.set('mode', 'login');
    if (nextFromQuery) params.set('next', nextFromQuery);
    const raw = params.toString();
    return raw ? `?${raw}` : '';
  }, [nextFromQuery]);
  const signupSearch = useMemo(() => {
    const params = new URLSearchParams();
    params.set('mode', 'signup');
    if (nextFromQuery) params.set('next', nextFromQuery);
    const raw = params.toString();
    return raw ? `?${raw}` : '';
  }, [nextFromQuery]);

  useEffect(() => {
    try {
      const saved = localStorage.getItem(REMEMBER_KEY);
      if (!saved) return;
      let parsed = null;
      try {
        parsed = JSON.parse(saved);
      } catch {
        localStorage.removeItem(REMEMBER_KEY);
        return;
      }
      const email = typeof parsed?.email === 'string' ? parsed.email.trim() : '';
      const hadLegacyPassword =
        parsed && Object.prototype.hasOwnProperty.call(parsed, 'password');

      if (!email) {
        if (hadLegacyPassword) {
          localStorage.removeItem(REMEMBER_KEY);
        }
        return;
      }

      // Migration: l'ancien format pouvait stocker { email, password }.
      // On purge tout résiduel password et on normalise vers email-only.
      if (hadLegacyPassword || parsed?.version !== 2) {
        try {
          localStorage.setItem(
            REMEMBER_KEY,
            JSON.stringify({ email, version: 2 }),
          );
        } catch { /* quota plein, on ignore */ }
      }

      // Préremplir uniquement l'email; ne JAMAIS repopuler le mot de passe.
      setLoginFormData((prev) => ({ ...prev, email }));
      setRememberMe(true);
    } catch { /* ignore corrupted data */ }
  }, []);

  useEffect(() => {
    if (justActivated) {
      setSuccessMessage('Compte activé avec succès ! Connectez-vous avec votre nouveau mot de passe.');
    }
    if (location.state?.signupSuccess === true) {
      setSuccessMessage('Inscription réussie. Connectez-vous pour continuer.');
    }
    if (location.state?.prefillEmail && typeof location.state.prefillEmail === 'string') {
      setLoginFormData((prev) => ({ ...prev, email: location.state.prefillEmail }));
    }
  }, [justActivated, location.state]);

  useEffect(() => {
    if (hasRecentExplicitLogout()) return;
    if (!authUser || !hasActiveSession()) return;
    const destination =
      firstAllowedReturnPath(authUser.role, nextFromQuery, safeReturnFromState) ||
      '/dashboard';
    navigate(destination, { replace: true });
  }, [authUser, navigate, nextFromQuery, safeReturnFromState]);

  useEffect(() => {
    setErrorMessage('');
    setIsCivilityOpen(false);
  }, [authMode]);

  useEffect(() => {
    if (!isCivilityOpen) return undefined;
    const handleDocumentMouseDown = (event) => {
      if (!civilityDropdownRef.current) return;
      if (!civilityDropdownRef.current.contains(event.target)) {
        setIsCivilityOpen(false);
      }
    };
    const handleDocumentEscape = (event) => {
      if (event.key === 'Escape') {
        setIsCivilityOpen(false);
      }
    };
    document.addEventListener('mousedown', handleDocumentMouseDown);
    document.addEventListener('keydown', handleDocumentEscape);
    return () => {
      document.removeEventListener('mousedown', handleDocumentMouseDown);
      document.removeEventListener('keydown', handleDocumentEscape);
    };
  }, [isCivilityOpen]);

  const handleLoginInputChange = (e) => {
    const { name, value } = e.target;
    setLoginFormData({ ...loginFormData, [name]: value });
    setErrorMessage('');
  };

  const handleSignupInputChange = (e) => {
    const { name, value } = e.target;
    const nextValue = name === 'lastName' ? value.toLocaleUpperCase('fr-CH') : value;
    setSignupFormData((prev) => {
      const next = { ...prev, [name]: nextValue };
      // Mot de passe uniquement avec email : on efface si l'email est vidé
      if (name === 'email' && !String(nextValue || '').trim()) {
        next.password = '';
        next.confirmPassword = '';
      }
      return next;
    });
    setErrorMessage('');
  };

  const handleSignupToggleChange = (name) => {
    setSignupFormData((prev) => ({
      ...prev,
      [name]: !prev[name],
    }));
    setErrorMessage('');
  };

  const handleCivilitySelect = (value) => {
    setSignupFormData((prev) => ({ ...prev, civility: value }));
    setIsCivilityOpen(false);
    setErrorMessage('');
  };

  const handleClearMobilityNeeds = () => {
    setSignupFormData((prev) => ({
      ...prev,
      needsWheelchair: false,
      needsElectricWheelchair: false,
      needsWalkingAid: false,
      needsDoorToDoorAssistance: false,
      assistanceLevel: '',
      emergencyContact: '',
      mobilityNotes: '',
    }));
    setErrorMessage('');
  };

  const validateLoginForm = () => {
    const { email, password } = loginFormData;
    const identifier = email.trim();

    if (!identifier || !password) {
      setErrorMessage('Veuillez remplir tous les champs.');
      return false;
    }

    if (identifier.includes('@')) {
      const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
      if (!emailRegex.test(identifier)) {
        setErrorMessage('Veuillez entrer une adresse email valide.');
        return false;
      }
    }

    if (password.length < 6) {
      setErrorMessage('Le mot de passe doit contenir au moins 6 caractères.');
      return false;
    }

    return true;
  };

  const validateSignupForm = () => {
    const { firstName, lastName, email, password, confirmPassword, phone } = signupFormData;
    const trimmedEmail = email.trim();
    const trimmedPhone = phone.trim();
    const hasEmail = Boolean(trimmedEmail);

    if (!firstName.trim() || !lastName.trim()) {
      setErrorMessage('Prénom et nom sont obligatoires.');
      return false;
    }

    if (!trimmedEmail && !trimmedPhone) {
      setErrorMessage('Indiquez une adresse email ou un numéro de téléphone.');
      return false;
    }

    if (hasEmail) {
      const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
      if (!emailRegex.test(trimmedEmail)) {
        setErrorMessage('Veuillez entrer une adresse email valide.');
        return false;
      }
      if (!password || !confirmPassword) {
        setErrorMessage('Le mot de passe est obligatoire lorsque vous indiquez un email.');
        return false;
      }
      if (password.length < 8) {
        setErrorMessage('Le mot de passe doit contenir au moins 8 caractères.');
        return false;
      }
      if (password !== confirmPassword) {
        setErrorMessage('Les mots de passe ne correspondent pas.');
        return false;
      }
    }

    if (trimmedPhone && trimmedPhone.length < 6) {
      setErrorMessage('Veuillez entrer un numéro de téléphone valide.');
      return false;
    }

    return true;
  };

  const handleLoginSubmit = async (e) => {
    e.preventDefault();

    if (!validateLoginForm()) return;

    setIsLoading(true);
    beginLoginSession();
    let loginSucceeded = false;
    try {
      const response = await apiClient.post(
        '/auth/login',
        {
          email: loginFormData.email,
          password: loginFormData.password,
          remember_me: rememberMe,
        },
        { skipCsrf: true },
      );
      const { token, user, refresh_token, target_env, redirect_to } = response.data;

      if (!user || !user.role || !user.public_id) {
        throw new Error('Aucune information utilisateur reçue.');
      }

      // ⚠️ Sécurité: on ne stocke JAMAIS le mot de passe en clair.
      // REMEMBER_KEY ne contient que l'email (et un marqueur de version)
      // pour pré-remplir l'identifiant et cocher la case au prochain chargement.
      if (rememberMe) {
        try {
          localStorage.setItem(
            REMEMBER_KEY,
            JSON.stringify({ email: loginFormData.email, version: 2 }),
          );
        } catch { /* quota plein, on ignore */ }
      } else {
        localStorage.removeItem(REMEMBER_KEY);
      }

      cleanLocalSession();
      try {
        const { clearTenantScopedClientCaches } = await import('../../utils/clearTenantScopedClientCaches');
        clearTenantScopedClientCaches();
      } catch {
        queryClient.clear();
      }
      clearExplicitLogoutMarker();

      let roleSegment;
      if (token && typeof token === 'string') {
        const decodedToken = jwtDecode(token);
        roleSegment = normalizeAuthRole(decodedToken.role || user.role);
      } else {
        roleSegment = normalizeAuthRole(user.role);
      }

      writeAuthSession({
        env: target_env,
        user,
        role: roleSegment,
        accessToken: token,
        refreshToken: refresh_token,
      });
      try {
        const { resumeSessionKeepAlive } = await import('../../utils/sessionKeepAlive');
        resumeSessionKeepAlive();
      } catch (_) {
        // ignore
      }
      removePendingActivationByEmail(loginFormData.email);
      linkMobilityProfileToUser({
        publicId: user.public_id,
        email: user.email || loginFormData.email,
      });

      window.dispatchEvent(new Event('auth-changed'));

      loginSucceeded = true;

      if (user.force_password_change) {
        navigate('/force-reset-password', { replace: true });
      } else {
        const preferredReturn = firstAllowedReturnPath(
          roleSegment,
          redirect_to,
          safeReturnFromState,
          nextFromQuery
        );
        const destination =
          preferredReturn ||
          `/dashboard/${roleSegment}/${user.public_id}`;
        navigate(destination, { replace: true });
      }
    } catch (error) {
      const responseData = error?.response?.data;
      const status = error?.response?.status;
      if (status === 401) {
        setErrorMessage("Les données de connexion sont incorrectes.");
        return;
      }
      if (status === 403 && responseData?.reason === 'account_pending_activation') {
        const pendingActivation = getPendingActivationByEmail(loginFormData.email);
        const sessionId =
          pendingActivation?.activation_session_id || responseData?.activation_session_id;
        const maskedEmail =
          pendingActivation?.masked_email || responseData?.masked_email || null;
        const maskedPhone =
          pendingActivation?.masked_phone || responseData?.masked_phone || null;
        if (sessionId) {
          setPendingActivationSession({
            email: loginFormData.email,
            activation_session_id: sessionId,
            masked_email: maskedEmail,
            masked_phone: maskedPhone,
          });
          const params = new URLSearchParams();
          params.set('activation_session_id', sessionId);
          navigate(`/activate-account?${params.toString()}`, {
            replace: true,
            state: {
              prefillEmail: loginFormData.email,
              maskedEmail,
              maskedPhone,
            },
          });
          return;
        }
        setErrorMessage('Compte en attente de validation email/SMS. Vérifiez vos messages ou réinscrivez-vous.');
        return;
      }
      console.error('Erreur lors de la connexion :', {
        status,
        url: error?.config?.url,
        baseURL: error?.config?.baseURL,
        data: responseData,
        code: error?.code,
      });

      setErrorMessage(getApiErrorMessage(error, 'Impossible de se connecter pour le moment.'));
    } finally {
      setIsLoading(false);
      if (!loginSucceeded) {
        endLoginSession();
      }
    }
  };

  const handleSignupSubmit = async (e) => {
    e.preventDefault();

    if (SIGNUP_DISABLED) {
      setErrorMessage("Les inscriptions sont temporairement suspendues. Contactez info@lirie.ch.");
      return;
    }

    if (!validateSignupForm()) return;

    setIsLoading(true);
    try {
      const normalizedFirstName = signupFormData.firstName.trim();
      const normalizedLastName = signupFormData.lastName.trim().toLocaleUpperCase('fr-CH');
      const trimmedEmail = signupFormData.email.trim();
      const trimmedPhone = signupFormData.phone.trim();
      const trimmedAddress = signupFormData.address.trim();
      const payload = {
        username: `${normalizedFirstName} ${normalizedLastName}`.trim(),
        first_name: normalizedFirstName,
        last_name: normalizedLastName,
        ...(trimmedEmail
          ? { email: trimmedEmail, password: signupFormData.password }
          : {}),
        ...(trimmedPhone ? { phone: trimmedPhone } : {}),
        ...(trimmedAddress ? { address: trimmedAddress } : {}),
      };
      const registerResponse = await apiClient.post('/auth/register', payload);
      const activationSessionId = registerResponse?.data?.activation_session_id;
      const maskedEmail = registerResponse?.data?.masked_email;
      const maskedPhone = registerResponse?.data?.masked_phone;
      if (!activationSessionId) {
        const backendMessage =
          registerResponse?.data?.message ||
          registerResponse?.data?.error ||
          "Inscription creee mais activation indisponible. Contactez le support.";
        setErrorMessage(
          `${backendMessage} (activation_session_id manquant)`
        );
        setIsLoading(false);
        return;
      }
      if (activationSessionId) {
        setPendingActivationSession({
          email: trimmedEmail || null,
          activation_session_id: activationSessionId,
          masked_email: maskedEmail,
          masked_phone: maskedPhone,
        });
      }
      if (trimmedEmail) {
        saveMobilityProfileForEmail(trimmedEmail, {
          needsWheelchair: signupFormData.needsWheelchair,
          needsElectricWheelchair: signupFormData.needsElectricWheelchair,
          needsWalkingAid: signupFormData.needsWalkingAid,
          needsDoorToDoorAssistance: signupFormData.needsDoorToDoorAssistance,
          assistanceLevel: signupFormData.assistanceLevel,
          emergencyContact: signupFormData.emergencyContact,
          notes: signupFormData.mobilityNotes,
        });
      }
      const params = new URLSearchParams();
      if (activationSessionId) {
        params.set('activation_session_id', activationSessionId);
      }
      const rawActivationQuery = params.toString();
      navigate(`/activate-account${rawActivationQuery ? `?${rawActivationQuery}` : ''}`, {
        replace: true,
        state: {
          signupSuccess: true,
          prefillEmail: trimmedEmail || null,
          maskedEmail: maskedEmail || null,
          maskedPhone: maskedPhone || null,
        },
      });
    } catch (error) {
      setErrorMessage(getApiErrorMessage(error, "Impossible de créer le compte pour le moment."));
    } finally {
      setIsLoading(false);
    }
  };

  const passwordChecks = useMemo(() => {
    const password = signupFormData.password || '';
    return {
      minLength: password.length >= 8,
      uppercase: /[A-Z]/.test(password),
      lowercase: /[a-z]/.test(password),
      digit: /\d/.test(password),
      special: /[^A-Za-z0-9]/.test(password),
    };
  }, [signupFormData.password]);

  const strengthScore = useMemo(() => {
    const values = [
      passwordChecks.minLength,
      passwordChecks.uppercase,
      passwordChecks.lowercase,
      passwordChecks.digit,
      passwordChecks.special,
    ];
    return values.filter(Boolean).length;
  }, [passwordChecks]);

  const strengthLabel = useMemo(() => {
    if (strengthScore <= 2) return 'Faible';
    if (strengthScore <= 4) return 'Moyen';
    return 'Fort';
  }, [strengthScore]);

  const strengthToneClass = useMemo(() => {
    if (strengthScore <= 2) return styles.passwordStrengthWeak;
    if (strengthScore <= 4) return styles.passwordStrengthMedium;
    return styles.passwordStrengthStrong;
  }, [strengthScore]);

  return (
    <div className={styles.pageWrapper}>
      <div className={styles.loginCard}>
        <div className={styles.header}>
          <div className={`${styles.modeSwitchBlock} ${styles.modeSwitchFieldScope}`}>
            <span id="login-auth-mode-label" className={styles.modeSwitchLabel}>
              Mode d&apos;authentification
            </span>
            <div
              className={`${institutionStyles.missionSegment} ${styles.modeSwitchSegment}`}
              role="tablist"
              aria-labelledby="login-auth-mode-label"
            >
              <Link
                to={`/login${loginSearch}`}
                replace
                role="tab"
                aria-selected={!isSignupMode}
                className={`${institutionStyles.missionBtn} ${styles.modeSwitchBtn} ${!isSignupMode ? institutionStyles.missionBtnActive : ''}`}
              >
                Connexion
              </Link>
              <Link
                to={`/login${signupSearch}`}
                replace
                role="tab"
                aria-selected={isSignupMode}
                className={`${institutionStyles.missionBtn} ${styles.modeSwitchBtn} ${isSignupMode ? institutionStyles.missionBtnActive : ''}`}
              >
                Inscription
              </Link>
            </div>
          </div>
          <img src="/logo-lirie.png" alt="Lirie" className={styles.logo} width="56" height="56" />
          <h1 className={styles.title}>{isSignupMode ? 'Créer un compte' : 'Connexion'}</h1>
          <p className={styles.subtitle}>
            {isSignupMode ? 'Inscrivez-vous pour démarrer rapidement vos réservations.' : 'Accédez à votre espace Lirie'}
          </p>
          {(() => {
            const next = nextFromQuery || safeReturnFromState || '';
            if (isSignupMode || !next) return null;
            if (
              next.includes('/client/payment/saferpay/return') ||
              next.includes('/client/payment/worldline/return')
            ) {
              return (
                <p className={styles.resumeHint} role="note">
                  Après connexion, vous serez renvoyé vers la page de statut de votre paiement (Saferpay).
                </p>
              );
            }
            if (
              next.includes('/client/payment/saferpay/start') ||
              next.includes('/client/payment/worldline/start')
            ) {
              return (
                <p className={styles.resumeHint} role="note">
                  Après connexion, vous pourrez poursuivre le paiement sécurisé de votre réservation
                  (Saferpay).
                </p>
              );
            }
            return null;
          })()}
          {hasSafeNext && !isSignupMode ? (
            <p className={styles.resumeHint} role="note">
              Après connexion, vous serez redirigé vers votre réservation.
            </p>
          ) : null}
        </div>

        <form className={styles.form} onSubmit={isSignupMode ? handleSignupSubmit : handleLoginSubmit} autoComplete="on" noValidate>
          {successMessage && (
            <p className={styles.successMessage} role="status">
              {successMessage}
            </p>
          )}

          {errorMessage && (
            <p className={styles.errorMessage} role="alert">
              {errorMessage}
            </p>
          )}

          {isSignupMode && SIGNUP_DISABLED ? (
            <p className={styles.resumeHint} role="note">
              Les inscriptions sont temporairement suspendues. Pour toute demande d&apos;accès, contactez
              {' '}
              <a href="mailto:info@lirie.ch">info@lirie.ch</a>.
            </p>
          ) : (
            <>
              {isSignupMode ? (
                <fieldset className={`${styles.inputGroup} ${styles.identityFieldset}`}>
                  <legend className={styles.label}>
                    Identité <span className={styles.requiredMark} aria-hidden="true">*</span>
                  </legend>
                  <div className={styles.identityRow}>
                    <div
                      ref={civilityDropdownRef}
                      className={`${styles.inputWrapper} ${styles.inputWrapperPlain} ${styles.identityCivility}`}
                    >
                      <input
                        type="text"
                        name="civility"
                        id="civility"
                        value={signupFormData.civility}
                        readOnly
                        tabIndex={-1}
                        aria-hidden="true"
                        autoComplete="honorific-prefix"
                        className={styles.civilityAutocompleteProxy}
                      />
                      <button
                        type="button"
                        className={styles.civilityTrigger}
                        aria-label="Civilité"
                        aria-haspopup="listbox"
                        aria-expanded={isCivilityOpen}
                        aria-controls="civility-listbox"
                        onClick={() => setIsCivilityOpen((prev) => !prev)}
                      >
                        <span
                          className={
                            signupFormData.civility
                              ? styles.civilityValue
                              : styles.civilityPlaceholder
                          }
                        >
                          {signupFormData.civility || 'Civilité'}
                        </span>
                        <span
                          className={`${styles.civilityChevron} ${isCivilityOpen ? styles.civilityChevronOpen : ''}`}
                          aria-hidden="true"
                        >
                          <svg viewBox="0 0 20 20" width="14" height="14" focusable="false">
                            <path d="M5.5 7.5 10 12l4.5-4.5" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
                          </svg>
                        </span>
                      </button>
                      {isCivilityOpen ? (
                        <ul
                          id="civility-listbox"
                          role="listbox"
                          aria-label="Civilité"
                          className={styles.civilityMenu}
                        >
                          {CIVILITY_OPTIONS.map((option) => (
                            <li key={option.label} role="none">
                              <button
                                type="button"
                                role="option"
                                aria-selected={signupFormData.civility === option.value}
                                className={`${styles.civilityOptionBtn} ${
                                  signupFormData.civility === option.value
                                    ? styles.civilityOptionBtnActive
                                    : ''
                                }`}
                                onClick={() => handleCivilitySelect(option.value)}
                              >
                                {option.label}
                              </button>
                            </li>
                          ))}
                        </ul>
                      ) : null}
                    </div>
                    <div className={`${styles.inputWrapper} ${styles.inputWrapperPlain}`}>
                      <input
                        type="text"
                        name="firstName"
                        id="firstName"
                        className={styles.input}
                        placeholder="Prénom *"
                        value={signupFormData.firstName}
                        onChange={handleSignupInputChange}
                        required
                        aria-required="true"
                        autoComplete="given-name"
                        autoFocus
                      />
                    </div>
                    <div className={`${styles.inputWrapper} ${styles.inputWrapperPlain}`}>
                      <input
                        type="text"
                        name="lastName"
                        id="lastName"
                        className={styles.input}
                        placeholder="NOM *"
                        value={signupFormData.lastName}
                        onChange={handleSignupInputChange}
                        required
                        aria-required="true"
                        autoComplete="family-name"
                        autoCapitalize="characters"
                        spellCheck={false}
                      />
                    </div>
                  </div>
                </fieldset>
              ) : null}

              <div className={styles.inputGroup}>
                <label htmlFor="email" className={styles.label}>
                  {isSignupMode ? (
                    <>
                      Adresse email{' '}
                      <span className={styles.optionalMark}>(ou téléphone)</span>
                    </>
                  ) : (
                    'Email ou identifiant'
                  )}
                </label>
                <div className={`${styles.inputWrapper} ${styles.inputWrapperPlain} ${styles.inputWrapper30}`}>
                  <input
                    type={isSignupMode ? 'email' : 'text'}
                    name="email"
                    id="email"
                    className={styles.input}
                    placeholder={isSignupMode ? 'nom@entreprise.ch' : 'nom@entreprise.ch ou j.drin'}
                    value={isSignupMode ? signupFormData.email : loginFormData.email}
                    onChange={isSignupMode ? handleSignupInputChange : handleLoginInputChange}
                    required={!isSignupMode}
                    aria-required={!isSignupMode}
                    autoComplete={isSignupMode ? 'email' : 'username'}
                    autoFocus={!isSignupMode}
                  />
                </div>
                {isSignupMode ? (
                  <p className={styles.fieldHint}>
                    Email ou téléphone : au moins l&apos;un des deux est obligatoire.
                  </p>
                ) : null}
              </div>

              {isSignupMode && signupFormData.email.trim() ? (
                <div className={styles.inputGroup}>
                  <div className={styles.passwordRow}>
                    <div className={styles.passwordColumn}>
                      <label htmlFor="password" className={styles.label}>
                        Mot de passe <span className={styles.requiredMark} aria-hidden="true">*</span>
                      </label>
                      <div className={`${styles.inputWrapper} ${styles.inputWrapperPlain} ${styles.inputWrapper30}`}>
                        <input
                          type={showSignupPassword ? 'text' : 'password'}
                          name="password"
                          id="password"
                          className={`${styles.input} ${styles.inputPasswordPadding}`}
                          placeholder="Créez un mot de passe"
                          value={signupFormData.password}
                          onChange={handleSignupInputChange}
                          required
                          aria-required="true"
                          autoComplete="new-password"
                        />
                        <button
                          type="button"
                          className={styles.togglePassword}
                          onClick={() => setShowSignupPassword(!showSignupPassword)}
                          aria-label={showSignupPassword ? 'Masquer le mot de passe' : 'Afficher le mot de passe'}
                        >
                          {showSignupPassword ? <EyeOffIcon /> : <EyeIcon />}
                        </button>
                      </div>
                    </div>
                    <div className={styles.passwordColumn}>
                      <label htmlFor="confirmPassword" className={styles.label}>
                        Confirmer le mot de passe <span className={styles.requiredMark} aria-hidden="true">*</span>
                      </label>
                      <div className={`${styles.inputWrapper} ${styles.inputWrapperPlain} ${styles.inputWrapper30}`}>
                        <input
                          type={showSignupPassword ? 'text' : 'password'}
                          name="confirmPassword"
                          id="confirmPassword"
                          className={styles.input}
                          placeholder="Confirmez le mot de passe"
                          value={signupFormData.confirmPassword}
                          onChange={handleSignupInputChange}
                          required
                          aria-required="true"
                          autoComplete="new-password"
                        />
                      </div>
                    </div>
                  </div>
                  <div className={styles.passwordStrengthWrap}>
                    <div className={styles.passwordStrengthBarTrack}>
                      <div
                        className={`${styles.passwordStrengthBarFill} ${strengthToneClass}`}
                        style={{ width: `${(strengthScore / 5) * 100}%` }}
                      />
                    </div>
                    <p className={styles.passwordStrengthLabel}>
                      Niveau de solidite: {strengthLabel}
                    </p>
                  </div>
                </div>
              ) : !isSignupMode ? (
                <div className={styles.inputGroup}>
                  <label htmlFor="password" className={styles.label}>Mot de passe</label>
                  <div className={`${styles.inputWrapper} ${styles.inputWrapperPlain} ${styles.inputWrapper30}`}>
                    <input
                      type={showLoginPassword ? 'text' : 'password'}
                      name="password"
                      id="password"
                      className={`${styles.input} ${styles.inputPasswordPadding}`}
                      placeholder="Entrez votre mot de passe"
                      value={loginFormData.password}
                      onChange={handleLoginInputChange}
                      required
                      autoComplete="current-password"
                    />
                    <button
                      type="button"
                      className={styles.togglePassword}
                      onClick={() => setShowLoginPassword(!showLoginPassword)}
                      aria-label={showLoginPassword ? 'Masquer le mot de passe' : 'Afficher le mot de passe'}
                    >
                      {showLoginPassword ? <EyeOffIcon /> : <EyeIcon />}
                    </button>
                  </div>
                </div>
              ) : null}

              {isSignupMode ? (
                <>
                  <div className={styles.inputGroup}>
                    <label htmlFor="phone" className={styles.label}>
                      Téléphone <span className={styles.optionalMark}>(ou email)</span>
                    </label>
                    <div className={`${styles.inputWrapper} ${styles.inputWrapperPlain} ${styles.inputWrapper30}`}>
                      <input
                        type="text"
                        name="phone"
                        id="phone"
                        className={styles.input}
                        placeholder="+41 ..."
                        value={signupFormData.phone}
                        onChange={handleSignupInputChange}
                        autoComplete="tel"
                      />
                    </div>
                  </div>
                  <div className={styles.inputGroup}>
                    <label htmlFor="address" className={styles.label}>
                      Adresse <span className={styles.optionalMark}>(optionnel)</span>
                    </label>
                    <div className={`${styles.inputWrapper} ${styles.inputWrapperPlain} ${styles.inputWrapper30}`}>
                      <AddressAutocomplete
                        inputId="address"
                        name="address"
                        inputClassName={styles.input}
                        placeholder="Votre adresse"
                        value={signupFormData.address}
                        onChange={handleSignupInputChange}
                        onSelect={(item) => {
                          setSignupFormData((prev) => ({
                            ...prev,
                            address: item?.label || item?.address || prev.address,
                          }));
                        }}
                        autoComplete="street-address"
                        autoCapitalize="words"
                        minChars={2}
                      />
                    </div>
                  </div>
                  <fieldset className={`${styles.inputGroup} ${styles.mobilityFieldset}`}>
                    <legend className={styles.label}>Besoins de mobilité (optionnel)</legend>
                    <p className={styles.mobilityHint}>
                      Indiquez seulement ce qui est utile pour vos trajets.
                    </p>
                    <div className={styles.mobilityChips}>
                      <button
                        type="button"
                        className={`${styles.mobilityChip} ${signupFormData.needsWheelchair ? styles.mobilityChipActive : ''}`}
                        aria-pressed={signupFormData.needsWheelchair}
                        onClick={() => handleSignupToggleChange('needsWheelchair')}
                      >
                        Fauteuil manuel
                      </button>
                      <button
                        type="button"
                        className={`${styles.mobilityChip} ${signupFormData.needsElectricWheelchair ? styles.mobilityChipActive : ''}`}
                        aria-pressed={signupFormData.needsElectricWheelchair}
                        onClick={() => handleSignupToggleChange('needsElectricWheelchair')}
                      >
                        Fauteuil électrique
                      </button>
                      <button
                        type="button"
                        className={`${styles.mobilityChip} ${signupFormData.needsWalkingAid ? styles.mobilityChipActive : ''}`}
                        aria-pressed={signupFormData.needsWalkingAid}
                        onClick={() => handleSignupToggleChange('needsWalkingAid')}
                      >
                        Aide à la marche
                      </button>
                      <button
                        type="button"
                        className={`${styles.mobilityChip} ${signupFormData.needsDoorToDoorAssistance ? styles.mobilityChipActive : ''}`}
                        aria-pressed={signupFormData.needsDoorToDoorAssistance}
                        onClick={() => handleSignupToggleChange('needsDoorToDoorAssistance')}
                      >
                        Porte-à-porte
                      </button>
                    </div>
                    <button
                      type="button"
                      className={styles.mobilityResetBtn}
                      onClick={handleClearMobilityNeeds}
                    >
                      Réinitialiser les besoins
                    </button>
                    <div className={styles.inputGroup}>
                      <label htmlFor="mobilityNotes" className={styles.label}>Précisions (optionnel)</label>
                      <div className={`${styles.inputWrapper} ${styles.inputWrapperPlain} ${styles.inputWrapper30}`}>
                        <input
                          type="text"
                          name="mobilityNotes"
                          id="mobilityNotes"
                          className={styles.input}
                          placeholder="Ex: étage sans ascenseur, digicode..."
                          value={signupFormData.mobilityNotes}
                          onChange={handleSignupInputChange}
                        />
                      </div>
                    </div>
                  </fieldset>
                </>
              ) : (
                <div className={styles.metaRow}>
                  <label className={styles.rememberLabel}>
                    <input
                      type="checkbox"
                      checked={rememberMe}
                      onChange={(e) => setRememberMe(e.target.checked)}
                      className={styles.rememberCheckbox}
                    />
                    <span className={styles.checkboxCustom} />
                    Se souvenir de moi
                  </label>
                  <Link to="/forgot-password" className={styles.forgotLink}>
                    Mot de passe oublié ?
                  </Link>
                </div>
              )}
            </>
          )}

          <button
            type="submit"
            className={styles.submitButton}
            disabled={isLoading || (isSignupMode && SIGNUP_DISABLED)}
          >
            {isLoading && <span className={styles.spinner} />}
            {isLoading
              ? (isSignupMode ? 'Inscription en cours...' : 'Connexion en cours...')
              : (isSignupMode ? "Créer mon compte" : 'Se connecter')}
          </button>
        </form>

        <div className={styles.footer}>
          <p className={styles.footerText}>
            {isSignupMode ? 'Déjà inscrit ? ' : 'Pas encore de compte ? '}
            <Link
              to={isSignupMode ? `/login${loginSearch}` : `/login${signupSearch}`}
              className={styles.forgotLink}
            >
              {isSignupMode ? 'Se connecter' : 'Créer un compte'}
            </Link>
          </p>
          <p className={styles.footerText}>
            Lirie — Plateforme de transport sanitaire
          </p>
        </div>
      </div>
    </div>
  );
};

export default Login;
