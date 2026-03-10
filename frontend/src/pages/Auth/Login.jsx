import React, { useState, useEffect } from 'react';
import { useNavigate, useLocation, Link } from 'react-router-dom';
import apiClient, { cleanLocalSession, setCurrentAuthEnv } from '../../utils/apiClient';
import { jwtDecode } from 'jwt-decode';
import { queryClient } from '../../App';
import styles from './Login.module.css';

const REMEMBER_KEY = 'lirie_remember_me';

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

const MailIcon = () => (
  <svg className={styles.inputIcon} width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="2" y="4" width="20" height="16" rx="2" />
    <path d="m22 7-8.97 5.7a1.94 1.94 0 0 1-2.06 0L2 7" />
  </svg>
);

const LockIcon = () => (
  <svg className={styles.inputIcon} width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
    <path d="M7 11V7a5 5 0 0 1 10 0v4" />
  </svg>
);

const Login = () => {
  const location = useLocation();
  const justActivated = location.state?.activated === true;

  const [formData, setFormData] = useState({ email: '', password: '' });
  const [showPassword, setShowPassword] = useState(false);
  const [rememberMe, setRememberMe] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');
  const [successMessage] = useState(
    justActivated ? 'Compte activé avec succès ! Connectez-vous avec votre nouveau mot de passe.' : ''
  );
  const [isLoading, setIsLoading] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    try {
      const saved = localStorage.getItem(REMEMBER_KEY);
      if (saved) {
        const { email, password } = JSON.parse(saved);
        setFormData({ email: email || '', password: password || '' });
        setRememberMe(true);
      }
    } catch { /* ignore corrupted data */ }
  }, []);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });
    setErrorMessage('');
  };

  const validateForm = () => {
    const { email, password } = formData;

    if (!email.trim() || !password) {
      setErrorMessage('Veuillez remplir tous les champs.');
      return false;
    }

    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
      setErrorMessage('Veuillez entrer une adresse email valide.');
      return false;
    }

    if (password.length < 6) {
      setErrorMessage('Le mot de passe doit contenir au moins 6 caractères.');
      return false;
    }

    return true;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!validateForm()) return;

    setIsLoading(true);
    try {
      const response = await apiClient.post('/auth/login', formData, { skipCsrf: true });
      const { token, user, refresh_token, target_env, redirect_to } = response.data;

      if (!user || !user.role || !user.public_id) {
        throw new Error('Aucune information utilisateur reçue.');
      }

      const authEnv = setCurrentAuthEnv(target_env);

      if (rememberMe) {
        localStorage.setItem(REMEMBER_KEY, JSON.stringify({
          email: formData.email,
          password: formData.password,
        }));
      } else {
        localStorage.removeItem(REMEMBER_KEY);
      }

      cleanLocalSession();
      queryClient.clear();

      let roleSegment;
      if (token && typeof token === 'string') {
        const decodedToken = jwtDecode(token);
        roleSegment = String(decodedToken.role || user.role || '').toLowerCase();
      } else {
        roleSegment = String(user.role || '').toLowerCase();
      }

      const userPayload = JSON.stringify({ ...user, role: roleSegment });
      localStorage.setItem(`${authEnv}_user`, userPayload);
      localStorage.setItem(`${authEnv}_public_id`, user.public_id);
      if (token) localStorage.setItem(`${authEnv}_access_token`, token);
      if (refresh_token) localStorage.setItem(`${authEnv}_refresh_token`, refresh_token);

      if (roleSegment === 'company' || roleSegment === 'admin') {
        localStorage.removeItem('driver_access_token');
        localStorage.removeItem('driver_refresh_token');
        localStorage.removeItem('driver_user');
        localStorage.removeItem('driver_public_id');
        localStorage.setItem('company_user', userPayload);
        localStorage.setItem('company_public_id', user.public_id);
        if (token) localStorage.setItem('company_access_token', token);
        if (refresh_token) localStorage.setItem('company_refresh_token', refresh_token);
      } else if (roleSegment === 'driver') {
        localStorage.removeItem('company_access_token');
        localStorage.removeItem('company_refresh_token');
        localStorage.removeItem('company_user');
        localStorage.removeItem('company_public_id');
        localStorage.setItem('driver_user', userPayload);
        localStorage.setItem('driver_public_id', user.public_id);
        if (token) localStorage.setItem('driver_access_token', token);
        if (refresh_token) localStorage.setItem('driver_refresh_token', refresh_token);
      } else if (roleSegment === 'institution') {
        localStorage.removeItem('company_access_token');
        localStorage.removeItem('company_refresh_token');
        localStorage.removeItem('company_user');
        localStorage.removeItem('company_public_id');
        localStorage.removeItem('driver_access_token');
        localStorage.removeItem('driver_refresh_token');
        localStorage.removeItem('driver_user');
        localStorage.removeItem('driver_public_id');
        localStorage.setItem('institution_user', userPayload);
        localStorage.setItem('institution_public_id', user.public_id);
        if (token) localStorage.setItem('institution_access_token', token);
        if (refresh_token) localStorage.setItem('institution_refresh_token', refresh_token);
      }

      localStorage.setItem('user', userPayload);
      localStorage.setItem('public_id', user.public_id);
      if (token) localStorage.setItem('authToken', token);
      if (refresh_token) localStorage.setItem('refreshToken', refresh_token);

      window.dispatchEvent(new Event('auth-changed'));

      if (user.force_password_change) {
        navigate(`/force-reset-password/${user.public_id}`, { replace: true });
      } else {
        navigate(redirect_to || `/dashboard/${roleSegment}/${user.public_id}`, { replace: true });
      }
    } catch (error) {
      const responseData = error?.response?.data;
      const status = error?.response?.status;
      console.error('Erreur lors de la connexion :', {
        status,
        url: error?.config?.url,
        baseURL: error?.config?.baseURL,
        data: responseData,
        code: error?.code,
      });

      const backendMessage =
        responseData?.message ??
        responseData?.detail ??
        responseData?.error ??
        (typeof responseData === 'string' ? responseData : null);
      const reason = responseData?.reason ? ` (${responseData.reason})` : '';
      const targetEnv = responseData?.target_env ? ` [env=${responseData.target_env}]` : '';
      const msg = `${backendMessage ?? error.message}${reason}${targetEnv}`;
      setErrorMessage(msg);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className={styles.pageWrapper}>
      <div className={styles.loginCard}>
        <div className={styles.header}>
          <img src="/logo-lirie.png" alt="Lirie" className={styles.logo} />
          <h1 className={styles.title}>Connexion</h1>
          <p className={styles.subtitle}>Accédez à votre espace Lirie</p>
        </div>

        <form className={styles.form} onSubmit={handleSubmit} noValidate>
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

          <div className={styles.inputGroup}>
            <label htmlFor="email" className={styles.label}>Adresse email</label>
            <div className={styles.inputWrapper}>
              <MailIcon />
              <input
                type="email"
                name="email"
                id="email"
                className={styles.input}
                placeholder="nom@entreprise.ch"
                value={formData.email}
                onChange={handleInputChange}
                required
                autoComplete="email"
                autoFocus
              />
            </div>
          </div>

          <div className={styles.inputGroup}>
            <label htmlFor="password" className={styles.label}>Mot de passe</label>
            <div className={styles.inputWrapper}>
              <LockIcon />
              <input
                type={showPassword ? 'text' : 'password'}
                name="password"
                id="password"
                className={`${styles.input} ${styles.inputPasswordPadding}`}
                placeholder="Entrez votre mot de passe"
                value={formData.password}
                onChange={handleInputChange}
                required
                autoComplete="current-password"
              />
              <button
                type="button"
                className={styles.togglePassword}
                onClick={() => setShowPassword(!showPassword)}
                tabIndex={-1}
                aria-label={showPassword ? 'Masquer le mot de passe' : 'Afficher le mot de passe'}
              >
                {showPassword ? <EyeOffIcon /> : <EyeIcon />}
              </button>
            </div>
          </div>

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

          <button
            type="submit"
            className={styles.submitButton}
            disabled={isLoading}
          >
            {isLoading && <span className={styles.spinner} />}
            {isLoading ? 'Connexion en cours...' : 'Se connecter'}
          </button>
        </form>

        <div className={styles.footer}>
          <p className={styles.footerText}>
            Lirie — Plateforme de transport sanitaire
          </p>
        </div>
      </div>
    </div>
  );
};

export default Login;
