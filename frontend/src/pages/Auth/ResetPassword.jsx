import React, { useState, useEffect, useMemo } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { jwtDecode } from 'jwt-decode';
import apiClient, { cleanLocalSession, setCurrentAuthEnv } from '../../utils/apiClient';
import { queryClient } from '../../App';
import {
  getActiveUser,
  writeAuthSession,
  normalizeAuthRole,
} from '../../utils/webAuthSession';
import styles from './Login.module.css';

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

const ResetPassword = ({ resetMode } = {}) => {
  const { token } = useParams();
  const navigate = useNavigate();
  const isForcedMode = resetMode === 'forced';
  const resetToken = isForcedMode ? null : token;

  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isForced, setIsForced] = useState(isForcedMode);

  useEffect(() => {
    const user = getActiveUser();
    if (user && user.force_password_change) {
      console.log('⚠️ Réinitialisation forcée du mot de passe requise.');
      setIsForced(true);
    }
  }, []);

  const passwordChecks = useMemo(() => {
    const password = newPassword || '';
    return {
      minLength: password.length >= 8,
      uppercase: /[A-Z]/.test(password),
      lowercase: /[a-z]/.test(password),
      digit: /\d/.test(password),
      special: /[^A-Za-z0-9]/.test(password),
    };
  }, [newPassword]);

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

  const autoLoginAfterForcedReset = async () => {
    const activeUser = getActiveUser();
    const identifier = activeUser?.email || activeUser?.username;

    // Sans identifiant connu, on ne peut pas reconnecter automatiquement :
    // repli vers le login (l'utilisateur saisira ses identifiants).
    if (!identifier) {
      setTimeout(() => navigate('/login', { replace: true }), 1200);
      return;
    }

    try {
      const loginResponse = await apiClient.post(
        '/auth/login',
        { email: identifier, password: newPassword },
        { skipCsrf: true }
      );
      const { token, user, refresh_token, target_env, redirect_to } = loginResponse.data;

      if (!user || !user.role || !user.public_id) {
        throw new Error('Aucune information utilisateur reçue.');
      }

      const authEnv = setCurrentAuthEnv(target_env);
      cleanLocalSession();
      queryClient.clear();

      let roleSegment;
      if (token && typeof token === 'string') {
        const decodedToken = jwtDecode(token);
        roleSegment = normalizeAuthRole(decodedToken.role || user.role);
      } else {
        roleSegment = normalizeAuthRole(user.role);
      }

      writeAuthSession({
        env: authEnv,
        user,
        role: roleSegment,
        accessToken: token,
        refreshToken: refresh_token,
      });
      window.dispatchEvent(new Event('auth-changed'));

      const destination = redirect_to || `/dashboard/${roleSegment}/${user.public_id}`;
      navigate(destination, { replace: true });
    } catch (loginError) {
      console.error('❌ Reconnexion automatique échouée :', loginError);
      // Le mot de passe a bien été changé : on invite à se reconnecter.
      setMessage('');
      setError(
        "Mot de passe mis à jour. Veuillez vous reconnecter avec votre nouveau mot de passe."
      );
      const activeUser2 = getActiveUser();
      const prefillEmail = activeUser2?.email || activeUser2?.username || '';
      setTimeout(
        () => navigate('/login', { replace: true, state: { prefillEmail } }),
        1800
      );
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setMessage('');
    setError('');

    if (newPassword !== confirmPassword) {
      setError('Les mots de passe ne correspondent pas.');
      return;
    }

    setIsLoading(true);

    try {
      let response;

      // ------------------------------------------------------------------
      // 1) Mode "réinitialisation par token" (lien e-mail)
      //    => URL /reset-password/:token
      // ------------------------------------------------------------------
      if (resetToken) {
        response = await apiClient.post('/auth/reset-password', {
          token: resetToken,
          new_password: newPassword,
          confirm_password: confirmPassword,
        });

        console.log('✅ Réinitialisation (par token) réussie :', response.data);
        setMessage('Votre mot de passe a été mis à jour avec succès !');
        navigate('/login');

        // ------------------------------------------------------------------
        // 2) Mode "réinitialisation forcée" (session JWT)
        //    => URL /force-reset-password
        // ------------------------------------------------------------------
      } else if (isForced || isForcedMode) {
        response = await apiClient.post('/auth/change-password', {
          new_password: newPassword,
          confirm_password: confirmPassword,
        });

        console.log('✅ Réinitialisation forcée réussie :', response.data);
        setMessage('Mot de passe mis à jour. Connexion en cours...');

        // Le changement forcé révoque toutes les sessions (token_version).
        // On se reconnecte avec le nouveau mot de passe.
        await autoLoginAfterForcedReset();
      } else {
        setError("Aucun token ni session d'authentification n'est fourni.");
        return;
      }
    } catch (err) {
      console.error('❌ Erreur lors de la réinitialisation :', err);
      setError(
        err.response?.data?.error ||
          'Une erreur est survenue lors de la réinitialisation du mot de passe.'
      );
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className={styles.pageWrapper}>
      <div className={styles.loginCard}>
        <div className={styles.header}>
          <img src="/logo-lirie.png" alt="Lirie" className={styles.logo} width="56" height="56" />
          <h1 className={styles.title}>
            {isForced ? 'Modification du mot de passe' : 'Réinitialiser le mot de passe'}
          </h1>
          <p className={styles.subtitle}>
            {isForced
              ? 'Pour sécuriser votre compte, définissez un nouveau mot de passe avant de continuer.'
              : 'Choisissez un nouveau mot de passe pour votre compte Lirie.'}
          </p>
        </div>

        <form className={styles.form} onSubmit={handleSubmit} noValidate>
          {message && (
            <p className={styles.successMessage} role="status">
              {message}
            </p>
          )}
          {error && (
            <p className={styles.errorMessage} role="alert">
              {error}
            </p>
          )}

          <div className={styles.inputGroup}>
            <label htmlFor="newPassword" className={styles.label}>
              Nouveau mot de passe
            </label>
            <div className={`${styles.inputWrapper} ${styles.inputWrapperPlain} ${styles.inputWrapper30}`}>
              <input
                type={showPassword ? 'text' : 'password'}
                id="newPassword"
                name="newPassword"
                className={`${styles.input} ${styles.inputPasswordPadding}`}
                placeholder="Créez un mot de passe"
                value={newPassword}
                onChange={(e) => setNewPassword(e.target.value)}
                required
                autoComplete="new-password"
                autoFocus
              />
              <button
                type="button"
                className={styles.togglePassword}
                onClick={() => setShowPassword((prev) => !prev)}
                aria-label={showPassword ? 'Masquer le mot de passe' : 'Afficher le mot de passe'}
              >
                {showPassword ? <EyeOffIcon /> : <EyeIcon />}
              </button>
            </div>
          </div>

          <div className={styles.inputGroup}>
            <label htmlFor="confirmPassword" className={styles.label}>
              Confirmer le mot de passe
            </label>
            <div className={`${styles.inputWrapper} ${styles.inputWrapperPlain} ${styles.inputWrapper30}`}>
              <input
                type={showPassword ? 'text' : 'password'}
                id="confirmPassword"
                name="confirmPassword"
                className={styles.input}
                placeholder="Confirmez le mot de passe"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                required
                autoComplete="new-password"
              />
            </div>
          </div>

          {newPassword ? (
            <div className={styles.passwordStrengthWrap}>
              <div className={styles.passwordStrengthBarTrack}>
                <div
                  className={`${styles.passwordStrengthBarFill} ${strengthToneClass}`}
                  style={{ width: `${(strengthScore / 5) * 100}%` }}
                />
              </div>
              <p className={styles.passwordStrengthLabel}>
                Niveau de solidité: {strengthLabel}
              </p>
            </div>
          ) : null}

          <button
            type="submit"
            className={styles.submitButton}
            disabled={isLoading}
          >
            {isLoading && <span className={styles.spinner} />}
            {isLoading ? 'Réinitialisation...' : 'Confirmer le changement'}
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

export default ResetPassword;
