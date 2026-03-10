import React, { useEffect, useMemo, useState } from 'react';
import { FiEye, FiEyeOff } from 'react-icons/fi';
import { useNavigate } from 'react-router-dom';
import useAuthToken from '../../hooks/useAuthToken';
import { trackDemoEvent } from '../../services/demoAnalyticsService';
import { setDemoPassword } from '../../services/demoAccessService';
import { setCurrentAuthEnv } from '../../utils/apiClient';
import styles from './DemoHome.module.css';

const normalizeDemoRole = (rawRole) => {
  const role = String(rawRole || '').trim().toLowerCase();
  if (!role) return '';
  if (role.startsWith('institution')) return 'institution';
  if (role.startsWith('company') || role.startsWith('transport_company')) return 'company';
  return role;
};

const DemoHome = () => {
  const user = useAuthToken();
  const navigate = useNavigate();
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [savingPassword, setSavingPassword] = useState(false);
  const [passwordError, setPasswordError] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [passwordSetupDone, setPasswordSetupDone] = useState(false);
  const passwordMismatch = confirmPassword.length > 0 && password !== confirmPassword;

  useEffect(() => {
    trackDemoEvent('demo_session_start', {
      role: String(user?.role || '').toLowerCase() || 'unknown',
    });
  }, [user?.role]);

  const role = normalizeDemoRole(user?.role);
  const publicId = user?.public_id;
  const mustSetPassword = Boolean(user?.force_password_change) && !passwordSetupDone;
  const recommendedJourney = useMemo(() => {
    const env = localStorage.getItem('lirie_auth_env') === 'demo' ? 'demo' : 'app';
    return String(
      localStorage.getItem(`${env}_demo_recommended_journey`) ||
        localStorage.getItem('demo_recommended_journey') ||
        'generic'
    )
      .trim()
      .toLowerCase();
  }, []);
  const showTransportCard = recommendedJourney !== 'institution';
  const showInstitutionCard = recommendedJourney !== 'transport';

  const navigateInDemo = (targetPath) => {
    // Verrouille l'environnement demo avant de basculer vers les écrans dashboard.
    setCurrentAuthEnv('demo');
    window.dispatchEvent(new Event('auth-changed'));
    navigate(targetPath);
  };

  const goTransporteur = () => {
    if (!publicId) return;
    navigateInDemo(`/demo/dashboard/company/${publicId}?demo_mission=transporteur`);
  };

  const goInstitution = () => {
    if (!publicId) return;
    navigateInDemo(`/demo/dashboard/institution/${publicId}?demo_mission=institution`);
  };

  const goFree = () => {
    if (!publicId) return;
    if (role === 'institution') {
      navigateInDemo(`/demo/dashboard/institution/${publicId}`);
      return;
    }
    navigateInDemo(`/demo/dashboard/company/${publicId}`);
  };

  const onSubmitPassword = async (event) => {
    event.preventDefault();
    setPasswordError('');

    if (password.length < 8) {
      setPasswordError('Le mot de passe doit contenir au moins 8 caracteres.');
      return;
    }
    if (password !== confirmPassword) {
      setPasswordError('Les mots de passe ne correspondent pas.');
      return;
    }

    try {
      setSavingPassword(true);
      const result = await setDemoPassword(password);
      const env = setCurrentAuthEnv(result?.target_env || 'demo');
      const rawUser = result?.user || null;
      const role = normalizeDemoRole(rawUser?.role);
      const userPayload = rawUser
        ? JSON.stringify({ ...rawUser, role, force_password_change: false })
        : null;
      const accessToken = result?.token || result?.access_token || null;
      const refreshToken = result?.refresh_token || null;

      if (userPayload) {
        localStorage.setItem(`${env}_user`, userPayload);
        localStorage.setItem('user', userPayload);
        localStorage.setItem(`${env}_public_id`, rawUser?.public_id || '');
        localStorage.setItem('public_id', rawUser?.public_id || '');
        if (role === 'company' || role === 'admin') {
          localStorage.setItem('company_user', userPayload);
          localStorage.setItem('company_public_id', rawUser?.public_id || '');
        } else if (role === 'institution') {
          localStorage.setItem('institution_user', userPayload);
          localStorage.setItem('institution_public_id', rawUser?.public_id || '');
        }
      }
      if (accessToken) {
        localStorage.setItem(`${env}_access_token`, accessToken);
        localStorage.setItem('authToken', accessToken);
        if (role === 'company' || role === 'admin') {
          localStorage.setItem('company_access_token', accessToken);
        } else if (role === 'institution') {
          localStorage.setItem('institution_access_token', accessToken);
        }
      }
      if (refreshToken) {
        localStorage.setItem(`${env}_refresh_token`, refreshToken);
        localStorage.setItem('refreshToken', refreshToken);
        if (role === 'company' || role === 'admin') {
          localStorage.setItem('company_refresh_token', refreshToken);
        } else if (role === 'institution') {
          localStorage.setItem('institution_refresh_token', refreshToken);
        }
      }
      window.dispatchEvent(new Event('auth-changed'));
      setPasswordSetupDone(true);
      navigate('/demo/home', { replace: true });
    } catch (error) {
      setPasswordError(
        error?.response?.data?.message ||
          "Impossible d'enregistrer le mot de passe pour le moment."
      );
    } finally {
      setSavingPassword(false);
    }
  };

  return (
    <div className={styles.page} data-tour-id="demo-home">
      <div className={styles.inner}>
        {mustSetPassword && (
          <div className={styles.modalBackdrop}>
            <div className={styles.modalCard}>
              <h2>Definir votre mot de passe demo</h2>
              <p>
                Avant de commencer la demonstration, choisissez un mot de passe personnel pour ce
                compte.
              </p>
              <form onSubmit={onSubmitPassword} className={styles.modalForm}>
                <label htmlFor="demo-password">Nouveau mot de passe</label>
                <div className={styles.passwordField}>
                  <input
                    id="demo-password"
                    type={showPassword ? 'text' : 'password'}
                    value={password}
                    onChange={(e) => {
                      setPassword(e.target.value);
                      setPasswordError('');
                    }}
                    minLength={8}
                    required
                  />
                  <button
                    type="button"
                    className={styles.passwordToggle}
                    onClick={() => setShowPassword((prev) => !prev)}
                    aria-label={showPassword ? 'Masquer le mot de passe' : 'Afficher le mot de passe'}
                    title={showPassword ? 'Masquer le mot de passe' : 'Afficher le mot de passe'}
                  >
                    {showPassword ? <FiEyeOff aria-hidden="true" /> : <FiEye aria-hidden="true" />}
                  </button>
                </div>
                <label htmlFor="demo-password-confirm">Confirmer le mot de passe</label>
                <div className={styles.passwordField}>
                  <input
                    id="demo-password-confirm"
                    type={showConfirmPassword ? 'text' : 'password'}
                    value={confirmPassword}
                    onChange={(e) => {
                      setConfirmPassword(e.target.value);
                      setPasswordError('');
                    }}
                    minLength={8}
                    required
                  />
                  <button
                    type="button"
                    className={styles.passwordToggle}
                    onClick={() => setShowConfirmPassword((prev) => !prev)}
                    aria-label={
                      showConfirmPassword ? 'Masquer la confirmation' : 'Afficher la confirmation'
                    }
                    title={
                      showConfirmPassword ? 'Masquer la confirmation' : 'Afficher la confirmation'
                    }
                  >
                    {showConfirmPassword ? (
                      <FiEyeOff aria-hidden="true" />
                    ) : (
                      <FiEye aria-hidden="true" />
                    )}
                  </button>
                </div>
                {passwordMismatch ? (
                  <p className={styles.modalError}>Les mots de passe ne correspondent pas.</p>
                ) : passwordError ? (
                  <p className={styles.modalError}>{passwordError}</p>
                ) : null}
                <button
                  type="submit"
                  className={styles.ctaPrimary}
                  disabled={savingPassword || passwordMismatch}
                >
                  {savingPassword ? 'Enregistrement...' : 'Demarrer la demo'}
                </button>
              </form>
            </div>
          </div>
        )}
        <div className={styles.contentShell}>
          <span className={styles.tag}>Démo LIRIE</span>
          <h1 className={styles.title}>Bienvenue dans la démonstration commerciale</h1>
          <p className={styles.subtitle}>
            Sélectionnez un parcours guidé pour comprendre la valeur métier en moins de 10 minutes.
          </p>

          <div className={styles.grid}>
            {showTransportCard && (
              <article className={styles.card}>
                <h3>Découvrir pour un transporteur</h3>
                <p>Dispatch, suivi opérationnel et facturation.</p>
                <button type="button" className={styles.ctaPrimary} onClick={goTransporteur}>
                  Commencer ce parcours
                </button>
              </article>
            )}

            {showInstitutionCard && (
              <article className={styles.card}>
                <h3>Découvrir pour une institution</h3>
                <p>Créer une demande, suivre son statut, puis consulter l’historique.</p>
                <button type="button" className={styles.ctaSecondary} onClick={goInstitution}>
                  Commencer ce parcours
                </button>
              </article>
            )}

            <article className={styles.card}>
              <h3>Explorer librement</h3>
              <p>Accès direct à la plateforme sans étapes de démonstration guidées.</p>
              <button type="button" className={styles.ctaSecondary} onClick={goFree}>
                Explorer
              </button>
            </article>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DemoHome;

