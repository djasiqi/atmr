import React, { useEffect, useRef, useState } from 'react';
import { Link, useNavigate, useSearchParams } from 'react-router-dom';
import { consumeDemoMagicLink } from '../../services/demoAccessService';
import { setCurrentAuthEnv } from '../../utils/apiClient';
import styles from './DemoAccessConsume.module.css';

const buildTokenStateKey = (token) => `demo_magic_link_state:${token}`;
const DEMO_SESSION_KEYS_TO_CLEAR = [
  'demo_institution_resume_step',
  'demo_institution_request_simulation_state',
  'demo_institution_journey_completed',
  'demo_dispatch_mini',
  'demo_invoices_mini',
];

const normalizeDemoRole = (rawRole) => {
  const role = String(rawRole || '').trim().toLowerCase();
  if (!role) return '';
  if (role.startsWith('institution')) return 'institution';
  if (role.startsWith('company') || role.startsWith('transport_company')) return 'company';
  return role;
};

const toDemoScopedPath = (target) => {
  const value = String(target || '').trim();
  if (!value) return '/demo/home';
  if (value.startsWith('/demo/')) return value;
  if (value.startsWith('/dashboard/')) return `/demo${value}`;
  return value;
};

const DemoAccessConsume = () => {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const [state, setState] = useState({ loading: true, message: '' });
  const [progress, setProgress] = useState(10);
  const pollTimerRef = useRef(null);
  const redirectTimerRef = useRef(null);

  useEffect(() => {
    const token = (searchParams.get('token') || '').trim();
    if (!token) {
      setState({
        loading: false,
        message: "Lien invalide: token manquant.",
      });
      return;
    }

    const storageKey = buildTokenStateKey(token);
    const resetDemoSessionState = () => {
      try {
        DEMO_SESSION_KEYS_TO_CLEAR.forEach((key) => sessionStorage.removeItem(key));
        // Nettoyage de tous les marqueurs de consume token afin de repartir a zero depuis le lien.
        const keysToDelete = [];
        for (let i = 0; i < sessionStorage.length; i += 1) {
          const key = sessionStorage.key(i);
          if (key && key.startsWith('demo_magic_link_state:')) {
            keysToDelete.push(key);
          }
        }
        keysToDelete.forEach((key) => sessionStorage.removeItem(key));
      } catch (_) {
        // no-op
      }
    };
    const readStored = () => {
      try {
        const raw = sessionStorage.getItem(storageKey);
        return raw ? JSON.parse(raw) : null;
      } catch (_) {
        return null;
      }
    };
    const writeStored = (value) => {
      try {
        sessionStorage.setItem(storageKey, JSON.stringify(value));
      } catch (_) {
        // no-op
      }
    };

    // Le lien magique doit toujours redemarrer une session demo "propre".
    resetDemoSessionState();

    const clearPollTimer = () => {
      if (pollTimerRef.current) {
        window.clearInterval(pollTimerRef.current);
        pollTimerRef.current = null;
      }
    };
    const scheduleRedirect = (redirectTo) => {
      const finalRedirectTo = toDemoScopedPath(redirectTo);
      setProgress(100);
      setState({
        loading: true,
        message: 'Validation terminee. Redirection en cours...',
      });
      if (redirectTimerRef.current) {
        window.clearTimeout(redirectTimerRef.current);
      }
      redirectTimerRef.current = window.setTimeout(() => {
        navigate(finalRedirectTo, { replace: true });
      }, 1000);
    };

    const run = async () => {
      const current = readStored();
      if (current?.status === 'pending') {
        setState({
          loading: true,
          message: 'Validation de votre compte demo en cours...',
        });
        pollTimerRef.current = window.setInterval(() => {
          const next = readStored();
          if (next?.status === 'success' && next?.redirectTo) {
            clearPollTimer();
            scheduleRedirect(next.redirectTo);
          } else if (next?.status === 'error') {
            clearPollTimer();
            setState({
              loading: false,
              message: next.message || "Le lien demo est invalide, deja utilise ou expire.",
            });
          }
        }, 200);
        return;
      }

      writeStored({ status: 'pending', startedAt: Date.now() });
      try {
        const result = await consumeDemoMagicLink(token);
        const nextEnv = setCurrentAuthEnv(result?.target_env || 'demo');
        const rawUser = result?.user || null;
        const role = normalizeDemoRole(rawUser?.role);
        const user = rawUser ? { ...rawUser, role } : null;
        const recommendedJourney =
          String(result?.recommended_journey || '').trim().toLowerCase() || 'generic';
        const accessToken = result?.token || result?.access_token || null;
        const refreshToken = result?.refresh_token || null;
        if (user) {
          const userPayload = JSON.stringify(user);
          localStorage.setItem(`${nextEnv}_user`, userPayload);
          localStorage.setItem(`${nextEnv}_public_id`, user.public_id || '');
          localStorage.setItem('user', userPayload);
          localStorage.setItem('public_id', user.public_id || '');

          // Sync role-specific storages so ProtectedRoute does not bounce to /login.
          if (role === 'company' || role === 'admin') {
            localStorage.setItem('company_user', userPayload);
            localStorage.setItem('company_public_id', user.public_id || '');
          } else if (role === 'institution') {
            localStorage.setItem('institution_user', userPayload);
            localStorage.setItem('institution_public_id', user.public_id || '');
          }
        }
        if (accessToken) {
          localStorage.setItem(`${nextEnv}_access_token`, accessToken);
          localStorage.setItem('authToken', accessToken);
          if (role === 'company' || role === 'admin') {
            localStorage.setItem('company_access_token', accessToken);
          } else if (role === 'institution') {
            localStorage.setItem('institution_access_token', accessToken);
          }
        }
        if (refreshToken) {
          localStorage.setItem(`${nextEnv}_refresh_token`, refreshToken);
          localStorage.setItem('refreshToken', refreshToken);
          if (role === 'company' || role === 'admin') {
            localStorage.setItem('company_refresh_token', refreshToken);
          } else if (role === 'institution') {
            localStorage.setItem('institution_refresh_token', refreshToken);
          }
        }
        localStorage.setItem(`${nextEnv}_demo_recommended_journey`, recommendedJourney);
        localStorage.setItem('demo_recommended_journey', recommendedJourney);
        window.dispatchEvent(new Event('auth-changed'));
        const redirectTo =
          String(result?.redirect_to || '/demo/home').trim() || '/demo/home';
        writeStored({ status: 'success', redirectTo, finishedAt: Date.now() });
        setState({
          loading: true,
          message: 'Validation terminee. Redirection en cours...',
        });
        scheduleRedirect(redirectTo);
      } catch (error) {
        const detail =
          error?.response?.data?.message ||
          error?.response?.data?.error ||
          "Le lien demo est invalide, deja utilise ou expire.";
        writeStored({ status: 'error', message: detail, finishedAt: Date.now() });
        setState({ loading: false, message: detail });
      }
    };

    run();
    return () => {
      if (pollTimerRef.current) {
        window.clearInterval(pollTimerRef.current);
      }
      if (redirectTimerRef.current) {
        window.clearTimeout(redirectTimerRef.current);
      }
    };
  }, [navigate, searchParams]);

  useEffect(() => {
    if (!state.loading) return undefined;
    const progressTimer = window.setInterval(() => {
      setProgress((prev) => {
        // Progression psychologique fluide jusqu'a 92% tant que le backend n'a pas fini.
        if (prev >= 92) return prev;
        const step = prev < 45 ? 2.4 : prev < 72 ? 1.3 : 0.65;
        return Math.min(92, Number((prev + step).toFixed(2)));
      });
    }, 180);
    return () => window.clearInterval(progressTimer);
  }, [state.loading]);

  return (
    <article className={styles.page}>
      <section className={styles.card}>
        <h1>Acces demo</h1>
        {state.loading ? (
          <>
            <p>{state.message || 'Validation de votre compte demo en cours...'}</p>
            <div className={styles.progressTrack} aria-hidden="true">
              <div className={styles.progressBar} style={{ width: `${progress}%` }} />
            </div>
          </>
        ) : (
          <p>{state.message}</p>
        )}
        {!state.loading && (
          <p className={styles.actions}>
            <Link to="/contact/demo">Revenir a la page de demonstration</Link>
          </p>
        )}
      </section>
    </article>
  );
};

export default DemoAccessConsume;

