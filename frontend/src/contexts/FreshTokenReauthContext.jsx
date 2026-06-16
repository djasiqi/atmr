import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
} from 'react';
import { getFreshToken } from '../services/authService';
import { registerFreshTokenReauthHandler } from '../utils/apiClient';
import styles from '../pages/company/Settings/CompanySettings.module.css';

const FreshTokenReauthContext = createContext(null);

const DEFAULT_TITLE = 'Confirmation requise';

export function FreshTokenReauthProvider({ children }) {
  const [open, setOpen] = useState(false);
  const [title, setTitle] = useState(DEFAULT_TITLE);
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [submitting, setSubmitting] = useState(false);

  const pendingRef = useRef(null);
  const reauthAttemptsRef = useRef(0);
  const maxAutoRetries = 1;

  const closeModal = useCallback(() => {
    setOpen(false);
    setPassword('');
    setError('');
    setTitle(DEFAULT_TITLE);
    if (pendingRef.current?.reject) {
      pendingRef.current.reject(new Error('Re-auth annulée'));
    }
    pendingRef.current = null;
    reauthAttemptsRef.current = 0;
  }, []);

  const runRetry = useCallback(async () => {
    const pending = pendingRef.current;
    if (!pending?.retryFn) return;
    if (reauthAttemptsRef.current >= maxAutoRetries) {
      setError('Action toujours bloquée. Réessayez après vérification.');
      return;
    }
    reauthAttemptsRef.current += 1;
    try {
      await pending.retryFn();
      pending.resolve?.();
      setOpen(false);
      setPassword('');
      setError('');
      pendingRef.current = null;
      reauthAttemptsRef.current = 0;
    } catch (retryErr) {
      if (retryErr?.isFreshTokenRequired) {
        setError('Mot de passe accepté mais action encore bloquée. Réessayez.');
      } else {
        pending.reject?.(retryErr);
        closeModal();
      }
    }
  }, [closeModal]);

  const requestFreshTokenReauth = useCallback(
    ({ retryFn, title: modalTitle } = {}) =>
      new Promise((resolve, reject) => {
        pendingRef.current = { retryFn, resolve, reject };
        reauthAttemptsRef.current = 0;
        setTitle(modalTitle || DEFAULT_TITLE);
        setOpen(true);
        setError('');
        setPassword('');
      }),
    []
  );

  useEffect(() => {
    registerFreshTokenReauthHandler(requestFreshTokenReauth);
    return () => registerFreshTokenReauthHandler(null);
  }, [requestFreshTokenReauth]);

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!password.trim()) {
      setError('Veuillez entrer votre mot de passe.');
      return;
    }
    setSubmitting(true);
    setError('');
    try {
      await getFreshToken(password);
      await runRetry();
    } catch (err) {
      const msg =
        err?.response?.data?.error ||
        err?.message ||
        'Mot de passe incorrect.';
      setError(msg);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <FreshTokenReauthContext.Provider value={{ requestFreshTokenReauth }}>
      {children}
      {open ? (
        <div className={styles.modalOverlay} role="presentation">
          <div
            className={styles.modalContent}
            role="dialog"
            aria-modal="true"
            aria-labelledby="fresh-token-reauth-title"
          >
            <h2 id="fresh-token-reauth-title">{title}</h2>
            <p>
              Pour des raisons de sécurité, confirmez votre mot de passe pour continuer.
            </p>
            <form onSubmit={handleSubmit}>
              <label htmlFor="fresh-token-password">Mot de passe</label>
              <input
                id="fresh-token-password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                autoComplete="current-password"
                disabled={submitting}
              />
              {error ? <p className={styles.error}>{error}</p> : null}
              <div className={styles.modalActions}>
                <button type="button" onClick={closeModal} disabled={submitting}>
                  Annuler
                </button>
                <button type="submit" disabled={submitting}>
                  {submitting ? 'Vérification…' : 'Confirmer'}
                </button>
              </div>
            </form>
          </div>
        </div>
      ) : null}
    </FreshTokenReauthContext.Provider>
  );
}

export function useFreshTokenReauth() {
  const ctx = useContext(FreshTokenReauthContext);
  if (!ctx) {
    throw new Error('useFreshTokenReauth doit être utilisé dans FreshTokenReauthProvider');
  }
  return ctx;
}

export default FreshTokenReauthContext;
