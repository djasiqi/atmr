import React, { useEffect, useMemo, useState } from 'react';
import { useNavigate, useSearchParams, useLocation } from 'react-router-dom';
import apiClient from '../../utils/apiClient';
import {
  removePendingActivationByEmail,
  removePendingActivationBySessionId,
  setPendingActivationSession,
} from '../../utils/activationSessionStore';
import styles from './SignupActivation.module.css';

const DEFAULT_COOLDOWN_SECONDS = 60;

const parseApiMessage = (error, fallbackMessage) => {
  const data = error?.response?.data;
  return (
    data?.message ||
    data?.error ||
    data?.detail ||
    (typeof data === 'string' ? data : null) ||
    fallbackMessage
  );
};

const SignupActivation = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const [searchParams] = useSearchParams();

  const token = String(searchParams.get('token') || '').trim();
  const sessionFromQuery = String(searchParams.get('activation_session_id') || '').trim();

  const [activationSessionId, setActivationSessionId] = useState(sessionFromQuery);
  const [prefillEmail] = useState(location.state?.prefillEmail || '');
  const [maskedEmail, setMaskedEmail] = useState(location.state?.maskedEmail || '');
  const [maskedPhone, setMaskedPhone] = useState(location.state?.maskedPhone || '');
  const [smsCode, setSmsCode] = useState('');
  const [showPhoneEdit, setShowPhoneEdit] = useState(false);
  const [newPhone, setNewPhone] = useState('');
  const [status, setStatus] = useState({
    email_verified: false,
    phone_verified: false,
    is_complete: false,
    is_finalized: false,
  });
  const [loading, setLoading] = useState(false);
  const [infoMessage, setInfoMessage] = useState('');
  const [errorMessage, setErrorMessage] = useState('');
  const [debugActivationLink, setDebugActivationLink] = useState('');
  const [debugSmsCode, setDebugSmsCode] = useState('');
  const [emailCooldown, setEmailCooldown] = useState(0);
  const [smsCooldown, setSmsCooldown] = useState(0);

  useEffect(() => {
    if (!emailCooldown && !smsCooldown) return undefined;
    const timer = setInterval(() => {
      setEmailCooldown((prev) => (prev > 0 ? prev - 1 : 0));
      setSmsCooldown((prev) => (prev > 0 ? prev - 1 : 0));
    }, 1000);
    return () => clearInterval(timer);
  }, [emailCooldown, smsCooldown]);

  const updateStatus = (newStatus) => {
    if (!newStatus || typeof newStatus !== 'object') return;
    setStatus({
      email_verified: Boolean(newStatus.email_verified),
      phone_verified: Boolean(newStatus.phone_verified),
      is_complete: Boolean(newStatus.is_complete),
      is_finalized: Boolean(newStatus.is_finalized),
    });
  };

  const fetchStatus = async (sessionId) => {
    if (!sessionId) return;
    const response = await apiClient.get(
      `/auth/activation/status?activation_session_id=${encodeURIComponent(sessionId)}`
    );
    if (response?.data?.masked_email) {
      setMaskedEmail(response.data.masked_email);
    }
    if (response?.data?.masked_phone) {
      setMaskedPhone(response.data.masked_phone);
    }
    updateStatus(response?.data?.activation_status);
  };

  useEffect(() => {
    const bootWithEmailToken = async () => {
      if (!token) return;
      setErrorMessage('');
      setInfoMessage('');
      setDebugActivationLink('');
      setDebugSmsCode('');
      setLoading(true);
      try {
        const verifyResponse = await apiClient.post('/auth/activation/verify-email', { token });
        const sessionId = verifyResponse?.data?.activation_session_id;
        if (verifyResponse?.data?.masked_email) {
          setMaskedEmail(verifyResponse.data.masked_email);
        }
        if (verifyResponse?.data?.masked_phone) {
          setMaskedPhone(verifyResponse.data.masked_phone);
        }
        if (sessionId) {
          setActivationSessionId(sessionId);
          await fetchStatus(sessionId);
          setInfoMessage('Email confirmé avec succès.');
        } else {
          setErrorMessage("Email confirmé, mais session d'activation introuvable.");
        }
      } catch (error) {
        setErrorMessage(parseApiMessage(error, "Impossible de charger la session d'activation."));
      } finally {
        setLoading(false);
      }
    };
    bootWithEmailToken();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token]);

  useEffect(() => {
    const bootWithSessionId = async () => {
      if (token) return;
      setErrorMessage('');
      setInfoMessage('');
      setDebugActivationLink('');
      setDebugSmsCode('');
      if (!activationSessionId) {
        setErrorMessage("Session d'activation introuvable. Recommencez l'inscription.");
        return;
      }
      setLoading(true);
      try {
        await fetchStatus(activationSessionId);
      } catch (error) {
        setErrorMessage(parseApiMessage(error, "Impossible de charger la session d'activation."));
      } finally {
        setLoading(false);
      }
    };
    bootWithSessionId();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activationSessionId, token]);

  const markCooldownFromError = (error, setCooldown) => {
    const retryAfter = Number(error?.response?.data?.details?.retry_after_seconds || 0);
    if (retryAfter > 0) {
      setCooldown(retryAfter);
    }
  };

  const handleResendEmail = async () => {
    if (!activationSessionId) return;
    setErrorMessage('');
    setInfoMessage('');
    setDebugActivationLink('');
    setDebugSmsCode('');
    setLoading(true);
    try {
      const response = await apiClient.post('/auth/activation/resend-email', {
        activation_session_id: activationSessionId,
      });
      setEmailCooldown(DEFAULT_COOLDOWN_SECONDS);
      const fallbackLink = String(response?.data?.debug_activation_link || '').trim();
      if (fallbackLink) {
        setInfoMessage(
          "Service email indisponible en local. Utilisez le lien d'activation ci-dessous."
        );
        setDebugActivationLink(fallbackLink);
      } else {
        setInfoMessage('Email renvoyé. Vérifiez votre boîte de réception.');
      }
    } catch (error) {
      markCooldownFromError(error, setEmailCooldown);
      setErrorMessage(parseApiMessage(error, "Impossible d'envoyer l'email."));
    } finally {
      setLoading(false);
    }
  };

  const handleResendSms = async () => {
    if (!activationSessionId) return;
    setErrorMessage('');
    setInfoMessage('');
    setDebugActivationLink('');
    setDebugSmsCode('');
    setLoading(true);
    try {
      const response = await apiClient.post('/auth/activation/resend-sms', {
        activation_session_id: activationSessionId,
      });
      setSmsCooldown(DEFAULT_COOLDOWN_SECONDS);
      const fallbackCode = String(response?.data?.debug_sms_code || '').trim();
      if (fallbackCode) {
        setInfoMessage(
          'Service SMS indisponible en local. Utilisez le code de secours ci-dessous.'
        );
        setDebugSmsCode(fallbackCode);
      } else {
        setInfoMessage('Code SMS renvoyé.');
      }
    } catch (error) {
      markCooldownFromError(error, setSmsCooldown);
      setErrorMessage(parseApiMessage(error, "Impossible d'envoyer le SMS."));
    } finally {
      setLoading(false);
    }
  };

  const handleVerifySms = async () => {
    if (!activationSessionId) return;
    const code = smsCode.trim();
    if (!/^\d{6}$/.test(code)) {
      setErrorMessage('Entrez un code SMS à 6 chiffres.');
      return;
    }
    setErrorMessage('');
    setInfoMessage('');
    setDebugActivationLink('');
    setDebugSmsCode('');
    setLoading(true);
    try {
      const response = await apiClient.post('/auth/activation/verify-sms', {
        activation_session_id: activationSessionId,
        code,
      });
      updateStatus(response?.data?.activation_status);
      setInfoMessage('Téléphone confirmé.');
      setSmsCode('');
    } catch (error) {
      setErrorMessage(parseApiMessage(error, 'Code SMS invalide ou expiré.'));
    } finally {
      setLoading(false);
    }
  };

  const handleUpdatePhone = async () => {
    if (!activationSessionId) return;
    const phone = newPhone.trim();
    if (phone.length < 7) {
      setErrorMessage('Entrez un numéro de téléphone valide.');
      return;
    }
    setErrorMessage('');
    setInfoMessage('');
    setDebugActivationLink('');
    setLoading(true);
    try {
      const response = await apiClient.post('/auth/activation/update-phone', {
        activation_session_id: activationSessionId,
        phone,
      });
      if (response?.data?.masked_phone) {
        setMaskedPhone(response.data.masked_phone);
      }
      const fallbackCode = String(response?.data?.debug_sms_code || '').trim();
      updateStatus(response?.data?.activation_status);
      setShowPhoneEdit(false);
      setNewPhone('');
      setSmsCooldown(DEFAULT_COOLDOWN_SECONDS);
      if (fallbackCode) {
        setInfoMessage(
          'Numéro mis à jour. Service SMS indisponible en local, utilisez le code de secours ci-dessous.'
        );
        setDebugSmsCode(fallbackCode);
      } else {
        setInfoMessage(response?.data?.message || 'Numéro mis à jour.');
      }
    } catch (error) {
      markCooldownFromError(error, setSmsCooldown);
      setErrorMessage(parseApiMessage(error, 'Impossible de mettre à jour le numéro.'));
    } finally {
      setLoading(false);
    }
  };

  const handleFinalize = async () => {
    if (!activationSessionId) return;
    setErrorMessage('');
    setInfoMessage('');
    setLoading(true);
    try {
      const response = await apiClient.post('/auth/activation/finalize', {
        activation_session_id: activationSessionId,
      });
      updateStatus(response?.data?.activation_status);
      setInfoMessage('Compte activé. Vous pouvez maintenant vous connecter.');
      removePendingActivationBySessionId(activationSessionId);
      if (prefillEmail) {
        removePendingActivationByEmail(prefillEmail);
      }
    } catch (error) {
      setErrorMessage(parseApiMessage(error, 'Activation finale impossible.'));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (!activationSessionId) return;
    setPendingActivationSession({
      email: prefillEmail,
      activation_session_id: activationSessionId,
      masked_email: maskedEmail || null,
      masked_phone: maskedPhone || null,
    });
  }, [activationSessionId, maskedEmail, maskedPhone, prefillEmail]);

  const finalizeEnabled = useMemo(
    () => Boolean(status.email_verified && status.phone_verified && !status.is_finalized),
    [status]
  );

  const handleLoginRedirect = () => {
    navigate('/login', {
      replace: true,
      state: { activated: true, prefillEmail },
    });
  };

  if (status.is_finalized) {
    return (
      <div className={styles.pageWrapper}>
        <div className={styles.card}>
          <h1 className={styles.title}>Activation terminee</h1>
          <p className={styles.subtitle}>
            Votre compte est actif. Connectez-vous pour continuer.
          </p>
          <p className={`${styles.statusBox} ${styles.statusSuccess}`}>
            Compte active avec succes.
          </p>
          <div className={styles.footerActions}>
            <button
              type="button"
              className={`${styles.button} ${styles.buttonPrimary}`}
              onClick={handleLoginRedirect}
              disabled={loading}
            >
              Se connecter
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={styles.pageWrapper}>
      <div className={styles.card}>
        <h1 className={styles.title}>Activation du compte</h1>
        <p className={styles.subtitle}>
          Validez votre email et votre téléphone pour activer le compte.
        </p>

        {infoMessage ? <p className={`${styles.statusBox} ${styles.statusInfo}`}>{infoMessage}</p> : null}
        {debugActivationLink ? (
          <p className={`${styles.statusBox} ${styles.statusInfo}`}>
            Lien de secours:{' '}
            <a href={debugActivationLink} className={styles.inlineLink}>
              Ouvrir le lien d'activation
            </a>
          </p>
        ) : null}
        {debugSmsCode ? (
          <p className={`${styles.statusBox} ${styles.statusInfo}`}>
            Code SMS de secours: <strong>{debugSmsCode}</strong>
          </p>
        ) : null}
        {errorMessage ? <p className={`${styles.statusBox} ${styles.statusError}`}>{errorMessage}</p> : null}
        <section className={styles.section}>
          <h2 className={styles.sectionTitle}>Email</h2>
          <p className={styles.sectionHint}>
            Un lien a été envoyé à {maskedEmail || prefillEmail || 'votre adresse email'}.
          </p>
          <span className={`${styles.badge} ${status.email_verified ? styles.badgeDone : styles.badgePending}`}>
            {status.email_verified ? 'Email confirmé' : 'En attente de confirmation'}
          </span>
          <div className={styles.row}>
            <button
              type="button"
              className={`${styles.button} ${styles.buttonSecondary}`}
              onClick={handleResendEmail}
              disabled={loading || status.email_verified || emailCooldown > 0 || !activationSessionId}
            >
              {emailCooldown > 0 ? `Renvoyer (${emailCooldown}s)` : "Renvoyer l'email"}
            </button>
          </div>
        </section>

        <section className={styles.section}>
          <h2 className={styles.sectionTitle}>SMS</h2>
          <p className={styles.sectionHint}>
            Saisissez le code reçu sur {maskedPhone || 'votre téléphone'}.
          </p>
          <button
            type="button"
            className={styles.linkButton}
            onClick={() => {
              setShowPhoneEdit((prev) => !prev);
              setErrorMessage('');
              setInfoMessage('');
            }}
            disabled={loading || status.phone_verified}
          >
            {showPhoneEdit ? 'Annuler' : 'Mauvais numéro ?'}
          </button>
          {showPhoneEdit ? (
            <div className={styles.row}>
              <input
                type="text"
                inputMode="tel"
                className={styles.input}
                placeholder="Nouveau numéro (+41...)"
                value={newPhone}
                onChange={(e) => setNewPhone(e.target.value)}
                disabled={loading || status.phone_verified}
              />
              <button
                type="button"
                className={`${styles.button} ${styles.buttonSecondary}`}
                onClick={handleUpdatePhone}
                disabled={loading || status.phone_verified || !activationSessionId}
              >
                Mettre à jour
              </button>
            </div>
          ) : null}
          <span className={`${styles.badge} ${status.phone_verified ? styles.badgeDone : styles.badgePending}`}>
            {status.phone_verified ? 'Téléphone confirmé' : 'En attente de confirmation'}
          </span>
          <div className={styles.row}>
            <input
              type="text"
              inputMode="numeric"
              maxLength={6}
              className={styles.input}
              placeholder="Code à 6 chiffres"
              value={smsCode}
              onChange={(e) => setSmsCode(e.target.value.replace(/[^\d]/g, ''))}
              disabled={loading || status.phone_verified}
            />
            <button
              type="button"
              className={`${styles.button} ${styles.buttonPrimary}`}
              onClick={handleVerifySms}
              disabled={loading || status.phone_verified || !activationSessionId}
            >
              Valider le code
            </button>
          </div>
          <div className={styles.row}>
            <button
              type="button"
              className={`${styles.button} ${styles.buttonSecondary}`}
              onClick={handleResendSms}
              disabled={loading || status.phone_verified || smsCooldown > 0 || !activationSessionId}
            >
              {smsCooldown > 0 ? `Renvoyer (${smsCooldown}s)` : 'Renvoyer le code'}
            </button>
          </div>
        </section>

        <div className={styles.footerActions}>
          <button
            type="button"
            className={`${styles.button} ${styles.buttonPrimary}`}
            onClick={handleFinalize}
            disabled={!finalizeEnabled || loading}
          >
            Activer mon compte
          </button>
        </div>

        {loading ? <p className={styles.subtitle}>Traitement en cours...</p> : null}
      </div>
    </div>
  );
};

export default SignupActivation;
