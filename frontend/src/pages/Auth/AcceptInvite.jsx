/**
 * Page publique d'activation de compte via invitation.
 *
 * Route: /invite/:token
 *
 * Flow:
 * 1. GET /api/v1/auth/invite/:token → vérifie validité du token
 * 2. L'utilisateur définit son mot de passe
 * 3. POST /api/v1/auth/activate-account → active le compte
 * 4. Redirection vers /login
 */

import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { apiClient } from '../../utils/apiClient';

const AcceptInvite = () => {
  const { token } = useParams();
  const navigate = useNavigate();

  const [loading, setLoading] = useState(true);
  const [inviteData, setInviteData] = useState(null);
  const [error, setError] = useState(null);
  const [errorCode, setErrorCode] = useState(null);

  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [activating, setActivating] = useState(false);
  const [success, setSuccess] = useState(false);
  const [formError, setFormError] = useState('');

  // Verify token on mount
  useEffect(() => {
    const verifyToken = async () => {
      try {
        const res = await apiClient.get(`/auth/invite/${token}`);
        setInviteData(res.data);
        setLoading(false);
      } catch (err) {
        const status = err.response?.status;
        const data = err.response?.data;
        const code = data?.code || (status === 404 ? 'invalid_token' : 'unknown');

        // Messages user-friendly par code d'erreur
        const ERROR_MESSAGES = {
          invalid_token: 'Ce lien d\'invitation est invalide ou a expiré.',
          expired: 'Ce lien d\'invitation a expiré. Demandez à votre administrateur d\'en envoyer un nouveau.',
          already_activated: 'Ce compte a déjà été activé.',
        };

        const msg = ERROR_MESSAGES[code]
          || data?.error
          || 'Ce lien d\'invitation est invalide ou a expiré.';

        setError(msg);
        setErrorCode(code);
        setLoading(false);
      }
    };

    if (token) {
      verifyToken();
    } else {
      setError('Aucun token d\'invitation fourni');
      setErrorCode('invalid_token');
      setLoading(false);
    }
  }, [token]);

  const handleActivate = async (e) => {
    e.preventDefault();
    setFormError('');

    if (password.length < 8) {
      setFormError('Le mot de passe doit contenir au moins 8 caractères');
      return;
    }

    if (password !== confirmPassword) {
      setFormError('Les mots de passe ne correspondent pas');
      return;
    }

    setActivating(true);
    try {
      await apiClient.post('/auth/activate-account', {
        token,
        password,
      });
      setSuccess(true);

      // Rediriger vers login avec message de succès
      // L'utilisateur se connecte et DashboardRedirect le redirige automatiquement
      setTimeout(() => {
        navigate('/login', { state: { activated: true } });
      }, 2500);
    } catch (err) {
      const msg = err.response?.data?.error || 'Erreur lors de l\'activation du compte';
      setFormError(msg);
      setActivating(false);
    }
  };

  // --- Styles ---
  const containerStyle = {
    minHeight: '100vh',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    padding: 20,
    fontFamily: "'Inter', 'Helvetica Neue', Helvetica, Arial, sans-serif",
  };

  const cardStyle = {
    background: '#fff',
    borderRadius: 12,
    padding: '40px 32px',
    maxWidth: 440,
    width: '100%',
    boxShadow: '0 20px 60px rgba(0,0,0,0.2)',
  };

  const headingStyle = {
    fontSize: 24,
    fontWeight: 700,
    color: '#333',
    marginBottom: 8,
    textAlign: 'center',
  };

  const subheadingStyle = {
    fontSize: 14,
    color: '#666',
    textAlign: 'center',
    marginBottom: 24,
  };

  const infoBoxStyle = {
    background: '#f8f9fa',
    borderLeft: '4px solid #667eea',
    padding: '12px 16px',
    borderRadius: 4,
    marginBottom: 24,
    fontSize: 14,
    color: '#555',
  };

  const labelStyle = {
    display: 'block',
    fontSize: 13,
    fontWeight: 600,
    color: '#333',
    marginBottom: 6,
  };

  const inputStyle = {
    width: '100%',
    padding: '10px 14px',
    border: '1px solid #ddd',
    borderRadius: 8,
    fontSize: 14,
    boxSizing: 'border-box',
    outline: 'none',
    transition: 'border-color 0.2s',
  };

  const btnStyle = {
    width: '100%',
    padding: '12px',
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    color: '#fff',
    border: 'none',
    borderRadius: 8,
    fontSize: 16,
    fontWeight: 600,
    cursor: activating ? 'not-allowed' : 'pointer',
    opacity: activating ? 0.7 : 1,
    marginTop: 8,
  };

  const errorStyle = {
    background: '#fff3f3',
    color: '#c62828',
    padding: '10px 14px',
    borderRadius: 6,
    fontSize: 13,
    marginBottom: 16,
    border: '1px solid #ffcdd2',
  };

  const successStyle = {
    background: '#e8f5e9',
    color: '#2e7d32',
    padding: '16px',
    borderRadius: 8,
    textAlign: 'center',
    fontSize: 15,
  };

  // --- Loading state ---
  if (loading) {
    return (
      <div style={containerStyle}>
        <div style={cardStyle}>
          <div style={{ textAlign: 'center', padding: '40px 0' }}>
            <div style={{ fontSize: 32, marginBottom: 16 }}>
              <span role="img" aria-label="loading">&#9203;</span>
            </div>
            <p style={{ color: '#666', fontSize: 15 }}>Vérification du lien d'invitation...</p>
          </div>
        </div>
      </div>
    );
  }

  // --- Error state (invalid/expired token) ---
  if (error) {
    // Titre et sous-titre contextuels
    const errorConfig = {
      expired: {
        icon: '\u23F0',
        title: 'Invitation expirée',
        color: '#c62828',
        subtitle: 'Ce lien d\'invitation a expiré.',
        hint: 'Contactez l\'administrateur de votre institution pour qu\'il vous renvoie une invitation.',
        btnLabel: 'Retour à la connexion',
        btnBg: '#666',
      },
      already_activated: {
        icon: '\u2705',
        title: 'Compte déjà actif',
        color: '#2e7d32',
        subtitle: 'Ce compte a déjà été activé.',
        hint: 'Vous pouvez vous connecter avec vos identifiants existants.',
        btnLabel: 'Se connecter',
        btnBg: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      },
      default: {
        icon: '\u{1F6D1}',
        title: 'Lien invalide',
        color: '#c62828',
        subtitle: 'Ce lien d\'invitation est invalide ou a expiré.',
        hint: 'Vérifiez que vous avez bien copié le lien complet, ou contactez votre administrateur.',
        btnLabel: 'Retour à la connexion',
        btnBg: '#666',
      },
    };

    const cfg = errorConfig[errorCode] || errorConfig.default;

    return (
      <div style={containerStyle}>
        <div style={cardStyle}>
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: 48, marginBottom: 16 }}>
              <span role="img" aria-label="error">{cfg.icon}</span>
            </div>
            <h2 style={{ ...headingStyle, color: cfg.color }}>{cfg.title}</h2>
            <p style={{ color: '#666', fontSize: 14, marginBottom: 8 }}>{cfg.subtitle}</p>
            <p style={{ color: '#888', fontSize: 13, marginBottom: 24 }}>{cfg.hint}</p>
            <button
              onClick={() => navigate('/login')}
              style={{
                ...btnStyle,
                background: cfg.btnBg,
                cursor: 'pointer',
                opacity: 1,
              }}
            >
              {cfg.btnLabel}
            </button>
          </div>
        </div>
      </div>
    );
  }

  // --- Success state ---
  if (success) {
    return (
      <div style={containerStyle}>
        <div style={cardStyle}>
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: 48, marginBottom: 16 }}>
              <span role="img" aria-label="success">&#9989;</span>
            </div>
            <h2 style={headingStyle}>Compte activé !</h2>
            <div style={successStyle}>
              <p>Votre compte a été activé avec succès.</p>
              <p style={{ marginTop: 8, fontSize: 13 }}>
                Vous allez être redirigé vers la page de connexion...
              </p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // --- Activation form ---
  return (
    <div style={containerStyle}>
      <div style={cardStyle}>
        <h2 style={headingStyle}>Créer votre accès</h2>
        <p style={subheadingStyle}>
          Bienvenue sur le portail Lirie
        </p>

        <div style={infoBoxStyle}>
          <div><strong>Institution :</strong> {inviteData?.institution_name || '-'}</div>
          <div style={{ marginTop: 4 }}><strong>Email :</strong> {inviteData?.email}</div>
          {inviteData?.first_name && (
            <div style={{ marginTop: 4 }}><strong>Bonjour</strong> {inviteData.first_name} !</div>
          )}
        </div>

        {formError && <div style={errorStyle}>{formError}</div>}

        <form onSubmit={handleActivate}>
          <div style={{ marginBottom: 16 }}>
            <label style={labelStyle}>Mot de passe</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Minimum 8 caractères"
              required
              minLength={8}
              autoComplete="new-password"
              style={inputStyle}
            />
          </div>

          <div style={{ marginBottom: 20 }}>
            <label style={labelStyle}>Confirmer le mot de passe</label>
            <input
              type="password"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              placeholder="Répétez votre mot de passe"
              required
              minLength={8}
              autoComplete="new-password"
              style={inputStyle}
            />
          </div>

          <button
            type="submit"
            style={btnStyle}
            disabled={activating}
          >
            {activating ? 'Activation en cours...' : 'Activer mon compte'}
          </button>
        </form>

        <p style={{ fontSize: 12, color: '#999', textAlign: 'center', marginTop: 20 }}>
          En activant votre compte, vous acceptez les conditions d'utilisation du portail.
        </p>
      </div>
    </div>
  );
};

export default AcceptInvite;
