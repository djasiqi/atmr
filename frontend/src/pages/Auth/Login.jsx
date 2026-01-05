import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import apiClient from '../../utils/apiClient';
import { jwtDecode } from 'jwt-decode'; // Utilisation de l'import nommé
import styles from './Login.module.css';

const Login = () => {
  const [formData, setFormData] = useState({
    email: '',
    password: '',
  });
  const [errorMessage, setErrorMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const navigate = useNavigate();

  // Gestion des changements d'input
  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });
    setErrorMessage(''); // Réinitialise le message d'erreur
  };

  // Validation du formulaire
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
      localStorage.removeItem('authToken');
      const response = await apiClient.post('/auth/login', formData);
      const { token, user, refresh_token } = response.data;

      // #region agent log
      console.log('[Login] Réponse API:', { hasToken: !!token, tokenType: typeof token, tokenValue: token, hasUser: !!user, responseData: response.data, cookies: document.cookie });
      // #endregion

      if (!user || !user.role || !user.public_id) {
        throw new Error('Aucune information utilisateur reçue.');
      }

      console.log('✅ Connexion réussie :', user);

      // ✅ Le backend utilise des cookies httpOnly pour l'authentification web
      // Le token peut être dans la réponse JSON (pour mobile) ou dans un cookie (pour web)
      // Si le token est présent dans la réponse, on le stocke dans localStorage (pour compatibilité mobile)
      // Sinon, on utilise les cookies httpOnly (pour web)
      let roleSegment;
      if (token && typeof token === 'string') {
        // Token présent dans la réponse JSON (mode mobile/compatibilité)
        localStorage.setItem('authToken', token);
        if (refresh_token) localStorage.setItem('refreshToken', refresh_token);
        
        // Décoder le token pour vérifier les informations (notamment le rôle)
        // #region agent log
        console.log('[Login] Token dans réponse JSON, décodage:', { token, tokenType: typeof token, tokenLength: token?.length });
        // #endregion
        const decodedToken = jwtDecode(token);
        roleSegment = String(decodedToken.role || user.role || '').toLowerCase();
      } else {
        // ✅ Mode web avec cookies httpOnly : le token est dans un cookie, pas besoin de le décoder
        // Le backend a déjà défini les cookies httpOnly, on utilise juste les infos utilisateur
        console.log('[Login] Mode cookies httpOnly, pas de token dans la réponse JSON');
        roleSegment = String(user.role || '').toLowerCase();
      }
      
      // Normaliser le rôle stocké (cohérent avec ProtectedRoute)
      localStorage.setItem('user', JSON.stringify({ ...user, role: roleSegment }));
      localStorage.setItem('public_id', user.public_id);

      // Vérification si l'utilisateur doit réinitialiser son mot de passe
      if (user.force_password_change) {
        console.log('🔄 Redirection vers la réinitialisation du mot de passe...');
        navigate(`/reset-password/${user.public_id}`, { replace: true });
      } else {
        // Redirection normale vers le dashboard
        navigate(`/dashboard/${roleSegment}/${user.public_id}`, {
          replace: true,
        });
      }
    } catch (error) {
      console.error('❌ Erreur lors de la connexion :', error);
      const msg =
        error.response?.data?.error ??
        error.response?.data?.message ??
        error.response?.data?.detail ??
        (typeof error.response?.data === 'string' ? error.response.data : null) ??
        error.message;
      setErrorMessage(msg);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className={styles.loginContainer}>
      <h1 className={styles.title}>Connexion</h1>
      <form className={styles.form} onSubmit={handleSubmit}>
        <div className={styles.inputWrapper}>
          <label htmlFor="email">Email</label>
          <input
            type="email"
            name="email"
            id="email"
            placeholder="Entrez votre email"
            value={formData.email}
            onChange={handleInputChange}
            required
            aria-label="Adresse email"
          />
        </div>

        <div className={styles.inputWrapper}>
          <label htmlFor="password">Mot de passe</label>
          <input
            type="password"
            name="password"
            id="password"
            placeholder="Entrez votre mot de passe"
            value={formData.password}
            onChange={handleInputChange}
            required
            aria-label="Mot de passe"
          />
        </div>

        {/* Lien pour mot de passe oublié */}
        <div className={styles.forgotPassword}>
          <Link to="/forgot-password">Mot de passe oublié ?</Link>
        </div>

        {errorMessage && (
          <p className={styles.errorMessage} role="alert">
            {errorMessage}
          </p>
        )}

        <button
          type="submit"
          className={`${styles.submitButton} ${isLoading ? styles.disabled : ''}`}
          disabled={isLoading}
        >
          {isLoading ? 'Connexion en cours...' : 'Connexion'}
        </button>
      </form>
    </div>
  );
};

export default Login;
