import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import apiClient from '../../utils/apiClient';
import styles from './Signup.module.css';

const SIGNUP_DISABLED =
  process.env.REACT_APP_SIGNUP_DISABLED === 'true' || process.env.REACT_APP_SIGNUP_DISABLED === '1';

const Signup = () => {
  const [formData, setFormData] = useState({
    username: '',
    email: '',
    password: '',
    phone: '',
    address: '',
  });

  const [errorMessage, setErrorMessage] = useState('');
  const [successMessage, setSuccessMessage] = useState('');
  const navigate = useNavigate();

  if (SIGNUP_DISABLED) {
    return (
      <div className={styles.signupContainer}>
        <h1 className={styles.title}>Inscriptions temporairement suspendues</h1>
        <p className={styles.infoMessage}>
          Cette fonctionnalité est en cours de développement. Pour toute demande d&apos;accès ou
          d&apos;information, veuillez écrire à{' '}
          <a href="mailto:info@lirie.ch" className={styles.contactLink}>
            info@lirie.ch
          </a>
          .
        </p>
      </div>
    );
  }

  // Gestion des changements dans les champs du formulaire
  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });
    setErrorMessage(''); // Réinitialise le message d'erreur
    setSuccessMessage(''); // Réinitialise le message de succès
  };

  // Validation du formulaire
  const validateForm = () => {
    const { username, email, password, phone } = formData;

    if (!username.trim() || !email.trim() || !password) {
      setErrorMessage(‘Tous les champs obligatoires doivent être remplis.’);
      return false;
    }

    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
      setErrorMessage(‘Veuillez entrer une adresse email valide.’);
      return false;
    }

    if (password.length < 8) {
      setErrorMessage(‘Le mot de passe doit contenir au moins 8 caractères.’);
      return false;
    }

    if (!phone.trim() || phone.trim().length < 7) {
      setErrorMessage(‘Un numéro de téléphone valide est requis.’);
      return false;
    }

    return true;
  };

  // Gestion de l’envoi du formulaire
  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!validateForm()) return;

    try {
      const response = await apiClient.post(‘/auth/register’, {
        username: formData.username.trim(),
        email: formData.email.trim(),
        password: formData.password,
        phone: formData.phone.trim(),
        address: formData.address.trim() || undefined,
      });

      const { activation_session_id, masked_email, masked_phone } = response.data || {};

      if (!activation_session_id) {
        setErrorMessage("Inscription créée mais session d’activation manquante. Contactez le support.");
        return;
      }

      navigate(
        `/activate-account?activation_session_id=${encodeURIComponent(activation_session_id)}`,
        {
          replace: true,
          state: {
            maskedEmail: masked_email ?? ‘’,
            maskedPhone: masked_phone ?? ‘’,
            prefillEmail: formData.email.trim(),
          },
        }
      );
    } catch (error) {
      const data = error?.response?.data || {};
      const sessionId = data.activation_session_id;
      if (error?.response?.status === 502 && sessionId) {
        navigate(
          `/activate-account?activation_session_id=${encodeURIComponent(sessionId)}`,
          {
            replace: true,
            state: {
              maskedEmail: data.masked_email ?? '',
              maskedPhone: data.masked_phone ?? '',
              prefillEmail: formData.email.trim(),
            },
          }
        );
        return;
      }
      const msg =
        data.message ||
        data.error ||
        (error?.message === 'Network Error'
          ? 'Impossible de communiquer avec le serveur.'
          : "Une erreur s'est produite.");
      setErrorMessage(msg);
      setSuccessMessage('');
    }
  };

  return (
    <div className={styles.signupContainer}>
      <h1 className={styles.title}>Créer un compte</h1>
      <form className={styles.form} onSubmit={handleSubmit}>
        <div className={styles.inputWrapper}>
          <label>Nom</label>
          <input
            type="text"
            name="username"
            placeholder="Entrez votre nom"
            value={formData.username}
            onChange={handleInputChange}
            required
          />
        </div>

        <div className={styles.inputWrapper}>
          <label>Email</label>
          <input
            type="email"
            name="email"
            placeholder="Entrez votre email"
            value={formData.email}
            onChange={handleInputChange}
            required
          />
        </div>

        <div className={styles.inputWrapper}>
          <label>Mot de passe</label>
          <input
            type="password"
            name="password"
            placeholder="Entrez votre mot de passe"
            value={formData.password}
            onChange={handleInputChange}
            required
          />
        </div>

        <div className={styles.inputWrapper}>
          <label>Téléphone</label>
          <input
            type="text"
            name="phone"
            placeholder="Entrez votre téléphone"
            value={formData.phone}
            onChange={handleInputChange}
          />
        </div>

        <div className={styles.inputWrapper}>
          <label>Adresse</label>
          <input
            type="text"
            name="address"
            placeholder="Entrez votre adresse"
            value={formData.address}
            onChange={handleInputChange}
          />
        </div>

        {/* Affichage des messages d'erreur ou de succès */}
        {errorMessage && <p className={styles.errorMessage}>{errorMessage}</p>}
        {successMessage && <p className={styles.successMessage}>{successMessage}</p>}

        <button type="submit" className={styles.submitButton}>
          S'inscrire
        </button>
      </form>
    </div>
  );
};

export default Signup;
