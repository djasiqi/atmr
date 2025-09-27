import React, { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import apiClient from "../../utils/apiClient";
import { jwtDecode } from "jwt-decode"; // Utilisation de l'import nommé
import styles from "./Login.module.css";

const Login = () => {
  const [formData, setFormData] = useState({
    email: "",
    password: "",
  });
  const [errorMessage, setErrorMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const navigate = useNavigate();

  // Gestion des changements d'input
  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });
    setErrorMessage(""); // Réinitialise le message d'erreur
  };

  // Validation du formulaire
  const validateForm = () => {
    const { email, password } = formData;

    if (!email.trim() || !password) {
      setErrorMessage("Veuillez remplir tous les champs.");
      return false;
    }

    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
      setErrorMessage("Veuillez entrer une adresse email valide.");
      return false;
    }

    if (password.length < 6) {
      setErrorMessage("Le mot de passe doit contenir au moins 6 caractères.");
      return false;
    }

    return true;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!validateForm()) return;

    setIsLoading(true);
    try {
      localStorage.removeItem("authToken");
      const response = await apiClient.post("/auth/login", formData);
      const { token, user } = response.data;

      if (!user || !user.role || !user.public_id) {
        throw new Error("Aucune information utilisateur reçue.");
      }

      console.log("✅ Connexion réussie :", user);

      // Stocker les infos utilisateur
      localStorage.setItem("authToken", token);
      localStorage.setItem("user", JSON.stringify(user));
      localStorage.setItem("public_id", user.public_id);

      // Décoder le token pour vérifier les informations (notamment le rôle)
      const decodedToken = jwtDecode(token);
      console.log("Token décodé :", decodedToken);
      console.log("Rôle dans le token :", decodedToken.role);

      // Vérification si l'utilisateur doit réinitialiser son mot de passe
      if (user.force_password_change) {
        console.log(
          "🔄 Redirection vers la réinitialisation du mot de passe..."
        );
        navigate(`/reset-password/${user.public_id}`);
      } else {
        // Redirection normale vers le dashboard, en utilisant le rôle extrait du token
        navigate(
          `/dashboard/${decodedToken.role.toLowerCase()}/${user.public_id}`
        );
      }
    } catch (error) {
      console.error("❌ Erreur lors de la connexion :", error);
      const msg =
        error.response?.data?.error ??
        error.response?.data?.message ??
        error.response?.data?.detail ??
        (typeof error.response?.data === "string" ? error.response.data : null) ??
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
          className={`${styles.submitButton} ${
            isLoading ? styles.disabled : ""
          }`}
          disabled={isLoading}
        >
          {isLoading ? "Connexion en cours..." : "Connexion"}
        </button>
      </form>
    </div>
  );
};

export default Login;
