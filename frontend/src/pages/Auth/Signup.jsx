import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import styles from "./Signup.module.css";

const Signup = () => {
  const [formData, setFormData] = useState({
    username: "",
    email: "",
    password: "",
    phone: "",
    address: "",
  });

  const [errorMessage, setErrorMessage] = useState("");
  const [successMessage, setSuccessMessage] = useState("");
  const navigate = useNavigate();

  // Gestion des changements dans les champs du formulaire
  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });
    setErrorMessage(""); // Réinitialise le message d'erreur
    setSuccessMessage(""); // Réinitialise le message de succès
  };

  // Validation du formulaire
  const validateForm = () => {
    const { username, email, password } = formData;

    if (!username || !email.trim() || !password) {
      setErrorMessage("Tous les champs obligatoires doivent être remplis.");
      return false;
    }

    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
      setErrorMessage("Veuillez entrer une adresse email valide.");
      return false;
    }

    if (password.length < 8) {
      setErrorMessage("Le mot de passe doit contenir au moins 8 caractères.");
      return false;
    }

    return true;
  };

  // Gestion de l'envoi du formulaire
  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!validateForm()) return; // Validation avant d'envoyer les données

    console.log("Données soumises :", formData); // Debugging : Vérifiez les données envoyées

    try {
      const response = await axios.post(
        "http://localhost:5000/auth/register",
        formData,
        {
          headers: {
            "Content-Type": "application/json",
          },
          withCredentials: true,
        }
      );

      console.log("Réponse du backend :", response.data); // Loguez la réponse pour confirmation
      setSuccessMessage(response.data.message);
      setErrorMessage("");

      // Redirection vers le tableau de bord
      const userId = response.data.user_id;
      navigate(`/dashboard/client/${userId}`);
    } catch (error) {
      // Détectez les erreurs de type CORS
      if (error.message === "Network Error") {
        console.error(
          "Erreur réseau détectée. Cela peut indiquer un problème CORS."
        );
        setErrorMessage(
          "Impossible de communiquer avec le serveur. Vérifiez la configuration CORS."
        );
      } else if (error.response) {
        // Loguez les détails de l'erreur côté serveur
        console.error("Erreur du serveur :", error.response.data);
        setErrorMessage(
          error.response.data.error || "Une erreur s'est produite."
        );
      } else {
        console.error("Erreur inattendue :", error);
        setErrorMessage("Une erreur inattendue est survenue.");
      }

      setSuccessMessage(""); // Réinitialiser les messages de succès
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
        {successMessage && (
          <p className={styles.successMessage}>{successMessage}</p>
        )}

        <button type="submit" className={styles.submitButton}>
          S'inscrire
        </button>
      </form>
    </div>
  );
};

export default Signup;
