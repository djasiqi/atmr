// src/pages/company/components/AddDriverForm.jsx
import React, { useState } from 'react';
import styles from './AddDriverForm.module.css';

const AddDriverForm = ({ onSubmit, onClose }) => {
  const [formData, setFormData] = useState({
    username: '',
    firstName: '',
    lastName: '',
    email: '',
    password: '',
    confirmPassword: '',
    vehicleAssigned: '',
    brand: '',
    licensePlate: '',
  });

  const [errors, setErrors] = useState({});
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prevData) => ({
      ...prevData,
      [name]: value,
    }));
  };

  const validateForm = () => {
    const newErrors = {};
    if (!formData.username) newErrors.username = "Le nom d'utilisateur est requis.";
    if (!formData.email) newErrors.email = "L'email est requis.";
    if (formData.password.length < 8)
      newErrors.password = 'Le mot de passe doit faire au moins 8 caractères.';
    if (formData.password !== formData.confirmPassword) {
      newErrors.confirmPassword = 'Les mots de passe ne correspondent pas.';
    }
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!validateForm()) return;
    setIsSubmitting(true);

    const payload = {
      username: formData.username,
      first_name: formData.firstName,
      last_name: formData.lastName,
      email: formData.email,
      password: formData.password,
      vehicle_assigned: formData.vehicleAssigned,
      brand: formData.brand,
      license_plate: formData.licensePlate,
    };

    try {
      await onSubmit(payload);
    } catch (error) {
      console.error('Submission failed:', error);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className={styles.form}>
      <div className={styles.formGroup}>
        <label>Nom d'utilisateur :</label>
        <input
          type="text"
          name="username"
          value={formData.username}
          onChange={handleChange}
          required
        />
        {errors.username && <span className={styles.error}>{errors.username}</span>}
      </div>

      <div className={styles.formGroup}>
        <label>Prénom :</label>
        <input
          type="text"
          name="firstName"
          value={formData.firstName}
          onChange={handleChange}
          required
        />
      </div>

      <div className={styles.formGroup}>
        <label>Nom :</label>
        <input
          type="text"
          name="lastName"
          value={formData.lastName}
          onChange={handleChange}
          required
        />
      </div>

      <div className={styles.formGroup}>
        <label>Adresse email :</label>
        <input type="email" name="email" value={formData.email} onChange={handleChange} required />
        {errors.email && <span className={styles.error}>{errors.email}</span>}
      </div>

      <div className={styles.formGroup}>
        <label>Mot de passe :</label>
        <input
          type="password"
          name="password"
          value={formData.password}
          onChange={handleChange}
          autoComplete="new-password"
          required
        />
        {errors.password && <span className={styles.error}>{errors.password}</span>}
      </div>

      <div className={styles.formGroup}>
        <label>Confirmer le mot de passe :</label>
        <input
          type="password"
          name="confirmPassword"
          value={formData.confirmPassword}
          onChange={handleChange}
          autoComplete="new-password"
          required
        />
        {errors.confirmPassword && <span className={styles.error}>{errors.confirmPassword}</span>}
      </div>

      <div className={styles.formGroup}>
        <label>Véhicule assigné :</label>
        <input
          type="text"
          name="vehicleAssigned"
          value={formData.vehicleAssigned}
          onChange={handleChange}
          placeholder="Nom ou modèle du véhicule"
          required
        />
      </div>

      <div className={styles.formGroup}>
        <label>Marque :</label>
        <input
          type="text"
          name="brand"
          value={formData.brand}
          onChange={handleChange}
          placeholder="Marque du véhicule"
          required
        />
      </div>

      <div className={styles.formGroup}>
        <label>Numéro de plaque :</label>
        <input
          type="text"
          name="licensePlate"
          value={formData.licensePlate}
          onChange={handleChange}
          placeholder="Numéro de plaque"
          required
        />
      </div>

      <div className={styles.formActions}>
        <button type="submit" className={styles.submitButton} disabled={isSubmitting}>
          {isSubmitting ? 'Ajout en cours...' : 'Ajouter'}
        </button>
        <button
          type="button"
          onClick={onClose}
          className={styles.cancelButton}
          disabled={isSubmitting}
        >
          Annuler
        </button>
      </div>
    </form>
  );
};

export default AddDriverForm;
