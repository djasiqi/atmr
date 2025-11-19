// src/pages/company/Driver/components/EditDriverForm.jsx
import React, { useState } from 'react';
import styles from './EditDriverForm.module.css';

const EditDriverForm = ({ driver, onSubmit, onClose }) => {
  // 1. État centralisé, initialisé avec les bonnes données du chauffeur
  const [formData, setFormData] = useState({
    vehicle_assigned: driver.vehicle_assigned || '',
    brand: driver.brand || '',
    license_plate: driver.license_plate || '',
    is_active: driver.is_active,
  });

  const [isSubmitting, setIsSubmitting] = useState(false);

  // 2. Gestionnaire de changement unique (pour les inputs texte et checkbox)
  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData((prevData) => ({
      ...prevData,
      [name]: type === 'checkbox' ? checked : value,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsSubmitting(true);
    try {
      // La prop onSubmit est asynchrone pour gérer l'appel API
      await onSubmit(driver.id, formData);
    } catch (error) {
      console.error('Failed to update driver:', error);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className={styles.form}>
      {/* Informations utilisateur en lecture seule pour contexte */}
      <div className={styles.readOnlySection}>
        <h4>Informations sur l'utilisateur</h4>
        <p>
          <strong>Nom d'utilisateur :</strong> {driver.username}
        </p>
        <p>
          <strong>Nom complet :</strong> {driver.first_name} {driver.last_name}
        </p>
        <p>
          <strong>Email :</strong> {driver.user?.email || 'N/A'}
        </p>
      </div>

      <hr className={styles.divider} />

      <h4>Informations sur le véhicule</h4>
      <div className={styles.formGroup}>
        <label>Véhicule assigné :</label>
        <input
          type="text"
          name="vehicle_assigned"
          value={formData.vehicle_assigned}
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
          name="license_plate"
          value={formData.license_plate}
          onChange={handleChange}
          placeholder="Numéro de plaque"
          required
        />
      </div>

      <hr className={styles.divider} />

      <h4>Statut</h4>
      <div className={styles.checkboxGroup}>
        <input
          type="checkbox"
          name="is_active"
          id="is_active_checkbox"
          checked={formData.is_active}
          onChange={handleChange}
        />
        <label htmlFor="is_active_checkbox">Chauffeur actif</label>
      </div>

      <div className={styles.formActions}>
        <button type="submit" className={styles.submitButton} disabled={isSubmitting}>
          {isSubmitting ? 'Enregistrement...' : 'Enregistrer'}
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

export default EditDriverForm;
