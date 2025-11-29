// src/pages/company/components/EditDriverForm.jsx
import React, { useState, useEffect } from 'react';
import { toast } from 'sonner';
import styles from './EditDriverForm.module.css';
import { fetchCompanyVehicles } from '../../../services/companyService';
import AddressAutocomplete from '../../../components/common/AddressAutocomplete';
import apiClient from '../../../utils/apiClient';

const EditDriverForm = ({ driver, onSubmit, onClose }) => {
  // État pour les informations utilisateur
  const [userData, setUserData] = useState({
    first_name: driver.first_name || driver.user?.first_name || '',
    last_name: driver.last_name || driver.user?.last_name || '',
    email: driver.email || driver.user?.email || '',
    address: driver.user?.address || '',
  });

  // État pour les informations du chauffeur
  const [formData, setFormData] = useState({
    vehicle_id: driver.vehicle_id || driver.vehicle?.id || null,
    is_active: driver.is_active !== undefined ? driver.is_active : true,
  });

  // État pour l'adresse de domiciliation
  const [domicileAddress, setDomicileAddress] = useState(driver.user?.address || '');

  // État pour les véhicules
  const [vehicles, setVehicles] = useState([]);
  const [loadingVehicles, setLoadingVehicles] = useState(true);

  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isResettingPassword, setIsResettingPassword] = useState(false);

  // Charger les véhicules de l'entreprise
  useEffect(() => {
    const loadVehicles = async () => {
      try {
        setLoadingVehicles(true);
        const vehiclesList = await fetchCompanyVehicles();
        // Filtrer uniquement les véhicules actifs
        const activeVehicles = vehiclesList.filter((v) => v.is_active !== false);
        setVehicles(activeVehicles || []);
      } catch (error) {
        console.error('Erreur lors du chargement des véhicules:', error);
        toast.error('Impossible de charger la liste des véhicules');
      } finally {
        setLoadingVehicles(false);
      }
    };

    loadVehicles();
  }, []);

  // Gestionnaire de changement pour les informations utilisateur
  const handleUserChange = (e) => {
    const { name, value } = e.target;
    setUserData((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  // Gestionnaire de changement pour les informations du chauffeur
  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData((prevData) => ({
      ...prevData,
      [name]: type === 'checkbox' ? checked : value === '' ? null : value,
    }));
  };

  // Gestion de l'adresse de domiciliation
  const handleDomicileAddressChange = (e) => {
    let address = '';
    if (e && typeof e === 'object' && e.target && typeof e.target === 'object') {
      address = e.target.value || '';
    } else if (typeof e === 'string') {
      address = e;
    }
    const cleanAddress = String(address || '').trim();
    setDomicileAddress(cleanAddress);
  };

  const handleDomicileAddressSelect = (item) => {
    let address = '';
    if (item && typeof item === 'object') {
      address = item.label || item.address || '';
    } else if (typeof item === 'string') {
      address = item;
    }
    const cleanAddress = String(address || '').trim();
    if (cleanAddress) {
      setDomicileAddress(cleanAddress);
    }
  };

  // Gestion de la réinitialisation du mot de passe
  const handleResetPassword = async () => {
    if (
      !window.confirm(
        'Êtes-vous sûr de vouloir réinitialiser le mot de passe de ce chauffeur ? Un nouveau mot de passe sera généré.'
      )
    ) {
      return;
    }

    setIsResettingPassword(true);
    try {
      // Utiliser la route spécifique pour réinitialiser le mot de passe du driver
      const response = await apiClient.post(
        `/companies/me/drivers/${driver.id}/reset-password`
      );
      if (response.data?.new_password) {
        toast.success(
          `Mot de passe réinitialisé avec succès. Nouveau mot de passe : ${response.data.new_password}`,
          { duration: 10000 }
        );
      } else {
        toast.error('Erreur lors de la réinitialisation du mot de passe.');
      }
    } catch (error) {
      console.error('Erreur réinitialisation mot de passe:', error);
      const errorMessage =
        error?.response?.data?.error ||
        error?.response?.data?.message ||
        'Erreur lors de la réinitialisation du mot de passe.';
      toast.error(errorMessage);
    } finally {
      setIsResettingPassword(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsSubmitting(true);
    try {
      // Préparer les données à envoyer
      const updateData = {
        // Informations utilisateur
        first_name: userData.first_name.trim(),
        last_name: userData.last_name.trim(),
        email: userData.email.trim(),
        address: domicileAddress.trim(),
        // Informations du chauffeur
        vehicle_id: formData.vehicle_id ? Number(formData.vehicle_id) : null,
        is_active: formData.is_active,
      };

      await onSubmit(driver.id, updateData);
    } catch (error) {
      console.error('Failed to update driver:', error);
      const errorMessage =
        error?.response?.data?.error ||
        error?.response?.data?.message ||
        'Erreur lors de la mise à jour du chauffeur.';
      toast.error(errorMessage);
    } finally {
      setIsSubmitting(false);
    }
  };

  // Véhicule sélectionné actuellement
  const selectedVehicle = vehicles.find((v) => v.id === Number(formData.vehicle_id));

  return (
    <form onSubmit={handleSubmit} className={styles.form}>
      {/* Section : Informations personnelles */}
      <div className={styles.section}>
        <h4>Informations personnelles</h4>

        <div className={styles.formGroup}>
          <label htmlFor="first_name">
            Prénom <span className={styles.required}>*</span>
          </label>
          <input
            type="text"
            id="first_name"
            name="first_name"
            value={userData.first_name}
            onChange={handleUserChange}
            placeholder="Prénom du chauffeur"
            required
            disabled={isSubmitting}
          />
        </div>

        <div className={styles.formGroup}>
          <label htmlFor="last_name">
            Nom <span className={styles.required}>*</span>
          </label>
          <input
            type="text"
            id="last_name"
            name="last_name"
            value={userData.last_name}
            onChange={handleUserChange}
            placeholder="Nom du chauffeur"
            required
            disabled={isSubmitting}
          />
        </div>

        <div className={styles.formGroup}>
          <label htmlFor="email">
            Email <span className={styles.required}>*</span>
          </label>
          <input
            type="email"
            id="email"
            name="email"
            value={userData.email}
            onChange={handleUserChange}
            placeholder="email@exemple.com"
            required
            disabled={isSubmitting}
          />
        </div>

        <div className={styles.formGroup} style={{ gridColumn: '1 / -1' }}>
          <label htmlFor="domicile_address">
            Adresse de domiciliation
          </label>
          <AddressAutocomplete
            id="domicile_address"
            name="domicile_address"
            value={domicileAddress}
            onChange={handleDomicileAddressChange}
            onSelect={handleDomicileAddressSelect}
            placeholder="Adresse complète du domicile"
            disabled={isSubmitting}
          />
        </div>
      </div>

      {/* Section : Véhicule assigné */}
      <div className={styles.section}>
        <h4>Véhicule assigné</h4>

        <div className={styles.formGroup} style={{ gridColumn: '1 / -1' }}>
          <label htmlFor="vehicle_id">Véhicule</label>
          {loadingVehicles ? (
            <div className={styles.loadingText}>Chargement des véhicules...</div>
          ) : vehicles.length === 0 ? (
            <div className={styles.warningText}>
              Aucun véhicule disponible. Veuillez d'abord créer un véhicule dans la gestion de la
              flotte.
            </div>
          ) : (
            <select
              id="vehicle_id"
              name="vehicle_id"
              value={formData.vehicle_id || ''}
              onChange={handleChange}
              disabled={isSubmitting}
              className={styles.select}
            >
              <option value="">Aucun véhicule assigné</option>
              {vehicles.map((vehicle) => (
                <option key={vehicle.id} value={vehicle.id}>
                  {vehicle.model} - {vehicle.license_plate}
                  {vehicle.year ? ` (${vehicle.year})` : ''}
                </option>
              ))}
            </select>
          )}
        </div>

        {selectedVehicle && (
          <div className={styles.vehicleInfo}>
            <p>
              <strong>Modèle :</strong> {selectedVehicle.model}
            </p>
            <p>
              <strong>Plaque :</strong> {selectedVehicle.license_plate}
            </p>
            {selectedVehicle.year && (
              <p>
                <strong>Année :</strong> {selectedVehicle.year}
              </p>
            )}
          </div>
        )}
      </div>

      {/* Section : Statut */}
      <div className={styles.section} style={{ gridColumn: '1 / -1' }}>
        <h4>Statut</h4>
        <div className={styles.checkboxGroup}>
          <input
            type="checkbox"
            name="is_active"
            id="is_active_checkbox"
            checked={formData.is_active}
            onChange={handleChange}
            disabled={isSubmitting}
          />
          <label htmlFor="is_active_checkbox">Chauffeur actif</label>
        </div>
      </div>

      {/* Actions */}
      <div className={styles.formActions}>
        <button
          type="button"
          onClick={handleResetPassword}
          className={styles.resetPasswordButton}
          disabled={isSubmitting || isResettingPassword}
        >
          {isResettingPassword ? 'Réinitialisation...' : '🔑 Réinitialiser le mot de passe'}
        </button>
        <div className={styles.buttonGroup}>
          <button
            type="button"
            onClick={onClose}
            className={styles.cancelButton}
            disabled={isSubmitting || isResettingPassword}
          >
            Annuler
          </button>
          <button
            type="submit"
            className={styles.submitButton}
            disabled={isSubmitting || isResettingPassword}
          >
            {isSubmitting ? 'Enregistrement...' : 'Enregistrer'}
          </button>
        </div>
      </div>
    </form>
  );
};

export default EditDriverForm;
