// frontend/src/pages/company/Settings/tabs/VehiclesTab.jsx
import React, { useState, useEffect } from 'react';
import {
  fetchCompanyVehicles,
  createVehicle,
  updateVehicle,
  deleteVehicle,
} from '../../../../services/companyService';
import styles from '../CompanySettings.module.css';
import modalStyles from '../../Clients/components/ClientFormModal.module.css';

const VehiclesTab = () => {
  const [vehicles, setVehicles] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [message, setMessage] = useState('');
  const [showModal, setShowModal] = useState(false);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [editingVehicle, setEditingVehicle] = useState(null);
  const [vehicleToDelete, setVehicleToDelete] = useState(null);

  const loadVehicles = async () => {
    try {
      setLoading(true);
      setError('');
      const data = await fetchCompanyVehicles();
      setVehicles(Array.isArray(data) ? data : []);
    } catch (err) {
      console.error('Erreur chargement véhicules:', err);
      setError('Impossible de charger les véhicules');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadVehicles();
  }, []);

  const handleCreate = () => {
    setEditingVehicle(null);
    setShowModal(true);
  };

  const handleEdit = (vehicle) => {
    setEditingVehicle(vehicle);
    setShowModal(true);
  };

  const handleDelete = (vehicle) => {
    setVehicleToDelete(vehicle);
    setShowDeleteModal(true);
  };

  const handleSave = async (vehicleData) => {
    try {
      setError('');
      setMessage('');
      if (editingVehicle) {
        await updateVehicle(editingVehicle.id, vehicleData);
        setMessage('✅ Véhicule mis à jour avec succès');
      } else {
        await createVehicle(vehicleData);
        setMessage('✅ Véhicule créé avec succès');
      }
      setShowModal(false);
      await loadVehicles();
      setTimeout(() => setMessage(''), 3000);
    } catch (err) {
      setError(err?.error || err?.message || 'Erreur lors de la sauvegarde');
    }
  };

  const handleConfirmDelete = async () => {
    if (!vehicleToDelete) return;
    try {
      setError('');
      await deleteVehicle(vehicleToDelete.id, false); // soft delete
      setMessage('✅ Véhicule supprimé avec succès');
      setShowDeleteModal(false);
      setVehicleToDelete(null);
      await loadVehicles();
      setTimeout(() => setMessage(''), 3000);
    } catch (err) {
      setError(err?.error || err?.message || 'Erreur lors de la suppression');
    }
  };

  if (loading) {
    return (
      <div className={styles.loadingContainer}>
        <div className={styles.spinner}></div>
        <p>Chargement des véhicules…</p>
      </div>
    );
  }

  return (
    <div className={styles.settingsForm} style={{ display: 'block' }}>
      {message && <div className={styles.success}>{message}</div>}
      {error && <div className={styles.error}>{error}</div>}

      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: 'var(--spacing-lg)',
        }}
      >
        <h2>🚗 Gestion de la flotte</h2>
        <button className="btn btn-primary" onClick={handleCreate}>
          ➕ Ajouter un véhicule
        </button>
      </div>

      {vehicles.length === 0 ? (
        <div style={{ textAlign: 'center', padding: 'var(--spacing-2xl)', color: 'var(--text-secondary)' }}>
          <div style={{ fontSize: '3rem', marginBottom: 'var(--spacing-md)' }}>🚗</div>
          <h3>Aucun véhicule</h3>
          <p>Créez votre premier véhicule pour commencer</p>
        </div>
      ) : (
        <div className="table-container">
          <table className="table">
            <thead>
              <tr>
                <th>Modèle</th>
                <th>Plaque</th>
                <th>Année</th>
                <th>Places</th>
                <th>FAH</th>
                <th>Statut</th>
                <th style={{ textAlign: 'right' }}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {vehicles.map((vehicle) => (
                <tr
                  key={vehicle.id}
                  style={{ opacity: vehicle.is_active ? 1 : 0.6 }}
                >
                  <td>
                    <strong>{vehicle.model || '-'}</strong>
                  </td>
                  <td style={{ fontFamily: 'monospace', fontWeight: 'var(--font-weight-medium)' }}>
                    {vehicle.license_plate || '-'}
                  </td>
                  <td>{vehicle.year || '-'}</td>
                  <td>{vehicle.seats || '-'}</td>
                  <td>{vehicle.wheelchair_accessible ? '✅' : '❌'}</td>
                  <td>
                    <span
                      style={{
                        padding: '4px 8px',
                        borderRadius: '4px',
                        fontSize: '0.85rem',
                        backgroundColor: vehicle.is_active
                          ? 'var(--success-light)'
                          : 'var(--error-light)',
                        color: vehicle.is_active ? 'var(--success)' : 'var(--error)',
                        fontWeight: 'var(--font-weight-medium)',
                      }}
                    >
                      {vehicle.is_active ? 'Actif' : 'Inactif'}
                    </span>
                  </td>
                  <td style={{ textAlign: 'right' }}>
                    <button
                      className="btn btn-secondary"
                      onClick={() => handleEdit(vehicle)}
                      style={{ marginRight: 'var(--spacing-xs)', padding: 'var(--spacing-xs)' }}
                      title="Modifier"
                    >
                      ✏️
                    </button>
                    <button
                      className="btn btn-secondary"
                      onClick={() => handleDelete(vehicle)}
                      style={{ padding: 'var(--spacing-xs)' }}
                      title="Supprimer"
                    >
                      🗑️
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Modal Création/Édition */}
      {showModal && (
        <VehicleModal
          vehicle={editingVehicle}
          onSave={handleSave}
          onClose={() => {
            setShowModal(false);
            setEditingVehicle(null);
            setError('');
          }}
        />
      )}

      {/* Modal Suppression */}
      {showDeleteModal && vehicleToDelete && (
        <div className="modal-overlay" onClick={() => setShowDeleteModal(false)}>
          <div className="modal-content modal-md" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h2 className="modal-title">🗑️ Confirmer la suppression</h2>
              <button className="modal-close" onClick={() => setShowDeleteModal(false)}>
                ✕
              </button>
            </div>
            <div className="modal-body">
              <p>
                Êtes-vous sûr de vouloir supprimer le véhicule{' '}
                <strong>{vehicleToDelete.model}</strong> ({vehicleToDelete.license_plate}) ?
              </p>
            </div>
            <div className="modal-footer">
              <button className="btn btn-secondary" onClick={() => setShowDeleteModal(false)}>
                Annuler
              </button>
              <button className="btn btn-primary" onClick={handleConfirmDelete}>
                Supprimer
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// Modal de formulaire véhicule
const VehicleModal = ({ vehicle, onSave, onClose }) => {
  const [formData, setFormData] = useState({
    model: '',
    license_plate: '',
    year: '',
    vin: '',
    seats: '',
    wheelchair_accessible: false,
    is_active: true,
  });
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (vehicle) {
      setFormData({
        model: vehicle.model || '',
        license_plate: vehicle.license_plate || '',
        year: vehicle.year || '',
        vin: vehicle.vin || '',
        seats: vehicle.seats || '',
        wheelchair_accessible: vehicle.wheelchair_accessible || false,
        is_active: vehicle.is_active !== undefined ? vehicle.is_active : true,
      });
    }
  }, [vehicle]);

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setSaving(true);
    try {
      // ✅ Construire le payload en omettant les champs vides (pas de null)
      const payload = {
        model: formData.model.trim(),
        license_plate: formData.license_plate.trim().toUpperCase(),
        wheelchair_accessible: formData.wheelchair_accessible,
        is_active: formData.is_active,
      };
      
      // Ajouter les champs optionnels seulement s'ils ont une valeur
      if (formData.year && formData.year.trim()) {
        payload.year = parseInt(formData.year);
      }
      if (formData.vin && formData.vin.trim()) {
        payload.vin = formData.vin.trim();
      }
      if (formData.seats && formData.seats.trim()) {
        payload.seats = parseInt(formData.seats);
      }
      
      await onSave(payload);
    } catch (err) {
      console.error('Erreur sauvegarde véhicule:', err);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content modal-md" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2 className="modal-title">
            {vehicle ? '✏️ Modifier le véhicule' : '➕ Ajouter un véhicule'}
          </h2>
          <button className="modal-close" onClick={onClose}>
            ✕
          </button>
        </div>

        <form onSubmit={handleSubmit} className={modalStyles.form}>
          {/* Informations du véhicule */}
          <div className={modalStyles.section}>
            <h3 className={modalStyles.sectionTitle}>🚗 Informations du véhicule</h3>

            <div className={modalStyles.formGroup}>
              <label htmlFor="model" className={modalStyles.label}>
                Modèle <span style={{ color: 'var(--danger-primary)' }}>*</span>
              </label>
              <input
                type="text"
                id="model"
                name="model"
                value={formData.model}
                onChange={handleChange}
                className={modalStyles.input}
                required
                placeholder="Ex: Peugeot Expert"
                disabled={saving}
              />
            </div>

            <div className={modalStyles.formGroup}>
              <label htmlFor="license_plate" className={modalStyles.label}>
                Plaque d'immatriculation <span style={{ color: 'var(--danger-primary)' }}>*</span>
              </label>
              <input
                type="text"
                id="license_plate"
                name="license_plate"
                value={formData.license_plate}
                onChange={handleChange}
                className={modalStyles.input}
                required
                placeholder="Ex: GE-123-456"
                style={{ fontFamily: 'monospace', textTransform: 'uppercase' }}
                disabled={saving}
              />
            </div>

            <div className={modalStyles.formRow}>
              <div className={modalStyles.formGroup}>
                <label htmlFor="year" className={modalStyles.label}>
                  Année
                </label>
                <input
                  type="number"
                  id="year"
                  name="year"
                  value={formData.year}
                  onChange={handleChange}
                  className={modalStyles.input}
                  min="1950"
                  max="2100"
                  placeholder="Ex: 2020"
                  disabled={saving}
                />
              </div>

              <div className={modalStyles.formGroup}>
                <label htmlFor="seats" className={modalStyles.label}>
                  Nombre de places
                </label>
                <input
                  type="number"
                  id="seats"
                  name="seats"
                  value={formData.seats}
                  onChange={handleChange}
                  className={modalStyles.input}
                  min="0"
                  placeholder="Ex: 7"
                  disabled={saving}
                />
              </div>
            </div>

            <div className={modalStyles.formGroup}>
              <label htmlFor="vin" className={modalStyles.label}>
                VIN (numéro de série)
              </label>
              <input
                type="text"
                id="vin"
                name="vin"
                value={formData.vin}
                onChange={handleChange}
                className={modalStyles.input}
                placeholder="Optionnel"
                disabled={saving}
              />
            </div>
          </div>

          {/* Caractéristiques */}
          <div className={modalStyles.section}>
            <h3 className={modalStyles.sectionTitle}>⚙️ Caractéristiques</h3>

            <div className={modalStyles.checkboxGroup}>
              <label className={modalStyles.checkboxLabel}>
                <input
                  type="checkbox"
                  name="wheelchair_accessible"
                  checked={formData.wheelchair_accessible}
                  onChange={handleChange}
                  disabled={saving}
                />
                <span className={modalStyles.checkboxText}>
                  <strong>Accessible aux fauteuils roulants (FAH)</strong>
                  <small>Véhicule équipé pour transporter des fauteuils roulants</small>
                </span>
              </label>
            </div>
          </div>

          {/* Statut */}
          <div className={modalStyles.section}>
            <h3 className={modalStyles.sectionTitle}>Statut</h3>

            <div className={modalStyles.checkboxGroup}>
              <label className={modalStyles.checkboxLabel}>
                <input
                  type="checkbox"
                  name="is_active"
                  checked={formData.is_active}
                  onChange={handleChange}
                  disabled={saving}
                />
                <span className={modalStyles.checkboxText}>
                  <strong>Véhicule actif</strong>
                  <small>Les véhicules inactifs ne sont pas disponibles pour assignation</small>
                </span>
              </label>
            </div>
          </div>

          {/* Actions */}
          <div className="modal-footer">
            <button type="button" className="btn btn-secondary" onClick={onClose} disabled={saving}>
              Annuler
            </button>
            <button type="submit" className="btn btn-primary" disabled={saving}>
              {saving ? '💾 Enregistrement...' : '💾 Enregistrer'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default VehiclesTab;

