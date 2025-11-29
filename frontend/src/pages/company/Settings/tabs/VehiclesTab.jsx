// frontend/src/pages/company/Settings/tabs/VehiclesTab.jsx
import React, { useState, useEffect } from 'react';
import {
  fetchCompanyVehicles,
  createVehicle,
  updateVehicle,
  deleteVehicle,
} from '../../../../services/companyService';
import styles from '../CompanySettings.module.css';

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

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--spacing-lg)' }}>
        <h2>🚗 Gestion de la flotte</h2>
        <button className={`${styles.button} ${styles.primary}`} onClick={handleCreate}>
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
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: '2px solid var(--border-color)' }}>
                <th style={{ padding: 'var(--spacing-md)', textAlign: 'left' }}>Modèle</th>
                <th style={{ padding: 'var(--spacing-md)', textAlign: 'left' }}>Plaque</th>
                <th style={{ padding: 'var(--spacing-md)', textAlign: 'left' }}>Année</th>
                <th style={{ padding: 'var(--spacing-md)', textAlign: 'left' }}>Places</th>
                <th style={{ padding: 'var(--spacing-md)', textAlign: 'left' }}>FAH</th>
                <th style={{ padding: 'var(--spacing-md)', textAlign: 'left' }}>Statut</th>
                <th style={{ padding: 'var(--spacing-md)', textAlign: 'right' }}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {vehicles.map((vehicle) => (
                <tr
                  key={vehicle.id}
                  style={{
                    borderBottom: '1px solid var(--border-color)',
                    opacity: vehicle.is_active ? 1 : 0.6,
                  }}
                >
                  <td style={{ padding: 'var(--spacing-md)' }}>{vehicle.model || '-'}</td>
                  <td style={{ padding: 'var(--spacing-md)', fontFamily: 'monospace' }}>
                    {vehicle.license_plate || '-'}
                  </td>
                  <td style={{ padding: 'var(--spacing-md)' }}>{vehicle.year || '-'}</td>
                  <td style={{ padding: 'var(--spacing-md)' }}>{vehicle.seats || '-'}</td>
                  <td style={{ padding: 'var(--spacing-md)' }}>
                    {vehicle.wheelchair_accessible ? '✅' : '❌'}
                  </td>
                  <td style={{ padding: 'var(--spacing-md)' }}>
                    <span
                      style={{
                        padding: '4px 8px',
                        borderRadius: '4px',
                        fontSize: '0.85rem',
                        backgroundColor: vehicle.is_active ? 'var(--success-light)' : 'var(--error-light)',
                        color: vehicle.is_active ? 'var(--success)' : 'var(--error)',
                      }}
                    >
                      {vehicle.is_active ? 'Actif' : 'Inactif'}
                    </span>
                  </td>
                  <td style={{ padding: 'var(--spacing-md)', textAlign: 'right' }}>
                    <button
                      className={`${styles.button} ${styles.secondary}`}
                      onClick={() => handleEdit(vehicle)}
                      style={{ marginRight: 'var(--spacing-xs)' }}
                    >
                      ✏️
                    </button>
                    <button
                      className={`${styles.button} ${styles.secondary}`}
                      onClick={() => handleDelete(vehicle)}
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
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <h3>Confirmer la suppression</h3>
            <p>
              Êtes-vous sûr de vouloir supprimer le véhicule{' '}
              <strong>{vehicleToDelete.model}</strong> ({vehicleToDelete.license_plate}) ?
            </p>
            <div style={{ display: 'flex', gap: 'var(--spacing-md)', justifyContent: 'flex-end', marginTop: 'var(--spacing-lg)' }}>
              <button className={`${styles.button} ${styles.secondary}`} onClick={() => setShowDeleteModal(false)}>
                Annuler
              </button>
              <button className={`${styles.button} ${styles.primary}`} onClick={handleConfirmDelete}>
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
      const payload = {
        model: formData.model.trim(),
        license_plate: formData.license_plate.trim().toUpperCase(),
        year: formData.year ? parseInt(formData.year) : null,
        vin: formData.vin.trim() || null,
        seats: formData.seats ? parseInt(formData.seats) : null,
        wheelchair_accessible: formData.wheelchair_accessible,
        is_active: formData.is_active,
      };
      await onSave(payload);
    } catch (err) {
      console.error('Erreur sauvegarde véhicule:', err);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()} style={{ maxWidth: '600px' }}>
        <h2>{vehicle ? '✏️ Modifier le véhicule' : '➕ Ajouter un véhicule'}</h2>
        <form onSubmit={handleSubmit}>
          <div className={styles.formGroup}>
            <label htmlFor="model">
              Modèle <span style={{ color: 'var(--error)' }}>*</span>
            </label>
            <input
              type="text"
              id="model"
              name="model"
              value={formData.model}
              onChange={handleChange}
              required
              placeholder="Ex: Peugeot Expert"
            />
          </div>

          <div className={styles.formGroup}>
            <label htmlFor="license_plate">
              Plaque d'immatriculation <span style={{ color: 'var(--error)' }}>*</span>
            </label>
            <input
              type="text"
              id="license_plate"
              name="license_plate"
              value={formData.license_plate}
              onChange={handleChange}
              required
              placeholder="Ex: GE-123-456"
              style={{ fontFamily: 'monospace', textTransform: 'uppercase' }}
            />
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--spacing-md)' }}>
            <div className={styles.formGroup}>
              <label htmlFor="year">Année</label>
              <input
                type="number"
                id="year"
                name="year"
                value={formData.year}
                onChange={handleChange}
                min="1950"
                max="2100"
                placeholder="Ex: 2020"
              />
            </div>

            <div className={styles.formGroup}>
              <label htmlFor="seats">Nombre de places</label>
              <input
                type="number"
                id="seats"
                name="seats"
                value={formData.seats}
                onChange={handleChange}
                min="0"
                placeholder="Ex: 7"
              />
            </div>
          </div>

          <div className={styles.formGroup}>
            <label htmlFor="vin">VIN (numéro de série)</label>
            <input
              type="text"
              id="vin"
              name="vin"
              value={formData.vin}
              onChange={handleChange}
              placeholder="Optionnel"
            />
          </div>

          <div className={styles.formGroup}>
            <label style={{ display: 'flex', alignItems: 'center', gap: 'var(--spacing-sm)' }}>
              <input
                type="checkbox"
                name="wheelchair_accessible"
                checked={formData.wheelchair_accessible}
                onChange={handleChange}
              />
              <span>Accessible aux fauteuils roulants (FAH)</span>
            </label>
          </div>

          <div className={styles.formGroup}>
            <label style={{ display: 'flex', alignItems: 'center', gap: 'var(--spacing-sm)' }}>
              <input
                type="checkbox"
                name="is_active"
                checked={formData.is_active}
                onChange={handleChange}
              />
              <span>Véhicule actif</span>
            </label>
          </div>

          <div style={{ display: 'flex', gap: 'var(--spacing-md)', justifyContent: 'flex-end', marginTop: 'var(--spacing-lg)' }}>
            <button type="button" className={`${styles.button} ${styles.secondary}`} onClick={onClose}>
              Annuler
            </button>
            <button type="submit" className={`${styles.button} ${styles.primary}`} disabled={saving}>
              {saving ? '💾 Enregistrement...' : '💾 Enregistrer'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default VehiclesTab;

