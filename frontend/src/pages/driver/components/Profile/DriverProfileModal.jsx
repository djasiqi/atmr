// src/components/DriverProfileModal.jsx
import React, { useState } from 'react';
import { updateDriverProfile } from '../../../../services/driverService';

const DriverProfileModal = ({ profile, onClose, onSave }) => {
  const [editedProfile, setEditedProfile] = useState({
    first_name: profile.first_name || '',
    last_name: profile.last_name || '',
    phone: profile.phone || '',
  });

  const handleChange = (e) => {
    setEditedProfile({
      ...editedProfile,
      [e.target.name]: e.target.value,
    });
  };

  const handleSubmit = async () => {
    const first_name = editedProfile.first_name.trim();
    const last_name = editedProfile.last_name.trim();
    const phone = editedProfile.phone.trim();

    if (first_name === '' || last_name === '') {
      alert('Le prénom et le nom ne peuvent pas être vides.');
      return;
    }

    const payload = { first_name, last_name, phone };

    try {
      const updatedData = await updateDriverProfile(payload);
      onSave(updatedData);
    } catch (error) {
      // eslint-disable-next-line no-console
      console.error('Erreur lors de la mise à jour du profil', error.response?.data);
      alert(
        'Une erreur est survenue lors de la mise à jour du profil : ' +
          JSON.stringify(error.response?.data)
      );
    }
  };

  return (
    <div className="modal-overlay">
      <div className="modal-content modal-md">
        <div className="modal-header">
          <h3 className="modal-title">Modifier le profil</h3>
          <button className="modal-close" onClick={onClose}>
            ✕
          </button>
        </div>
        <div className="modal-body">
          <div className="form-group">
            <label htmlFor="first_name" className="form-label">
              Prénom
            </label>
            <input
              type="text"
              id="first_name"
              name="first_name"
              className="form-input"
              value={editedProfile.first_name}
              onChange={handleChange}
            />
          </div>
          <div className="form-group">
            <label htmlFor="last_name" className="form-label">
              Nom
            </label>
            <input
              type="text"
              id="last_name"
              name="last_name"
              className="form-input"
              value={editedProfile.last_name}
              onChange={handleChange}
            />
          </div>
          <div className="form-group">
            <label htmlFor="phone" className="form-label">
              Téléphone
            </label>
            <input
              type="text"
              id="phone"
              name="phone"
              className="form-input"
              value={editedProfile.phone}
              onChange={handleChange}
            />
          </div>
          <div className="form-group">
            <label htmlFor="vehicle" className="form-label">
              Véhicule
            </label>
            <input
              type="text"
              id="vehicle"
              className="form-input"
              value={profile.vehicle || ''}
              readOnly
            />
          </div>
        </div>
        <div className="modal-footer">
          <button className="btn btn-secondary" onClick={onClose}>
            Annuler
          </button>
          <button className="btn btn-primary" onClick={handleSubmit}>
            Enregistrer
          </button>
        </div>
      </div>
    </div>
  );
};

export default DriverProfileModal;
