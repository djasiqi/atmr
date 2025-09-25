// src/components/DriverProfileModal.jsx
import React, { useState } from "react";
import styles from "./DriverProfileModal.module.css";
import { updateDriverProfile } from "../../../../services/driverService";

const DriverProfileModal = ({ profile, onClose, onSave }) => {
  const [editedProfile, setEditedProfile] = useState({
    first_name: profile.first_name || "",
    last_name: profile.last_name || "",
    phone: profile.phone || "",
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

    if (first_name === "" || last_name === "") {
      alert("Le prénom et le nom ne peuvent pas être vides.");
      return;
    }

    const payload = { first_name, last_name, phone };

    console.log("Payload complet envoyé:", payload);

    try {
      const updatedData = await updateDriverProfile(payload);
      onSave(updatedData);
    } catch (error) {
      console.error(
        "Erreur lors de la mise à jour du profil",
        error.response?.data
      );
      alert(
        "Une erreur est survenue lors de la mise à jour du profil : " +
          JSON.stringify(error.response?.data)
      );
    }
  };

  return (
    <div className={styles.modal}>
      <div className={styles.modalContent}>
        <h3>Modifier le profil</h3>
        <div className={styles.field}>
          <label htmlFor="first_name">Prénom :</label>
          <input
            type="text"
            id="first_name"
            name="first_name"
            value={editedProfile.first_name}
            onChange={handleChange}
          />
        </div>
        <div className={styles.field}>
          <label htmlFor="last_name">Nom :</label>
          <input
            type="text"
            id="last_name"
            name="last_name"
            value={editedProfile.last_name}
            onChange={handleChange}
          />
        </div>
        <div className={styles.field}>
          <label htmlFor="phone">Téléphone :</label>
          <input
            type="text"
            id="phone"
            name="phone"
            value={editedProfile.phone}
            onChange={handleChange}
          />
        </div>
        <div className={styles.field}>
          <label htmlFor="vehicle">Véhicule :</label>
          <input
            type="text"
            id="vehicle"
            value={profile.vehicle || ""}
            readOnly
          />
        </div>
        <div className={styles.actions}>
          <button onClick={handleSubmit}>Enregistrer</button>
          <button onClick={onClose}>Annuler</button>
        </div>
      </div>
    </div>
  );
};

export default DriverProfileModal;
