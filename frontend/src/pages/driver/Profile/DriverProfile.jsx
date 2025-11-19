// src/components/DriverProfile.jsx
import React, { useState, useEffect } from 'react';
import styles from './DriverProfile.module.css';
import { fetchDriverProfile } from '../../../services/driverService';
import PhotoCaptureModal from '../components/Dashboard/PhotoCaptureModal';

const DriverProfile = ({ public_id, onEdit }) => {
  const [profile, setProfile] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [profilePic, setProfilePic] = useState('/images/default-driver.png');
  const [photoModalOpen, setPhotoModalOpen] = useState(false);

  useEffect(() => {
    const loadProfile = async () => {
      try {
        setLoading(true);
        setError('');
        const data = await fetchDriverProfile();
        if (data && data.profile) {
          setProfile(data.profile);
        } else if (data) {
          setProfile(data);
        } else {
          setError('Profil introuvable.');
        }
      } catch (err) {
        console.error('Erreur lors du chargement du profil', err);
        setError('Erreur lors du chargement du profil.');
      } finally {
        setLoading(false);
      }
    };

    loadProfile();
  }, [public_id]);

  useEffect(() => {
    if (profile) {
      setProfilePic(profile.photo || '/images/default-driver.png');
    }
  }, [profile]);

  const handlePhotoModalClose = () => {
    setPhotoModalOpen(false);
  };

  const handlePhotoCapture = (newPhoto) => {
    setProfilePic(newPhoto);
    // Vous pouvez ajouter ici un appel pour mettre à jour la photo dans le backend
  };

  if (loading) {
    return <div className={styles.profile}>Chargement du profil...</div>;
  }

  if (error || !profile) {
    return <div className={styles.profile}>{error || 'Profil introuvable'}</div>;
  }

  const { firstName, lastName, phone, vehicle, status } = profile;

  return (
    <section className={styles.profileCard}>
      <div className={styles.profileHeader}>
        <img
          src={profilePic}
          alt="Profil"
          className={styles.profilePhoto}
          onClick={() => setPhotoModalOpen(true)}
          title="Cliquez pour changer la photo"
        />
        <div className={styles.profileInfo}>
          <h2 className={styles.profileName}>
            {firstName || 'Non spécifié'} {lastName || ''}
          </h2>
          <p className={styles.profileStatus}>
            Statut:{' '}
            <span className={status === 'Disponible' ? styles.available : styles.busy}>
              {status || 'Non spécifié'}
            </span>
          </p>
        </div>
      </div>
      <div className={styles.profileDetails}>
        <div className={styles.detailItem}>
          <strong>Téléphone :</strong> <span>{phone || 'Non spécifié'}</span>
        </div>
        <div className={styles.detailItem}>
          <strong>Véhicule :</strong> <span>{vehicle || 'Non spécifié'}</span>
        </div>
      </div>
      {onEdit && (
        <div className={styles.profileActions}>
          <button className={styles.editButton} onClick={() => onEdit(profile)}>
            Modifier le profil
          </button>
        </div>
      )}

      {photoModalOpen && (
        <PhotoCaptureModal onClose={handlePhotoModalClose} onCapture={handlePhotoCapture} />
      )}
    </section>
  );
};

export default DriverProfile;
