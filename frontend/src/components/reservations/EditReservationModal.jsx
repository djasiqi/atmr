// src/components/reservations/EditReservationModal.jsx
import React, { useState, useEffect } from 'react';
import Modal from '../common/Modal';
import Button from '../common/Button';
import AddressAutocomplete from '../common/AddressAutocomplete';
import styles from './EditReservationModal.module.css';

/**
 * Modal pour éditer une réservation existante
 * Permet de modifier : adresses, heure, informations médicales, notes
 */
const EditReservationModal = ({ isOpen, onClose, reservation, onConfirm }) => {
  const [pickupLocation, setPickupLocation] = useState('');
  const [dropoffLocation, setDropoffLocation] = useState('');
  const [pickupCoords, setPickupCoords] = useState({ lat: null, lon: null });
  const [dropoffCoords, setDropoffCoords] = useState({ lat: null, lon: null });
  const [scheduledDate, setScheduledDate] = useState('');
  const [scheduledTime, setScheduledTime] = useState('');
  const [medicalFacility, setMedicalFacility] = useState('');
  const [doctorName, setDoctorName] = useState('');
  const [notesMedical, setNotesMedical] = useState('');
  const [loading, setLoading] = useState(false);

  // Initialiser les valeurs depuis la réservation
  useEffect(() => {
    if (isOpen && reservation) {
      // S'assurer que les valeurs initiales sont toujours des chaînes valides
      const initialPickup = reservation.pickup_location;
      const initialDropoff = reservation.dropoff_location;
      setPickupLocation(initialPickup && typeof initialPickup === 'string' ? initialPickup : '');
      setDropoffLocation(
        initialDropoff && typeof initialDropoff === 'string' ? initialDropoff : ''
      );
      setPickupCoords({
        lat: reservation.pickup_lat || null,
        lon: reservation.pickup_lon || null,
      });
      setDropoffCoords({
        lat: reservation.dropoff_lat || null,
        lon: reservation.dropoff_lon || null,
      });
      setMedicalFacility(String(reservation.medical_facility || ''));
      setDoctorName(String(reservation.doctor_name || ''));
      setNotesMedical(String(reservation.notes_medical || ''));

      // Parser la date/heure
      if (reservation.scheduled_time) {
        const dateObj = new Date(reservation.scheduled_time);
        const year = dateObj.getFullYear();
        const month = String(dateObj.getMonth() + 1).padStart(2, '0');
        const day = String(dateObj.getDate()).padStart(2, '0');
        setScheduledDate(`${year}-${month}-${day}`);
        const hours = String(dateObj.getHours()).padStart(2, '0');
        const minutes = String(dateObj.getMinutes()).padStart(2, '0');
        setScheduledTime(`${hours}:${minutes}`);
      }
    }
  }, [isOpen, reservation]);

  const handlePickupAddressChange = (e) => {
    // AddressAutocomplete passe { target: { name, value } }
    // Extraire la valeur de manière sûre
    let address = '';
    if (e && typeof e === 'object' && e.target && typeof e.target === 'object') {
      address = e.target.value || '';
    } else if (typeof e === 'string') {
      address = e;
    }
    // Toujours convertir en chaîne et nettoyer
    const cleanAddress = String(address || '').trim();
    setPickupLocation(cleanAddress);
    // Réinitialiser les coordonnées si l'adresse est vidée
    if (!cleanAddress) {
      setPickupCoords({ lat: null, lon: null });
    }
  };

  const handlePickupAddressSelect = (item) => {
    // onSelect reçoit l'item complet avec coordonnées
    if (item?.lat && item?.lon) {
      setPickupCoords({ lat: item.lat, lon: item.lon });
    }
    // Extraire l'adresse de manière sûre
    let address = '';
    if (item && typeof item === 'object') {
      address = item.label || item.address || '';
    } else if (typeof item === 'string') {
      address = item;
    }
    const cleanAddress = String(address || '').trim();
    if (cleanAddress) {
      setPickupLocation(cleanAddress);
    }
  };

  const handleDropoffAddressChange = (e) => {
    // AddressAutocomplete passe { target: { name, value } }
    // Extraire la valeur de manière sûre
    let address = '';
    if (e && typeof e === 'object' && e.target && typeof e.target === 'object') {
      address = e.target.value || '';
    } else if (typeof e === 'string') {
      address = e;
    }
    // Toujours convertir en chaîne et nettoyer
    const cleanAddress = String(address || '').trim();
    setDropoffLocation(cleanAddress);
    // Réinitialiser les coordonnées si l'adresse est vidée
    if (!cleanAddress) {
      setDropoffCoords({ lat: null, lon: null });
    }
  };

  const handleDropoffAddressSelect = (item) => {
    // onSelect reçoit l'item complet avec coordonnées
    if (item?.lat && item?.lon) {
      setDropoffCoords({ lat: item.lat, lon: item.lon });
    }
    // Extraire l'adresse de manière sûre
    let address = '';
    if (item && typeof item === 'object') {
      address = item.label || item.address || '';
    } else if (typeof item === 'string') {
      address = item;
    }
    const cleanAddress = String(address || '').trim();
    if (cleanAddress) {
      setDropoffLocation(cleanAddress);
    }
  };

  const handleConfirm = async () => {
    // S'assurer que les adresses sont des chaînes valides
    const pickupLoc = String(pickupLocation || '').trim();
    const dropoffLoc = String(dropoffLocation || '').trim();

    if (!pickupLoc || !dropoffLoc || !scheduledDate || !scheduledTime) {
      alert('Veuillez remplir tous les champs obligatoires');
      return;
    }

    setLoading(true);
    try {
      // Format datetime ISO local (sans Z)
      const scheduledDateTime = `${scheduledDate}T${scheduledTime}:00`;

      const updateData = {
        pickup_location: pickupLoc,
        dropoff_location: dropoffLoc,
        scheduled_time: scheduledDateTime,
      };

      // Ajouter les coordonnées si disponibles (convertir en nombres)
      if (pickupCoords.lat != null && pickupCoords.lon != null) {
        updateData.pickup_lat = Number(pickupCoords.lat);
        updateData.pickup_lon = Number(pickupCoords.lon);
      }
      if (dropoffCoords.lat != null && dropoffCoords.lon != null) {
        updateData.dropoff_lat = Number(dropoffCoords.lat);
        updateData.dropoff_lon = Number(dropoffCoords.lon);
      }

      // Ajouter les informations médicales si fournies (convertir en chaînes)
      if (medicalFacility) updateData.medical_facility = String(medicalFacility).trim();
      if (doctorName) updateData.doctor_name = String(doctorName).trim();
      if (notesMedical) updateData.notes_medical = String(notesMedical).trim();

      await onConfirm(updateData);
    } catch (error) {
      console.error('Erreur lors de la mise à jour:', error);
      const errorMessage =
        error?.response?.data?.error || error?.message || 'Erreur lors de la mise à jour';
      alert(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  if (!isOpen) return null;

  const minDate = new Date().toISOString().split('T')[0];
  const today = new Date().toISOString().split('T')[0];
  const minTime =
    scheduledDate === today
      ? `${String(new Date().getHours()).padStart(2, '0')}:${String(
          new Date().getMinutes()
        ).padStart(2, '0')}`
      : '00:00';

  return (
    <Modal onClose={onClose}>
      <div className={styles.modalWrapper}>
        <h3>Éditer la réservation #{reservation?.id}</h3>

        {reservation && (
          <div className={styles.reservationInfo}>
            <div className={styles.infoRow}>
              <span className={styles.label}>Client :</span>
              <strong>{reservation.customer_name || reservation.client?.full_name}</strong>
            </div>
            <div className={styles.infoRow}>
              <span className={styles.label}>Statut :</span>
              <span>{reservation.status}</span>
            </div>
          </div>
        )}

        <div className={styles.formGroup}>
          <label htmlFor="pickup-location" className={styles.label}>
            Adresse de départ <span>*</span>
          </label>
          <AddressAutocomplete
            id="pickup-location"
            value={typeof pickupLocation === 'string' ? pickupLocation : ''}
            onChange={handlePickupAddressChange}
            onSelect={handlePickupAddressSelect}
            placeholder="Adresse de prise en charge"
            disabled={loading}
          />
        </div>

        <div className={styles.formGroup}>
          <label htmlFor="dropoff-location" className={styles.label}>
            Adresse d'arrivée <span>*</span>
          </label>
          <AddressAutocomplete
            id="dropoff-location"
            value={typeof dropoffLocation === 'string' ? dropoffLocation : ''}
            onChange={handleDropoffAddressChange}
            onSelect={handleDropoffAddressSelect}
            placeholder="Adresse de destination"
            disabled={loading}
          />
        </div>

        <div className={styles.formGroup}>
          <label htmlFor="scheduled-date" className={styles.label}>
            Date <span>*</span>
          </label>
          <input
            type="date"
            id="scheduled-date"
            className={styles.input}
            value={scheduledDate}
            onChange={(e) => setScheduledDate(e.target.value)}
            min={minDate}
            required
            disabled={loading}
          />
        </div>

        <div className={styles.formGroup}>
          <label htmlFor="scheduled-time" className={styles.label}>
            Heure <span>*</span>
          </label>
          <input
            type="time"
            id="scheduled-time"
            className={styles.input}
            value={scheduledTime}
            onChange={(e) => setScheduledTime(e.target.value)}
            min={scheduledDate === today ? minTime : undefined}
            required
            disabled={loading}
          />
          <small className={styles.hint}>Format 24h (ex: 14:30)</small>
        </div>

        <div className={styles.formGroup}>
          <label htmlFor="medical-facility" className={styles.label}>
            Établissement médical
          </label>
          <input
            type="text"
            id="medical-facility"
            className={styles.input}
            value={medicalFacility}
            onChange={(e) => setMedicalFacility(e.target.value)}
            placeholder="Hôpital, clinique, etc."
            disabled={loading}
          />
        </div>

        <div className={styles.formGroup}>
          <label htmlFor="doctor-name" className={styles.label}>
            Nom du médecin
          </label>
          <input
            type="text"
            id="doctor-name"
            className={styles.input}
            value={doctorName}
            onChange={(e) => setDoctorName(e.target.value)}
            placeholder="Dr. Nom Prénom"
            disabled={loading}
          />
        </div>

        <div className={styles.formGroup}>
          <label htmlFor="notes-medical" className={styles.label}>
            Notes médicales
          </label>
          <textarea
            id="notes-medical"
            className={styles.textarea}
            value={notesMedical}
            onChange={(e) => setNotesMedical(e.target.value)}
            placeholder="Instructions particulières, bâtiment, étage…"
            rows={3}
            disabled={loading}
          />
        </div>

        <div className={styles.buttonGroup}>
          <Button variant="secondary" onClick={onClose} disabled={loading}>
            Annuler
          </Button>
          <Button
            variant="primary"
            onClick={handleConfirm}
            loading={loading}
            disabled={!pickupLocation || !dropoffLocation || !scheduledDate || !scheduledTime}
          >
            Enregistrer les modifications
          </Button>
        </div>
      </div>
    </Modal>
  );
};

export default EditReservationModal;
