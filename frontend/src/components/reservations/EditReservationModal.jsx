// src/components/reservations/EditReservationModal.jsx
import React, { useState, useEffect } from 'react';
import {
  FiX, FiMapPin, FiClock, FiFileText, FiUser, FiTruck, FiPackage,
} from 'react-icons/fi';
import AddressAutocomplete from '../common/AddressAutocomplete';
import { hasScheduledPickupTime } from '../../utils/bookingScheduling';
import styles from './EditReservationModal.module.css';

const STATUS_LABELS = {
  pending: 'En attente',
  accepted: 'Acceptee',
  assigned: 'Assignee',
  en_route: 'En route',
  in_progress: 'En cours',
  completed: 'Terminee',
  return_completed: 'Retour termine',
  canceled: 'Annulee',
  cancelled: 'Annulee',
  rejected: 'Refusee',
  no_show: 'Non presentee',
};

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
  const [deliveryDescription, setDeliveryDescription] = useState('');
  const [loading, setLoading] = useState(false);

  const isMaterialDelivery =
    (reservation?.mission_type || 'patient_transport') === 'material_delivery';

  useEffect(() => {
    if (isOpen && reservation) {
      const initialPickup = reservation.pickup_location;
      const initialDropoff = reservation.dropoff_location;
      setPickupLocation(initialPickup && typeof initialPickup === 'string' ? initialPickup : '');
      setDropoffLocation(initialDropoff && typeof initialDropoff === 'string' ? initialDropoff : '');
      setPickupCoords({ lat: reservation.pickup_lat || null, lon: reservation.pickup_lon || null });
      setDropoffCoords({ lat: reservation.dropoff_lat || null, lon: reservation.dropoff_lon || null });
      setMedicalFacility(String(reservation.medical_facility || ''));
      setDoctorName(String(reservation.doctor_name || ''));
      setNotesMedical(String(reservation.notes_medical || ''));
      setDeliveryDescription(String(reservation.delivery_description || ''));

      if (reservation.scheduled_time) {
        const dateObj = new Date(reservation.scheduled_time);
        const year = dateObj.getFullYear();
        const month = String(dateObj.getMonth() + 1).padStart(2, '0');
        const day = String(dateObj.getDate()).padStart(2, '0');
        setScheduledDate(`${year}-${month}-${day}`);
        if (hasScheduledPickupTime(reservation)) {
          const hours = String(dateObj.getHours()).padStart(2, '0');
          const minutes = String(dateObj.getMinutes()).padStart(2, '0');
          setScheduledTime(`${hours}:${minutes}`);
        } else {
          setScheduledTime('');
        }
      }
    }
  }, [isOpen, reservation]);

  const handlePickupAddressChange = (e) => {
    let address = '';
    if (e && typeof e === 'object' && e.target && typeof e.target === 'object') {
      address = e.target.value || '';
    } else if (typeof e === 'string') {
      address = e;
    }
    const cleanAddress = String(address || '').trim();
    setPickupLocation(cleanAddress);
    if (!cleanAddress) setPickupCoords({ lat: null, lon: null });
  };

  const handlePickupAddressSelect = (item) => {
    if (item?.lat && item?.lon) setPickupCoords({ lat: item.lat, lon: item.lon });
    let address = '';
    if (item && typeof item === 'object') address = item.label || item.address || '';
    else if (typeof item === 'string') address = item;
    const cleanAddress = String(address || '').trim();
    if (cleanAddress) setPickupLocation(cleanAddress);
  };

  const handleDropoffAddressChange = (e) => {
    let address = '';
    if (e && typeof e === 'object' && e.target && typeof e.target === 'object') {
      address = e.target.value || '';
    } else if (typeof e === 'string') {
      address = e;
    }
    const cleanAddress = String(address || '').trim();
    setDropoffLocation(cleanAddress);
    if (!cleanAddress) setDropoffCoords({ lat: null, lon: null });
  };

  const handleDropoffAddressSelect = (item) => {
    if (item?.lat && item?.lon) setDropoffCoords({ lat: item.lat, lon: item.lon });
    let address = '';
    if (item && typeof item === 'object') address = item.label || item.address || '';
    else if (typeof item === 'string') address = item;
    const cleanAddress = String(address || '').trim();
    if (cleanAddress) setDropoffLocation(cleanAddress);
  };

  const handleConfirm = async () => {
    const pickupLoc = String(pickupLocation || '').trim();
    const dropoffLoc = String(dropoffLocation || '').trim();

    if (!pickupLoc || !dropoffLoc || !scheduledDate || !scheduledTime) {
      alert('Veuillez remplir tous les champs obligatoires');
      return;
    }
    if (isMaterialDelivery && !(deliveryDescription || '').trim()) {
      alert('Veuillez saisir la description de la livraison');
      return;
    }

    setLoading(true);
    try {
      const scheduledDateTime = `${scheduledDate}T${scheduledTime}:00`;
      const updateData = {
        pickup_location: pickupLoc,
        dropoff_location: dropoffLoc,
        scheduled_time: scheduledDateTime,
      };
      if (pickupCoords.lat != null && pickupCoords.lon != null) {
        updateData.pickup_lat = Number(pickupCoords.lat);
        updateData.pickup_lon = Number(pickupCoords.lon);
      }
      if (dropoffCoords.lat != null && dropoffCoords.lon != null) {
        updateData.dropoff_lat = Number(dropoffCoords.lat);
        updateData.dropoff_lon = Number(dropoffCoords.lon);
      }
      if (medicalFacility) updateData.medical_facility = String(medicalFacility).trim();
      if (doctorName) updateData.doctor_name = String(doctorName).trim();
      if (notesMedical) updateData.notes_medical = String(notesMedical).trim();
      if (isMaterialDelivery && deliveryDescription) {
        updateData.delivery_description = String(deliveryDescription).trim();
      }
      await onConfirm(updateData);
    } catch (error) {
      console.error('Erreur lors de la mise a jour:', error);
      const errorMessage = error?.response?.data?.error || error?.message || 'Erreur lors de la mise a jour';
      alert(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  if (!isOpen) return null;

  const minDate = new Date().toISOString().split('T')[0];
  const today = new Date().toISOString().split('T')[0];
  const minTime = scheduledDate === today
    ? `${String(new Date().getHours()).padStart(2, '0')}:${String(new Date().getMinutes()).padStart(2, '0')}`
    : '00:00';

  const clientName = reservation?.client_name || reservation?.client?.full_name || '-';
  const statusRaw = reservation?.status?.toLowerCase() || 'pending';
  const statusLabel = STATUS_LABELS[statusRaw] || statusRaw;
  const isDisabled = !pickupLocation || !dropoffLocation || !scheduledDate || !scheduledTime
    || (isMaterialDelivery && !(deliveryDescription || '').trim());

  return (
    <div className={styles.overlay} onClick={(e) => e.target === e.currentTarget && onClose()}>
      <div className={styles.modal}>
        {/* Header */}
        <div className={styles.header}>
          <div className={styles.headerLeft}>
            <h2 className={styles.headerTitle}>
              Reservation #{reservation?.id}
            </h2>
            <span className={`${styles.statusBadge} ${styles[`status_${statusRaw}`] || ''}`}>
              {statusLabel}
            </span>
          </div>
          <button className={styles.closeBtn} onClick={onClose} aria-label="Fermer">
            <FiX size={18} />
          </button>
        </div>

        {/* Body */}
        <div className={styles.body}>
          {/* Context card */}
          {reservation && (
            <div className={styles.contextCard}>
              <div className={styles.contextItem}>
                <FiUser size={14} className={styles.contextIcon} />
                <span className={styles.contextLabel}>Client</span>
                <span className={styles.contextValue}>{clientName}</span>
              </div>
              {reservation.client?.institution_name && (
                <div className={styles.contextItem}>
                  <FiTruck size={14} className={styles.contextIcon} />
                  <span className={styles.contextLabel}>Institution</span>
                  <span className={styles.contextValue}>{reservation.client.institution_name}</span>
                </div>
              )}
            </div>
          )}

          {/* Section Trajet */}
          <div className={styles.section}>
            <div className={styles.sectionHeader}>
              <div className={styles.sectionIcon}><FiMapPin size={14} /></div>
              <h3 className={styles.sectionTitle}>Trajet</h3>
            </div>
            <div className={styles.fieldGroup}>
              <label className={styles.label}>Depart <span className={styles.required}>*</span></label>
              <div className={styles.inputWrap}>
                <AddressAutocomplete
                  id="pickup-location"
                  value={typeof pickupLocation === 'string' ? pickupLocation : ''}
                  onChange={handlePickupAddressChange}
                  onSelect={handlePickupAddressSelect}
                  placeholder="Adresse de prise en charge"
                  disabled={loading}
                />
              </div>
            </div>
            <div className={styles.fieldGroup}>
              <label className={styles.label}>Arrivee <span className={styles.required}>*</span></label>
              <div className={styles.inputWrap}>
                <AddressAutocomplete
                  id="dropoff-location"
                  value={typeof dropoffLocation === 'string' ? dropoffLocation : ''}
                  onChange={handleDropoffAddressChange}
                  onSelect={handleDropoffAddressSelect}
                  placeholder="Adresse de destination"
                  disabled={loading}
                />
              </div>
            </div>
          </div>

          {/* Section Horaire */}
          <div className={styles.section}>
            <div className={styles.sectionHeader}>
              <div className={styles.sectionIcon}><FiClock size={14} /></div>
              <h3 className={styles.sectionTitle}>Horaire</h3>
            </div>
            <div className={styles.fieldRow}>
              <div className={styles.fieldGroup}>
                <label className={styles.label}>Date <span className={styles.required}>*</span></label>
                <input
                  type="date"
                  className={styles.input}
                  value={scheduledDate}
                  onChange={(e) => setScheduledDate(e.target.value)}
                  min={minDate}
                  required
                  disabled={loading}
                />
              </div>
              <div className={styles.fieldGroup}>
                <label className={styles.label}>Heure <span className={styles.required}>*</span></label>
                <input
                  type="time"
                  className={styles.input}
                  value={scheduledTime}
                  onChange={(e) => setScheduledTime(e.target.value)}
                  min={scheduledDate === today ? minTime : undefined}
                  required
                  disabled={loading}
                />
              </div>
            </div>
          </div>

          {/* Section Informations medicales (patient transport) */}
          {!isMaterialDelivery && (
            <div className={styles.section}>
              <div className={styles.sectionHeader}>
                <div className={`${styles.sectionIcon} ${styles.sectionIconMuted}`}><FiFileText size={14} /></div>
                <h3 className={styles.sectionTitle}>Informations complementaires</h3>
              </div>
              <div className={styles.fieldRow}>
                <div className={styles.fieldGroup}>
                  <label className={styles.label}>Etablissement medical</label>
                  <input
                    type="text"
                    className={styles.input}
                    value={medicalFacility}
                    onChange={(e) => setMedicalFacility(e.target.value)}
                    placeholder="Hopital, clinique..."
                    disabled={loading}
                  />
                </div>
                <div className={styles.fieldGroup}>
                  <label className={styles.label}>Medecin</label>
                  <input
                    type="text"
                    className={styles.input}
                    value={doctorName}
                    onChange={(e) => setDoctorName(e.target.value)}
                    placeholder="Dr. Nom Prenom"
                    disabled={loading}
                  />
                </div>
              </div>
              <div className={styles.fieldGroup}>
                <label className={styles.label}>Instructions</label>
                <textarea
                  className={styles.textarea}
                  value={notesMedical}
                  onChange={(e) => setNotesMedical(e.target.value)}
                  placeholder="Batiment, etage, instructions particulieres..."
                  rows={2}
                  disabled={loading}
                />
              </div>
            </div>
          )}

          {/* Section Livraison materiel */}
          {isMaterialDelivery && (
            <div className={styles.section}>
              <div className={styles.sectionHeader}>
                <div className={styles.sectionIcon}><FiPackage size={14} /></div>
                <h3 className={styles.sectionTitle}>Livraison</h3>
              </div>
              <div className={styles.fieldGroup}>
                <label className={styles.label}>Description <span className={styles.required}>*</span></label>
                <input
                  type="text"
                  className={styles.input}
                  value={deliveryDescription}
                  onChange={(e) => setDeliveryDescription(e.target.value)}
                  placeholder="Medicament, oxygene, documents..."
                  required
                  disabled={loading}
                />
                <span className={styles.hint}>Requis pour la facturation des livraisons</span>
              </div>
              <div className={styles.fieldGroup}>
                <label className={styles.label}>Instructions</label>
                <textarea
                  className={styles.textarea}
                  value={notesMedical}
                  onChange={(e) => setNotesMedical(e.target.value)}
                  placeholder="Instructions particulieres..."
                  rows={2}
                  disabled={loading}
                />
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className={styles.footer}>
          <button
            type="button"
            className={styles.btnCancel}
            onClick={onClose}
            disabled={loading}
          >
            Annuler
          </button>
          <button
            type="button"
            className={styles.btnSave}
            onClick={handleConfirm}
            disabled={isDisabled || loading}
          >
            {loading ? 'Enregistrement...' : 'Enregistrer'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default EditReservationModal;
