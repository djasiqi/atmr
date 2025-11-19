// src/components/CourseDetailsModal.jsx
import React from 'react';
import styles from './CourseDetailsModal.module.css';
import { startBooking, completeBooking, reportBookingIssue } from '../../services/driverService';
import { renderBookingDateTime } from '../../utils/formatDate';

const CourseDetailsModal = ({ course, onClose }) => {
  const handleStart = async () => {
    try {
      await startBooking(course.id);
      // Vous pouvez ajouter ici un rafraîchissement de données ou une notification
      onClose(); // Ferme la modale après l'action
    } catch (error) {
      // eslint-disable-next-line no-console
      console.error('Error starting course:', error);
    }
  };

  const handleReport = async () => {
    const issue = prompt('Décrivez le problème pour cette course:');
    if (issue) {
      try {
        await reportBookingIssue(course.id, issue);
        onClose(); // Ferme la modale après l'action
      } catch (error) {
        // eslint-disable-next-line no-console
        console.error('Error reporting issue:', error);
      }
    }
  };

  const handleComplete = async () => {
    try {
      await completeBooking(course.id);
      onClose(); // Ferme la modale après l'action
    } catch (error) {
      // eslint-disable-next-line no-console
      console.error('Error completing course:', error);
    }
  };

  return (
    <div className="modal-overlay">
      <div className="modal-content modal-md">
        <div className="modal-header">
          <h3 className="modal-title">Détails de la course #{course.id}</h3>
          <button className="modal-close" onClick={onClose}>
            ✕
          </button>
        </div>

        <div className="modal-body">
          <div className="flex-col gap-sm mb-lg">
            <p>
              <strong>Client :</strong> {course.customer_name}
            </p>
            <p>
              <strong>Heure :</strong> {renderBookingDateTime(course)}
            </p>
            <p>
              <strong>Départ :</strong> {course.pickup_location}
            </p>
            <p>
              <strong>Destination :</strong> {course.dropoff_location}
            </p>
          </div>

          {/* Informations chaise roulante */}
          {(course.wheelchair_client_has || course.wheelchair_need) && (
            <div className={styles.wheelchairInfo}>
              {course.wheelchair_client_has && (
                <p className={styles.wheelchairBadge}>
                  ♿ <strong>Client en chaise roulante</strong>
                </p>
              )}
              {course.wheelchair_need && (
                <p className={styles.wheelchairBadge}>
                  🏥 <strong>Prendre une chaise roulante</strong>
                </p>
              )}
            </div>
          )}

          {/* Informations médicales */}
          {(course.medical_facility ||
            course.doctor_name ||
            course.hospital_service ||
            course.notes_medical) && (
            <div className={styles.medicalInfo}>
              <p>
                <strong>🏥 Informations médicales :</strong>
              </p>
              {course.medical_facility && (
                <p className={styles.medicalDetail}>📍 {course.medical_facility}</p>
              )}
              {course.doctor_name && (
                <p className={styles.medicalDetail}>👨‍⚕️ Dr {course.doctor_name}</p>
              )}
              {course.hospital_service && (
                <p className={styles.medicalDetail}>🚪 {course.hospital_service}</p>
              )}
              {course.notes_medical && (
                <p className={styles.medicalDetail}>📝 {course.notes_medical}</p>
              )}
            </div>
          )}

          {course.instructions && (
            <p>
              <strong>Instructions :</strong> {course.instructions}
            </p>
          )}

          <div className="flex gap-sm mt-lg">
            <button className="btn btn-primary btn-sm" onClick={handleStart}>
              ▶️ Démarrer
            </button>
            <button className="btn btn-warning btn-sm" onClick={handleReport}>
              ⚠️ Signaler un problème
            </button>
            <button className="btn btn-success btn-sm" onClick={handleComplete}>
              ✅ Terminer
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CourseDetailsModal;
