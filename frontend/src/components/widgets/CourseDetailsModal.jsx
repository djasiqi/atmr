// src/components/CourseDetailsModal.jsx
import React from "react";
import styles from "./CourseDetailsModal.module.css";
import {
  startBooking,
  completeBooking,
  reportBookingIssue,
} from "../../services/driverService";
import { renderBookingDateTime } from "../../utils/formatDate";

const CourseDetailsModal = ({ course, onClose }) => {
  const handleStart = async () => {
    try {
      await startBooking(course.id);
      console.log("Course started successfully");
      // Vous pouvez ajouter ici un rafraîchissement de données ou une notification
      onClose(); // Ferme la modale après l'action
    } catch (error) {
      console.error("Error starting course:", error);
    }
  };

  const handleReport = async () => {
    const issue = prompt("Décrivez le problème pour cette course:");
    if (issue) {
      try {
        await reportBookingIssue(course.id, issue);
        console.log("Issue reported successfully");
        onClose(); // Ferme la modale après l'action
      } catch (error) {
        console.error("Error reporting issue:", error);
      }
    }
  };

  const handleComplete = async () => {
    try {
      await completeBooking(course.id);
      console.log("Course completed successfully");
      onClose(); // Ferme la modale après l'action
    } catch (error) {
      console.error("Error completing course:", error);
    }
  };

  return (
    <div className={styles.modal}>
      <div className={styles.modalContent}>
        <h3>Détails de la course #{course.id}</h3>
        <p>
          <strong>Client :</strong> {course.customer_name}
        </p>
        <p>
          <strong>Heure :</strong>{" "}
          {renderBookingDateTime(course)}
        </p>
        <p>
          <strong>Départ :</strong> {course.pickup_location}
        </p>
        <p>
          <strong>Destination :</strong> {course.dropoff_location}
        </p>
        
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
        {(course.medical_facility || course.doctor_name || course.hospital_service || course.notes_medical) && (
          <div className={styles.medicalInfo}>
            <p><strong>🏥 Informations médicales :</strong></p>
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
        <div className={styles.actions}>
          <button onClick={handleStart}>Démarrer</button>
          <button onClick={handleReport}>Signaler un problème</button>
          <button onClick={handleComplete}>Terminer</button>
        </div>
        <button className={styles.closeButton} onClick={onClose}>
          Fermer
        </button>
      </div>
    </div>
  );
};

export default CourseDetailsModal;
