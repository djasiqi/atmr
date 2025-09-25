// src/components/CourseDetailsModal.jsx
import React from "react";
import styles from "./CourseDetailsModal.module.css";
import {
  startBooking,
  completeBooking,
  reportBookingIssue,
} from "../../services/driverService";

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
          {new Date(course.scheduled_time).toLocaleString("fr-FR")}
        </p>
        <p>
          <strong>Départ :</strong> {course.pickup_location}
        </p>
        <p>
          <strong>Destination :</strong> {course.dropoff_location}
        </p>
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
