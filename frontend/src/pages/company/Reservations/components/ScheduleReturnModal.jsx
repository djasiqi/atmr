import React, { useState } from "react";
import styles from "./ScheduleReturnModal.module.css";

const ScheduleReturnModal = ({ isOpen, onClose, reservation, onConfirm }) => {
  const [date, setDate] = useState("");
  const [time, setTime] = useState("");
  const [loading, setLoading] = useState(false);

  React.useEffect(() => {
    if (isOpen && reservation) {
      // Si le retour a déjà une heure définie, la pré-remplir
      if (reservation.scheduled_time) {
        const scheduledDate = new Date(reservation.scheduled_time);
        const year = scheduledDate.getFullYear();
        const month = String(scheduledDate.getMonth() + 1).padStart(2, "0");
        const day = String(scheduledDate.getDate()).padStart(2, "0");
        setDate(`${year}-${month}-${day}`);

        // Pré-remplir l'heure existante
        const hours = scheduledDate.getHours();
        const minutes = scheduledDate.getMinutes();
        const hh = String(hours).padStart(2, "0");
        const mm = String(minutes).padStart(2, "0");
        setTime(`${hh}:${mm}`);
      } else {
        // Pas de scheduled_time : utiliser la date du jour et suggérer 18:00
        const today = new Date();
        const year = today.getFullYear();
        const month = String(today.getMonth() + 1).padStart(2, "0");
        const day = String(today.getDate()).padStart(2, "0");
        setDate(`${year}-${month}-${day}`);
        setTime("18:00");
      }
    }
  }, [isOpen, reservation]);

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!date || !time) {
      alert("Veuillez renseigner la date et l'heure");
      return;
    }

    setLoading(true);
    try {
      const isoDatetime = `${date} ${time}`;
      await onConfirm(isoDatetime);
      onClose();
    } catch (error) {
      console.error("Erreur lors de la planification:", error);
      alert(error?.response?.data?.error || "Erreur lors de la planification");
    } finally {
      setLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className={styles.overlay} onClick={onClose}>
      <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
        <div className={styles.header}>
          <h2 className={styles.title}>🕐 Planifier l'heure de retour</h2>
          <button
            onClick={onClose}
            className={styles.closeButton}
            disabled={loading}
          >
            ✕
          </button>
        </div>

        <div className={styles.body}>
          {reservation && (
            <div className={styles.reservationInfo}>
              <div className={styles.infoRow}>
                <span className={styles.label}>Client :</span>
                <strong>{reservation.customer_name}</strong>
              </div>
              <div className={styles.infoRow}>
                <span className={styles.label}>Aller :</span>
                <span>
                  {reservation.scheduled_time
                    ? new Date(reservation.scheduled_time).toLocaleString(
                        "fr-CH"
                      )
                    : "Non spécifié"}
                </span>
              </div>
              <div className={styles.infoRow}>
                <span className={styles.label}>Trajet retour :</span>
                <span>
                  {reservation.pickup_location} → {reservation.dropoff_location}
                </span>
              </div>
            </div>
          )}

          <form onSubmit={handleSubmit} className={styles.form}>
            <div className={styles.formGroup}>
              <label htmlFor="return-date" className={styles.label}>
                Date du retour *
              </label>
              <input
                type="date"
                id="return-date"
                value={date}
                onChange={(e) => setDate(e.target.value)}
                className={styles.input}
                required
                disabled={loading}
              />
            </div>

            <div className={styles.formGroup}>
              <label htmlFor="return-time" className={styles.label}>
                Heure du retour *
              </label>
              <input
                type="time"
                id="return-time"
                value={time}
                onChange={(e) => setTime(e.target.value)}
                className={styles.input}
                required
                disabled={loading}
              />
              <small className={styles.hint}>
                ℹ️ Heure approximative. Vous pourrez ajuster si nécessaire.
              </small>
            </div>

            <div className={styles.actions}>
              <button
                type="button"
                onClick={onClose}
                className={styles.cancelButton}
                disabled={loading}
              >
                Annuler
              </button>
              <button
                type="submit"
                className={styles.confirmButton}
                disabled={loading}
              >
                {loading ? "Planification..." : "Confirmer l'heure"}
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};

export default ScheduleReturnModal;
