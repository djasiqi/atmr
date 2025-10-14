// frontend/src/pages/company/Settings/tabs/NotificationsTab.jsx
import React, { useState } from "react";
import styles from "../CompanySettings.module.css";
import ToggleField from "../../../../components/ui/ToggleField";

const NotificationsTab = () => {
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  const [form, setForm] = useState({
    // Notifications email
    notify_new_booking: true,
    notify_booking_confirmed: true,
    notify_booking_canceled: true,
    notify_dispatch_completed: true,
    notify_delays: true,
    notify_weekly_analytics: false,

    // Destinataires
    notification_emails: "",
  });

  const handleToggle = (e) => {
    const { name, checked } = e.target;
    setForm((prev) => ({
      ...prev,
      [name]: checked,
    }));
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setForm((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setMessage("");
    setError("");
    setSaving(true);

    try {
      // TODO: API call pour sauvegarder les notifications
      // await updateNotificationSettings(form);

      // Simulation temporaire
      await new Promise((resolve) => setTimeout(resolve, 1000));

      setMessage("✅ Paramètres de notifications enregistrés avec succès.");
    } catch (err) {
      console.error("Failed to update notification settings:", err);
      setError("Erreur lors de la sauvegarde.");
    } finally {
      setSaving(false);
    }
  };

  return (
    <form className={styles.settingsForm} onSubmit={handleSubmit}>
      {message && <div className={styles.success}>{message}</div>}
      {error && <div className={styles.error}>{error}</div>}

      {/* Notifications par email */}
      <section className={styles.section}>
        <h2>📧 Notifications par email</h2>

        <ToggleField
          label="Nouvelle réservation"
          name="notify_new_booking"
          value={form.notify_new_booking}
          onChange={handleToggle}
          hint="Recevoir un email à chaque nouvelle réservation"
        />

        <ToggleField
          label="Réservation confirmée"
          name="notify_booking_confirmed"
          value={form.notify_booking_confirmed}
          onChange={handleToggle}
          hint="Notification quand une réservation est confirmée par le client"
        />

        <ToggleField
          label="Réservation annulée"
          name="notify_booking_canceled"
          value={form.notify_booking_canceled}
          onChange={handleToggle}
          hint="Alerte en cas d'annulation de réservation"
        />

        <ToggleField
          label="Dispatch terminé"
          name="notify_dispatch_completed"
          value={form.notify_dispatch_completed}
          onChange={handleToggle}
          hint="Email quotidien avec résumé du dispatch"
        />

        <ToggleField
          label="Retards détectés"
          name="notify_delays"
          value={form.notify_delays}
          onChange={handleToggle}
          hint="Alerte immédiate en cas de retard significatif"
        />

        <ToggleField
          label="Rapports Analytics hebdomadaires"
          name="notify_weekly_analytics"
          value={form.notify_weekly_analytics}
          onChange={handleToggle}
          hint="Résumé de performance envoyé chaque lundi"
        />
      </section>

      {/* Destinataires */}
      <section className={styles.section}>
        <h2>👥 Destinataires des notifications</h2>

        <div className={styles.formGroup}>
          <label htmlFor="notification_emails">Emails supplémentaires</label>
          <input
            id="notification_emails"
            name="notification_emails"
            value={form.notification_emails}
            onChange={handleChange}
            placeholder="admin@emmenezmoi.ch, manager@emmenezmoi.ch"
          />
          <small className={styles.hint}>
            Séparez plusieurs adresses par des virgules
          </small>
        </div>
      </section>

      {/* Boutons */}
      <div className={styles.actionsRow}>
        <button
          type="submit"
          className={`${styles.button} ${styles.primary}`}
          disabled={saving}
        >
          {saving ? "💾 Enregistrement…" : "💾 Enregistrer"}
        </button>
      </div>
    </form>
  );
};

export default NotificationsTab;
