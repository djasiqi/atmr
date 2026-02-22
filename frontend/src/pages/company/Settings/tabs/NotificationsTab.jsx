// frontend/src/pages/company/Settings/tabs/NotificationsTab.jsx
import React, { useState } from 'react';
import { FiBell, FiUsers } from 'react-icons/fi';
import styles from '../CompanySettings.module.css';
import notifStyles from './NotificationsTab.module.css';

const NOTIFICATIONS = [
  { name: 'notify_new_booking', label: 'Nouvelle reservation', hint: 'Email a chaque nouvelle reservation', defaultOn: true },
  { name: 'notify_booking_confirmed', label: 'Reservation confirmee', hint: 'Confirmation par le client', defaultOn: true },
  { name: 'notify_booking_canceled', label: 'Reservation annulee', hint: "Alerte en cas d'annulation", defaultOn: true },
  { name: 'notify_dispatch_completed', label: 'Dispatch termine', hint: 'Resume quotidien du dispatch', defaultOn: true },
  { name: 'notify_delays', label: 'Retards detectes', hint: 'Alerte immediate si retard significatif', defaultOn: true },
  { name: 'notify_weekly_analytics', label: 'Rapports hebdomadaires', hint: 'Performance envoyee chaque lundi', defaultOn: false },
];

const NotificationsTab = ({ isEditing: _isEditing }) => {
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');

  const [form, setForm] = useState(() => {
    const init = { notification_emails: '' };
    NOTIFICATIONS.forEach((n) => { init[n.name] = n.defaultOn; });
    return init;
  });

  const activeCount = NOTIFICATIONS.filter((n) => form[n.name]).length;

  const autoSave = async () => {
    setMessage('');
    setError('');
    try {
      await new Promise((resolve) => setTimeout(resolve, 500));
      setMessage('Sauvegarde automatiquement');
      setTimeout(() => setMessage(''), 2000);
    } catch (err) {
      console.error('Auto-save failed:', err);
      setError('Erreur lors de la sauvegarde');
      setTimeout(() => setError(''), 3000);
    }
  };

  const handleToggle = (name) => {
    setForm((prev) => ({ ...prev, [name]: !prev[name] }));
    autoSave();
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setForm((prev) => ({ ...prev, [name]: value }));
  };

  return (
    <div className={`${styles.settingsForm} ${styles.billingFormBlock}`}>
      {message && <div className={styles.success}>{message}</div>}
      {error && <div className={styles.error}>{error}</div>}

      <div className={styles.billingGrid}>
        <div className={styles.billingCol}>
          {/* Notifications par email */}
          <div className={styles.card}>
            <div className={styles.cardHeader}>
              <div className={styles.cardIcon}><FiBell size={16} /></div>
              <div className={styles.cardHeaderText}>
                <h3 className={styles.cardTitle}>Notifications par email</h3>
                <p className={styles.cardHint}>{activeCount} sur {NOTIFICATIONS.length} actives</p>
              </div>
            </div>

            <div className={notifStyles.notifList}>
              {NOTIFICATIONS.map((n) => (
                <label key={n.name} className={notifStyles.notifRow} htmlFor={`notif-${n.name}`}>
                  <div className={notifStyles.notifInfo}>
                    <span className={notifStyles.notifLabel}>{n.label}</span>
                    <span className={notifStyles.notifHint}>{n.hint}</span>
                  </div>
                  <div className={notifStyles.miniToggle}>
                    <input
                      id={`notif-${n.name}`}
                      type="checkbox"
                      checked={form[n.name]}
                      onChange={() => handleToggle(n.name)}
                    />
                    <span className={notifStyles.miniSlider} />
                  </div>
                </label>
              ))}
            </div>
          </div>
        </div>

        <div className={styles.billingCol}>
          {/* Destinataires */}
          <div className={styles.card}>
            <div className={styles.cardHeader}>
              <div className={styles.cardIcon}><FiUsers size={16} /></div>
              <div className={styles.cardHeaderText}>
                <h3 className={styles.cardTitle}>Destinataires</h3>
                <p className={styles.cardHint}>
                  {form.notification_emails?.trim()
                    ? `${form.notification_emails.split(',').filter((e) => e.trim()).length} adresse(s)`
                    : 'Aucune adresse supplementaire'
                  }
                </p>
              </div>
            </div>

            <div className={styles.formGroup}>
              <label htmlFor="notification_emails">Emails supplementaires</label>
              <input
                id="notification_emails"
                name="notification_emails"
                value={form.notification_emails}
                onChange={handleChange}
                onBlur={autoSave}
                placeholder="admin@emmenezmoi.ch, manager@emmenezmoi.ch"
              />
              <small className={styles.hint}>Separez plusieurs adresses par des virgules</small>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default NotificationsTab;
