// src/pages/DriverSettingsPage.jsx
import React, { useState } from 'react';
import HeaderDashboard from '../../../components/layout/Header/HeaderDashboard';
import DriverSidebar from '../../../components/layout/Sidebar/DriverSidebar/DriverSidebar';
import styles from '../Dashboard/DriverDashboard.module.css';

const DriverSettingsPage = () => {
  const [settings, setSettings] = useState({
    notifications: true,
    darkMode: false,
    language: 'fr',
  });

  const handleChange = (e) => {
    const { name, type, value, checked } = e.target;
    setSettings({
      ...settings,
      [name]: type === 'checkbox' ? checked : value,
    });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    // Sauvegarder les paramètres via API ou dans l'état local
    console.log('Settings saved:', settings);
  };

  return (
    <div className={styles.driverDashboard}>
      <HeaderDashboard />
      <DriverSidebar />
      <main className={styles.mainContent}>
        <h1>Paramètres</h1>
        <form onSubmit={handleSubmit}>
          <div>
            <label>
              Notifications:
              <input
                type="checkbox"
                name="notifications"
                checked={settings.notifications}
                onChange={handleChange}
              />
            </label>
          </div>
          <div>
            <label>
              Mode sombre:
              <input
                type="checkbox"
                name="darkMode"
                checked={settings.darkMode}
                onChange={handleChange}
              />
            </label>
          </div>
          <div>
            <label>
              Langue:
              <select name="language" value={settings.language} onChange={handleChange}>
                <option value="fr">Français</option>
                <option value="en">English</option>
                <option value="de">Deutsch</option>
              </select>
            </label>
          </div>
          <button type="submit">Enregistrer</button>
        </form>
      </main>
    </div>
  );
};

export default DriverSettingsPage;
