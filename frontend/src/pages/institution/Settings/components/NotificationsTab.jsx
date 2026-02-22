// pages/institution/Settings/components/NotificationsTab.jsx
/**
 * Onglet Notifications pour les paramètres institution.
 *
 * Sections:
 * - Emails de notification (multi-email input)
 * - Toggles événements (request_sent, offer_accepted, request_expired)
 */

import React, { useState, useEffect } from 'react';
import { FaSave, FaPlus, FaTimes } from 'react-icons/fa';
import { toast } from 'sonner';
import { useInstitutionSettings, useUpdateInstitutionSettings } from '../../../../hooks/useInstitutionData';
import { useInstitutionMe } from '../../../../hooks/useInstitutionData';
import { isAdmin } from '../../../../utils/institutionPermissions';
import styles from '../InstitutionSettings.module.css';

const NotificationsTab = () => {
  const { data: meData } = useInstitutionMe();
  const { data, isLoading, isError } = useInstitutionSettings();
  const updateMutation = useUpdateInstitutionSettings();

  const canEdit = isAdmin(meData?.institution_role);

  const [emails, setEmails] = useState([]);
  const [newEmail, setNewEmail] = useState('');
  const [notifyRequestSent, setNotifyRequestSent] = useState(true);
  const [notifyOfferAccepted, setNotifyOfferAccepted] = useState(true);
  const [notifyRequestExpired, setNotifyRequestExpired] = useState(true);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    if (data && !loaded) {
      const settings = data.settings || {};
      // Normaliser les emails existants (trim + lowercase)
      const rawEmails = settings.notification_emails || [];
      setEmails(rawEmails.map(e => e.trim().toLowerCase()).filter(Boolean));
      setNotifyRequestSent(settings.notify_request_sent ?? true);
      setNotifyOfferAccepted(settings.notify_offer_accepted ?? true);
      setNotifyRequestExpired(settings.notify_request_expired ?? true);
      setLoaded(true);
    }
  }, [data, loaded]);

  const addEmail = () => {
    const email = newEmail.trim().toLowerCase();
    if (!email) return;

    // Validation basique
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
      toast.error('Adresse email invalide');
      return;
    }

    // Comparaison case-insensitive pour éviter doublons
    if (emails.some(e => e.toLowerCase() === email)) {
      toast.error('Email déjà ajouté');
      return;
    }

    setEmails(prev => [...prev, email]);
    setNewEmail('');
  };

  const removeEmail = (emailToRemove) => {
    setEmails(prev => prev.filter(e => e !== emailToRemove));
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      addEmail();
    }
  };

  // Dirty state: compare current state vs server data
  const isDirty = (() => {
    if (!data) return false;
    const settings = data.settings || {};
    const currentEmails = settings.notification_emails || [];
    const emailsChanged = JSON.stringify([...emails].sort()) !== JSON.stringify([...currentEmails].sort());
    return (
      emailsChanged ||
      notifyRequestSent !== (settings.notify_request_sent ?? true) ||
      notifyOfferAccepted !== (settings.notify_offer_accepted ?? true) ||
      notifyRequestExpired !== (settings.notify_request_expired ?? true)
    );
  })();

  const handleSave = async () => {
    const settings = data?.settings || {};
    const payload = {};

    // Compare emails
    const currentEmails = settings.notification_emails || [];
    const emailsChanged = JSON.stringify(emails.sort()) !== JSON.stringify([...currentEmails].sort());
    if (emailsChanged) payload.notification_emails = emails;

    // Compare toggles
    if (notifyRequestSent !== (settings.notify_request_sent ?? true))
      payload.notify_request_sent = notifyRequestSent;
    if (notifyOfferAccepted !== (settings.notify_offer_accepted ?? true))
      payload.notify_offer_accepted = notifyOfferAccepted;
    if (notifyRequestExpired !== (settings.notify_request_expired ?? true))
      payload.notify_request_expired = notifyRequestExpired;

    if (Object.keys(payload).length === 0) {
      toast.info('Aucune modification');
      return;
    }

    try {
      const result = await updateMutation.mutateAsync(payload);
      if (result?.settings) {
        setEmails(result.settings.notification_emails || []);
        setNotifyRequestSent(result.settings.notify_request_sent ?? true);
        setNotifyOfferAccepted(result.settings.notify_offer_accepted ?? true);
        setNotifyRequestExpired(result.settings.notify_request_expired ?? true);
      }
      toast.success('Notifications mises à jour');
    } catch (err) {
      toast.error(err?.response?.data?.error || 'Erreur lors de la sauvegarde');
    }
  };

  if (isLoading) {
    return (
      <div className={styles.section}>
        <p>Chargement des paramètres...</p>
      </div>
    );
  }

  if (isError) {
    return (
      <div className={styles.section}>
        <p style={{ color: '#c62828' }}>Erreur lors du chargement des paramètres.</p>
      </div>
    );
  }

  return (
    <div className={styles.section}>
      {/* Emails de notification */}
      <div className={styles.sectionHeader}>
        <h3>Emails de notification</h3>
        <p>Adresses qui recevront les notifications de transport</p>
      </div>

      <div className={styles.profileForm}>
        {/* Email list */}
        <div className={styles.emailTags}>
          {emails.map((email) => (
            <span key={email} className={styles.emailTag}>
              {email}
              {canEdit && (
                <button
                  className={styles.emailTagRemove}
                  onClick={() => removeEmail(email)}
                  title="Supprimer"
                >
                  <FaTimes />
                </button>
              )}
            </span>
          ))}
          {emails.length === 0 && (
            <span className={styles.emptyEmails}>Aucune adresse configurée</span>
          )}
        </div>

        {/* Add email */}
        {canEdit && (
          <div className={styles.addEmailRow}>
            <input
              type="email"
              value={newEmail}
              onChange={(e) => setNewEmail(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="ajouter@email.ch"
              className={styles.addEmailInput}
            />
            <button
              className={styles.addEmailBtn}
              onClick={addEmail}
              disabled={!newEmail.trim()}
            >
              <FaPlus /> Ajouter
            </button>
          </div>
        )}
      </div>

      {/* Toggles notifications */}
      <div className={styles.sectionHeader} style={{ marginTop: 32 }}>
        <h3>Événements notifiés</h3>
        <p>Choisissez les événements pour lesquels vous souhaitez recevoir une notification</p>
      </div>

      <div className={styles.toggleList}>
        <label className={styles.toggleItem}>
          <input
            type="checkbox"
            checked={notifyRequestSent}
            onChange={(e) => setNotifyRequestSent(e.target.checked)}
            disabled={!canEdit}
          />
          <div>
            <span className={styles.toggleLabel}>Demande envoyée</span>
            <span className={styles.toggleDesc}>
              Notification quand une demande de transport est envoyée aux entreprises
            </span>
          </div>
        </label>

        <label className={styles.toggleItem}>
          <input
            type="checkbox"
            checked={notifyOfferAccepted}
            onChange={(e) => setNotifyOfferAccepted(e.target.checked)}
            disabled={!canEdit}
          />
          <div>
            <span className={styles.toggleLabel}>Offre acceptée</span>
            <span className={styles.toggleDesc}>
              Notification quand une entreprise de transport accepte une demande
            </span>
          </div>
        </label>

        <label className={styles.toggleItem}>
          <input
            type="checkbox"
            checked={notifyRequestExpired}
            onChange={(e) => setNotifyRequestExpired(e.target.checked)}
            disabled={!canEdit}
          />
          <div>
            <span className={styles.toggleLabel}>Demande expirée</span>
            <span className={styles.toggleDesc}>
              Notification quand aucune entreprise n'a accepté dans le délai imparti
            </span>
          </div>
        </label>
      </div>

      {/* Save */}
      {canEdit && (
        <button
          className={styles.saveBtn}
          onClick={handleSave}
          disabled={updateMutation.isPending || !isDirty}
          style={{ marginTop: 24 }}
        >
          <FaSave /> {updateMutation.isPending ? 'Enregistrement...' : 'Enregistrer'}
        </button>
      )}
    </div>
  );
};

export default NotificationsTab;
