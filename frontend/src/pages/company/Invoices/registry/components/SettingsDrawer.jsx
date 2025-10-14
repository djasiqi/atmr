import React, { useState, useEffect } from 'react';
import styles from './SettingsDrawer.module.css';

const SettingsDrawer = ({ open, settings, onClose, onSave }) => {
  const [formData, setFormData] = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (settings) {
      setFormData({
        payment_terms_days: settings.payment_terms_days || 10,
        overdue_fee: settings.overdue_fee || 15,
        reminder1_fee: settings.reminder1_fee || 0,
        reminder2_fee: settings.reminder2_fee || 40,
        reminder3_fee: settings.reminder3_fee || 0,
        reminder_schedule_days: settings.reminder_schedule_days || { "1": 10, "2": 5, "3": 5 },
        auto_reminders_enabled: settings.auto_reminders_enabled !== false,
        email_sender: settings.email_sender || '',
        invoice_number_format: settings.invoice_number_format || '{PREFIX}-{YYYY}-{MM}-{SEQ4}',
        invoice_prefix: settings.invoice_prefix || 'EM',
        iban: settings.iban || '',
        qr_iban: settings.qr_iban || '',
        esr_ref_base: settings.esr_ref_base || '',
        invoice_message_template: settings.invoice_message_template || '',
        reminder1_template: settings.reminder1_template || '',
        reminder2_template: settings.reminder2_template || '',
        reminder3_template: settings.reminder3_template || '',
        legal_footer: settings.legal_footer || '',
        pdf_template_variant: settings.pdf_template_variant || 'default'
      });
    }
  }, [settings]);

  const handleInputChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value
    }));
  };

  const handleScheduleChange = (level, days) => {
    setFormData(prev => ({
      ...prev,
      reminder_schedule_days: {
        ...prev.reminder_schedule_days,
        [level]: parseInt(days)
      }
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    try {
      setLoading(true);
      setError(null);
      
      await onSave(formData);
    } catch (err) {
      setError(err.message || 'Erreur lors de la sauvegarde des paramètres');
    } finally {
      setLoading(false);
    }
  };

  const handleClose = () => {
    setError(null);
    onClose();
  };

  if (!open) return null;

  return (
    <div className={styles.drawerOverlay}>
      <div className={styles.drawer}>
        <div className={styles.drawerHeader}>
          <h2>Paramètres de facturation</h2>
          <button className={styles.closeBtn} onClick={handleClose}>
            ✕
          </button>
        </div>

        <div className={styles.drawerBody}>
          <form onSubmit={handleSubmit} className={styles.form}>
            {/* Délais et frais */}
            <div className={styles.section}>
              <h3>Délais et frais</h3>
              
              <div className={styles.formGroup}>
                <label htmlFor="payment_terms_days" className={styles.label}>
                  Délai de paiement (jours)
                </label>
                <input
                  type="number"
                  id="payment_terms_days"
                  name="payment_terms_days"
                  value={formData.payment_terms_days || ''}
                  onChange={handleInputChange}
                  className={styles.input}
                  min="1"
                  max="90"
                />
              </div>

              <div className={styles.formGroup}>
                <label htmlFor="overdue_fee" className={styles.label}>
                  Frais de retard (CHF)
                </label>
                <input
                  type="number"
                  id="overdue_fee"
                  name="overdue_fee"
                  value={formData.overdue_fee || ''}
                  onChange={handleInputChange}
                  className={styles.input}
                  step="0.01"
                  min="0"
                />
              </div>

              <div className={styles.feeGrid}>
                <div className={styles.formGroup}>
                  <label htmlFor="reminder1_fee" className={styles.label}>
                    Frais 1er rappel (CHF)
                  </label>
                  <input
                    type="number"
                    id="reminder1_fee"
                    name="reminder1_fee"
                    value={formData.reminder1_fee || ''}
                    onChange={handleInputChange}
                    className={styles.input}
                    step="0.01"
                    min="0"
                  />
                </div>

                <div className={styles.formGroup}>
                  <label htmlFor="reminder2_fee" className={styles.label}>
                    Frais 2e rappel (CHF)
                  </label>
                  <input
                    type="number"
                    id="reminder2_fee"
                    name="reminder2_fee"
                    value={formData.reminder2_fee || ''}
                    onChange={handleInputChange}
                    className={styles.input}
                    step="0.01"
                    min="0"
                  />
                </div>

                <div className={styles.formGroup}>
                  <label htmlFor="reminder3_fee" className={styles.label}>
                    Frais dernier rappel (CHF)
                  </label>
                  <input
                    type="number"
                    id="reminder3_fee"
                    name="reminder3_fee"
                    value={formData.reminder3_fee || ''}
                    onChange={handleInputChange}
                    className={styles.input}
                    step="0.01"
                    min="0"
                  />
                </div>
              </div>
            </div>

            {/* Planning des rappels */}
            <div className={styles.section}>
              <h3>Planning des rappels (jours)</h3>
              
              <div className={styles.scheduleGrid}>
                <div className={styles.formGroup}>
                  <label htmlFor="schedule1" className={styles.label}>
                    1er rappel après échéance
                  </label>
                  <input
                    type="number"
                    id="schedule1"
                    value={formData.reminder_schedule_days?.["1"] || ''}
                    onChange={(e) => handleScheduleChange("1", e.target.value)}
                    className={styles.input}
                    min="1"
                    max="30"
                  />
                </div>

                <div className={styles.formGroup}>
                  <label htmlFor="schedule2" className={styles.label}>
                    2e rappel après le 1er
                  </label>
                  <input
                    type="number"
                    id="schedule2"
                    value={formData.reminder_schedule_days?.["2"] || ''}
                    onChange={(e) => handleScheduleChange("2", e.target.value)}
                    className={styles.input}
                    min="1"
                    max="30"
                  />
                </div>

                <div className={styles.formGroup}>
                  <label htmlFor="schedule3" className={styles.label}>
                    3e rappel après le 2e
                  </label>
                  <input
                    type="number"
                    id="schedule3"
                    value={formData.reminder_schedule_days?.["3"] || ''}
                    onChange={(e) => handleScheduleChange("3", e.target.value)}
                    className={styles.input}
                    min="1"
                    max="30"
                  />
                </div>
              </div>

              <div className={styles.formGroup}>
                <div className={styles.checkboxGroup}>
                  <input
                    type="checkbox"
                    id="auto_reminders_enabled"
                    name="auto_reminders_enabled"
                    checked={formData.auto_reminders_enabled || false}
                    onChange={handleInputChange}
                    className={styles.checkbox}
                  />
                  <label htmlFor="auto_reminders_enabled" className={styles.checkboxLabel}>
                    Rappels automatiques activés
                  </label>
                </div>
              </div>
            </div>

            {/* Configuration */}
            <div className={styles.section}>
              <h3>Configuration</h3>
              
              <div className={styles.formGroup}>
                <label htmlFor="email_sender" className={styles.label}>
                  Email d'envoi
                </label>
                <input
                  type="email"
                  id="email_sender"
                  name="email_sender"
                  value={formData.email_sender || ''}
                  onChange={handleInputChange}
                  className={styles.input}
                  placeholder="facturation@entreprise.ch"
                />
              </div>

              <div className={styles.formGroup}>
                <label htmlFor="invoice_prefix" className={styles.label}>
                  Préfixe facture
                </label>
                <input
                  type="text"
                  id="invoice_prefix"
                  name="invoice_prefix"
                  value={formData.invoice_prefix || ''}
                  onChange={handleInputChange}
                  className={styles.input}
                  placeholder="EM"
                />
              </div>

              <div className={styles.formGroup}>
                <label htmlFor="invoice_number_format" className={styles.label}>
                  Format numérotation
                </label>
                <input
                  type="text"
                  id="invoice_number_format"
                  name="invoice_number_format"
                  value={formData.invoice_number_format || ''}
                  onChange={handleInputChange}
                  className={styles.input}
                  placeholder="{PREFIX}-{YYYY}-{MM}-{SEQ4}"
                />
              </div>
            </div>

            {/* Informations bancaires */}
            <div className={styles.section}>
              <h3>Informations bancaires</h3>
              
              <div className={styles.formGroup}>
                <label htmlFor="iban" className={styles.label}>
                  IBAN
                </label>
                <input
                  type="text"
                  id="iban"
                  name="iban"
                  value={formData.iban || ''}
                  onChange={handleInputChange}
                  className={styles.input}
                  placeholder="CH93 0076 2011 6238 5295 7"
                />
              </div>

              <div className={styles.formGroup}>
                <label htmlFor="qr_iban" className={styles.label}>
                  QR-IBAN
                </label>
                <input
                  type="text"
                  id="qr_iban"
                  name="qr_iban"
                  value={formData.qr_iban || ''}
                  onChange={handleInputChange}
                  className={styles.input}
                  placeholder="CH93 0076 2011 6238 5295 7"
                />
              </div>

              <div className={styles.formGroup}>
                <label htmlFor="esr_ref_base" className={styles.label}>
                  Référence ESR de base
                </label>
                <input
                  type="text"
                  id="esr_ref_base"
                  name="esr_ref_base"
                  value={formData.esr_ref_base || ''}
                  onChange={handleInputChange}
                  className={styles.input}
                  placeholder="1234567890123456789"
                />
              </div>
            </div>

            {error && (
              <div className={styles.error}>
                {error}
              </div>
            )}

            <div className={styles.formActions}>
              <button
                type="button"
                className={styles.cancelBtn}
                onClick={handleClose}
                disabled={loading}
              >
                Annuler
              </button>
              <button
                type="submit"
                className={styles.submitBtn}
                disabled={loading}
              >
                {loading ? 'Sauvegarde...' : 'Sauvegarder'}
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};

export default SettingsDrawer;
